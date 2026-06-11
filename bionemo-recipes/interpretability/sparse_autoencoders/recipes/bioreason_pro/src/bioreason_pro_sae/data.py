# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data wiring for BioReason-Pro SAE extraction.

We IMPORT and CALL the authors' pipeline unchanged (``load_cafa5_dataset`` +
``qwen_protein_collate_fn``). The only SAE-side glue here:

* ``wanglab/cafa5`` is gated; the SFT reasoning split is published separately as the (ungated)
  repo ``wanglab/bioreason-pro-sft-reasoning-data``. We monkeypatch the ``load_dataset`` symbol
  *inside the authors' load module* to redirect the gated path to that repo. The processing
  (prompt formatting, GO-aspect tagging, splitting) runs unchanged.
* ``interpro_dataset_name=None`` skips the gated InterPro-metadata fetch. With
  ``interpro_in_prompt=True`` the reasoning prompt uses each row's pre-formatted
  ``interpro_formatted`` field, so the prompt is identical to the SFT run.

Faithful SFT data config (from scripts/sh_train_protein_qwen_staged.sh):
reasoning dataset, go_gpt_predictions_column="go_pred", add_uniprot_summary=True,
interpro_in_prompt=True, ppi_in_prompt=True, include_protein_function_summary=True,
split_go_aspects=False, val_split_ratio=0.1, seed=23, max_length(protein)=2000.
"""

from functools import partial

import numpy as np


REASONING_DATASET_NAME = "bioreason-pro-sft-reasoning-data"
REASONING_DATASET_REPO = "wanglab/bioreason-pro-sft-reasoning-data"
GATED_DATASET = "wanglab/cafa5"

# Faithful SFT prompt-shaping config.
SFT_DATA_KWARGS = dict(
    reasoning_dataset_name=REASONING_DATASET_NAME,
    interpro_dataset_name=None,  # rows carry interpro_formatted; skip gated metadata fetch
    val_split_ratio=0.1,
    seed=23,
    return_as_chat_template=True,
    structure_dir=None,
    split_go_aspects=False,
    interpro_in_prompt=True,
    ppi_in_prompt=True,
    include_protein_function_summary=True,
    add_uniprot_summary=True,
    is_swissprot=False,
    go_gpt_predictions_column="go_pred",
    include_ground_truth_in_final_answer=False,
)


def _install_dataset_redirect(load_module) -> None:
    """Redirect the gated ``wanglab/cafa5`` reasoning load to the ungated standalone repo."""
    if getattr(load_module, "_sae_redirect_installed", False):
        return
    orig = load_module.load_dataset

    def patched(path, name=None, **kw):
        if path == GATED_DATASET and name == REASONING_DATASET_NAME:
            kw.pop("dataset_subset", None)
            return orig(REASONING_DATASET_REPO, **kw)
        return orig(path, name=name, **kw)

    load_module.load_dataset = patched
    load_module._sae_redirect_installed = True


def load_reasoning_splits(max_length_protein: int = 2000, debug: bool = False, include_go_pred: bool = True):
    """Return (train, val, test) HF datasets in collate-ready form (faithful SFT processing).

    Requires the BioReason-Pro repo importable (caller installs the unsloth stub + sys.path via
    ``model_loader`` first, or imports this only inside that context).

    ``include_go_pred=False`` drops the GO-GPT predictions ("go_speculations") from the prompt — used
    by the fusion-control experiment to test whether the text band encodes GO *beyond* the handed-in
    predictions.
    """
    import bioreason2.dataset.cafa5.load as load_module

    _install_dataset_redirect(load_module)
    kwargs = dict(SFT_DATA_KWARGS)
    if not include_go_pred:
        kwargs["go_gpt_predictions_column"] = None
    return load_module.load_cafa5_dataset(
        dataset=GATED_DATASET,
        max_length=max_length_protein,
        debug=debug,
        **kwargs,
    )


def make_collate_fn(tokenizer, max_length_text: int = 10000, max_length_protein: int = 2000,
                    inference_mode: bool = False):
    """Build the authors' collate fn with a fresh PLProcessor.

    ``inference_mode=True`` truncates after the assistant-start marker (drops the reasoning/answer
    tokens) — used by the fusion control so the text band has no explicit model-stated GO terms.
    """
    from bioreason2.dataset.cafa5.collate import qwen_protein_collate_fn
    from bioreason2.models.pl.processing_pl import PLProcessor

    processor = PLProcessor(tokenizer=tokenizer)
    return partial(
        qwen_protein_collate_fn,
        processor=processor,
        max_length_text=max_length_text,
        max_length_protein=max_length_protein,
        return_answer_in_batch=True,
        inference_mode=inference_mode,
    )


# Per-token modality tags. protein_pad/go_graph_pad ids come from the model.
TAG_PROTEIN = "protein"
TAG_GO = "go"
TAG_TEXT = "text"


def tag_and_keep_mask(input_ids_row, protein_token_id: int, go_token_id: int, pad_token_id: int):
    """Tag each token (protein/go/text) and return a keep-mask dropping pad tokens.

    Args:
        input_ids_row: 1-D LongTensor of token ids for one sequence (left-padded).
        protein_token_id / go_token_id / pad_token_id: special ids.

    Returns:
        (tags, keep) numpy arrays of equal length. ``tags`` holds the position_type string for
        every non-pad token (length = keep.sum()); ``keep`` is a boolean mask over the full row.
    """
    ids = input_ids_row.detach().cpu().numpy()
    keep = ids != pad_token_id
    kept = ids[keep]
    tags = np.full(kept.shape, TAG_TEXT, dtype=object)
    tags[kept == protein_token_id] = TAG_PROTEIN
    tags[kept == go_token_id] = TAG_GO
    return tags, keep
