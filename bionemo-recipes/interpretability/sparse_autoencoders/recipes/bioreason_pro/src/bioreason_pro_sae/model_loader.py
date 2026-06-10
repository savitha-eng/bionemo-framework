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

"""Load the BioReason-Pro SFT model as a *hookable* HuggingFace tree for SAE extraction.

This is the faithful fused-forward path scoped in Phase 2: the text model is a plain
``AutoModelForCausalLM`` (so we can register a forward hook on ``text_model.model.layers[L]``),
real ESM3 protein embeddings + the trained GO band are spliced into the placeholder token slots
by the authors' own ``protein_llm.ProteinLLMModel.forward``.

We import and reuse the authors' model code unchanged. Two small, NON-invasive glue steps live
here on the SAE side (no edits to the BioReason-Pro repo):

1. A ``sys.modules`` stub for ``unsloth`` — ``bioreason2/models/protein_llm.py`` does a top-level
   ``from unsloth import FastLanguageModel`` which is only *used* on the ``use_unsloth=True``
   training path. The inference env (Env A) does not install unsloth, so we register a dummy
   module before import. ``FastLanguageModel`` is never called on our ``use_unsloth=False`` path.
2. The GO band: ``predict.py`` (production inference) builds the model with
   ``precomputed_embeddings_path=None`` and relies on the cached ``go_embedding.pt`` instead of
   running the GO graph encoder. The authors' *training* ``ProteinLLMModel`` only creates
   ``go_encoder``/``go_projection`` when both GO paths are given, so we construct it without GO,
   then attach ``go_projection`` (weights from ``go_projection.pt``) and patch
   ``process_go_aspects`` to return ``go_projection(cached_go_embedding)`` — identical to the vLLM
   path (``protein_vllm.process_go_aspects`` applies the projection *after* the cache lookup).
"""

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


# Faithful arch config — mirrors predict.py MODEL_ARCH (the production loader).
DEFAULT_PROTEIN_MODEL_NAME = "esm3_sm_open_v1"
DEFAULT_PROTEIN_EMBEDDING_LAYER = 37  # ESM3-internal layer (NOT the SAE hook layer)
DEFAULT_GO_EMBEDDING_DIM = 2560
DEFAULT_NUM_GO_TOKENS = 200
DEFAULT_MAX_LENGTH_TEXT = 10000
DEFAULT_MAX_LENGTH_PROTEIN = 2000

# Special-token ids in the SFT tokenizer (for position_type tagging downstream).
PROTEIN_PAD_TOKEN = "<|protein_pad|>"
GO_PAD_TOKEN = "<|go_graph_pad|>"


def _install_unsloth_stub() -> None:
    """Register a dummy ``unsloth`` module so importing protein_llm succeeds without unsloth.

    Only matters for the top-level ``from unsloth import FastLanguageModel``; the symbol is never
    invoked on the ``use_unsloth=False`` path we use.
    """
    if "unsloth" in sys.modules:
        return
    stub = types.ModuleType("unsloth")
    stub.FastLanguageModel = None  # never called when use_unsloth=False
    sys.modules["unsloth"] = stub


def _ensure_bioreason_on_path(bioreason_pro_root: str) -> None:
    """Put the BioReason-Pro repo root on sys.path so ``import bioreason2`` works."""
    root = str(Path(bioreason_pro_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_bioreason_pro_sft(
    ckpt_dir: str,
    bioreason_pro_root: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    protein_model_name: str = DEFAULT_PROTEIN_MODEL_NAME,
    protein_embedding_layer: int = DEFAULT_PROTEIN_EMBEDDING_LAYER,
    max_length_text: int = DEFAULT_MAX_LENGTH_TEXT,
    max_length_protein: int = DEFAULT_MAX_LENGTH_PROTEIN,
    attn_implementation: str = "flash_attention_2",
    cache_dir: str | None = None,
):
    """Build the faithful, hookable BioReason-Pro SFT model.

    Args:
        ckpt_dir: Path to the SFT checkpoint snapshot dir (contains ``model-*.safetensors``,
            ``config.json``, ``protein_projection.pt``, ``go_projection.pt``, ``go_embedding.pt``).
        bioreason_pro_root: Path to the BioReason-Pro repo (for ``import bioreason2``).
        device: Device to place the model on.
        dtype: Compute dtype for the text model + projections (bf16, matching training).
        protein_model_name: ESM3 model id (default ``esm3_sm_open_v1``).
        protein_embedding_layer: ESM3 internal layer to read protein embeddings from (37).
        max_length_text: Text cap used at training (SFT used 10000).
        max_length_protein: Protein cap used at training (2000).
        attn_implementation: HF attention impl (flash_attention_2, as in the Phase-2 gate).
        cache_dir: HF cache dir; defaults to ``HF_HOME`` env if unset.

    Returns:
        A ``ProteinLLMModel`` in eval mode on ``device`` with all custom components loaded and the
        GO band wired via the cached embedding. The hookable LLM tree is at
        ``model.text_model.model.layers``; ``model.protein_token_id`` / ``model.go_token_id`` give
        the placeholder ids for position_type tagging.
    """
    _install_unsloth_stub()
    _ensure_bioreason_on_path(bioreason_pro_root)

    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME")

    from bioreason2.models.protein_llm import ProteinLLMModel  # noqa: E402 (after stub + path)

    # Build WITHOUT GO paths -> go_encoder/go_projection are None; text weights load from ckpt_dir.
    model = ProteinLLMModel(
        text_model_name=ckpt_dir,
        protein_model_name=protein_model_name,
        cache_dir=cache_dir,
        max_length_protein=max_length_protein,
        max_length_text=max_length_text,
        text_model_finetune=False,
        protein_model_finetune=False,
        go_model_finetune=False,
        protein_embedding_layer=protein_embedding_layer,
        attn_implementation=attn_implementation,
        go_obo_path=None,
        precomputed_embeddings_path=None,
        go_embedding_dim=DEFAULT_GO_EMBEDDING_DIM,
        unified_go_encoder=True,
        use_unsloth=False,
    )

    text_hidden = model.text_config.hidden_size

    # ---- Load protein projection (1536 -> 2560 -> 2560) ----
    proj_path = os.path.join(ckpt_dir, "protein_projection.pt")
    if not os.path.exists(proj_path):
        raise FileNotFoundError(f"protein_projection.pt missing in {ckpt_dir}")
    model.protein_projection.load_state_dict(torch.load(proj_path, map_location="cpu"), strict=True)

    # ---- Attach GO projection + cached GO band (matches predict.py / protein_vllm) ----
    go_proj_path = os.path.join(ckpt_dir, "go_projection.pt")
    go_emb_path = os.path.join(ckpt_dir, "go_embedding.pt")
    if not (os.path.exists(go_proj_path) and os.path.exists(go_emb_path)):
        raise FileNotFoundError(f"go_projection.pt / go_embedding.pt missing in {ckpt_dir}")

    go_projection = nn.Sequential(
        nn.Linear(DEFAULT_GO_EMBEDDING_DIM, text_hidden),
        nn.GELU(),
        nn.Linear(text_hidden, text_hidden),
    )
    go_projection.load_state_dict(torch.load(go_proj_path, map_location="cpu"), strict=True)
    model.go_projection = go_projection

    cached_go = torch.load(go_emb_path, map_location="cpu")  # (200, 2560), pre-projection
    model._cached_go_embedding = cached_go

    def process_go_aspects(go_aspects, batch_size):
        """Return the trained GO band per batch item (cached embedding -> go_projection)."""
        if go_aspects is None:
            return None
        proj_w = model.go_projection[0].weight
        band = model._cached_go_embedding.to(device=proj_w.device, dtype=proj_w.dtype)
        band = model.go_projection(band)  # (200, text_hidden)
        return [band for _ in range(batch_size)]

    model.process_go_aspects = process_go_aspects  # bound override (matches authors' signature)

    # ---- Place on device + eval ----
    model.text_model.to(device=device, dtype=dtype)
    model.protein_projection.to(device=device, dtype=dtype)
    model.go_projection.to(device=device, dtype=dtype)
    # ESM3 protein encoder: move underlying model to device (keep its native dtype).
    try:
        model.protein_model.to(device)
    except Exception:
        pass
    model.eval()
    return model
