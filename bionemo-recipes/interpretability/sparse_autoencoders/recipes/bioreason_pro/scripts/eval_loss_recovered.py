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

r"""CE-based loss-recovered for a trained BioReason-Pro SAE (Env A — loads the model).

loss-recovered fundamentally needs an LLM forward with the SAE reconstruction substituted into the
residual stream, so unlike the other eval metrics it runs in Env A (the model env). On a small
held-out set (validation split), for each batch we run the faithful fused forward three ways:
  * ce_original : unmodified residual at layer L
  * ce_sae      : layer-L residual replaced by sae.decode(sae.encode(h))
  * ce_zero     : layer-L residual replaced by zeros (zero-ablation baseline)
loss_recovered = 1 - (ce_sae - ce_original) / (ce_zero - ce_original).

Usage (Env A, GPU 3):
    CUDA_VISIBLE_DEVICES=3 python scripts/eval_loss_recovered.py \
        --sae /data/.../sae_layer28/checkpoint.pt --layer 28 \
        --split validation --num-proteins 150 --out-json /data/.../loss_recovered_layer28.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_RECIPE_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_RECIPE_SRC))

DEFAULT_CKPT = (
    "/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
    "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b"
)
DEFAULT_BIOREASON_ROOT = "/data/savithas/bioreason-pro"


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="CE loss-recovered for a BioReason-Pro SAE")
    p.add_argument("--sae", required=True, help="Trained SAE checkpoint (.pt)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default=DEFAULT_BIOREASON_ROOT)
    p.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    p.add_argument("--num-proteins", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def load_sae(path, device):
    from sae.architectures import TopKSAE

    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("model_config") or {k: ckpt[k] for k in ("input_dim", "hidden_dim", "top_k") if k in ckpt}
    sae = TopKSAE(**cfg)
    sae.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return sae.to(device).eval()


def main():  # noqa: D103
    args = parse_args()
    torch.manual_seed(args.seed)

    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft

    model = load_bioreason_pro_sft(
        ckpt_dir=args.ckpt_dir, bioreason_pro_root=args.bioreason_root, device=args.device,
        max_length_text=args.max_length_text, max_length_protein=args.max_length_protein,
    )
    sae = load_sae(args.sae, args.device)
    layer_module = model.text_model.model.layers[args.layer]

    # Hook modes: None (passthrough), "sae", "zero". A forward hook may return a replacement output.
    mode = {"v": None}

    def hook(module, inp, out):
        if mode["v"] is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        if mode["v"] == "zero":
            h2 = torch.zeros_like(h)
        else:  # "sae"
            dtype = h.dtype
            recon, _ = sae(h.reshape(-1, h.shape[-1]).float())
            h2 = recon.reshape(h.shape).to(dtype)
        return (h2, *out[1:]) if isinstance(out, tuple) else h2

    handle = layer_module.register_forward_hook(hook)

    train_ds, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=args.max_length_protein)
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[args.split]
    ds = ds.select(range(min(args.num_proteins, len(ds))))
    collate_fn = brp_data.make_collate_fn(model.text_tokenizer, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sums = {"original": 0.0, "sae": 0.0, "zero": 0.0}
    n_batches = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="loss-recovered"):
            kw = dict(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                protein_sequences=batch.get("protein_sequences"),
                batch_idx_map=batch.get("batch_idx_map"),
                structure_coords=batch.get("structure_coords"),
                labels=batch["labels"].to(args.device),
                go_aspects=batch.get("batch_go_aspects"),
            )
            for m in ("original", "sae", "zero"):
                mode["v"] = None if m == "original" else m
                out = model(**kw)
                sums[m] += float(out.loss.item())
            mode["v"] = None
            n_batches += 1

    handle.remove()
    ce = {m: sums[m] / max(1, n_batches) for m in sums}
    denom = ce["zero"] - ce["original"]
    loss_recovered = 1.0 - (ce["sae"] - ce["original"]) / (denom + 1e-8)
    report = {
        "sae": args.sae, "layer": args.layer, "split": args.split, "n_proteins": int(len(ds)),
        "ce_original": round(ce["original"], 4), "ce_sae": round(ce["sae"], 4), "ce_zero": round(ce["zero"], 4),
        "loss_recovered": round(float(loss_recovered), 4), "wall_time_s": round(time.time() - t0, 1),
    }
    print(json.dumps(report, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"[loss-recovered] wrote {args.out_json}")


if __name__ == "__main__":
    main()
