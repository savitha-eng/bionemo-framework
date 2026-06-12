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

"""Honesty check on SAE-V protein<->text fusion: does it survive when the text band is restricted to the
model's REASONING/RESPONSE tokens (label != -100), excluding the fixed system-prompt prefix?

Reproduces per-token labels via the collate (no model forward needed beyond load), splits text tokens
into prompt vs response, and recomputes the cross-modal omega for the cross-modal features three ways:
all-text / prompt-text / response-text, each vs a random baseline restricted to that text subset.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--features", type=int, nargs="+", required=True)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft
    from sae.architectures import TopKSAE

    m = load_bioreason_pro_sft(ckpt_dir=DEFAULT_CKPT, bioreason_pro_root="/data/savithas/bioreason-pro",
                               device=dev, max_length_text=10000, max_length_protein=2000)
    tok = m.text_tokenizer
    pad_tok = tok.pad_token_id
    _, val, _ = brp_data.load_reasoning_splits()
    ds = val.select(range(300))
    coll = brp_data.make_collate_fn(tok, 10000, 2000)
    # per-protein: is each (unpadded) token position a RESPONSE token (label != -100)?
    pid_isresp = {}
    for i, b in enumerate(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=coll)):
        ids = b["input_ids"][0]
        lab = b["labels"][0]
        keep = ids != pad_tok
        pid_isresp[str(ds[i].get("protein_id", f"row{i}"))] = (lab[keep] != -100).cpu().numpy()

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    row_pid = tl.column("protein_id").to_pylist()
    row_tidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    band = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    # response mask aligned to store rows
    is_resp = np.zeros(len(row_pid), dtype=bool)
    for r in range(len(row_pid)):
        v = pid_isresp.get(row_pid[r])
        if v is not None and row_tidx[r] < len(v):
            is_resp[r] = v[row_tidx[r]]

    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    Z = np.concatenate([
        np.column_stack([t.column(c).to_numpy(zero_copy_only=False)
                         for c in sorted([c for c in t.column_names if c.startswith("dim_")],
                                         key=lambda c: int(c.split("_")[1]))]).astype(np.float32)
        for t in (pq.read_table(sp) for sp in shards)])
    Zt = torch.from_numpy(Z).to(dev)
    Zn = torch.nn.functional.normalize(Zt, dim=1)
    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict(ck["model_state_dict"])

    protein = np.where(band == "protein")[0]
    text_all = np.where(band == "text")[0]
    text_resp = np.where((band == "text") & is_resp)[0]
    text_prompt = np.where((band == "text") & ~is_resp)[0]
    print(f"text tokens: all={len(text_all)} prompt={len(text_prompt)} response={len(text_resp)} "
          f"({100*len(text_resp)/len(text_all):.0f}% response)")
    rng = np.random.default_rng(0)

    def baseline(textset):
        a = torch.from_numpy(rng.choice(protein, 20000)).to(dev)
        b = torch.from_numpy(rng.choice(textset, 20000)).to(dev)
        return float((Zn[a] * Zn[b]).sum(1).mean())

    feats = torch.tensor(args.features, device=dev)
    acts = np.zeros((Z.shape[0], len(args.features)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, Z.shape[0], 8192):
            e = min(Z.shape[0], s + 8192)
            acts[s:e] = sae.encode(Zt[s:e])[:, feats].cpu().numpy()
    K = args.top_k

    def omega(col, textset):
        if len(textset) < K:
            return None
        pa = protein[np.argsort(-col[protein])[:K]]
        ta = textset[np.argsort(-col[textset])[:K]]
        return float((Zn[torch.from_numpy(pa).to(dev)] * Zn[torch.from_numpy(ta).to(dev)]).sum(1).mean())

    b_all, b_resp, b_prompt = baseline(text_all), baseline(text_resp), baseline(text_prompt)
    print(f"\nbaselines: all-text={b_all:+.3f}  prompt-text={b_prompt:+.3f}  response-text={b_resp:+.3f}")
    print(f"\n{'feat':>6} | {'omega(all)':>11} {'lift':>7} | {'omega(prompt)':>14} {'lift':>7} | {'omega(RESP)':>12} {'lift':>7}")
    lifts_resp = []
    for jf, f in enumerate(args.features):
        col = acts[:, jf]
        oa, op, orr = omega(col, text_all), omega(col, text_prompt), omega(col, text_resp)
        la = oa - b_all if oa is not None else None
        lp = op - b_prompt if op is not None else None
        lr = orr - b_resp if orr is not None else None
        if lr is not None:
            lifts_resp.append(lr)
        fmt = lambda o, l: f"{o:+.3f} {l:+.3f}" if o is not None else "   n/a      "
        print(f"{f:>6} | {fmt(oa,la):>19} | {fmt(op,lp):>22} | {fmt(orr,lr):>20}")
    if lifts_resp:
        print(f"\nmean RESPONSE-text lift over baseline (n={len(lifts_resp)}): {np.mean(lifts_resp):+.3f}")
        print("=> if response-text lift ~ 0, the protein<->text 'fusion' was the prompt prefix, not reasoning.")


if __name__ == "__main__":
    main()
