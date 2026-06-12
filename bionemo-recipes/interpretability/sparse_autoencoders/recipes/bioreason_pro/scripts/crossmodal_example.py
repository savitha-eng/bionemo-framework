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

"""Concrete SAE-V example: for a few cross-modal features, show their top PROTEIN-band tokens (which
proteins + GO terms) and top TEXT-band tokens (decoded reasoning), and the paired cosine vs the random
baseline that defines 'fused'. (Env A — needs the tokenizer to decode text windows.)
"""

import argparse
import glob
import json
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
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--window", type=int, default=10)
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
    pid_tok, gid_tok, pad_tok = m.protein_token_id, m.go_token_id, tok.pad_token_id
    _, val, _ = brp_data.load_reasoning_splits()
    ds = val.select(range(300))
    coll = brp_data.make_collate_fn(tok, 10000, 2000)
    pid_ids = {}
    for i, b in enumerate(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=coll)):
        row = b["input_ids"][0]
        pid_ids[str(ds[i].get("protein_id", f"row{i}"))] = row[row != pad_tok].tolist()

    # row meta + GO ids per protein
    prot = pq.read_table(Path(args.store) / "proteins.parquet")
    go_of = {pid: json.loads(g) for pid, g in zip(prot.column("protein_id").to_pylist(),
                                                  prot.column("go_ids").to_pylist())}
    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    row_pid = tl.column("protein_id").to_pylist()
    row_tidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    band = np.asarray(tl.column("position_type").to_pylist(), dtype=object)

    # activations into RAM (for cosine) + SAE
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    Z = []
    for sp in shards:
        t = pq.read_table(sp)
        cols = sorted([c for c in t.column_names if c.startswith("dim_")], key=lambda c: int(c.split("_")[1]))
        Z.append(np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32))
    Z = np.concatenate(Z)
    Zt = torch.from_numpy(Z).to(dev)
    Znorm = torch.nn.functional.normalize(Zt, dim=1)
    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict(ck["model_state_dict"])

    prot_idx = np.where(band == "protein")[0]
    text_idx = np.where(band == "text")[0]
    rng = np.random.default_rng(0)
    # random baseline: mean cosine of random protein-text token pairs
    ra = torch.from_numpy(rng.choice(prot_idx, 20000)).to(dev)
    rb = torch.from_numpy(rng.choice(text_idx, 20000)).to(dev)
    base = float((Znorm[ra] * Znorm[rb]).sum(1).mean())

    # feature activations (encode in batches, keep only target feature columns)
    feats = torch.tensor(args.features, device=dev)
    acts = np.zeros((Z.shape[0], len(args.features)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, Z.shape[0], 8192):
            e = min(Z.shape[0], s + 8192)
            acts[s:e] = sae.encode(Zt[s:e])[:, feats].cpu().numpy()

    def decode_win(r):
        ids = pid_ids.get(row_pid[r]); ti = int(row_tidx[r])
        if ids is None:
            return "?"
        lo, hi = max(0, ti - args.window), min(len(ids), ti + args.window + 1)
        out = []
        for j in range(lo, hi):
            t = ids[j]
            piece = "<protein>" if t == pid_tok else "<go>" if t == gid_tok else tok.decode([t])
            out.append(f"⟦{piece.strip()}⟧" if j == ti else piece)
        return "".join(out).replace("\n", " ")

    print(f"\nrandom protein↔text baseline cosine = {base:+.3f}\n" + "=" * 70)
    K = args.top_k
    for jf, f in enumerate(args.features):
        col = acts[:, jf]
        pa = prot_idx[np.argsort(-col[prot_idx])[:K]]
        ta = text_idx[np.argsort(-col[text_idx])[:K]]
        za = Znorm[torch.from_numpy(pa).to(dev)]
        zb = Znorm[torch.from_numpy(ta).to(dev)]
        omega = float((za * zb).sum(1).mean())
        print(f"\nFEATURE {f}  |  paired cosine={omega:+.3f}  baseline={base:+.3f}  LIFT={omega-base:+.3f}")
        print(f"  top PROTEIN-band tokens (protein → its GO terms):")
        for r in pa[:5]:
            print(f"    [{col[r]:.1f}] {row_pid[r]}  GO={go_of.get(row_pid[r], [])[:4]}")
        print(f"  top TEXT-band tokens (decoded reasoning window):")
        for r in ta[:5]:
            print(f"    [{col[r]:.1f}] {row_pid[r]}: …{decode_win(r)}…")


if __name__ == "__main__":
    main()
