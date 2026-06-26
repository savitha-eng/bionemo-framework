#!/usr/bin/env python
"""Test whether cross-modal features are UNDERSAMPLED by the small eval subset.

Streams a chosen number of shards (no Z accumulation -> no OOM), encodes with the SAE, and accumulates
per-feature per-band activation MASS (protein/go/text). Reports, as a function of #proteins:
  - how many features fire at all
  - how many are "balanced" cross-modal (both bands' mass fraction > thresh)
If these counts rise sharply with more proteins, the 555-protein dashboards were undersampling fusion.
Usage: crossmodal_scaling.py --sae <ckpt> --store <full_store> --layer L --shards N
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import sys
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--shards", type=int, default=10)
    p.add_argument("--batch", type=int, default=16384)
    args = p.parse_args()
    dev = "cuda"

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    row_band = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    row_pid = np.asarray(tl.column("protein_id").to_pylist(), dtype=object)

    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:args.shards]

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim

    BANDS = ["protein", "go", "text"]
    mass = {b: torch.zeros(H, device=dev) for b in BANDS}
    fire = torch.zeros(H, device=dev)
    row0 = 0
    seen_pids = set()
    with torch.no_grad():
        for sp in shards:
            X = shard_table_to_array(pq.read_table(sp)).astype(np.float32)
            n = X.shape[0]
            rb = row_band[row0:row0 + n]
            seen_pids.update(row_pid[row0:row0 + n].tolist())
            for s in range(0, n, args.batch):
                e = min(n, s + args.batch)
                codes = sae.encode(torch.from_numpy(X[s:e]).to(dev))
                fire += (codes > 0).sum(0).float()
                rbs = rb[s:e]
                for b in BANDS:
                    m = torch.from_numpy((rbs == b)).to(dev)
                    if m.any():
                        mass[b] += codes[m].sum(0)
            row0 += n

    pf = (mass["protein"] / (mass["protein"] + mass["go"] + mass["text"] + 1e-9)).cpu().numpy()
    gf = (mass["go"] / (mass["protein"] + mass["go"] + mass["text"] + 1e-9)).cpu().numpy()
    tf = (mass["text"] / (mass["protein"] + mass["go"] + mass["text"] + 1e-9)).cpu().numpy()
    fire = fire.cpu().numpy()
    active = fire > 0
    pt = ((pf > 0.2) & (tf > 0.2) & active).sum()
    pg = ((pf > 0.2) & (gf > 0.2) & active).sum()
    print(f"[scaling] shards={args.shards}  proteins={len(seen_pids)}  features_firing={int(active.sum())}/{H}")
    print(f"[scaling]   balanced protein+text (both mass-frac>0.2): {int(pt)}")
    print(f"[scaling]   balanced protein+go  (both mass-frac>0.2): {int(pg)}")


if __name__ == "__main__":
    main()
