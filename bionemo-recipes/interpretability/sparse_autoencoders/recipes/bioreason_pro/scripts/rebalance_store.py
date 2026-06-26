#!/usr/bin/env python
"""Build a MODALITY-BALANCED activation store (the fast subset test for modality balancing).

Streams a layer's shards + the aligned token_labels, DROPS the <go> graph tokens (model-unused per
the ablation), and DOWNSAMPLES text tokens to ~1:1 with protein -> a balanced protein/text store. Train
an SAE on it and compare per-band feature coverage / protein AUC vs the text-dominated baseline.
Writes OUT/layer{L}/shard_*.parquet (+ metadata.json) and OUT/token_labels.parquet (+ proteins.parquet).
Usage: rebalance_store.py --store <full_store> --layer 24 --out <dir> [--text-keep auto|<p>]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae.activation_store import shard_table_to_array

SHARD = 200_000


def _write_shard(X, path):  # X: (n, 2560) float32 -> FixedSizeList parquet
    flat = pa.array(X.reshape(-1), type=pa.float32())
    arr = pa.FixedSizeListArray.from_arrays(flat, 2560)
    pq.write_table(pa.table({"act": arr}), path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True); p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--text-keep", default="auto", help="'auto' (balance text to bio count) or a float prob")
    p.add_argument("--drop-go", action="store_true", help="also drop <go> tokens (default: keep them)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    pos = tl.column("position_type").to_numpy(zero_copy_only=False)   # to_numpy >> to_pylist on 421M rows
    pid = tl.column("protein_id").to_numpy(zero_copy_only=False)
    n_prot = int((pos == "protein").sum()); n_go = int((pos == "go").sum()); n_text = int((pos == "text").sum())
    bio = n_prot + (0 if args.drop_go else n_go)   # balance text against the bio tokens we keep
    p_text = (bio / n_text) if args.text_keep == "auto" else float(args.text_keep)
    print(f"[rebal] protein={n_prot:,} go={n_go:,} ({'drop' if args.drop_go else 'keep'}) text={n_text:,} | "
          f"text-keep-prob={p_text:.3f} -> ~{int(n_text*p_text):,} text (bio≈text balanced)")

    outL = Path(args.out) / f"layer{args.layer}"; outL.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    buf_X, buf_pid, buf_pt = [], [], []   # accumulate numpy chunks (vectorized, not per-row Python)
    n_out_shards = [0]; kept = {"protein": 0, "go": 0, "text": 0}; row0 = 0

    def flush(force=False):
        if not buf_X:
            return
        X = np.concatenate(buf_X); P = np.concatenate(buf_pid); T = np.concatenate(buf_pt)
        buf_X.clear(); buf_pid.clear(); buf_pt.clear()
        off = 0
        while X.shape[0] - off >= SHARD or (force and X.shape[0] - off > 0):
            take = min(SHARD, X.shape[0] - off)
            _write_shard(X[off:off + take], outL / f"shard_{n_out_shards[0]:05d}.parquet")
            n_out_shards[0] += 1; off += take
            if force and off >= X.shape[0]:
                break
        if off < X.shape[0]:                      # carry remainder
            buf_X.append(X[off:]); buf_pid.append(P[off:]); buf_pt.append(T[off:])

    for sp in shards:
        X = shard_table_to_array(pq.read_table(sp)).astype(np.float32)   # (n,2560) vectorized
        n = X.shape[0]; band = pos[row0:row0 + n]; pids = pid[row0:row0 + n]
        text_keep = (band == "text") & (rng.random(n) < p_text)            # downsample text
        if args.drop_go:
            keep = (band == "protein") | text_keep                        # drop go
        else:
            keep = (band != "text") | text_keep                           # keep protein + go
        buf_X.append(X[keep]); buf_pid.append(pids[keep]); buf_pt.append(band[keep])
        for b in ("protein", "go", "text"):
            kept[b] += int((band[keep] == b).sum())
        flush()
        row0 += n
    flush(force=True)
    buf_pid_all = np.concatenate(buf_pid) if buf_pid else np.array([], dtype=object)
    buf_pt_all = np.concatenate(buf_pt) if buf_pt else np.array([], dtype=object)

    # sidecars
    pq.write_table(pa.table({"protein_id": pa.array(buf_pid), "token_index": pa.array(np.arange(len(buf_pid))),
                             "position_type": pa.array(buf_pt)}), Path(args.out) / "token_labels.parquet")
    upids = sorted(set(buf_pid))
    pq.write_table(pa.table({"protein_id": pa.array(upids)}), Path(args.out) / "proteins.parquet")
    n_samples = sum(kept.values())
    ptypes = "|".join(b for b in ("protein", "go", "text") if kept[b] > 0)
    json.dump({"model": "bioreason-pro-sft", "split": "train", "position_types": ptypes,
               "layer": args.layer, "hidden_dim": 2560, "n_samples": n_samples,
               "n_shards": n_out_shards[0], "balanced": True, "drop_go": args.drop_go, "kept": kept},
              open(outL / "metadata.json", "w"), indent=2)
    print(f"[rebal] wrote {n_out_shards[0]} shards, {n_samples:,} tokens (kept={kept}) -> {args.out}")


if __name__ == "__main__":
    main()
