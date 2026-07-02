#!/usr/bin/env python
"""Write a modality-BALANCED copy of a DNA activation store ONCE, so SAE training reads a small store
instead of re-filtering the full 151M-token store every epoch (the naive --balance-modality path re-reads
all shards each epoch -> ~15x wasted NFS I/O + stalls). Keeps ALL text tokens + downsamples dna to hit the
target bio fraction. Row-aligned token_labels are filtered identically and rewritten.

Usage: pre_materialize_balanced.py --src <store> --dst <store> --layer 16 --bio-frac 0.7 --bio-band dna
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sae" / "src"))
from sae.activation_store import shard_table_to_array

SHARD = 200000


def to_fsl(arr, dim):
    flat = pa.array(arr.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--bio-frac", type=float, default=0.7); p.add_argument("--bio-band", default="dna")
    p.add_argument("--seed", type=int, default=23)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    (dst / f"layer{a.layer}").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    tl = pq.read_table(src / "token_labels.parquet")
    band = tl.column("position_type").to_numpy(zero_copy_only=False)
    N = len(band); is_bio = band == a.bio_band
    n_b, n_t = int(is_bio.sum()), int((~is_bio).sum())
    bio_keep = min(1.0, (a.bio_frac / (1 - a.bio_frac)) * (n_t / n_b))  # downsample the majority bio band
    keep = np.ones(N, bool)
    keep[is_bio] = rng.random(n_b) < bio_keep
    print(f"[premat] {a.bio_band}={n_b:,} text={n_t:,} bio_keep={bio_keep:.4f} -> keep {int(keep.sum()):,}/{N:,} "
          f"({int(keep[is_bio].sum()):,} {a.bio_band} + {n_t:,} text)", flush=True)

    shards = sorted(glob.glob(str(src / f"layer{a.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    dim = json.load(open(src / f"layer{a.layer}" / "metadata.json"))["hidden_dim"]
    buf = []; buf_rows = 0; row0 = 0; out_i = 0; total = 0
    keep_idx_all = []  # global row indices kept (to filter token_labels)

    def flush(final=False):
        nonlocal buf, buf_rows, out_i
        while buf_rows >= SHARD or (final and buf_rows > 0):
            cat = np.concatenate(buf) if len(buf) > 1 else buf[0]
            take = min(SHARD, cat.shape[0]); part, rest = cat[:take], cat[take:]
            pq.write_table(pa.table({"act": to_fsl(part, dim)}), dst / f"layer{a.layer}" / f"shard_{out_i:05d}.parquet")
            out_i += 1; buf = [rest] if rest.shape[0] else []; buf_rows = rest.shape[0]
            if final and buf_rows == 0: break

    for sp in shards:
        X = shard_table_to_array(pq.read_table(sp)).astype(np.float32); n = X.shape[0]
        m = keep[row0:row0 + n]
        gi = np.nonzero(m)[0] + row0; keep_idx_all.append(gi)
        row0 += n
        if m.any():
            buf.append(X[m]); buf_rows += int(m.sum()); total += int(m.sum())
            if buf_rows >= SHARD:
                flush()
    flush(final=True)

    kept_global = np.concatenate(keep_idx_all)
    # cast string columns to large_string first: take() on >few-M rows overflows 32-bit string offsets
    tl_ls = pa.table({n: (tl.column(n).cast(pa.large_string()) if pa.types.is_string(tl.column(n).type)
                          else tl.column(n)) for n in tl.column_names})
    tl_kept = tl_ls.take(pa.array(kept_global))
    pq.write_table(tl_kept, dst / "token_labels.parquet")
    meta = json.load(open(src / f"layer{a.layer}" / "metadata.json"))
    meta.update({"n_samples": int(total), "n_shards": out_i,
                 "note": f"modality-balanced ({a.bio_band}-frac={a.bio_frac}) pre-materialized from {src.name}"})
    json.dump(meta, open(dst / f"layer{a.layer}" / "metadata.json", "w"), indent=2)
    print(f"[premat] wrote {total:,} tokens over {out_i} shards -> {dst}", flush=True)


if __name__ == "__main__":
    main()
