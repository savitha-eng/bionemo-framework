#!/usr/bin/env python
"""Make a small dashboard-sized subset store from a big full store (file-level, NO re-extraction).

build_dashboard sizes per-protein arrays as (n_proteins, H), so it can't run on a 117k-protein full
store (OOM). This copies the first M shards of each requested layer + the row-aligned token_labels +
the covered proteins into a new store, so build_dashboard/dashboard.py run cheaply and stay aligned.
Usage: make_dash_subset.py --src <full_store> --dst <out_store> --layers 16,18,22 --shards 145
"""
import argparse, glob, json, shutil
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--layers", required=True); p.add_argument("--shards", type=int, default=145)
    args = p.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    layers = [int(x) for x in args.layers.split(",")]
    dst.mkdir(parents=True, exist_ok=True)

    # 1. copy the first --shards shards of each layer; track row count from the FIRST layer
    n_rows = None
    for L in layers:
        sh = sorted(glob.glob(str(src / f"layer{L}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:args.shards]
        (dst / f"layer{L}").mkdir(parents=True, exist_ok=True)
        rows = 0
        for s in sh:
            shutil.copy(s, dst / f"layer{L}" / Path(s).name)
            rows += pq.read_metadata(s).num_rows
        if n_rows is None:
            n_rows = rows
        elif rows != n_rows:
            raise SystemExit(f"layer{L} rows {rows} != layer{layers[0]} rows {n_rows} (stores misaligned)")
        # copy metadata.json, fix n_samples/n_shards
        meta = json.load(open(src / f"layer{L}" / "metadata.json"))
        meta["n_samples"] = rows; meta["n_shards"] = len(sh)
        json.dump(meta, open(dst / f"layer{L}" / "metadata.json", "w"), indent=2)
    print(f"[subset] copied {len(layers)} layers x {args.shards} shards = {n_rows:,} rows each")

    # 2. truncate token_labels (+ role sidecar) to n_rows
    for f in ("token_labels.parquet", "token_labels_with_role.parquet"):
        if (src / f).exists():
            t = pq.read_table(src / f).slice(0, n_rows)
            pq.write_table(t, dst / f)
    tl = pq.read_table(dst / "token_labels.parquet")
    covered = list(dict.fromkeys(tl.column("protein_id").to_pylist()))  # proteins in the kept rows, in order
    print(f"[subset] token_labels -> {n_rows:,} rows covering {len(covered)} proteins")

    # 3. proteins.parquet -> only covered proteins (preserve column schema/order)
    prot = pq.read_table(src / "proteins.parquet")
    pid_col = prot.column("protein_id").to_pylist()
    keep_idx = [i for i, pid in enumerate(pid_col) if pid in set(covered)]
    pq.write_table(prot.take(pa.array(keep_idx)), dst / "proteins.parquet")
    print(f"[subset] proteins.parquet -> {len(keep_idx)} proteins  | wrote {dst}")


if __name__ == "__main__":
    main()
