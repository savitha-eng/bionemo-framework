#!/usr/bin/env python
"""Stage OBELICS parquet shards locally for robust extraction.

The extractor can read a local parquet file/dir/glob via --dataset. Staging the
needed shards first avoids long extraction runs depending on remote HF range
reads, which can fail after hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo-id", default="HuggingFaceM4/OBELICS")
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--out", default="/data/savithas/saev-llava/obelics-parquet")
    p.add_argument("--split-prefix", default="data/train-")
    p.add_argument("--count", type=int, default=1, help="Number of train parquet shards to download")
    p.add_argument("--start", type=int, default=0, help="Zero-based train shard index to start from")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = sorted(
        f
        for f in api.list_repo_files(args.repo_id, repo_type=args.repo_type)
        if f.startswith(args.split_prefix) and f.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No parquet files found in {args.repo_id} with prefix {args.split_prefix!r}")

    selected = files[args.start : args.start + args.count]
    if len(selected) < args.count:
        raise ValueError(f"Requested {args.count} shards from {args.start}, only {len(selected)} available")

    rows_total = 0
    for filename in selected:
        local = Path(
            hf_hub_download(
                args.repo_id,
                filename=filename,
                repo_type=args.repo_type,
                local_dir=out,
            )
        )
        rows = pq.read_metadata(local).num_rows
        rows_total += rows
        print(f"{local}\trows={rows}", flush=True)

    print(f"downloaded={len(selected)} rows_total={rows_total} out={out}", flush=True)


if __name__ == "__main__":
    main()
