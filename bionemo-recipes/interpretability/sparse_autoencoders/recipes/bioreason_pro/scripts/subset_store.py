# Build a matched multi-layer SMALL subset store from the full store, by copying the first K shards of
# each layer + slicing the shared sidecar. All layers share the same proteins (shards are in the same
# protein order across layers) -> a clean, comparable per-layer store for the expansion experiments.
# No GPU, no model — just file copy + parquet slicing. Writes to shared /data so any pod can train on it.
#
#   python scripts/subset_store.py --full <full_store> --out <dst> --layers 24,26,28,30,32 --shards 145
import argparse, json, shutil
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="24,26,28,30,32")
    ap.add_argument("--shards", type=int, default=145, help="first K shards (~200k tokens each)")
    a = ap.parse_args()
    full = Path(a.full); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    K = a.shards

    R = None
    for L in layers:
        src = full / f"layer{L}"; dst = out / f"layer{L}"; dst.mkdir(exist_ok=True)
        rows = 0
        for i in range(K):
            sp = src / f"shard_{i:05d}.parquet"
            if not sp.exists():
                print(f"  WARN missing {sp}"); break
            d = dst / sp.name
            if not d.exists():
                shutil.copy2(sp, d)
            rows += pq.read_metadata(sp).num_rows
            if i % 25 == 0:
                print(f"  layer{L}: copied shard {i}/{K}", flush=True)
        meta = json.load(open(src / "metadata.json"))
        meta["n_samples"] = rows; meta["n_shards"] = K
        json.dump(meta, open(dst / "metadata.json", "w"), indent=2)
        R = rows
        print(f"  layer{L} done: {rows} tokens, {K} shards", flush=True)

    # shared sidecar: first R rows of token_labels (row-aligned to concatenated shards), proteins covered
    print(f"slicing sidecar to first {R} rows...", flush=True)
    tl = pq.read_table(full / "token_labels.parquet").slice(0, R)
    pq.write_table(tl, out / "token_labels.parquet")
    pids = set(tl.column("protein_id").to_pylist())
    pr = pq.read_table(full / "proteins.parquet")
    mask = pa.array([p in pids for p in pr.column("protein_id").to_pylist()])
    pq.write_table(pr.filter(mask), out / "proteins.parquet")
    em = json.load(open(full / "extract_metadata.json"))
    em["n_proteins"] = len(pids); em["subset_first_shards"] = K; em["subset_tokens"] = R
    json.dump(em, open(out / "extract_metadata.json", "w"), indent=2)
    print(f"DONE: {len(layers)} layers x {K} shards, {R} tokens, {len(pids)} proteins -> {out}", flush=True)
    print("(note: role sidecar token_labels_with_role.parquet not built here — regenerate via add_role_sidecar if needed for analysis)")


if __name__ == "__main__":
    main()
