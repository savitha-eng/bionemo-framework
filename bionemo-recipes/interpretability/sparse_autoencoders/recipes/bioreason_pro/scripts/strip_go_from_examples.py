#!/usr/bin/env python
"""Strip <go> graph-slot tokens from a dashboard's feature_examples.parquet (for go-dropped/balanced models).

Balancing drops <go> from TRAINING, but example windows were built from a store that kept <go> for
alignment -> the dashboard still shows <go> slots. This removes them post-hoc, with no re-encoding:
  (1) drop example rows whose peak band is 'go',
  (2) in remaining windows, delete <go> tokens AND their parallel activation entries,
  (3) recompute max_activation + re-rank examples per feature.
Text mentions of GO (e.g. "GO:0005622 (organelle)") are real prompt text and are KEPT.
Usage: strip_go_from_examples.py <public/model_dir>
"""
import json, shutil, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

GO = "<go>"


def main():
    d = Path(sys.argv[1])
    fp = d / "feature_examples.parquet"
    bak = d / "feature_examples.pre_gostrip.parquet"
    # Refresh the backup from the CURRENT file each run, then strip the current file. (Reading a
    # persistent stale backup clobbered rebuilds — e.g. a fresh 1.3M-row build got overwritten by an
    # old 163k-row backup. The strip is idempotent, so re-running on already-stripped data is a no-op.)
    shutil.copy(fp, bak)
    rows = pq.read_table(fp).to_pylist()
    out, dropped_go_band, stripped = [], 0, 0
    by_feat = {}
    for r in rows:
        if r["band"] == "go":
            dropped_go_band += 1
            continue
        toks = r["sequence"].split(" ")
        acts = json.loads(r["activations"]) if isinstance(r["activations"], str) else list(r["activations"])
        if len(toks) == len(acts) and GO in toks:
            keep = [i for i, t in enumerate(toks) if t != GO]
            toks = [toks[i] for i in keep]; acts = [acts[i] for i in keep]
            stripped += 1
        if not acts or max(acts) <= 0:   # feature only fired on <go> here -> drop the window
            continue
        r = dict(r)
        r["sequence"] = " ".join(toks)
        r["activations"] = json.dumps(acts)
        r["max_activation"] = float(max(acts))
        by_feat.setdefault(r["feature_id"], []).append(r)
    # re-rank per feature by max_activation desc
    for fid, lst in by_feat.items():
        lst.sort(key=lambda r: -r["max_activation"])
        for rank, r in enumerate(lst):
            r["example_rank"] = rank
            out.append(r)
    cols = list(rows[0].keys())
    pq.write_table(pa.Table.from_pylist([{c: r[c] for c in cols} for r in out]), fp)
    print(f"dropped go-band rows: {dropped_go_band} | windows stripped of <go>: {stripped} | "
          f"final rows: {len(out)} (was {len(rows)})")


if __name__ == "__main__":
    main()
