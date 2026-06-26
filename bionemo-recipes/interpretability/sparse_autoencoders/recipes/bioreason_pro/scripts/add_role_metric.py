#!/usr/bin/env python
"""Add per-feature reasoning/answer/prompt fractions to feature_metadata.parquet.

dashboard.py now splits the text band into prompt/reasoning/answer (via <think>...</think>). This
summarizes, per feature, what fraction of its response-text example windows fall in each role:
reasoning_frac / answer_frac / prompt_frac (over prompt+reasoning+answer+text example rows). Lets the
dashboard filter text (and cross-modal) features by whether they fire on REASONING vs the ANSWER.
Approximate (based on the top-N example windows per band). Re-run after rebuilding feature_examples.
"""
import argparse
import collections

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROLE_BANDS = {"prompt", "reasoning", "answer", "text"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    args = p.parse_args()

    ex = pq.read_table(f"{args.dir}/feature_examples.parquet")
    fid = ex.column("feature_id").to_pylist()
    band = ex.column("band").to_pylist()
    cnt = collections.defaultdict(collections.Counter)  # feature -> {role: n example rows}
    for f, b in zip(fid, band):
        if b in ROLE_BANDS:
            cnt[f][b] += 1

    m = pq.read_table(f"{args.dir}/feature_metadata.parquet")
    fids = m.column("feature_id").to_pylist()

    def frac(f, role):
        c = cnt.get(f)
        if not c:
            return 0.0
        tot = sum(c.values())
        return c.get(role, 0) / tot if tot else 0.0

    reasoning_frac = [frac(f, "reasoning") for f in fids]
    answer_frac = [frac(f, "answer") for f in fids]
    prompt_frac = [frac(f, "prompt") for f in fids]

    keep = [c for c in m.column_names if c not in ("reasoning_frac", "answer_frac", "prompt_frac")]
    m = m.select(keep)
    for name, col in [("reasoning_frac", reasoning_frac), ("answer_frac", answer_frac),
                      ("prompt_frac", prompt_frac)]:
        m = m.append_column(name, pa.array(col, pa.float32()))
    pq.write_table(m, f"{args.dir}/feature_metadata.parquet")

    r, a = np.array(reasoning_frac), np.array(answer_frac)
    print(f"[role] wrote reasoning_frac/answer_frac/prompt_frac for {len(fids)} features")
    print(f"[role] reasoning-dominant (>=0.6): {(r >= 0.6).sum()} | answer-dominant (>=0.6): {(a >= 0.6).sum()}")


if __name__ == "__main__":
    main()
