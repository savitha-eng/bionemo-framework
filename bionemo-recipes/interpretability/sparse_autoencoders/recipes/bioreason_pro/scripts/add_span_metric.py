#!/usr/bin/env python
"""Add per-feature activation-SPAN metrics to feature_metadata.parquet.

For each feature, looks at its example windows (feature_examples.parquet) and counts how many
tokens are *meaningfully active* (> frac * the window peak) per example, averaged per band.
A single-token detector -> span ~= 1; a feature that fires over a multi-token span -> span > 1.

Emits columns: text_span, protein_span, go_span, max_span (max over the three).
Lets you sort/color/brush the dashboard to find text features that light up over MULTIPLE tokens
rather than a single character. Re-run after rebuilding feature_examples.parquet.
"""
import argparse
import collections

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="dashboard public dir (has feature_examples + feature_metadata)")
    p.add_argument("--frac", type=float, default=0.5, help="active threshold = frac * per-example window peak "
                   "(0.5 = positions at NEAR-MAX activation, i.e. a meaningful multi-token pattern)")
    args = p.parse_args()

    ex = pq.read_table(f"{args.dir}/feature_examples.parquet")
    fid = ex.column("feature_id").to_numpy()
    band = ex.column("band").to_pylist()
    acts = ex.column("activations").to_pylist()

    # per (feature, band): list of active-token counts, one per example window
    spans = collections.defaultdict(lambda: collections.defaultdict(list))
    for f, b, a in zip(fid, band, acts):
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            continue
        peak = a.max()
        if peak <= 0:
            continue
        n_active = int((a > args.frac * peak).sum())  # tokens firing alongside the peak
        spans[int(f)][b].append(n_active)

    # text band is split into prompt/reasoning/answer (+ 'text' fallback) -> aggregate them for text_span
    TEXT_BANDS = ("text", "prompt", "reasoning", "answer")

    def mean_bands(f, bands):
        v = []
        for b in bands:
            v += spans.get(f, {}).get(b, [])
        return float(np.mean(v)) if v else 0.0

    m = pq.read_table(f"{args.dir}/feature_metadata.parquet")
    feat_ids = m.column("feature_id").to_pylist()
    text_span = [mean_bands(f, TEXT_BANDS) for f in feat_ids]
    protein_span = [mean_bands(f, ("protein",)) for f in feat_ids]
    go_span = [mean_bands(f, ("go",)) for f in feat_ids]
    max_span = [max(t, pr, g) for t, pr, g in zip(text_span, protein_span, go_span)]

    # drop any pre-existing span columns, then append fresh
    keep = [c for c in m.column_names if c not in ("text_span", "protein_span", "go_span", "max_span")]
    m = m.select(keep)
    for name, col in [("text_span", text_span), ("protein_span", protein_span),
                      ("go_span", go_span), ("max_span", max_span)]:
        m = m.append_column(name, pa.array(col, type=pa.float32()))
    pq.write_table(m, f"{args.dir}/feature_metadata.parquet")

    ts = np.array(text_span)
    nz = ts[ts > 0]
    print(f"[span] wrote text_span/protein_span/go_span/max_span for {len(feat_ids)} features")
    print(f"[span] text_span (firing features): mean={nz.mean():.2f} median={np.median(nz):.2f} "
          f"| span>=2: {(nz >= 2).sum()} features | span>=3: {(nz >= 3).sum()} | ~1 (single-tok): {(nz < 1.5).sum()}")


if __name__ == "__main__":
    main()
