#!/usr/bin/env python
"""Add a GO-annotation-text metric to feature_metadata.parquet.

The <go> GRAPH slots are NOT per-protein GO terms (the GO graph encoder cross-attention-reduces the
whole ontology into 200 fixed query vectors -- slot k != go_ids[k]). The protein's REAL GO terms enter
as literal TEXT in the prompt (the go_pred section: "GO:0006887 (endocytosis) ..."), tokenized
digit-by-digit -> they render as "GO : 0 0 0 6 8 8 7" in example windows.

So a genuine GO feature is a TEXT feature whose activating windows land on GO: accession strings.
This computes, per feature, the fraction of its TEXT example windows that contain a GO: accession
(go_text_frac) and the same for InterPro domain accessions (ipr_text_frac). Sort/filter the dashboard
by go_text_frac to find label-grounded GO features (e.g. F8). Re-run after rebuilding examples.
"""
import argparse
import collections
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GO_PAT = re.compile(r"GO\s*:\s*(?:\d\s*){5,}")     # "GO : 0 0 0 6 8 8 7" (digits split per-token)
IPR_PAT = re.compile(r"I\s*PR\s*(?:\d\s*){5,}")    # "I PR 0 0 0 6 3 5" InterPro domain accession


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    args = p.parse_args()

    ex = pq.read_table(f"{args.dir}/feature_examples.parquet").to_pylist()
    go_hit = collections.Counter()
    ipr_hit = collections.Counter()
    txt_tot = collections.Counter()
    TEXT_BANDS = {"text", "prompt", "reasoning", "answer"}  # text band is split into roles now
    for r in ex:
        if r["band"] not in TEXT_BANDS:
            continue
        f = r["feature_id"]
        txt_tot[f] += 1
        pcs = r["sequence"].split(" ")
        acts = r["activations"]
        if len(pcs) != len(acts) or not pcs:
            continue
        j = int(np.argmax(acts))                          # PEAK token of this window
        lo, hi = max(0, j - 2), min(len(pcs), j + 3)      # peak-local: does the peak land ON the accession?
        win = " ".join(pcs[lo:hi])
        if GO_PAT.search(win) or "GO" in pcs[lo:hi]:
            go_hit[f] += 1
        if IPR_PAT.search(win) or "PR" in pcs[lo:hi]:
            ipr_hit[f] += 1

    m = pq.read_table(f"{args.dir}/feature_metadata.parquet")
    fids = m.column("feature_id").to_pylist()
    go_text_frac = [go_hit[f] / txt_tot[f] if txt_tot[f] else 0.0 for f in fids]
    ipr_text_frac = [ipr_hit[f] / txt_tot[f] if txt_tot[f] else 0.0 for f in fids]

    keep = [c for c in m.column_names if c not in ("go_text_frac", "ipr_text_frac")]
    m = m.select(keep)
    m = m.append_column("go_text_frac", pa.array(go_text_frac, pa.float32()))
    m = m.append_column("ipr_text_frac", pa.array(ipr_text_frac, pa.float32()))
    pq.write_table(m, f"{args.dir}/feature_metadata.parquet")

    g = np.array(go_text_frac)
    i = np.array(ipr_text_frac)
    print(f"[go-text] wrote go_text_frac / ipr_text_frac for {len(fids)} features")
    print(f"[go-text] GO features (go_text_frac>=0.5): {(g >= 0.5).sum()} | >=0.8: {(g >= 0.8).sum()}")
    print(f"[go-text] InterPro features (ipr_text_frac>=0.5): {(i >= 0.5).sum()}")
    top = sorted(zip(fids, go_text_frac), key=lambda x: -x[1])[:12]
    print("[go-text] top GO-text features:", [(f"F{f}", round(v, 2)) for f, v in top if v > 0])


if __name__ == "__main__":
    main()
