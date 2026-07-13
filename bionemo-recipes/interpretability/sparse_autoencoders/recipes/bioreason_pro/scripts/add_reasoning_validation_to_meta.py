#!/usr/bin/env python
"""Surface the reasoning-feature validation on the dashboard + build a review shortlist.

Adds to feature_metadata.parquet:
  go_auc_reasoning     held-out reasoning-band GO-AUROC (best specific term)
  reasoning_go_term    that term
  validated_reasoning  bool: held_auc>=0.75 AND coherence>=0.4 AND beats raw+random (margin>0)
  reasoning_span       multi-token metric: mean # reasoning tokens per example with act > 0.3*peak
                       (single-token spikes -> ~1; coherent multi-token features -> higher)

Also writes analysis/review_shortlist_l16_reasoning.csv: the validated reasoning features filtered to
the clean/monosemantic profile (log_freq < -3, max_act > 20) and ranked by span, for manual review.

Usage: add_reasoning_validation_to_meta.py <pub_dir> <validated_reasoning_csv>
"""
import csv, sys, shutil
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

pub, valcsv = sys.argv[1], sys.argv[2]

rows = list(csv.DictReader(open(valcsv)))
auc = {int(r["feature"]): float(r["held_auc"]) for r in rows}
term = {int(r["feature"]): r["go_term"] for r in rows}
validated = {int(r["feature"]): (float(r["held_auc"]) >= 0.75 and float(r["coherence"]) >= 0.4 and float(r["margin"]) > 0)
             for r in rows}

# multi-token span from reasoning-band example windows
ex = pq.read_table(f"{pub}/feature_examples.parquet", columns=["feature_id", "band", "activations"]).to_pandas()
ex = ex[ex.band == "reasoning"]
span = {}
for fid, g in ex.groupby("feature_id"):
    spans = []
    for a in g.activations:
        a = list(a)
        if not a:
            continue
        pk = max(a)
        if pk <= 0:
            continue
        spans.append(sum(1 for x in a if x > 0.3 * pk))
    if spans:
        span[int(fid)] = round(sum(spans) / len(spans), 1)

meta = pq.read_table(f"{pub}/feature_metadata.parquet")
fids = meta.column("feature_id").to_pylist()
for col in ["go_auc_reasoning", "reasoning_go_term", "validated_reasoning", "reasoning_span"]:
    if col in meta.column_names:
        meta = meta.drop([col])
meta = meta.append_column("go_auc_reasoning", pa.array([float(auc.get(f, 0.0)) for f in fids], pa.float32()))
meta = meta.append_column("reasoning_go_term", pa.array([term.get(f, "") for f in fids], pa.string()))
meta = meta.append_column("validated_reasoning", pa.array([bool(validated.get(f, False)) for f in fids], pa.bool_()))
meta = meta.append_column("reasoning_span", pa.array([float(span.get(f, 0.0)) for f in fids], pa.float32()))
shutil.copy(f"{pub}/feature_metadata.parquet", f"{pub}/feature_metadata.parquet.pre_reasoning.bak")
pq.write_table(meta, f"{pub}/feature_metadata.parquet")
n_val = sum(validated.values())
print(f"added reasoning validation columns to {pub} ({n_val} validated_reasoning=True)")

# review shortlist: validated + clean profile (low freq, high activation), ranked by span
md = meta.to_pandas()
sub = md[(md.validated_reasoning) & (md.log_frequency < -3) & (md.max_activation > 20)].copy()
sub = sub.sort_values(["reasoning_span", "go_auc_reasoning"], ascending=False)
cols = ["feature_id", "reasoning_go_term", "go_auc_reasoning", "reasoning_span", "log_frequency", "max_activation", "xmodal_caption"]
cols = [c for c in cols if c in sub.columns]
out = Path("analysis/review_shortlist_l16_reasoning.csv")
sub[cols].to_csv(out, index=False)
print(f"review shortlist (validated + log_freq<-3 + max_act>20): {len(sub)} features -> {out}")
for _, r in sub.head(15).iterrows():
    print(f"  F{int(r.feature_id):<6} {str(r.reasoning_go_term)[:26]:26} auc={r.go_auc_reasoning:.2f} span={r.reasoning_span:.0f} logfreq={r.log_frequency:.1f} maxact={r.max_activation:.0f}")
