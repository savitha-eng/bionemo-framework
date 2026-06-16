# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Token-distribution analysis for a BioReason-Pro activation store (Env B, no model).

Reads the per-protein sidecar (`proteins.parquet`: n_protein / n_go / n_text counts per example) and
reports (1) the corpus-level token mix per band, and (2) the per-protein distribution of each band's
token count (mean/median/std/min/percentiles/max). The per-protein protein-token count varies a lot
(protein length, capped at max_length_protein), so the corpus mix ≠ any single example.

    python scripts/token_distribution.py --store <activation-store> [--out-json out.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BANDS = ["protein", "go", "text"]


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True, help="Activation store dir (has proteins.parquet)")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    t = pq.read_table(Path(args.store) / "proteins.parquet")
    cols = t.column_names
    # n_protein / n_go / n_text are per-example token counts written by extract.py
    counts = {b: np.asarray(t.column(f"n_{b}").to_numpy(zero_copy_only=False), dtype=np.int64) for b in BANDS}
    n_prot = len(counts["protein"])
    total_per = counts["protein"] + counts["go"] + counts["text"]
    grand = int(total_per.sum())

    print(f"store: {args.store}")
    print(f"examples (proteins): {n_prot:,}   total tokens: {grand:,}   (~{grand/n_prot:,.0f} tokens/example)\n")

    print("CORPUS token mix (share of all tokens):")
    corpus = {}
    for b in BANDS:
        s = int(counts[b].sum()); corpus[b] = s
        print(f"  {b:8}: {s:>12,}  ({100*s/grand:5.1f}%)")

    def stats(a):
        return dict(mean=float(a.mean()), median=float(np.median(a)), std=float(a.std()),
                    min=int(a.min()), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                    max=int(a.max()))

    print("\nPER-PROTEIN token counts (how many of each band per example):")
    print(f"  {'band':8}{'mean':>8}{'median':>8}{'std':>8}{'min':>7}{'p10':>7}{'p90':>8}{'max':>8}")
    per = {}
    for b in BANDS:
        st = stats(counts[b]); per[b] = st
        print(f"  {b:8}{st['mean']:>8.0f}{st['median']:>8.0f}{st['std']:>8.0f}{st['min']:>7d}"
              f"{st['p10']:>7.0f}{st['p90']:>8.0f}{st['max']:>8d}")
    # protein-token count is the variable one (protein length); show why corpus% != per-example%
    print(f"\nnote: corpus protein-share {100*corpus['protein']/grand:.1f}% is dominated by LONG proteins; "
          f"the median example is only {per['protein']['median']:.0f} protein tokens "
          f"({100*per['protein']['median']/np.median(total_per):.1f}% of a median example).")

    out = {"store": args.store, "n_examples": n_prot, "total_tokens": grand,
           "corpus_mix": {b: {"tokens": corpus[b], "pct": round(100 * corpus[b] / grand, 2)} for b in BANDS},
           "per_protein": per, "has_columns": cols}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
