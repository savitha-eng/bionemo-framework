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

r"""Layer selection by linear probing for the BioReason-Pro SAE hook.

Picks the SAE hook layer the *proper* way: design several biologically-meaningful probes (known
labels from the dataset), fit ridge (continuous) / logistic (categorical / multi-label) on the
per-protein pooled residual at each candidate layer, and rank layers by cross-validated probe
performance. Layer choice is about network depth, so this is dataset-size-independent and far more
rigorous than a single near-tied probe.

Probes (from CAFA5 reasoning dataset fields):
  * GO-MF / GO-BP / GO-CC presence (top-K terms each)  -> multi-label logistic, macro ROC-AUC
  * subcellular_location (top-N classes)               -> logistic, balanced accuracy
  * organism (top-N classes)                           -> logistic, accuracy
  * protein length                                     -> ridge, CV R^2
  * #InterPro domains                                  -> ridge, CV R^2

Pure-sae (Env B) + the BioReason-Pro dataset for labels (no model load). Runs on a multi-layer
diagnostic store from extract.py (e.g., diag500 with layers {24,28,32,35}).

    python scripts/layer_probe.py --store <.../diag500_L24-28-32-35> --layers 24 28 32 35 \
        --split train --pool mean --out-json <out.json>
"""

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BANDS = ["protein", "go", "text"]
_RECIPE_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_RECIPE_SRC))


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Layer selection by linear probing")
    p.add_argument("--store", required=True, help="Multi-layer diagnostic store (layer<L>/ + sidecars)")
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--pool", default="mean", choices=["mean", "text", "protein", "go"],
                   help="Pool per-protein over all tokens (mean) or a single band")
    p.add_argument("--top-k-go", type=int, default=20, help="Top-K GO terms per aspect to probe")
    p.add_argument("--top-n-class", type=int, default=10, help="Top-N classes for categorical probes")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def _shard_paths(layer_dir):
    return sorted(glob.glob(str(Path(layer_dir) / "shard_*.parquet")))


def _read_shard(path):
    t = pq.read_table(path)
    cols = sorted(t.column_names, key=lambda c: int(c.split("_")[1]))
    return np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)


def pooled_all_bands(store, layer, protein_index, pos_code, n_prot, hidden):
    """One streaming pass -> per-band mean-pooled residual per protein.

    Returns {band: (n_prot, hidden)}. The probe reps are then: each band alone, plus their
    concatenation [protein|go|text] (= the balanced all-modality input; avoids the magnitude-
    dominated plain mean over all tokens).
    """
    psum = {b: np.zeros((n_prot, hidden), dtype=np.float64) for b in BANDS}
    pcnt = {b: np.zeros(n_prot, dtype=np.int64) for b in BANDS}
    row0 = 0
    for sp in _shard_paths(Path(store) / f"layer{layer}"):
        acts = _read_shard(sp)
        n = acts.shape[0]
        pidx = protein_index[row0:row0 + n]
        pc = pos_code[row0:row0 + n]
        for bi, b in enumerate(BANDS):
            m = pc == bi
            if m.any():
                np.add.at(psum[b], pidx[m], acts[m])
                np.add.at(pcnt[b], pidx[m], 1)
        row0 += n
    return {b: (psum[b] / np.maximum(pcnt[b][:, None], 1)).astype(np.float32) for b in BANDS}


def build_labels(protein_ids, split):
    """Join protein_ids -> dataset fields for probe targets."""
    import bioreason_pro_sae.data as brp_data  # installs unsloth stub + path indirectly? no — guard:
    # data.load_reasoning_splits needs bioreason2 importable; we only need raw fields, so load directly.
    import sys as _sys
    import types as _types
    if "unsloth" not in _sys.modules:
        st = _types.ModuleType("unsloth"); st.FastLanguageModel = None; _sys.modules["unsloth"] = st
    _sys.path.insert(0, "/data/savithas/bioreason-pro")
    from datasets import load_dataset

    ds = load_dataset("wanglab/bioreason-pro-sft-reasoning-data")[split]
    by_id = {}
    cols = set(ds.column_names)
    for r in ds:
        by_id[str(r["protein_id"])] = r
    rows = [by_id.get(pid) for pid in protein_ids]
    return rows, cols


def main():  # noqa: D103
    args = parse_args()
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
    from sklearn.preprocessing import StandardScaler

    store = Path(args.store)
    labels = pq.read_table(store / "token_labels.parquet")
    pos_type = np.asarray(labels.column("position_type").to_pylist(), dtype=object)
    pos_code = np.select([pos_type == b for b in BANDS], list(range(3)), default=-1).astype(np.int64)
    prot = pq.read_table(store / "proteins.parquet").to_pylist()
    block = np.array([r["n_protein"] + r["n_go"] + r["n_text"] for r in prot], dtype=np.int64)
    protein_index = np.repeat(np.arange(len(prot), dtype=np.int64), block)
    protein_ids = [r["protein_id"] for r in prot]
    n_prot = len(prot)
    meta0 = json.loads((store / f"layer{args.layers[0]}" / "metadata.json").read_text())
    hidden = meta0["hidden_dim"]
    print(f"[probe] {n_prot} proteins, hidden={hidden}, pool={args.pool}, layers={args.layers}")

    rows, cols = build_labels(protein_ids, args.split)
    have = np.array([r is not None for r in rows])
    print(f"[probe] matched {int(have.sum())}/{n_prot} protein_ids to dataset rows")

    # ---- build probe targets (mask to matched rows) ----
    def col(name):
        return [r.get(name) if r else None for r in rows]

    targets = {}  # name -> (kind, y, scorer)
    # continuous: length, #interpro
    length = np.array([float(r["length"]) if (r and r.get("length") not in (None, "")) else np.nan for r in rows])
    targets["length"] = ("ridge", length)
    n_ipr = np.array([len(r.get("interpro_ids") or []) if r else np.nan for r in rows], dtype=float)
    if np.nanstd(n_ipr) > 0:
        targets["n_interpro_domains"] = ("ridge", n_ipr)

    # categorical: subcellular_location, organism (top-N classes)
    def top_class_target(field):
        vals = []
        for r in rows:
            v = r.get(field) if r else None
            if isinstance(v, list):
                v = v[0] if v else None
            vals.append(v if v not in (None, "") else None)
        top = [c for c, _ in Counter([v for v in vals if v]).most_common(args.top_n_class)]
        idx = {c: i for i, c in enumerate(top)}
        y = np.array([idx.get(v, -1) for v in vals])
        return y, top
    for field in ["subcellular_location", "organism"]:
        if field in cols:
            y, top = top_class_target(field)
            if len(top) >= 2:
                targets[field] = ("logreg", y)

    # multi-label GO aspects
    def go_multilabel(field):
        sets = [set(r.get(field) or []) if r else set() for r in rows]
        freq = Counter(t for s in sets for t in s)
        top = [t for t, _ in freq.most_common(args.top_k_go)]
        Y = np.array([[1 if t in s else 0 for t in top] for s in sets])
        return Y, top
    for field in ["go_mf", "go_bp", "go_cc"]:
        if field in cols:
            Y, top = go_multilabel(field)
            if Y.shape[1] >= 3:
                targets[field] = ("multilabel", Y)

    # ---- probe scoring helper (closes over sklearn fns, targets, have, args) ----
    def score_targets(X_all):
        scores = {}
        for name, (kind, y) in targets.items():
            if kind == "ridge":
                m = have & ~np.isnan(y)
                if m.sum() < 30 or np.std(y[m]) == 0:
                    continue
                X = StandardScaler().fit_transform(X_all[m])
                cv = KFold(args.cv, shuffle=True, random_state=args.seed)
                scores[name] = round(float(cross_val_score(Ridge(alpha=10.0), X, y[m], cv=cv,
                                                            scoring="r2").mean()), 4)
            elif kind == "logreg":
                m = have & (y >= 0)
                if m.sum() < 30 or len(set(y[m])) < 2:
                    continue
                X = StandardScaler().fit_transform(X_all[m])
                cv = KFold(args.cv, shuffle=True, random_state=args.seed)
                yp = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                       X, y[m], cv=cv)
                scores[name] = round(float(balanced_accuracy_score(y[m], yp)), 4)
            else:  # multilabel -> mean per-term CV AUC
                X = StandardScaler().fit_transform(X_all[have])
                Ym = y[have]
                cv = KFold(args.cv, shuffle=True, random_state=args.seed)
                aucs = []
                for j in range(Ym.shape[1]):
                    yj = Ym[:, j]
                    if yj.sum() < 5 or yj.sum() > len(yj) - 5:
                        continue
                    try:
                        proba = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                                  X, yj, cv=cv, method="predict_proba")[:, 1]
                        aucs.append(roc_auc_score(yj, proba))
                    except Exception:
                        pass
                if aucs:
                    scores[name] = round(float(np.mean(aucs)), 4)
        return scores

    # ---- probe each layer, for each representation (per-band + concat all-modality) ----
    REPS = ["protein", "go", "text", "concat"]
    report = {"store": str(store), "n_proteins": n_prot, "layers": {},
              "probes": list(targets.keys()), "reps": REPS}
    for L in args.layers:
        print(f"[probe] layer {L}: pooling all bands ...")
        bands = pooled_all_bands(store, L, protein_index, pos_code, n_prot, hidden)
        reps = {"protein": bands["protein"], "go": bands["go"], "text": bands["text"],
                "concat": np.hstack([bands[b] for b in BANDS])}
        report["layers"][str(L)] = {}
        for rep_name in REPS:
            sc = score_targets(reps[rep_name])
            report["layers"][str(L)][rep_name] = sc
            print(f"   [{rep_name:8s}] {json.dumps(sc)}")

    # ---- rank layers (per representation) by mean rank across probes ----
    report["mean_rank_by_layer"] = {}
    report["recommended_layer"] = {}
    for rep_name in REPS:
        probe_names = sorted({p for L in report["layers"] for p in report["layers"][L][rep_name]})
        ranks = {str(L): [] for L in args.layers}
        for p in probe_names:
            vals = [(str(L), report["layers"][str(L)][rep_name].get(p)) for L in args.layers]
            vals = [(L, v) for L, v in vals if v is not None]
            vals.sort(key=lambda kv: kv[1], reverse=True)
            for rank, (L, _) in enumerate(vals):
                ranks[L].append(rank + 1)
        mean_rank = {L: round(float(np.mean(r)), 3) for L, r in ranks.items() if r}
        best = min(mean_rank, key=mean_rank.get) if mean_rank else None
        report["mean_rank_by_layer"][rep_name] = mean_rank
        report["recommended_layer"][rep_name] = int(best) if best else None
        print(f"\n[probe] [{rep_name}] mean rank by layer (lower=better): {mean_rank} -> best L{best}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"[probe] wrote {args.out_json}")


if __name__ == "__main__":
    main()
