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

r"""Step B layer diagnostic for BioReason-Pro SAE: pick ONE layer over {24,28,32,35}.

Reads the multi-layer activation store written by extract.py (one ``layer<L>/`` dir per layer +
``token_labels.parquet`` + ``proteins.parquet``). Pure-sae (Env B) — never loads the model.

Per layer, ONE streaming pass over shards computes:
  * Tier 1: per position_type {protein, go, text} — mean residual L2 norm, activation density
    (fraction of |dim|>eps), and %dead dims (dims ~0 across all tokens of that band).
  * Tier 2: GO logistic-probe macro-F1 — per-protein mean-pooled band vectors predict the top-K
    most frequent GO terms (one-vs-rest logistic regression, train/test split).
Layer 35 additionally gets the Fig-1H pipeline (PCA 2560->50, UMAP n_neighbors=30 min_dist=0.3,
HDBSCAN min_cluster_size=20 min_samples=5) on a token sample as a hook-correctness check.

Usage (Env B):
    python scripts/diagnostic.py --store /data/.../diag500_L24-28-32-35 --layers 24 28 32 35 \
        --out-json /data/.../diag_report.json --fig-dir /data/.../figs
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

EPS = 1e-3
BANDS = ["protein", "go", "text"]


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="BioReason-Pro SAE layer diagnostic")
    p.add_argument("--store", required=True, help="Extraction output dir (contains layer<L>/ + sidecars)")
    p.add_argument("--layers", type=int, nargs="+", default=[24, 28, 32, 35])
    p.add_argument("--top-k-go", type=int, default=30, help="Number of most-frequent GO terms to probe")
    p.add_argument("--probe-band", default="go", choices=BANDS, help="Band to pool for the GO probe")
    p.add_argument("--fig35-sample", type=int, default=8000, help="Tokens sampled for the layer-35 Fig-1H check")
    p.add_argument("--fig-layer", type=int, default=35)
    p.add_argument("--out-json", default=None)
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def _shard_paths(layer_dir: Path):
    return sorted(glob.glob(str(layer_dir / "shard_*.parquet")))


def _read_shard(path: str) -> np.ndarray:
    """Read a shard parquet (cols dim_0..dim_{H-1}) into a (n, H) float32 array in dim order."""
    t = pq.read_table(path)
    # Columns are written dim_0..dim_{H-1} in order; reorder defensively by index.
    cols = sorted(t.column_names, key=lambda c: int(c.split("_")[1]))
    return np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)


def load_row_metadata(store: Path):
    """Return (protein_index per row, position_type per row, go_ids per protein, protein_ids)."""
    labels = pq.read_table(store / "token_labels.parquet")
    pos_type = np.asarray(labels.column("position_type").to_pylist(), dtype=object)
    prot = pq.read_table(store / "proteins.parquet").to_pylist()
    block = np.array([r["n_protein"] + r["n_go"] + r["n_text"] for r in prot], dtype=np.int64)
    protein_index = np.repeat(np.arange(len(prot), dtype=np.int64), block)
    if protein_index.shape[0] != pos_type.shape[0]:
        raise ValueError(f"row mismatch: proteins sum {protein_index.shape[0]} vs sidecar {pos_type.shape[0]}")
    go_ids = [json.loads(r["go_ids"]) for r in prot]
    pids = [r["protein_id"] for r in prot]
    return protein_index, pos_type, go_ids, pids


def tier1_and_pool(layer_dir: Path, protein_index, pos_type, n_proteins, hidden, probe_band):
    """One streaming pass: Tier-1 band stats + per-protein pooled vectors for the probe band."""
    band_count = {b: 0 for b in BANDS}
    band_norm_sum = {b: 0.0 for b in BANDS}
    band_density_sum = {b: 0.0 for b in BANDS}
    band_absdim_sum = {b: np.zeros(hidden, dtype=np.float64) for b in BANDS}  # for %dead dims
    pool_sum = np.zeros((n_proteins, hidden), dtype=np.float64)
    pool_cnt = np.zeros(n_proteins, dtype=np.int64)

    row0 = 0
    for sp in _shard_paths(layer_dir):
        acts = _read_shard(sp)
        n = acts.shape[0]
        sl = slice(row0, row0 + n)
        pt = pos_type[sl]
        pidx = protein_index[sl]
        norms = np.linalg.norm(acts, axis=1)
        density = (np.abs(acts) > EPS).mean(axis=1)
        absacts = np.abs(acts)
        for b in BANDS:
            m = pt == b
            if not m.any():
                continue
            band_count[b] += int(m.sum())
            band_norm_sum[b] += float(norms[m].sum())
            band_density_sum[b] += float(density[m].sum())
            band_absdim_sum[b] += absacts[m].sum(axis=0)
        # pool the probe band per protein
        pm = pt == probe_band
        if pm.any():
            np.add.at(pool_sum, pidx[pm], acts[pm])
            np.add.at(pool_cnt, pidx[pm], 1)
        row0 += n

    tier1 = {}
    for b in BANDS:
        c = max(1, band_count[b])
        mean_abs_dim = band_absdim_sum[b] / c
        dead_dims = int((mean_abs_dim < EPS).sum())
        tier1[b] = {
            "n_tokens": band_count[b],
            "mean_l2_norm": round(band_norm_sum[b] / c, 3),
            "mean_density": round(band_density_sum[b] / c, 4),
            "pct_dead_dims": round(100.0 * dead_dims / hidden, 2),
        }
    pooled = pool_sum / np.maximum(pool_cnt[:, None], 1)
    return tier1, pooled, pool_cnt


def go_probe_f1(pooled, pool_cnt, go_ids, top_k, seed):
    """One-vs-rest logistic probe macro-F1 for the top-K most frequent GO terms."""
    from collections import Counter

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    valid = pool_cnt > 0
    X = pooled[valid]
    gi = [g for g, v in zip(go_ids, valid) if v]
    freq = Counter(t for terms in gi for t in terms)
    top = [t for t, _ in freq.most_common(top_k)]
    if not top or X.shape[0] < 20:
        return {"macro_f1": None, "n_terms": len(top), "note": "insufficient data"}
    Y = np.array([[1 if t in set(terms) else 0 for t in top] for terms in gi], dtype=int)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=seed)
    f1s = []
    for j in range(len(top)):
        ytr, yte = Ytr[:, j], Yte[:, j]
        if ytr.sum() < 3 or yte.sum() < 1:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, ytr)
        f1s.append(f1_score(yte, clf.predict(Xte), zero_division=0))
    return {"macro_f1": round(float(np.mean(f1s)), 4) if f1s else None,
            "n_terms_scored": len(f1s), "n_terms_requested": len(top)}


def fig1h(layer_dir: Path, pos_type, sample_n, fig_dir, seed):
    """PCA->UMAP->HDBSCAN hook-correctness check on a token sample (layer 35)."""
    import hdbscan
    import umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    # gather a sample across shards
    chunks, tags, got = [], [], 0
    for sp in _shard_paths(layer_dir):
        acts = _read_shard(sp)
        take = min(acts.shape[0], max(1, sample_n // 4))
        idx = rng.choice(acts.shape[0], size=take, replace=False)
        chunks.append(acts[idx])
        got += take
        if got >= sample_n:
            break
    X = np.concatenate(chunks)[:sample_n]
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=50, random_state=seed).fit_transform(Xs)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(pca)
    cl = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(emb)
    n_clusters = int(len(set(cl)) - (1 if -1 in cl else 0))
    noise = float((cl == -1).mean())
    result = {"n_sample": int(X.shape[0]), "n_clusters": n_clusters, "noise_frac": round(noise, 3),
              "pca_var_explained_50": round(float(PCA(n_components=50, random_state=seed)
                                                  .fit(Xs).explained_variance_ratio_.sum()), 3)}
    if fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 6))
        plt.scatter(emb[:, 0], emb[:, 1], c=cl, s=3, cmap="tab20")
        plt.title(f"Layer-35 Fig1H: {n_clusters} clusters, {noise:.0%} noise")
        plt.tight_layout()
        plt.savefig(Path(fig_dir) / "fig1h_layer35_umap.png", dpi=120)
        plt.close()
        result["figure"] = str(Path(fig_dir) / "fig1h_layer35_umap.png")
    return result


def main():  # noqa: D103
    args = parse_args()
    store = Path(args.store)
    protein_index, pos_type, go_ids, pids = load_row_metadata(store)
    n_proteins = len(pids)
    # hidden dim from first layer's metadata
    meta0 = json.loads((store / f"layer{args.layers[0]}" / "metadata.json").read_text())
    hidden = meta0["hidden_dim"]
    print(f"[diag] {n_proteins} proteins, {pos_type.shape[0]} tokens, hidden={hidden}, layers={args.layers}")

    report = {"store": str(store), "n_proteins": n_proteins, "n_tokens": int(pos_type.shape[0]),
              "hidden": hidden, "layers": {}}
    for L in args.layers:
        print(f"[diag] layer {L}: streaming Tier1 + pooling ...")
        tier1, pooled, pool_cnt = tier1_and_pool(
            store / f"layer{L}", protein_index, pos_type, n_proteins, hidden, args.probe_band)
        probe = go_probe_f1(pooled, pool_cnt, go_ids, args.top_k_go, args.seed)
        report["layers"][str(L)] = {"tier1": tier1, "go_probe": probe}
        print(f"   tier1={json.dumps(tier1)}")
        print(f"   go_probe(band={args.probe_band})={json.dumps(probe)}")

    if args.fig_layer in args.layers:
        print(f"[diag] Fig1H on layer {args.fig_layer} ...")
        report["fig1h"] = fig1h(store / f"layer{args.fig_layer}", pos_type, args.fig35_sample,
                                args.fig_dir, args.seed)
        print(f"   fig1h={json.dumps(report['fig1h'])}")

    # recommend: best GO-probe macro-F1 (tie-break higher protein-band norm/richness)
    scored = [(int(L), v["go_probe"].get("macro_f1") or -1) for L, v in report["layers"].items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    report["recommended_layer"] = scored[0][0]
    report["ranking_by_go_probe_f1"] = scored
    print(f"\n[diag] RECOMMENDED LAYER: {report['recommended_layer']} (by GO-probe macro-F1) | ranking={scored}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"[diag] wrote {args.out_json}")


if __name__ == "__main__":
    main()
