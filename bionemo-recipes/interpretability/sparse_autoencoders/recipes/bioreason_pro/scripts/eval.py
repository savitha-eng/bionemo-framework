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

r"""Step 3 eval (pure-sae / Env B) for a trained BioReason-Pro SAE.

Computes the go/no-go metrics that don't need the LLM:
  * reconstruction R^2 (variance explained) + normalized MSE  (sae.eval)
  * sparsity: mean L0                                          (sae.eval)
  * %dead latents over the whole eval store                   (feature firing counts)
  * GO-feature F1: per-protein, best single SAE feature vs each top-K GO term, OVERALL and
    per position_type {protein, go, text}; plus a count of cross-position_type features.

The CE-based loss-recovered metric needs an LLM forward and lives in eval_loss_recovered.py (Env A).

Usage (Env B):
    python scripts/eval.py --sae /data/.../sae_layer28/checkpoint.pt \
        --store /data/.../layer28_eval_val300 --layer 28 --out-json /data/.../eval_layer28.json
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

BANDS = ["protein", "go", "text"]


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Evaluate a trained BioReason-Pro SAE (pure-sae metrics)")
    p.add_argument("--sae", required=True, help="Path to trained SAE checkpoint (.pt)")
    p.add_argument("--store", required=True, help="Eval activation store (layer<L>/ + sidecars) — held-out split")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--top-k-go", type=int, default=30)
    p.add_argument("--go-min-prev", type=float, default=0.02,
                   help="Min protein-prevalence for a GO term to be probed (drops ultra-rare)")
    p.add_argument("--go-max-prev", type=float, default=0.5,
                   help="Max prevalence (drops near-ubiquitous ontology roots that inflate F1)")
    p.add_argument("--recon-sample", type=int, default=200_000, help="Tokens sampled for recon/sparsity metrics")
    p.add_argument("--encode-batch", type=int, default=8192,
                   help="Mini-batch for SAE encode/decode (a dense [n,hidden] code tensor is large)")
    p.add_argument("--fire-threshold", type=float, default=0.0, help="Feature activation > thr counts as 'fires'")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def load_sae(path, device):
    from sae.architectures import TopKSAE

    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("model_config") or {k: ckpt[k] for k in ("input_dim", "hidden_dim", "top_k") if k in ckpt}
    sae = TopKSAE(**cfg)
    sae.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return sae.to(device).eval()


def _shard_paths(layer_dir):
    return sorted(glob.glob(str(Path(layer_dir) / "shard_*.parquet")))


def _read_shard(path):
    t = pq.read_table(path)
    cols = sorted(t.column_names, key=lambda c: int(c.split("_")[1]))
    return np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)


def load_row_meta(store):
    labels = pq.read_table(Path(store) / "token_labels.parquet")
    pos_type = np.asarray(labels.column("position_type").to_pylist(), dtype=object)
    prot = pq.read_table(Path(store) / "proteins.parquet").to_pylist()
    block = np.array([r["n_protein"] + r["n_go"] + r["n_text"] for r in prot], dtype=np.int64)
    protein_index = np.repeat(np.arange(len(prot), dtype=np.int64), block)
    go_ids = [json.loads(r["go_ids"]) for r in prot]
    return protein_index, pos_type, go_ids, len(prot)


def main():  # noqa: D103
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = args.device if torch.cuda.is_available() else "cpu"

    sae = load_sae(args.sae, dev)
    H = sae.hidden_dim
    layer_dir = Path(args.store) / f"layer{args.layer}"
    protein_index, pos_type, go_ids, n_prot = load_row_meta(args.store)
    print(f"[eval] SAE hidden={H} | store {n_prot} proteins, {pos_type.shape[0]} tokens")

    # Per-protein/per-band max feature activation, dead-latent firing counts, recon sample.
    band_idx = {b: i for i, b in enumerate(BANDS)}
    pmax = {b: np.zeros((n_prot, H), dtype=np.float32) for b in BANDS}
    ever_fired = np.zeros(H, dtype=bool)
    recon_chunks, recon_orig = [], []
    l0_sum, l0_count = 0.0, 0
    row0 = 0

    bs = args.encode_batch
    with torch.no_grad():
        for sp in _shard_paths(layer_dir):
            acts = _read_shard(sp)
            n = acts.shape[0]
            pt = pos_type[row0:row0 + n]
            pidx = protein_index[row0:row0 + n]
            # Encode/decode in mini-batches: a dense [n, H] code tensor is huge (n*H*4 bytes).
            for s in range(0, n, bs):
                e = min(n, s + bs)
                x = torch.from_numpy(acts[s:e]).to(dev)
                codes = sae.encode(x)
                recon = sae.decode(codes)
                codes_np = codes.float().cpu().numpy()
                ever_fired |= (codes_np > args.fire_threshold).any(axis=0)
                l0_sum += float((codes_np > 0).sum())
                l0_count += codes_np.shape[0]
                pt_c, pidx_c = pt[s:e], pidx[s:e]
                for b in BANDS:
                    m = pt_c == b
                    if m.any():
                        np.maximum.at(pmax[b], pidx_c[m], codes_np[m])
                if sum(c.shape[0] for c in recon_chunks) < args.recon_sample:
                    recon_chunks.append(recon.float().cpu().numpy())
                    recon_orig.append(acts[s:e])
            row0 += n

    # ---- reconstruction R^2 / normalized MSE + sparsity ----
    X = np.concatenate(recon_orig)[: args.recon_sample]
    R = np.concatenate(recon_chunks)[: args.recon_sample]
    ss_res = float(((X - R) ** 2).sum())
    ss_tot = float(((X - X.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    nmse = ss_res / (float((X ** 2).sum()) + 1e-8)
    mean_l0 = round(l0_sum / max(1, l0_count), 2)
    pct_dead = round(100.0 * float((~ever_fired).mean()), 3)

    # ---- GO-feature F1 (per-protein, best single feature per top-K GO term) ----
    # Select INFORMATIVE GO terms: most-frequent terms whose protein-prevalence is in
    # [go_min_prev, go_max_prev]. This drops the ontology roots (present in ~all proteins) whose
    # F1 is dominated by base rate rather than feature selectivity.
    freq = Counter(t for terms in go_ids for t in terms)
    sets = [set(g) for g in go_ids]
    top = []
    for t, _ in freq.most_common():
        prev = sum(t in s for s in sets) / n_prot
        if args.go_min_prev <= prev <= args.go_max_prev:
            top.append(t)
        if len(top) >= args.top_k_go:
            break
    has_term = {t: np.array([1 if t in s else 0 for s in sets]) for t in top}
    print(f"[eval] probing {len(top)} informative GO terms (prevalence in "
          f"[{args.go_min_prev},{args.go_max_prev}])")

    # Honest selection: pick the best feature per term on a TRAIN protein split, report its metric
    # on a held-out TEST split. Primary metric is ROC-AUC (prevalence-robust; random ~ 0.5);
    # F1 kept for the go/no-go language.
    rng_sel = np.random.default_rng(args.seed)
    perm = rng_sel.permutation(n_prot)
    n_te = max(1, int(0.4 * n_prot))
    test_idx = np.zeros(n_prot, dtype=bool)
    test_idx[perm[:n_te]] = True
    train_idx = ~test_idx

    def _auc_vec(A, y):
        """Per-feature ROC-AUC (Mann-Whitney, ordinal ranks). A:(P,H) scores, y:(P,) 0/1 -> (H,)."""
        P = A.shape[0]
        pos = y.astype(bool)
        n_pos, n_neg = int(pos.sum()), P - int(pos.sum())
        if n_pos == 0 or n_neg == 0:
            return np.full(A.shape[1], 0.5)
        order = np.argsort(A, axis=0)
        ranks = np.empty_like(order, dtype=np.float64)
        np.put_along_axis(ranks, order, (np.arange(P, dtype=np.float64) + 1.0)[:, None], axis=0)
        r_pos = ranks[pos].sum(axis=0)
        return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def _f1_of_feature(act_col, y):
        fires = act_col > args.fire_threshold
        yb = y.astype(bool)
        tp = int((fires & yb).sum()); fp = int((fires & ~yb).sum()); fn = int((~fires & yb).sum())
        d = 2 * tp + fp + fn
        return (2 * tp / d) if d > 0 else 0.0

    def best_feature_metrics(activation_matrix):
        """Best feature chosen on train by AUC; AUC + F1 reported on held-out test."""
        out = {}
        for t in top:
            y = has_term[t]
            if y[train_idx].sum() < 3 or y[test_idx].sum() < 1:
                continue
            auc_tr = _auc_vec(activation_matrix[train_idx], y[train_idx])
            j = int(auc_tr.argmax())
            auc_te = float(_auc_vec(activation_matrix[test_idx][:, j:j + 1], y[test_idx])[0])
            f1_te = _f1_of_feature(activation_matrix[test_idx][:, j], y[test_idx])
            out[t] = {"best_feature": j, "auc": round(auc_te, 4), "f1": round(f1_te, 4),
                      "auc_train": round(float(auc_tr[j]), 4), "n_pos": int(y.sum())}
        return out

    go_f1 = {"overall": {}, "by_band": {}}
    # overall = max activation across all bands
    allmax = np.maximum.reduce([pmax[b] for b in BANDS])
    go_f1["overall"] = best_feature_metrics(allmax)
    for b in BANDS:
        go_f1["by_band"][b] = best_feature_metrics(pmax[b])

    def summarize(d):
        f1s = [v["f1"] for v in d.values()]
        aucs = [v["auc"] for v in d.values()]
        return {"n_terms": len(f1s),
                "mean_best_f1": round(float(np.mean(f1s)), 4) if f1s else None,
                "mean_best_auc": round(float(np.mean(aucs)), 4) if aucs else None,
                "n_f1_gt_0.5": int(sum(v > 0.5 for v in f1s)),
                "n_auc_gt_0.7": int(sum(v > 0.7 for v in aucs))}

    # ---- cross-position_type features (fire in >1 band somewhere in the corpus) ----
    band_fire = {b: (pmax[b] > args.fire_threshold).any(axis=0) for b in BANDS}
    n_bands_per_feature = np.sum([band_fire[b] for b in BANDS], axis=0)
    cross_modal = int((n_bands_per_feature > 1).sum())

    report = {
        "sae": args.sae, "layer": args.layer, "hidden_dim": H, "n_proteins": n_prot,
        "reconstruction": {"r2_variance_explained": round(r2, 4), "normalized_mse": round(nmse, 4)},
        "sparsity": {"mean_l0": round(mean_l0, 2), "top_k": int(getattr(sae, "top_k", -1))},
        "pct_dead_latents": pct_dead,
        "go_feature_f1": {
            "overall_summary": summarize(go_f1["overall"]),
            "by_band_summary": {b: summarize(go_f1["by_band"][b]) for b in BANDS},
            "overall_detail": go_f1["overall"],
        },
        "cross_position_type_features": cross_modal,
        "n_active_features": int(ever_fired.sum()),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "go_feature_f1"}, indent=2))
    print("GO-F1 overall:", json.dumps(report["go_feature_f1"]["overall_summary"]))
    print("GO-F1 by band:", json.dumps(report["go_feature_f1"]["by_band_summary"]))
    print(f"cross-position_type features: {cross_modal} / {int(ever_fired.sum())} active")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"[eval] wrote {args.out_json}")


if __name__ == "__main__":
    main()
