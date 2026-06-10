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
    p.add_argument("--recon-sample", type=int, default=200_000, help="Tokens sampled for recon/sparsity metrics")
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
    row0 = 0

    with torch.no_grad():
        for sp in _shard_paths(layer_dir):
            acts = _read_shard(sp)
            n = acts.shape[0]
            x = torch.from_numpy(acts).to(dev)
            codes = sae.encode(x)  # (n, H), sparse
            recon = sae.decode(codes)
            codes_np = codes.float().cpu().numpy()
            ever_fired |= (codes_np > args.fire_threshold).any(axis=0)

            pt = pos_type[row0:row0 + n]
            pidx = protein_index[row0:row0 + n]
            for b in BANDS:
                m = pt == b
                if m.any():
                    # per-protein max activation within this band
                    np.maximum.at(pmax[b], pidx[m], codes_np[m])

            # reservoir-ish recon sample
            if sum(c.shape[0] for c in recon_chunks) < args.recon_sample:
                take = min(n, max(1, args.recon_sample // 8))
                sel = rng.choice(n, size=take, replace=False)
                recon_chunks.append(recon[sel].float().cpu().numpy())
                recon_orig.append(acts[sel])
            row0 += n

    # ---- reconstruction R^2 / normalized MSE + sparsity ----
    X = np.concatenate(recon_orig)[: args.recon_sample]
    R = np.concatenate(recon_chunks)[: args.recon_sample]
    ss_res = float(((X - R) ** 2).sum())
    ss_tot = float(((X - X.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    nmse = ss_res / (float((X ** 2).sum()) + 1e-8)
    with torch.no_grad():
        codes_s = sae.encode(torch.from_numpy(X).to(dev))
        mean_l0 = float((codes_s > 0).float().sum(dim=1).mean().item())
    pct_dead = round(100.0 * float((~ever_fired).mean()), 3)

    # ---- GO-feature F1 (per-protein, best single feature per top-K GO term) ----
    from sklearn.metrics import f1_score

    freq = Counter(t for terms in go_ids for t in terms)
    top = [t for t, _ in freq.most_common(args.top_k_go)]
    has_term = {t: np.array([1 if t in set(g) else 0 for g in go_ids]) for t in top}

    def best_feature_f1(activation_matrix):
        """For each top GO term, best single feature's F1 (feature fires == activation>thr)."""
        fires = (activation_matrix > args.fire_threshold)  # (P, H) bool
        out = {}
        for t in top:
            y = has_term[t]
            if y.sum() < 3:
                continue
            # F1 of each feature vs y; take best. Vectorized over features.
            tp = (fires & y[:, None].astype(bool)).sum(axis=0)
            fp = (fires & ~y[:, None].astype(bool)).sum(axis=0)
            fn = (~fires & y[:, None].astype(bool)).sum(axis=0)
            denom = 2 * tp + fp + fn
            f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
            j = int(f1.argmax())
            out[t] = {"best_feature": j, "f1": round(float(f1[j]), 4), "n_pos": int(y.sum())}
        return out

    go_f1 = {"overall": {}, "by_band": {}}
    # overall = max activation across all bands
    allmax = np.maximum.reduce([pmax[b] for b in BANDS])
    go_f1["overall"] = best_feature_f1(allmax)
    for b in BANDS:
        go_f1["by_band"][b] = best_feature_f1(pmax[b])

    def summarize(d):
        vals = [v["f1"] for v in d.values()]
        return {"n_terms": len(vals), "mean_best_f1": round(float(np.mean(vals)), 4) if vals else None,
                "n_f1_gt_0.5": int(sum(v > 0.5 for v in vals))}

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
