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

r"""SAE-V cross-modal feature analysis for BioReason-Pro (arXiv:2502.17514, Eq. 7).

For each SAE feature, take its top-K activating tokens **within each modality band**
(protein / go / text) and compute the mean rank-paired cosine similarity between those tokens'
residual-stream vectors across each band pair:

    omega_k(A,B) = (1/K) * sum_i cos( z[topk_A[i]], z[topk_B[i]] )

A feature is "multimodal" for a band pair if it activates (> delta) on >= K tokens in BOTH bands.
omega near 1 => the feature's strongly-activating tokens point the same direction across modalities
=> a genuinely fused/shared multimodal feature (the model represents the concept modality-agnostically).

Cosine is per-token L2-normalized, so the large per-band activation-norm disparity does NOT bias it.

Pure-sae (Env B). Runs on the held-out eval store + a trained SAE checkpoint (no re-extraction):
    python scripts/crossmodal.py --sae <ckpt.pt> --store <.../layer24_eval_val300> --layer 24 \
        --top-k 16 --delta 1.0 --out-json <out.json>
"""

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

BANDS = ["protein", "go", "text"]


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="SAE-V cross-modal feature analysis")
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True, help="Eval store (layer<L>/ + token_labels.parquet)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--top-k", type=int, default=16, help="K activating tokens per modality (SAE-V Eq.7)")
    p.add_argument("--delta", type=float, default=1.0, help="Activation threshold for 'fires' (SAE-V delta)")
    p.add_argument("--encode-batch", type=int, default=8192)
    p.add_argument("--omega-thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--top-features-report", type=int, default=25)
    p.add_argument("--device", default="cuda")
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


def main():  # noqa: D103
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    sae = load_sae(args.sae, dev)
    H = sae.hidden_dim
    K = args.top_k
    layer_dir = Path(args.store) / f"layer{args.layer}"

    labels = pq.read_table(Path(args.store) / "token_labels.parquet")
    pos_type = np.asarray(labels.column("position_type").to_pylist(), dtype=object)
    band_of = {b: i for i, b in enumerate(BANDS)}
    pos_code = np.select([pos_type == b for b in BANDS], list(range(len(BANDS))), default=-1).astype(np.int64)

    # Load all residual vectors into RAM (N x 2560) so we can index top-K tokens directly.
    shards = _shard_paths(layer_dir)
    Z = np.concatenate([_read_shard(sp) for sp in shards], axis=0)  # (N, d_model)
    N, d_model = Z.shape
    assert N == pos_type.shape[0], f"row mismatch {N} vs {pos_type.shape[0]}"
    print(f"[crossmodal] N={N} tokens, d_model={d_model}, SAE H={H}, K={K}, delta={args.delta}")

    # Streaming per-(feature, band) top-K activation + global token index, on GPU.
    NEG = -1e30
    topv = {b: torch.full((H, K), NEG, device=dev) for b in BANDS}        # top-K activation values
    topi = {b: torch.full((H, K), -1, dtype=torch.long, device=dev) for b in BANDS}  # token global idx
    pcode_t = torch.from_numpy(pos_code).to(dev)

    with torch.no_grad():
        for s in range(0, N, args.encode_batch):
            e = min(N, s + args.encode_batch)
            x = torch.from_numpy(Z[s:e]).to(dev)
            codes = sae.encode(x)  # (n, H)
            gidx = torch.arange(s, e, device=dev)
            for b in BANDS:
                m = pcode_t[s:e] == band_of[b]
                if not bool(m.any()):
                    continue
                cb = codes[m]              # (nb, H)
                ib = gidx[m]               # (nb,)
                # merge with running top-K: concat (H, K) with (H, nb) along dim=1, take top-K
                vals = torch.cat([topv[b], cb.T], dim=1)                       # (H, K+nb)
                idxs = torch.cat([topi[b], ib.unsqueeze(0).expand(H, -1)], 1)  # (H, K+nb)
                tv, ord_ = torch.topk(vals, K, dim=1)
                topv[b] = tv
                topi[b] = torch.gather(idxs, 1, ord_)

    topv = {b: topv[b].cpu().numpy() for b in BANDS}
    topi = {b: topi[b].cpu().numpy() for b in BANDS}

    # A feature "fires" in a band if it has >= K activations above delta there.
    fires = {b: (topv[b][:, -1] > args.delta) for b in BANDS}  # K-th best > delta => >=K above delta
    n_active = {b: int(fires[b].sum()) for b in BANDS}
    print(f"[crossmodal] features with >={K} activations>delta per band: {n_active}")

    # Precompute L2-normalized residuals for fast cosine.
    Zt = torch.from_numpy(Z).to(dev)
    Znorm = torch.nn.functional.normalize(Zt, dim=1)

    def omega(feature_ids, ba, bb):
        """SAE-V Eq.7 mean rank-paired cosine for a band pair over the given features."""
        out = {}
        for k in feature_ids:
            ia = topi[ba][k]
            ib = topi[bb][k]
            za = Znorm[torch.from_numpy(ia).to(dev)]  # (K, d)
            zb = Znorm[torch.from_numpy(ib).to(dev)]  # (K, d)
            cos = (za * zb).sum(dim=1)                # rank-paired cosine, (K,)
            out[int(k)] = float(cos.mean().item())
        return out

    report = {"sae": args.sae, "layer": args.layer, "K": K, "delta": args.delta,
              "n_tokens": int(N), "sae_hidden": H,
              "n_features_firing_per_band": n_active, "pairs": {}}
    for ba, bb in combinations(BANDS, 2):
        both = np.where(fires[ba] & fires[bb])[0]
        pair = f"{ba}-{bb}"
        if both.size == 0:
            report["pairs"][pair] = {"n_multimodal_features": 0}
            print(f"[crossmodal] {pair}: 0 multimodal features")
            continue
        w = omega(both, ba, bb)
        vals = np.array(list(w.values()))
        top_feats = sorted(w.items(), key=lambda kv: kv[1], reverse=True)[: args.top_features_report]
        report["pairs"][pair] = {
            "n_multimodal_features": int(both.size),
            "omega_mean": round(float(vals.mean()), 4),
            "omega_median": round(float(np.median(vals)), 4),
            "omega_max": round(float(vals.max()), 4),
            **{f"n_omega_gt_{t}": int((vals > t).sum()) for t in args.omega_thresholds},
            "top_features": [{"feature": f, "omega": round(o, 4)} for f, o in top_feats],
        }
        print(f"[crossmodal] {pair}: {both.size} multimodal feats | omega mean={vals.mean():.3f} "
              f"median={np.median(vals):.3f} max={vals.max():.3f} | "
              + " ".join(f">{t}:{int((vals>t).sum())}" for t in args.omega_thresholds))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"[crossmodal] wrote {args.out_json}")


if __name__ == "__main__":
    main()
