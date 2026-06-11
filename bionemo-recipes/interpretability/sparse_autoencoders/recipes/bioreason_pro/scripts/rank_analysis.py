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

r"""Effective-rank / redundancy diagnostic — is the high SAE R^2 an artifact of low-rank activations?

Two checks:
  (A) ACTIVATION effective rank per layer, OVERALL and PER BAND (protein/go/text):
        participation ratio  PR = (sum s)^2 / sum(s^2)   on mean-centered activations,
        and 99%-variance-k = #components for 99% of variance.
      Low PR (<< d_model=2560) => the residual lives in a small subspace, so an SAE reaches
      R^2~1 with few features (and most latents die). The GO band (fixed 200-vector memory) and
      protein band (ESM3 + linear projection) are expected to be low-rank.
  (B) SAE DECODER redundancy for a trained checkpoint: effective rank (PR of singular values) of
      the active-feature decoder directions, and the off-diagonal |cosine| distribution.
      If the decoder's effective rank << #active features, the learned features are near-linear-
      combinations of each other (the failure Polina described).

Pure-sae (Env B). Subsamples tokens; SVD on GPU.
    python scripts/rank_analysis.py --store <diag500> --layers 24 28 32 35 \
        --sae <ckpt.pt> --out-json <out.json>
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

BANDS = ["protein", "go", "text"]


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Effective-rank / redundancy diagnostic")
    p.add_argument("--store", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--sae", default=None, help="Optional trained SAE checkpoint for decoder-redundancy check")
    p.add_argument("--sample-tokens", type=int, default=60000, help="Tokens subsampled per band for SVD")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def _shard_paths(layer_dir):
    return sorted(glob.glob(str(Path(layer_dir) / "shard_*.parquet")))


def _read_shard(path):
    t = pq.read_table(path)
    cols = sorted(t.column_names, key=lambda c: int(c.split("_")[1]))
    return np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)


def svd_stats(X, dev):
    """participation ratio + 99%-variance-k of mean-centered rows X (n, d)."""
    Xt = torch.from_numpy(X).to(dev).float()
    Xt = Xt - Xt.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xt)              # singular values (descending)
    s2 = (s ** 2)
    pr = float((s.sum() ** 2) / (s2.sum() + 1e-12))     # effective rank
    cum = torch.cumsum(s2, 0) / (s2.sum() + 1e-12)
    k99 = int((cum < 0.99).sum().item()) + 1
    k90 = int((cum < 0.90).sum().item()) + 1
    return {"n": int(X.shape[0]), "d_model": int(X.shape[1]),
            "participation_ratio": round(pr, 1), "k_99pct_var": k99, "k_90pct_var": k90,
            "top_sv": [round(float(v), 1) for v in s[:5].tolist()]}


def activation_rank(store, layer, pos_type, sample_tokens, dev, rng):
    """Per-band + overall effective rank from a token subsample (reads enough shards to fill)."""
    layer_dir = Path(store) / f"layer{layer}"
    # collect a subsample per band by streaming shards until each band is filled
    need = sample_tokens
    buf = {b: [] for b in BANDS}
    got = {b: 0 for b in BANDS}
    overall = []
    row0 = 0
    for sp in _shard_paths(layer_dir):
        if all(got[b] >= need for b in BANDS):
            break
        acts = _read_shard(sp)
        n = acts.shape[0]
        pt = pos_type[row0:row0 + n]
        for b in BANDS:
            if got[b] >= need:
                continue
            m = pt == b
            if m.any():
                a = acts[m]
                take = min(a.shape[0], need - got[b])
                sel = rng.choice(a.shape[0], take, replace=False)
                buf[b].append(a[sel]); got[b] += take
        # overall sample
        if len(overall) * 200000 < need:
            sel = rng.choice(n, min(n, need // 4), replace=False)
            overall.append(acts[sel])
        row0 += n
    res = {}
    for b in BANDS:
        if buf[b]:
            res[b] = svd_stats(np.concatenate(buf[b])[:need], dev)
    res["overall"] = svd_stats(np.concatenate(overall)[:need], dev)
    return res


def decoder_redundancy(sae_path, dev):
    """Effective rank + off-diagonal |cosine| of the active-feature decoder directions."""
    from sae.architectures import TopKSAE

    ckpt = torch.load(sae_path, map_location="cpu")
    cfg = ckpt.get("model_config")
    sae = TopKSAE(**cfg)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.to(dev).eval()
    d_in, H = cfg["input_dim"], cfg["hidden_dim"]
    # find the decoder weight: a param whose shape is a permutation of (H, d_in)
    Wdec = None
    for name, p in sae.named_parameters():
        if tuple(sorted(p.shape)) == tuple(sorted((H, d_in))) and "dec" in name.lower():
            Wdec = p.detach()
            break
    if Wdec is None:
        for name, p in sae.named_parameters():
            if tuple(sorted(p.shape)) == tuple(sorted((H, d_in))):
                Wdec = p.detach(); break
    if Wdec is None:
        return {"error": "decoder weight not found"}
    # orient as (H, d_in): each row = one feature's decoder direction
    if Wdec.shape[0] != H:
        Wdec = Wdec.T
    # active features = nonzero decoder norm
    norms = Wdec.norm(dim=1)
    active = norms > 1e-8
    D = torch.nn.functional.normalize(Wdec[active], dim=1)  # (n_active, d_in)
    n_active = int(active.sum())
    # effective rank of decoder via singular values
    s = torch.linalg.svdvals(Wdec[active].float())
    s2 = s ** 2
    pr = float((s.sum() ** 2) / (s2.sum() + 1e-12))
    # off-diagonal |cosine| distribution on a random subset (avoid O(n^2) blowup)
    idx = torch.randperm(n_active, device=dev)[: min(4000, n_active)]
    G = (D[idx] @ D[idx].T).abs()
    off = G[~torch.eye(idx.numel(), dtype=torch.bool, device=dev)]
    return {"n_active_features": n_active, "d_model": d_in, "dict_size": H,
            "decoder_effective_rank": round(pr, 1),
            "mean_abs_offdiag_cosine": round(float(off.mean()), 4),
            "frac_pairs_cos_gt_0.5": round(float((off > 0.5).float().mean()), 4),
            "frac_pairs_cos_gt_0.9": round(float((off > 0.9).float().mean()), 4)}


def main():  # noqa: D103
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    pos_type = np.asarray(pq.read_table(Path(args.store) / "token_labels.parquet")
                          .column("position_type").to_pylist(), dtype=object)
    report = {"store": str(args.store), "layers": {}}
    for L in args.layers:
        print(f"[rank] layer {L}: activation effective rank ...")
        r = activation_rank(args.store, L, pos_type, args.sample_tokens, dev, rng)
        report["layers"][str(L)] = r
        for b in BANDS + ["overall"]:
            if b in r:
                print(f"   {b:8s} PR={r[b]['participation_ratio']:7.1f} "
                      f"k99={r[b]['k_99pct_var']:5d} k90={r[b]['k_90pct_var']:5d} (d_model={r[b]['d_model']})")
    if args.sae:
        print(f"[rank] decoder redundancy for {args.sae} ...")
        report["decoder"] = decoder_redundancy(args.sae, dev)
        print(f"   {json.dumps(report['decoder'])}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"[rank] wrote {args.out_json}")


if __name__ == "__main__":
    main()
