#!/usr/bin/env python
"""Positive-control verification of the SAE-V Eq.7 cross-modal cosine.

The cross-modal null (0 aligned, cos~0.04) is suspiciously robust. This checks whether the metric
CAN return a high cosine at all, and whether it's dominated by residual-subspace separation.

For features that co-fire on BOTH protein & text bands, compute the rank-paired top-K cosine in TWO
representations:
  RAW   = cosine of the raw residual-stream vectors (what crossmodal_cooccur.py currently uses)
  CODE  = cosine of the SAE code vectors (feature-space; arguably the right space for 'feature alignment')
each under THREE conditions:
  P-T   = protein top-K  vs  text top-K      (the real cross-modal test)
  P-P   = protein top-K  vs  protein top-K    (disjoint even/odd rank halves)  <- positive control
  T-T   = text    top-K  vs  text top-K       (disjoint even/odd rank halves)  <- positive control

If P-P / T-T are high but P-T ~0.04:
  - RAW high within / low across  -> null reflects residual SUBSPACE separation (metric geometry), not features
  - CODE high within / low across -> the null is REAL in feature space (protein & text fire disjoint features)
"""
import argparse, glob
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array


def paired_cos(A, B, k):
    """rank-paired mean cosine between rows of A (top-k) and rows of B (top-k), each L2-normalized."""
    An = torch.nn.functional.normalize(A, dim=1); Bn = torch.nn.functional.normalize(B, dim=1)
    return (An * Bn).sum(-1).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True); p.add_argument("--shards", type=int, default=10)
    p.add_argument("--tau", type=float, default=1.0); p.add_argument("--topk", type=int, default=5)
    p.add_argument("--min-proteins", type=int, default=5); p.add_argument("--max-feats", type=int, default=200)
    p.add_argument("--band-a", default="protein", help="modality A band (DNA store: dna)")
    p.add_argument("--band-b", default="text", help="modality B band")
    p.add_argument("--id-col", default="protein_id", help="per-sample id col (DNA store: sequence_id)")
    p.add_argument("--token-labels", default="token_labels.parquet", help="per-token band sidecar filename")
    args = p.parse_args(); dev = "cuda"
    BAND_A = args.band_a; BAND_B = args.band_b

    tl = pq.read_table(Path(args.store) / args.token_labels)
    row_pid = tl.column(args.id_col).to_numpy(zero_copy_only=False)
    row_band = tl.column("position_type").to_numpy(zero_copy_only=False)
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:args.shards]
    R = sum(pq.read_metadata(s).num_rows for s in shards)
    row_pid, row_band = row_pid[:R], row_band[:R]

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    H = sae.hidden_dim

    def stream():
        buf_X, buf_b, cur, row0 = [], [], None, 0
        for sp in shards:
            Xs = shard_table_to_array(pq.read_table(sp)).astype(np.float32); n = Xs.shape[0]
            ps = row_pid[row0:row0 + n]; bs = row_band[row0:row0 + n]; row0 += n
            bnd = [0] + (np.where(ps[1:] != ps[:-1])[0] + 1).tolist() + [n]
            for kk in range(len(bnd) - 1):
                s, e = bnd[kk], bnd[kk + 1]; pp = ps[s]
                if cur is None: cur = pp
                if pp != cur: yield np.concatenate(buf_X), np.concatenate(buf_b); buf_X, buf_b, cur = [], [], pp
                buf_X.append(Xs[s:e]); buf_b.append(bs[s:e])
        if buf_X: yield np.concatenate(buf_X), np.concatenate(buf_b)

    # accumulate per-condition cosine sums over co-firing features.
    # BIN_* = cosine of BINARIZED codes (feature-set overlap, magnitude removed) -> rules out a few
    # high-magnitude ubiquitous features inflating the raw code cosine.
    acc = {c: [0.0, 0] for c in ["RAW_PT", "RAW_PP", "RAW_TT", "CODE_PT", "CODE_PP", "CODE_TT",
                                  "BIN_PT", "BIN_PP", "BIN_TT"]}
    cofire = np.zeros(H, dtype=np.int32); n_samples = 0
    K = args.topk
    with torch.no_grad():
        for Xp, bp in stream():
            n_samples += 1
            ia = np.where(bp == BAND_A)[0]; ib = np.where(bp == BAND_B)[0]
            if len(ia) < 2 * K or len(ib) < 2 * K: continue
            Xa = torch.from_numpy(Xp[ia]).to(dev); Xb = torch.from_numpy(Xp[ib]).to(dev)
            ca = sae.encode(Xa); cb = sae.encode(Xb)
            fa = (ca.max(0).values > args.tau); fb = (cb.max(0).values > args.tau)
            both = (fa & fb).cpu().numpy(); cofire += both
            fi = np.where(both)[0]
            for f in fi:
                # top-2K tokens each band by feature f; even/odd rank halves for within-modality control
                ra = torch.topk(ca[:, f], 2 * K).indices; rb = torch.topk(cb[:, f], 2 * K).indices
                pa1, pa2 = ra[0::2][:K], ra[1::2][:K]; pb1, pb2 = rb[0::2][:K], rb[1::2][:K]
                ta, tb = ra[:K], rb[:K]  # top-K for the cross test
                acc["RAW_PT"][0] += paired_cos(Xa[ta], Xb[tb], K); acc["RAW_PT"][1] += 1
                acc["RAW_PP"][0] += paired_cos(Xa[pa1], Xa[pa2], K); acc["RAW_PP"][1] += 1
                acc["RAW_TT"][0] += paired_cos(Xb[pb1], Xb[pb2], K); acc["RAW_TT"][1] += 1
                acc["CODE_PT"][0] += paired_cos(ca[ta], cb[tb], K); acc["CODE_PT"][1] += 1
                acc["CODE_PP"][0] += paired_cos(ca[pa1], ca[pa2], K); acc["CODE_PP"][1] += 1
                acc["CODE_TT"][0] += paired_cos(cb[pb1], cb[pb2], K); acc["CODE_TT"][1] += 1
                ba_, bb_ = (ca > 0).float(), (cb > 0).float()
                acc["BIN_PT"][0] += paired_cos(ba_[ta], bb_[tb], K); acc["BIN_PT"][1] += 1
                acc["BIN_PP"][0] += paired_cos(ba_[pa1], ba_[pa2], K); acc["BIN_PP"][1] += 1
                acc["BIN_TT"][0] += paired_cos(bb_[pb1], bb_[pb2], K); acc["BIN_TT"][1] += 1
            if n_samples % 2000 == 0: print(f"  {n_samples} proteins, cofire feats so far={int((cofire>=args.min_proteins).sum())}", flush=True)

    print(f"\n=== Eq.7 verification (l={args.layer}, {n_samples} proteins, tau={args.tau}, K={K}) ===")
    print(f"co-firing features (>= {args.min_proteins} samples): {int((cofire>=args.min_proteins).sum())}")
    print(f"{'condition':>10} {'mean cosine':>12} {'n':>10}")
    for c in ["RAW_PT", "RAW_PP", "RAW_TT", "CODE_PT", "CODE_PP", "CODE_TT", "BIN_PT", "BIN_PP", "BIN_TT"]:
        s, nn = acc[c]; print(f"{c:>10} {(s/max(nn,1)):>12.4f} {nn:>10}")
    print("\nInterpretation:")
    print("  RAW_PP/RAW_TT high & RAW_PT low  -> raw metric dominated by modality subspace separation")
    print("  CODE_PT high                     -> features DO align (null was a raw-space artifact)")
    print("  CODE_PT ~ RAW_PT ~ 0 & CODE_PP high -> null is REAL: protein & text fire disjoint features")


if __name__ == "__main__":
    main()
