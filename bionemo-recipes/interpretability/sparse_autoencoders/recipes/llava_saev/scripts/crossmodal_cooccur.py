#!/usr/bin/env python
"""SAE-V-faithful, PER-SAMPLE cross-modal detection (the paper requires co-firing within one sample).

For each protein (sample), encodes its tokens and finds features that fire MEANINGFULLY on BOTH its
protein residues AND its text tokens *within that same protein*. A feature is genuinely cross-modal
if it co-fires across MANY samples. For the top such features we compute SAE-V Eq.7 (rank-paired cosine
of the top-K protein vs top-K text token activations, normalized, within-sample) and dump a same-protein
example showing the residue-side and text-side together.

Usage: crossmodal_cooccur.py --sae <ckpt> --store <store> --layer L --shards N [--tau 2.0 --min-proteins 5]
"""
import argparse, glob
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
import sys; sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True); p.add_argument("--shards", type=int, default=40)
    p.add_argument("--tau", type=float, default=2.0, help="min activation to count as 'fires' in a band")
    p.add_argument("--min-proteins", type=int, default=5, help="co-fire in >=this many samples to be cross-modal")
    p.add_argument("--topk", type=int, default=8, help="K for the Eq.7 within-sample paired cosine")
    p.add_argument("--dump-json", default=None, help="write the full cross-modal feature list to this JSON (for the explorer page)")
    p.add_argument("--pair", default="protein-text",
                   choices=["image-text", "protein-text", "protein-go", "go-text", "protein-reasoning",
                            "protein-answer", "protein-prompt", "go-reasoning", "go-answer"])
    args = p.parse_args()
    dev = "cuda"
    ba, bb = args.pair.split("-")

    # role bands (prompt/reasoning/answer) need the role sidecar; else use position_type
    role_bands = {"prompt", "reasoning", "answer"}
    use_role = bool(role_bands & {ba, bb})
    labels_file = "token_labels_with_role.parquet" if use_role else "token_labels.parquet"
    col = "role" if use_role else "position_type"
    lp = Path(args.store) / labels_file
    if use_role and not lp.exists():
        raise SystemExit(f"[cooccur] {labels_file} missing — run make_role_sidecar.py --store {args.store} first")
    tl = pq.read_table(lp)
    row_pid = tl.column("protein_id").to_numpy(zero_copy_only=False)   # to_numpy >> to_pylist on 421M rows
    row_band = tl.column(col).to_numpy(zero_copy_only=False)
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:args.shards]
    R = sum(pq.read_metadata(s).num_rows for s in shards)
    row_pid, row_band = row_pid[:R], row_band[:R]

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim

    # STREAM per-protein from ordered shards (proteins are contiguous in token order) -> holds ONE
    # protein in RAM at a time, so this scales to the full 117k-protein / 421M-token store (no OOM).
    def stream_proteins():
        buf_X, buf_b, cur, row0 = [], [], None, 0
        for sp in shards:
            Xs = shard_table_to_array(pq.read_table(sp)).astype(np.float32)
            n = Xs.shape[0]
            ps = row_pid[row0:row0 + n]; bs = row_band[row0:row0 + n]; row0 += n
            bnd = [0] + (np.where(ps[1:] != ps[:-1])[0] + 1).tolist() + [n]
            for k in range(len(bnd) - 1):
                s, e = bnd[k], bnd[k + 1]; p = ps[s]
                if cur is None:
                    cur = p
                if p != cur:
                    yield np.concatenate(buf_X), np.concatenate(buf_b); buf_X, buf_b, cur = [], [], p
                buf_X.append(Xs[s:e]); buf_b.append(bs[s:e])
        if buf_X:
            yield np.concatenate(buf_X), np.concatenate(buf_b)

    cofire = np.zeros(H, dtype=np.int32)          # # samples where feature fires on BOTH bands (>tau)
    cos_sum = np.zeros(H, dtype=np.float64); cos_n = np.zeros(H, dtype=np.int32)
    fire_a = np.zeros(H, dtype=np.int32); fire_b = np.zeros(H, dtype=np.int32)  # single-band sample counts
    n_samples = 0
    with torch.no_grad():
        for Xp, bp in stream_proteins():
            n_samples += 1
            ia = np.where(bp == ba)[0]; ib = np.where(bp == bb)[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            Xa = torch.from_numpy(Xp[ia]).to(dev); Xb = torch.from_numpy(Xp[ib]).to(dev)
            ca = sae.encode(Xa); cb = sae.encode(Xb)
            fa = (ca.max(0).values > args.tau).cpu().numpy(); fb = (cb.max(0).values > args.tau).cpu().numpy()
            fire_a += fa; fire_b += fb
            both = fa & fb
            cofire += both
            fi = np.where(both)[0]
            if len(fi):
                k = min(args.topk, ca.shape[0], cb.shape[0])
                ta = torch.topk(ca[:, fi], k, dim=0).indices; tb = torch.topk(cb[:, fi], k, dim=0).indices
                Xan = torch.nn.functional.normalize(Xa, dim=1); Xbn = torch.nn.functional.normalize(Xb, dim=1)
                cs = (Xan[ta] * Xbn[tb]).sum(-1).mean(0).cpu().numpy()   # rank-paired Eq.7 cosine
                cos_sum[fi] += cs; cos_n[fi] += 1
            if n_samples % 5000 == 0:
                print(f"  [{args.pair}] {n_samples} proteins...", flush=True)
    xm = cofire >= args.min_proteins
    cos_mean = np.where(cos_n > 0, cos_sum / np.maximum(cos_n, 1), 0.0)
    print(f"[cooccur] pair={args.pair} samples={n_samples} tau={args.tau} K={args.topk}")
    print(f"[cooccur] features co-firing on BOTH bands in >= {args.min_proteins} samples: {int(xm.sum())} / {H}")
    print(f"[cooccur]   ...and within-sample Eq.7 cosine > 0.3: {int((xm & (cos_mean > 0.3)).sum())}")
    print(f"[cooccur]   ...cosine > 0.5: {int((xm & (cos_mean > 0.5)).sum())}")
    top = np.argsort(-(cofire + cos_mean))[:15]
    print("[cooccur] top cross-modal features (id, #co-fire-samples, mean within-sample cosine):")
    for f in top:
        if cofire[f] >= args.min_proteins:
            print(f"    F{int(f):<6} cofire={int(cofire[f])}/{n_samples}  cos={cos_mean[f]:.3f}  (fires {ba} in {int(fire_a[f])}, {bb} in {int(fire_b[f])})")

    if args.dump_json:
        import json
        xf = np.where(xm)[0]
        xf = xf[np.argsort(-cos_mean[xf])]  # most-aligned first
        feats = [{"feature_id": int(f), "cofire": int(cofire[f]), "cofire_frac": round(float(cofire[f]) / n_samples, 4),
                  "cosine": round(float(cos_mean[f]), 3), f"fire_{ba}": int(fire_a[f]), f"fire_{bb}": int(fire_b[f])}
                 for f in xf]
        out = {"pair": args.pair, "layer": args.layer, "n_samples": int(n_samples), "n_latents": int(H),
               "tau": args.tau, "topk": args.topk, "min_proteins": args.min_proteins,
               "n_cofire": int(xm.sum()), "n_aligned_0.3": int((xm & (cos_mean > 0.3)).sum()),
               "n_aligned_0.5": int((xm & (cos_mean > 0.5)).sum()), "features": feats}
        json.dump(out, open(args.dump_json, "w"), indent=2)
        print(f"[cooccur] dumped {len(feats)} cross-modal features -> {args.dump_json}", flush=True)


if __name__ == "__main__":
    main()
