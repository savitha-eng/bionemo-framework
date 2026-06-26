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
    p.add_argument("--pair", default="protein-text",
                   choices=["protein-text", "protein-go", "go-text", "protein-reasoning",
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
    row_pid = np.asarray(tl.column("protein_id").to_pylist(), dtype=object)
    row_band = np.asarray(tl.column(col).to_pylist(), dtype=object)
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:args.shards]
    R = sum(pq.read_metadata(s).num_rows for s in shards)
    row_pid, row_band = row_pid[:R], row_band[:R]
    X = np.concatenate([shard_table_to_array(pq.read_table(s)) for s in shards], axis=0)[:R].astype(np.float32)

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim

    # group rows by protein (contiguous-ish; just bucket)
    order = np.argsort(row_pid, kind="stable")
    pids, starts = np.unique(row_pid[order], return_index=True)
    groups = np.split(order, starts[1:])
    # NOTE: normalize per-protein inside the loop (normalizing all ~8M tokens on GPU OOMs)

    cofire = np.zeros(H, dtype=np.int32)          # # samples where feature fires on BOTH bands (>tau)
    cos_sum = np.zeros(H, dtype=np.float64); cos_n = np.zeros(H, dtype=np.int32)
    fire_a = np.zeros(H, dtype=np.int32); fire_b = np.zeros(H, dtype=np.int32)  # single-band sample counts
    with torch.no_grad():
        for g in groups:
            rb = row_band[g]
            ia = g[rb == ba]; ib = g[rb == bb]
            if len(ia) == 0 or len(ib) == 0:
                continue
            Xa = torch.from_numpy(X[ia]).to(dev); Xb = torch.from_numpy(X[ib]).to(dev)  # per-protein, small
            ca = sae.encode(Xa)   # (na, H)
            cb = sae.encode(Xb)
            amax = ca.max(0).values; bmax = cb.max(0).values
            fa = (amax > args.tau).cpu().numpy(); fb = (bmax > args.tau).cpu().numpy()
            fire_a += fa; fire_b += fb
            both = fa & fb
            cofire += both
            # Eq.7 within-sample cosine for features co-firing in THIS sample
            fi = np.where(both)[0]
            if len(fi):
                # shared k so the rank-paired tensors match even when a band is short (< K tokens)
                k = min(args.topk, ca.shape[0], cb.shape[0])
                ta = torch.topk(ca[:, fi], k, dim=0).indices  # (k, nf)
                tb = torch.topk(cb[:, fi], k, dim=0).indices
                Xan = torch.nn.functional.normalize(Xa, dim=1)  # per-protein normalize (no global OOM)
                Xbn = torch.nn.functional.normalize(Xb, dim=1)
                za = Xan[ta]   # (k, nf, d)
                zb = Xbn[tb]
                cs = (za * zb).sum(-1).mean(0).cpu().numpy()  # rank-paired cosine, mean over K -> (nf,)
                cos_sum[fi] += cs; cos_n[fi] += 1

    n_samples = len(groups)
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


if __name__ == "__main__":
    main()
