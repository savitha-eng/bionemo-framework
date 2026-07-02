#!/usr/bin/env python
"""Per-feature cross-modal CO-ACTIVATION score in the SHARED SAE feature basis.

Motivation: SAE-V's ω_k cosines RAW hidden states, which is blind to cross-modality when the two
modalities occupy different residual subspaces (as protein/text do here). The SAE feature dictionary
IS the shared space — feature k is the same latent for both modalities. So the right question is
whether k is USED by both modalities on the SAME samples.

For each feature k, across samples s:
  pf = k fires (>tau) on the PROTEIN tokens of s ;  tf = k fires (>tau) on the TEXT tokens of s
Build the 2x2 contingency over samples and report the phi coefficient:
  phi = (n11*n00 - n10*n01) / sqrt((n11+n10)(n01+n00)(n11+n01)(n10+n00))
phi>0  => fires on protein & text of the SAME samples (cross-modal binding of a shared concept).
Generic always-on feats have ~no variance => phi~0 (correctly excluded, unlike co-fire counts).
Also reports Pearson r on the continuous per-sample max activations. Selective = fires on a moderate
fraction of samples (not ~0, not ~all).
"""
import argparse, glob, json, math
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True); p.add_argument("--shards", type=int, default=10)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--band-a", default="protein", help="modality A band (DNA store: dna)")
    p.add_argument("--band-b", default="text", help="modality B band")
    p.add_argument("--id-col", default="protein_id", help="per-sample id col in token_labels (DNA: sequence_id)")
    p.add_argument("--token-labels", default="token_labels.parquet", help="per-token band sidecar filename")
    p.add_argument("--out-json", required=True); p.add_argument("--device", default="cuda")
    args = p.parse_args(); dev = args.device if torch.cuda.is_available() else "cpu"
    PROT = args.band_a; TEXT = args.band_b

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

    # 2x2 contingency counts per feature; + sums for Pearson on continuous max-acts
    n11 = np.zeros(H); n10 = np.zeros(H); n01 = np.zeros(H); n00 = np.zeros(H)
    sp_ = np.zeros(H); st_ = np.zeros(H); spp = np.zeros(H); stt = np.zeros(H); spt = np.zeros(H)
    ns = 0
    with torch.no_grad():
        for Xp, bp in stream():
            ia = np.where(bp == PROT)[0]; ib = np.where(bp == TEXT)[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            ns += 1
            ca = sae.encode(torch.from_numpy(Xp[ia]).to(dev)).max(0).values.cpu().numpy()  # [H] max over protein toks
            cb = sae.encode(torch.from_numpy(Xp[ib]).to(dev)).max(0).values.cpu().numpy()  # [H] max over text toks
            pf = ca > args.tau; tf = cb > args.tau
            n11 += pf & tf; n10 += pf & ~tf; n01 += ~pf & tf; n00 += ~pf & ~tf
            sp_ += ca; st_ += cb; spp += ca * ca; stt += cb * cb; spt += ca * cb

    # phi coefficient (binary) + Pearson r (continuous)
    def safe(x): return np.maximum(x, 1e-9)
    denom = np.sqrt(safe((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    phi = (n11 * n00 - n10 * n01) / denom
    cov = spt / ns - (sp_ / ns) * (st_ / ns)
    vp = spp / ns - (sp_ / ns) ** 2; vt = stt / ns - (st_ / ns) ** 2
    r = cov / np.sqrt(safe(vp) * safe(vt))
    fire_p = n11 + n10; fire_t = n11 + n01  # samples firing on each side
    sel = (fire_p > 0.02 * ns) & (fire_p < 0.9 * ns) & (fire_t > 0.02 * ns) & (fire_t < 0.9 * ns)

    feats = [{"feature_id": int(k), "phi": round(float(phi[k]), 3), "pearson_r": round(float(r[k]), 3),
              "fire_protein": int(fire_p[k]), "fire_text": int(fire_t[k]), "fire_both": int(n11[k]),
              "selective": bool(sel[k])} for k in range(H) if fire_p[k] > 0 and fire_t[k] > 0]
    feats.sort(key=lambda d: -d["phi"])
    json.dump({"n_samples": ns, "tau": args.tau, "layer": args.layer, "features": feats},
              open(args.out_json, "w"), indent=2)
    print(f"=== cross-modal co-activation (l{args.layer}, {ns} samples, tau={args.tau}) ===")
    print(f"features firing on both sides (>=1 sample): {len(feats)}")
    print(f"  phi>0.3: {sum(f['phi']>0.3 for f in feats)}   phi>0.5: {sum(f['phi']>0.5 for f in feats)}"
          f"   selective & phi>0.3: {sum(f['phi']>0.3 and f['selective'] for f in feats)}")
    print("top by phi (selective ones marked *):")
    for f in feats[:20]:
        print(f"  F{f['feature_id']:<6} phi={f['phi']:+.3f} r={f['pearson_r']:+.3f} "
              f"fireP={f['fire_protein']} fireT={f['fire_text']} both={f['fire_both']} {'*SEL' if f['selective'] else ''}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
