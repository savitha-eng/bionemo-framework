#!/usr/bin/env python
"""Per-prefix analysis of a Matryoshka TopK SAE — the hierarchy readout (coarse->fine).

Each nested prefix [0:b] is itself a valid SAE. For each prefix this reports:
  - var_exp_norm  : reconstruction quality using only the first b latents
  - per-band coverage: # features (within the prefix) that ever fire on protein / text / both
The question it answers: do COARSE (small-prefix) latents carry MODALITY-level structure while FINE
(later) latents add specific concepts? And does the protein/text/both mix shift with scale?
Usage: matryoshka_prefix.py --sae <ckpt> --store <store> --layer L [--out-json out.json]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

BANDS = ["protein", "go", "text"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True); p.add_argument("--out-json", default=None)
    p.add_argument("--max-shards", type=int, default=20, help="cap shards (var_exp needs only a representative sample)")
    p.add_argument("--bounds", default=None, help="override prefix bounds as comma frac list, e.g. 0.5,0.75,0.875,0.9375,1.0 (impose Matryoshka nesting on a FLAT sae for the control)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    H = sae.hidden_dim
    if args.bounds:
        bounds = sorted(set(min(H, max(1, int(round(float(fr) * H)))) for fr in args.bounds.split(",")))
    else:
        bounds = sae.matryoshka_bounds or [H]   # falls back to single full prefix for a vanilla TopK
    print(f"[prefix] H={H} bounds={bounds}", flush=True)

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    band = tl.column("position_type").to_numpy(zero_copy_only=False)
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:args.max_shards]
    R = sum(pq.read_metadata(s).num_rows for s in shards)
    band = band[:R]
    bcode = np.select([band == b for b in BANDS], list(range(len(BANDS))), default=-1)

    # accumulate, per prefix: which latents fire on each band; and prefix recon error for var_exp.
    # BATCHED (dense [batch,H] codes would OOM on a full shard) — encode 8192 tokens at a time.
    EB = 8192
    fired = {b: torch.zeros(len(BANDS), H, dtype=torch.bool, device=dev) for b in bounds}
    sse = {b: 0.0 for b in bounds}; sst = 0.0; row0 = 0
    bt = torch.from_numpy(bcode).to(dev)
    with torch.no_grad():
        for sp in shards:
            X = shard_table_to_array(pq.read_table(sp)).astype(np.float32)
            n = X.shape[0]
            for s in range(0, n, EB):
                e = min(n, s + EB)
                x = torch.from_numpy(X[s:e]).to(dev)
                codes = sae.encode(x)
                xn, info = sae._normalize(x) if sae.normalize_input else (x, {})
                sst += float((xn ** 2).sum())               # normalized tokens ~ zero mean -> var ~ sum(xn^2)
                bb = bt[row0 + s: row0 + e]
                for b in bounds:
                    cp = codes.clone(); cp[:, b:] = 0.0      # prefix = first b latents
                    recon = sae.decode(cp, info)
                    rn = (recon - info["mu"]) / info["std"] if sae.normalize_input else recon
                    sse[b] += float(((rn - xn) ** 2).sum())
                    act = cp > 0
                    for bi in range(len(BANDS)):
                        m = bb == bi
                        if m.any(): fired[b][bi] |= act[m].any(0)
            row0 += n

    out = {"sae": args.sae, "layer": args.layer, "H": H, "bounds": list(map(int, bounds)), "prefixes": []}
    print(f"{'prefix(latents)':>16} {'var_exp':>8} {'protein':>8} {'text':>7} {'both':>6} {'dead':>6}")
    for b in bounds:
        f = fired[b].cpu().numpy()
        # only latents within the prefix count
        prot = int(f[0, :b].sum()); txt = int(f[2, :b].sum())
        both = int((f[0, :b] & f[2, :b]).sum()); live = int((f[:, :b].any(0)).sum())
        ve = 1.0 - sse[b] / (sst + 1e-8)
        print(f"{b:>16} {ve:>8.3f} {prot:>8} {txt:>7} {both:>6} {b-live:>6}")
        out["prefixes"].append({"latents": int(b), "var_exp_norm": round(ve, 4),
                                "protein_feats": prot, "text_feats": txt, "both_feats": both, "dead": b - live})
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2); print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
