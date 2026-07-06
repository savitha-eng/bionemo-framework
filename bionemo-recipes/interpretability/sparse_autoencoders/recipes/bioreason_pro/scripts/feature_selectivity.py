#!/usr/bin/env python
"""Per-FEATURE modality selectivity, balanced vs unbalanced loss.

Unlike the code UMAP (one dot per token, dominated by the fixed modality orthogonality), this puts
ONE DOT PER SAE FEATURE: x = fraction of TEXT tokens it fires on, y = fraction of PROTEIN tokens it
fires on. Protein-selective features sit in the top-left, text-selective in the bottom-right, shared
on the diagonal. Balancing the loss should visibly populate the protein-selective (top-left) region.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae.activation_store import shard_table_to_array
from sae.architectures import TopKSAE
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sample_bands(store, layer, bands, n_per, max_shards=12, seed=0):
    rng = np.random.default_rng(seed)
    tl = pq.read_table(Path(store) / "token_labels.parquet").column("position_type").to_numpy(zero_copy_only=False)
    shards = sorted(glob.glob(str(Path(store) / f"layer{layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:max_shards]
    got = {b: [] for b in bands}; row0 = 0
    for sp in shards:
        X = shard_table_to_array(pq.read_table(sp)).astype(np.float32); n = X.shape[0]
        bnd = tl[row0:row0 + n]; row0 += n
        for b in bands:
            if sum(len(g) for g in got[b]) >= n_per:
                continue
            idx = np.where(bnd == b)[0]
            if len(idx):
                need = n_per - sum(len(g) for g in got[b])
                got[b].append(X[rng.choice(idx, size=min(len(idx), need), replace=False)])
        if all(sum(len(g) for g in got[b]) >= n_per for b in bands):
            break
    return {b: (np.concatenate(v)[:n_per] if v else np.zeros((0, X.shape[1]), np.float32)) for b, v in got.items()}


def fire_fracs(path, X, lab, dev, bs=8192):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                         for k, v in ck["model_state_dict"].items()}, strict=False)
    sae = sae.to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append((sae.encode(torch.from_numpy(X[s:s + bs]).to(dev)) > 0).cpu().numpy())
    A = np.concatenate(out)
    del sae
    if dev == "cuda":
        torch.cuda.empty_cache()
    return A[lab == 0].mean(0), A[lab == 1].mean(0)   # protein-fire-frac, text-fire-frac per feature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protein-store", required=True)
    p.add_argument("--unbalanced-sae", required=True)
    p.add_argument("--balanced-sae", required=True)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pr = sample_bands(a.protein_store, a.layer, ["protein", "text"], a.n)
    X = np.concatenate([pr["protein"], pr["text"]])
    lab = np.array([0] * len(pr["protein"]) + [1] * len(pr["text"]))
    print(f"protein={len(pr['protein'])} text={len(pr['text'])}", flush=True)

    eps = 1.0 / (2 * a.n)   # floor so a feature that never fires plots on the axis edge, not at -inf
    panels = []
    for tag, path in [("unbalanced loss", a.unbalanced_sae), ("balanced loss", a.balanced_sae)]:
        pf, tf = fire_fracs(path, X, lab, dev)
        live = (pf > 0) | (tf > 0)
        pf, tf = pf[live], tf[live]
        prot_sel = int(((pf > 0.01) & (tf < 0.001)).sum())
        text_sel = int(((tf > 0.01) & (pf < 0.001)).sum())
        panels.append((tag, tf, pf, prot_sel, text_sel, int(live.sum())))
        print(f"  {tag}: live={int(live.sum())} protein-selective={prot_sel} text-selective={text_sel}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    for A, (tag, tf, pf, ps, ts, live) in zip(ax, panels):
        prot_m = (pf > 0.01) & (tf < 0.001)      # protein-selective (top-left)
        text_m = (tf > 0.01) & (pf < 0.001)      # text-selective (bottom-right)
        rest_m = ~(prot_m | text_m)
        A.scatter(tf[rest_m] + eps, pf[rest_m] + eps, s=5, alpha=0.2, c="#cbd5e1", edgecolors="none")
        A.scatter(tf[text_m] + eps, pf[text_m] + eps, s=11, alpha=0.55, c="#9333ea",
                  edgecolors="none", label=f"text-selective ({ts})")
        A.scatter(tf[prot_m] + eps, pf[prot_m] + eps, s=16, alpha=0.8, c="#2563eb",
                  edgecolors="none", label=f"protein-selective ({ps})")
        A.set_xscale("log"); A.set_yscale("log")
        A.plot([eps, 1], [eps, 1], ls="--", lw=0.8, c="#94a3b8")   # shared diagonal
        A.set_xlim(eps * 0.8, 1); A.set_ylim(eps * 0.8, 1)
        A.set_xlabel("fires on TEXT tokens (fraction)")
        A.set_title(f"{tag}", fontsize=12)
        A.legend(loc="lower left", fontsize=9, framealpha=0.9, markerscale=1.6)
        A.axhline(0.01, color="#2563eb", lw=0.5, ls=":"); A.axvline(0.001, color="#2563eb", lw=0.5, ls=":")
    ax[0].set_ylabel("fires on PROTEIN tokens (fraction)")
    fig.suptitle(f"Layer {a.layer}: per-feature modality selectivity — balancing populates the "
                 f"protein-selective corner", fontsize=12)
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
