#!/usr/bin/env python
"""UMAP of the SAE-PRODUCED feature codes (not raw residuals), balanced vs unbalanced loss.

For each of two SAEs trained on the same layer-16 protein+text activations
  - unbalanced (text-dominated mix)
  - balanced   (~50/50 protein:text mix)
we sample the SAME protein/text residual tokens, encode them through the SAE, and UMAP the
resulting sparse CODES (feature-activation vectors), colored by modality.

The point of the figure: what changes in the SAE's own feature space when we rebalance. With the
unbalanced loss the protein tokens collapse into a thin sliver of code space (few protein features
to spread them out); with the balanced loss the protein tokens open up into their own structured
region -- the picture behind the ~100 -> ~1,800 protein-feature count in Figure 1.

Codes are L2-normalized before UMAP (cosine geometry) so the plot reflects WHICH features fire, not
raw magnitude.
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
import umap


def sample_bands(store, layer, bands, n_per, tl_name="token_labels.parquet", max_shards=12, seed=0):
    """Return {band: [n_per, H] raw residuals} sampled row-aligned to token_labels position_type."""
    rng = np.random.default_rng(seed)
    tl = pq.read_table(Path(store) / tl_name).column("position_type").to_numpy(zero_copy_only=False)
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
                take = rng.choice(idx, size=min(len(idx), need), replace=False)
                got[b].append(X[take])
        if all(sum(len(g) for g in got[b]) >= n_per for b in bands):
            break
    return {b: (np.concatenate(v)[:n_per] if v else np.zeros((0, X.shape[1]), np.float32)) for b, v in got.items()}


def load_sae(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                         for k, v in ck["model_state_dict"].items()}, strict=False)
    return sae.to(dev).eval()


def encode(sae, X, dev, bs=8192):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x = torch.from_numpy(X[s:s + bs]).to(dev)
            out.append(sae.encode(x).float().cpu().numpy())
    return np.concatenate(out)


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

    print("sampling protein+text residuals...", flush=True)
    pr = sample_bands(a.protein_store, a.layer, ["protein", "text"], a.n)
    X = np.concatenate([pr["protein"], pr["text"]])
    lab = np.array([0] * len(pr["protein"]) + [1] * len(pr["text"]))
    print(f"  protein={len(pr['protein'])} text={len(pr['text'])}", flush=True)

    panels = []
    for tag, sae_path in [("unbalanced loss", a.unbalanced_sae), ("balanced loss", a.balanced_sae)]:
        print(f"encoding + UMAP: {tag} ({sae_path})", flush=True)
        sae = load_sae(sae_path, dev)
        C = encode(sae, X, dev)
        # fraction of protein tokens' active features that live in features text ~never uses
        act = C > 0
        pfrac = act[lab == 0].mean(0); tfrac = act[lab == 1].mean(0)
        prot_only = int(((pfrac > 0.01) & (tfrac < 0.001)).sum())
        print(f"  code dim={C.shape[1]}  live feats={int((act.any(0)).sum())}  "
              f"protein-preferring feats={prot_only}", flush=True)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(Cn)
        panels.append((tag, emb, prot_only))
        del sae
        if dev == "cuda":
            torch.cuda.empty_cache()

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for A, (tag, emb, prot_only) in zip(ax, panels):
        for c, name, col in [(0, "protein", "#2563eb"), (1, "text", "#9333ea")]:
            m = lab == c
            A.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.45, c=col, label=name)
        A.set_title(f"SAE feature codes — {tag}\n{prot_only} protein-preferring features", fontsize=12)
        A.legend(markerscale=3, fontsize=10); A.set_xticks([]); A.set_yticks([])
    fig.suptitle(f"Layer {a.layer}: SAE-produced feature space, balanced vs unbalanced loss", fontsize=13)
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
