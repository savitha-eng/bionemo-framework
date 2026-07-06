#!/usr/bin/env python
"""UMAP of the RAW residual embeddings vs the SAE code embeddings (unbalanced & balanced).

Three panels, SAME protein/text tokens throughout, cosine-metric UMAP, colored by modality:
  (a) raw residual stream h  (the non-SAE embedding)
  (b) SAE codes z = encode(h), UNBALANCED loss
  (c) SAE codes z = encode(h), BALANCED loss

Point: the SAE encode is (roughly) an overcomplete rotation + sparsify of h, so the gross
protein/text separation is inherited from the raw geometry and survives in code space regardless of
balancing. What balancing changes is which/how-many features carry protein (see feature_selectivity.py),
not the 2D token layout.
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


def encode(path, X, dev, bs=8192):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                         for k, v in ck["model_state_dict"].items()}, strict=False)
    sae = sae.to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(sae.encode(torch.from_numpy(X[s:s + bs]).to(dev)).float().cpu().numpy())
    del sae
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def umap2d(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(Mn)


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

    print("UMAP raw residuals...", flush=True); e_raw = umap2d(X)
    print("encode + UMAP unbalanced...", flush=True); e_unb = umap2d(encode(a.unbalanced_sae, X, dev))
    print("encode + UMAP balanced...", flush=True); e_bal = umap2d(encode(a.balanced_sae, X, dev))

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    for A, emb, title in [
        (ax[0], e_raw, "(a) raw residual embedding\n(no SAE)"),
        (ax[1], e_unb, "(b) SAE code embedding\nunbalanced loss"),
        (ax[2], e_bal, "(c) SAE code embedding\nbalanced loss")]:
        for c, name, col in [(0, "protein", "#2563eb"), (1, "text", "#9333ea")]:
            m = lab == c
            A.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.45, c=col, label=name)
        A.set_title(title, fontsize=12); A.legend(markerscale=3, fontsize=10)
        A.set_xticks([]); A.set_yticks([])
    fig.suptitle(f"Layer {a.layer}: raw residual geometry vs SAE code geometry (same tokens)", fontsize=13)
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
