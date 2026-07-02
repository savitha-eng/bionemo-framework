#!/usr/bin/env python
"""Illustrate the cross-modal RESIDUAL-STREAM geometry that underlies the cross-modal null.

Makes a 4-panel figure:
  (a) UMAP of PROTEIN vs TEXT raw residuals (cosine metric) -> two separated clouds = orthogonal subspaces
  (b) UMAP of DNA vs TEXT raw residuals                      -> overlapping = shared subspace
  (c) per-modality vector-NORM distributions                 -> bio tokens carry far larger magnitude
  (d) RAW / CODE / BIN cross-modal cosine bars               -> CODE (magnitude-weighted) looks aligned,
      but BIN (magnitude removed) collapses -> the apparent code alignment is a MAGNITUDE artifact.

Raw residuals are sampled row-aligned to token_labels position_type. Cosine/norm stats are printed too.
"""
import argparse, glob
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae.activation_store import shard_table_to_array
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap


def sample_bands(store, layer, bands, n_per, tl_name="token_labels.parquet", max_shards=12, seed=0):
    """Return {band: [n_per, H] raw residuals} sampled row-aligned to token_labels."""
    rng = np.random.default_rng(seed)
    tl = pq.read_table(Path(store) / tl_name).column("position_type").to_numpy(zero_copy_only=False)
    shards = sorted(glob.glob(str(Path(store) / f"layer{layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:max_shards]
    got = {b: [] for b in bands}; row0 = 0
    for sp in shards:
        X = shard_table_to_array(pq.read_table(sp)).astype(np.float32); n = X.shape[0]
        bnd = tl[row0:row0 + n]; row0 += n
        for b in bands:
            if len(got[b]) >= n_per:
                continue
            idx = np.where(bnd == b)[0]
            if len(idx):
                take = rng.choice(idx, size=min(len(idx), n_per - len(got[b])), replace=False)
                got[b].append(X[take])
        if all(len(np.concatenate(got[b])) >= n_per for b in bands if got[b]):
            if all(got[b] for b in bands):
                break
    return {b: (np.concatenate(v)[:n_per] if v else np.zeros((0, X.shape[1]))) for b, v in got.items()}


def mean_cos(A, B, npair=20000, seed=0):
    rng = np.random.default_rng(seed)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    i = rng.integers(0, len(An), npair); j = rng.integers(0, len(Bn), npair)
    return float((An[i] * Bn[j]).sum(1).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protein-store", required=True); p.add_argument("--dna-store", required=True)
    p.add_argument("--layer", type=int, default=16); p.add_argument("--n", type=int, default=2500)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    print("sampling protein store...", flush=True)
    pr = sample_bands(a.protein_store, a.layer, ["protein", "text"], a.n)
    print("sampling dna store...", flush=True)
    dn = sample_bands(a.dna_store, a.layer, ["dna", "text"], a.n)

    # cosine + norm stats
    stats = {}
    stats["protein RAW cos(protein,text)"] = mean_cos(pr["protein"], pr["text"])
    stats["protein RAW cos(protein,protein)"] = mean_cos(pr["protein"], pr["protein"])
    stats["dna RAW cos(dna,text)"] = mean_cos(dn["dna"], dn["text"])
    stats["dna RAW cos(dna,dna)"] = mean_cos(dn["dna"], dn["dna"])
    for k, v in stats.items():
        print(f"  {k} = {v:+.3f}")
    norms = {"protein": np.linalg.norm(pr["protein"], axis=1), "text(pro)": np.linalg.norm(pr["text"], axis=1),
             "dna": np.linalg.norm(dn["dna"], axis=1), "text(dna)": np.linalg.norm(dn["text"], axis=1)}
    for k, v in norms.items():
        print(f"  |{k}| median={np.median(v):.1f}")

    def embed(A, B):
        X = np.concatenate([A, B]); lab = np.array([0] * len(A) + [1] * len(B))
        emb = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(X)
        return emb, lab

    print("UMAP protein...", flush=True); ep, lp = embed(pr["protein"], pr["text"])
    print("UMAP dna...", flush=True); ed, ld = embed(dn["dna"], dn["text"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    for axi, (emb, lab, names, cols, title, cos) in enumerate([
        (ep, lp, ["protein", "text"], ["#2563eb", "#9333ea"],
         f"(a) Protein L{a.layer}: protein vs text residuals", stats["protein RAW cos(protein,text)"]),
        (ed, ld, ["dna", "text"], ["#0891b2", "#9333ea"],
         f"(b) DNA L{a.layer}: dna vs text residuals", stats["dna RAW cos(dna,text)"])]):
        A = ax[0, axi]
        for c in (0, 1):
            m = lab == c
            A.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.4, c=cols[c], label=names[c])
        A.set_title(f"{title}\nmean cross-modal cos = {cos:+.2f}", fontsize=11)
        A.legend(markerscale=3, fontsize=9); A.set_xticks([]); A.set_yticks([])

    # (c) norm distributions
    A = ax[1, 0]
    for k, c in zip(["protein", "text(pro)", "dna", "text(dna)"], ["#2563eb", "#c084fc", "#0891b2", "#9333ea"]):
        A.hist(norms[k], bins=60, alpha=0.5, density=True, label=f"{k} (med {np.median(norms[k]):.0f})", color=c)
    A.set_title("(c) residual vector norms per modality\n(bio tokens carry larger magnitude)", fontsize=11)
    A.set_xlabel("||residual||"); A.legend(fontsize=8)

    # (d) RAW/CODE/BIN cross-modal cosine bars (from eq7_verify, hard-coded measured values)
    A = ax[1, 1]
    labels = ["RAW\n(residual)", "CODE\n(mag-weighted)", "BIN\n(mag removed)"]
    prot = [-0.06, 0.77, 0.19]; dna = [0.37, 0.56, 0.10]
    x = np.arange(3); w = 0.35
    A.bar(x - w / 2, prot, w, label="protein", color="#2563eb")
    A.bar(x + w / 2, dna, w, label="dna", color="#0891b2")
    A.axhline(0, color="#888", lw=0.8); A.set_xticks(x); A.set_xticklabels(labels, fontsize=9)
    A.set_ylabel("cross-modal cosine (co-firing feats)")
    A.set_title("(d) apparent CODE alignment collapses when\nmagnitude is removed (BIN) = magnitude artifact", fontsize=11)
    A.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
