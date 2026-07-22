#!/usr/bin/env python
"""Across all layers that have a balanced+unbalanced SAE pair, build two grids:

  1. per-feature modality selectivity scatter  (one dot per feature; x=text fire-frac, y=protein
     fire-frac), unbalanced vs balanced -> fig_layer_selectivity_grid.png
  2. token UMAP: raw residual / unbalanced code / balanced code (same tokens)
     -> fig_layer_umap_grid.png

Also prints/saves a cross-layer counts table (protein-selective / text-selective / dead per layer).
The scatter part is cheap; the UMAP part is the slow one (--no-umap to skip it).
"""
import argparse, glob, json
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


def load_sae(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                         for k, v in ck["model_state_dict"].items()}, strict=False)
    return sae.to(dev).eval()


def codes(sae, X, dev, bs=8192):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(sae.encode(torch.from_numpy(X[s:s + bs]).to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="14,16,18,20,22,24,28,30,32")
    p.add_argument("--store-tmpl", default="/data/savithas/phase3_subset_l{N}_dash")
    p.add_argument("--unbal-tmpl", default="/data/savithas/phase3_full/sae-l{N}-exp16-full/checkpoint_final.pt")
    p.add_argument("--bal-tmpl", default="/data/savithas/phase3_full/sae-l{N}-exp16-balanced/checkpoint_final.pt")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--outdir", default="analysis/figures")
    p.add_argument("--no-umap", action="store_true")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in a.layers.split(",")]
    eps = 1.0 / (2 * a.n)

    umap2d = None
    if not a.no_umap:
        import umap
        from sklearn.decomposition import PCA

        def umap2d(M):
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            if Mn.shape[1] > 50:   # PCA pre-reduction (UMAP docs recommend for high-dim) — ~10x faster
                Mn = PCA(n_components=50, svd_solver="randomized", random_state=0).fit_transform(Mn)
            return umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(Mn)

    per_layer = {}   # N -> dict with tf/pf arrays (unbal,bal), 2D embeddings, counts (NO big code matrices)
    for N in layers:
        print(f"\n=== L{N} ===", flush=True)
        pr = sample_bands(a.store_tmpl.format(N=N), N, ["protein", "text"], a.n)
        X = np.concatenate([pr["protein"], pr["text"]])
        lab = np.array([0] * len(pr["protein"]) + [1] * len(pr["text"]))
        d = {"lab": lab, "np": len(pr["protein"]), "nt": len(pr["text"])}
        if umap2d is not None:
            d["emb_raw"] = umap2d(X)
        for tag, tmpl in [("unbal", a.unbal_tmpl), ("bal", a.bal_tmpl)]:
            sae = load_sae(tmpl.format(N=N), dev)
            C = codes(sae, X, dev)
            act = C > 0
            pf = act[lab == 0].mean(0); tf = act[lab == 1].mean(0)
            live = (pf > 0) | (tf > 0)
            d[f"{tag}_pf"] = pf; d[f"{tag}_tf"] = tf
            d[f"{tag}_ps"] = int(((pf > 0.01) & (tf < 0.001)).sum())
            d[f"{tag}_ts"] = int(((tf > 0.01) & (pf < 0.001)).sum())
            d[f"{tag}_live"] = int(live.sum()); d[f"{tag}_dead"] = int(C.shape[1] - live.sum())
            if umap2d is not None:
                d[f"emb_{tag}"] = umap2d(C)
            print(f"  {tag}: protein-sel={d[f'{tag}_ps']:5d}  text-sel={d[f'{tag}_ts']:5d}  "
                  f"live={d[f'{tag}_live']:6d}  dead={d[f'{tag}_dead']:6d}", flush=True)
            del sae, C
            if dev == "cuda":
                torch.cuda.empty_cache()
        per_layer[N] = d

    # ---- counts table ----
    tbl = {N: {k: per_layer[N][k] for k in
               ["unbal_ps", "bal_ps", "unbal_ts", "bal_ts", "unbal_dead", "bal_dead"]} for N in layers}
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    json.dump(tbl, open(Path(a.outdir) / "layer_balance_counts.json", "w"), indent=2)
    print("\nLAYER  protein-sel(unbal->bal)  text-sel(unbal->bal)   dead%(unbal->bal)")
    for N in layers:
        d = per_layer[N]
        print(f"L{N:<4} {d['unbal_ps']:6d} -> {d['bal_ps']:6d}      "
              f"{d['unbal_ts']:6d} -> {d['bal_ts']:6d}     "
              f"{100*d['unbal_dead']/40960:5.1f} -> {100*d['bal_dead']/40960:5.1f}")

    # ---- selectivity scatter grid ----
    ncol = 2; nrow = len(layers)
    fig, ax = plt.subplots(nrow, ncol, figsize=(9, 3.4 * nrow), sharex=True, sharey=True)
    for r, N in enumerate(layers):
        d = per_layer[N]
        for c, tag, ttl in [(0, "unbal", "unbalanced"), (1, "bal", "balanced")]:
            A = ax[r, c]
            pf, tf = d[f"{tag}_pf"], d[f"{tag}_tf"]
            live = (pf > 0) | (tf > 0)
            psel = live & (pf > 0.01) & (tf < 0.001)     # protein-selective population (top-left)
            rest = live & ~psel
            A.scatter(tf[rest] + eps, pf[rest] + eps, s=3, alpha=0.18, c="#b8bec6", edgecolors="none")
            A.scatter(tf[psel] + eps, pf[psel] + eps, s=9, alpha=0.8, c="#2563eb", edgecolors="none",
                      label="protein-selective")
            A.set_xscale("log"); A.set_yscale("log")
            A.plot([eps, 1], [eps, 1], ls="--", lw=0.7, c="#cbd5e1")
            A.axhline(0.01, color="#9aa3ad", lw=0.5, ls=":"); A.axvline(0.001, color="#9aa3ad", lw=0.5, ls=":")
            A.set_xlim(eps * 0.8, 1); A.set_ylim(eps * 0.8, 1)
            A.set_title(f"L{N} {ttl} — {d[f'{tag}_ps']} protein-sel", fontsize=10)
            if r == nrow - 1:
                A.set_xlabel("text fire-frac")
            if c == 0:
                A.set_ylabel("protein fire-frac")
    fig.suptitle("Per-feature modality selectivity across layers: unbalanced vs balanced loss", fontsize=12, y=1.001)
    fig.tight_layout(); fig.savefig(Path(a.outdir) / "fig_layer_selectivity_grid.png", dpi=115, bbox_inches="tight")
    print(f"wrote {a.outdir}/fig_layer_selectivity_grid.png")

    if a.no_umap:
        return

    fig, ax = plt.subplots(nrow, 3, figsize=(13, 4.2 * nrow))
    for r, N in enumerate(layers):
        d = per_layer[N]; lab = d["lab"]
        embs = [("(raw)", d["emb_raw"]),
                ("unbalanced code", d["emb_unbal"]),
                ("balanced code", d["emb_bal"])]
        for c, (ttl, emb) in enumerate(embs):
            A = ax[r, c]
            for cl, name, col in [(0, "protein", "#2563eb"), (1, "text", "#e8710a")]:
                m = lab == cl
                A.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.4, c=col, label=name)
            A.set_title(f"L{N} {ttl}", fontsize=10); A.set_xticks([]); A.set_yticks([])
            if r == 0 and c == 0:
                A.legend(markerscale=3, fontsize=8)
    fig.suptitle("Token geometry across layers: raw residual vs SAE code (unbalanced / balanced)", fontsize=12, y=1.001)
    fig.tight_layout(); fig.savefig(Path(a.outdir) / "fig_layer_umap_grid.png", dpi=110, bbox_inches="tight")
    print(f"wrote {a.outdir}/fig_layer_umap_grid.png")


if __name__ == "__main__":
    main()
