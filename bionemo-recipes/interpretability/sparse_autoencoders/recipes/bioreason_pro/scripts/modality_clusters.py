# "Investigate clustering" (Jared): do PROTEIN-position features and TEXT-position features form
# separate clusters? UMAP the SAE feature directions (decoder weights), classify each feature by which
# band it fires on (protein/go/text, from a val store), and measure k-NN cluster purity. Clean
# separation => the SAE learned meaningful modality structure; muddy => undertrained.
#
#   python scripts/modality_clusters.py --sae <ckpt> --store <val300_L*> --layer L --out <atlas.parquet>
import argparse, glob
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True); ap.add_argument("--store", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--max-shards", type=int, default=0, help="0=all; else cap shards for speed (band coverage)")
    ap.add_argument("--roles", action="store_true",
                    help="split text into prompt vs reasoning via token_labels_with_role (5-band: protein/go/prompt/reasoning)")
    a = ap.parse_args(); dev = "cuda"

    ck = torch.load(a.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}  # strip DDP prefix
    sae.load_state_dict(sd)
    H = sae.hidden_dim
    # feature directions = decoder weight (d_model x H) -> transpose to (H, d_model)
    sd = ck["model_state_dict"]
    dkey = "decoder.weight" if "decoder.weight" in sd else next(k for k in sd if "decoder" in k and "weight" in k)
    W = sd[dkey].float().cpu().numpy()
    Wf = W.T if W.shape[0] != H else W                      # (H, d_model)
    print(f"[clusters] decoder weights {dkey} -> feature matrix {Wf.shape}", flush=True)

    # ---- band coverage (which modality/role each feature fires on) ----
    # 3-band default (protein/go/text); with --roles, split text -> prompt vs reasoning via the role
    # sidecar, isolating reasoning-text features from prompt boilerplate/annotation features.
    use_roles = a.roles and (Path(a.store) / "token_labels_with_role.parquet").exists()
    tl = pq.read_table(f"{a.store}/{'token_labels_with_role.parquet' if use_roles else 'token_labels.parquet'}")
    pos = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    if use_roles:
        role = np.asarray(tl.column("role").to_pylist(), dtype=object)
        bands = {"protein": pos == "protein", "go": pos == "go",
                 "prompt": (pos == "text") & (role == "prompt"),       # boilerplate + handed-in annotations
                 "reasoning": (pos == "text") & (role == "response")}  # model's own reasoning + answer
    else:
        bands = {"protein": pos == "protein", "go": pos == "go", "text": pos == "text"}
    names = list(bands)
    print(f"[clusters] bands: {names}  (roles={'on' if use_roles else 'off'})", flush=True)
    bmask = {k: torch.from_numpy(v).to(dev) for k, v in bands.items()}
    mass = {k: torch.zeros(H, device=dev) for k in names}
    row0 = 0
    with torch.no_grad():
        _shards = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
        if a.max_shards:
            _shards = _shards[:a.max_shards]
        for sp in _shards:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            for s in range(0, n, 8192):
                e = min(n, s + 8192); c = sae.encode(torch.from_numpy(X[s:e]).to(dev))
                for k in names:
                    band = bmask[k][row0 + s:row0 + e]
                    if band.any(): mass[k] += c[band].sum(0)
            row0 += n
    M = np.stack([mass[k].cpu().numpy() for k in names], 1)   # (H, nbands)
    tot = M.sum(1) + 1e-9
    fracs = M / tot[:, None]
    band_class = np.array(["dead"] * H, dtype=object)
    alive = tot > 1e-6
    band_class[alive] = np.array(names)[fracs[alive].argmax(1)]
    print(f"[clusters] band_class counts: " + ", ".join(f"{l}={int((band_class==l).sum())}" for l in names + ["dead"]), flush=True)

    # ---- UMAP the feature directions ----
    import umap
    print("[clusters] UMAP...", flush=True)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(Wf)

    # ---- cluster purity: for each feature, fraction of UMAP k-NN sharing its band_class ----
    from sklearn.neighbors import NearestNeighbors
    al = np.where(alive)[0]
    nn = NearestNeighbors(n_neighbors=a.knn+1).fit(xy[al])
    _, idx = nn.kneighbors(xy[al])
    bc_al = band_class[al]
    purity = np.array([(bc_al[idx[i,1:]] == bc_al[i]).mean() for i in range(len(al))])
    print("\n[clusters] k-NN purity (do same-band features cluster together?):", flush=True)
    for l in names:
        m = bc_al == l
        if m.any(): print(f"   {l:10s}: purity {purity[m].mean():.2f}  (n={int(m.sum())}, chance={(band_class[alive]==l).mean():.3f})", flush=True)
    print(f"   OVERALL : {purity.mean():.2f}   (1.0=perfect separation, ~base-rate=no structure)", flush=True)

    cols = {"feature_id": np.arange(H), "x": xy[:, 0], "y": xy[:, 1], "band_class": band_class}
    for i, k in enumerate(names):
        cols[f"{k}_frac"] = fracs[:, i]
    pq.write_table(pa.table(cols), a.out)
    print(f"\n[clusters] wrote atlas -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
