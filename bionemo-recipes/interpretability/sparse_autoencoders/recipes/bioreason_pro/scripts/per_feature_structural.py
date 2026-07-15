#!/usr/bin/env python
"""Q1: are there INDIVIDUAL SAE features that fire on structural RESIDUE patterns?

For each SAE feature, compute per-feature AUROC on the RESIDUE band against structural labels:
  - buried (3D core) vs surface  (AlphaFold contact number)
  - in-domain vs not, for the top abundant InterPro domains  (interpro_location)
This is the residue-band analog of validate_features.py (which found the reasoning band rich but the
protein band weak). If the top-AUROC feature per structural label is clean (>~0.8), the SAE isolates a
structural-residue feature; if the best is ~0.5-0.65, the SAE does NOT decompose residue structure into
interpretable features (consistent with 'ESM3 residue rep is the ceiling, SAE re-expresses it').

Usage: per_feature_structural.py <sae.pt> <store> <layer> <af_dir> [--shard-stride N] [--max-res N]
"""
import glob, json, sys, argparse, os
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
import biotite.structure.io.pdbx as pdbx, biotite.structure as struc
import bioreason_pro_sae.data as brp_data

p = argparse.ArgumentParser()
p.add_argument("sae"); p.add_argument("store"); p.add_argument("layer", type=int); p.add_argument("af_dir")
p.add_argument("--shard-stride", type=int, default=2); p.add_argument("--max-res", type=int, default=25000)
a = p.parse_args(); dev = "cuda"


def contact_numbers(acc):
    fp = os.path.join(a.af_dir, f"{acc}.cif")
    if not os.path.exists(fp) or os.path.getsize(fp) == 0: return None
    try:
        arr = pdbx.get_structure(pdbx.CIFFile.read(fp), model=1); arr = arr[struc.filter_amino_acids(arr)]
        cb = arr[(arr.atom_name == "CB") | ((arr.res_name == "GLY") & (arr.atom_name == "CA"))]
        xyz = cb.coord
        if len(xyz) < 5: return None
        d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        return ((d < 8) & (d > 0.1)).sum(1)
    except Exception:
        return None


# residue rows + within-protein residue index
tl = pq.read_table(f"{a.store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64)
N = len(band); prot = band == "protein"
df = pd.DataFrame({"pid": rpid[prot].astype(str), "tidx": rtidx[prot], "row": np.arange(N)[prot]})
df = df.sort_values(["pid", "tidx"]).reset_index(drop=True); df["res"] = df.groupby("pid").cumcount()

# interpro domain spans per protein
tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
loc_of = {}
for pid, loc in zip(tr["protein_id"], tr["interpro_location"]):
    if loc:
        try: loc_of[str(pid)] = json.loads(loc) if isinstance(loc, str) else loc
        except Exception: pass
from collections import Counter
dc = Counter()
for pid in df["pid"].unique():
    for d in loc_of.get(pid, {}): dc[d] += 1
DOMAINS = [d for d, c in dc.most_common(12) if c >= 30]

# per-residue-row labels: buried(1)/surface(0)/drop(-1); in-domain per top domain
buried = np.full(len(df), -1, np.int8)
domlab = {d: np.zeros(len(df), np.int8) for d in DOMAINS}
have = 0
for pid, g in df.groupby("pid", sort=False):
    res = g["res"].to_numpy(); rows = g.index.to_numpy()
    cn = contact_numbers(pid)
    if cn is not None:
        have += 1; ok = res < len(cn); c = cn[res[ok]]
        if len(c) >= 10:
            hi, lo = np.quantile(c, 0.67), np.quantile(c, 0.33)
            buried[rows[ok]] = np.where(c >= hi, 1, np.where(c <= lo, 0, -1))
    dloc = loc_of.get(pid, {})
    for d in DOMAINS:
        span = dloc.get(d)
        if span:
            s = span if isinstance(span[0], list) else [span]
            ind = np.array([any(x[0] <= r + 1 <= x[1] for x in s) for r in res], bool)
            domlab[d][rows] = ind.astype(np.int8)
print(f"[perfeat] {have} proteins w/ structure; buried {int((buried==1).sum())} surface {int((buried==0).sum())}", flush=True)

# subsample residue rows, gather SAE activations for ALL features
keep_idx = df.index.to_numpy()
rng = np.random.default_rng(0)
if len(keep_idx) > a.max_res: keep_idx = np.sort(rng.choice(keep_idx, a.max_res, replace=False))
keep_rows = df.loc[keep_idx, "row"].to_numpy().astype(np.int64)
row2i = {int(r): i for i, r in enumerate(keep_rows)}; nR = len(keep_rows)
ck = torch.load(a.sae, map_location="cpu"); sae = TopKSAE(**ck["model_config"]).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
H = sae.hidden_dim; SAE = np.zeros((nR, H), np.float16); got = np.zeros(nR, bool)
keepset = set(keep_rows.tolist()); row0 = 0
order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si, sp in enumerate(order):
        nrows = pq.read_metadata(sp).num_rows
        if si % a.shard_stride == 0:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            lr = keep_rows[(keep_rows >= row0) & (keep_rows < row0 + n)]
            if len(lr):
                enc = sae.encode(torch.from_numpy(np.ascontiguousarray(X[lr - row0])).to(dev)).half().cpu().numpy()
                for j, gr in enumerate(lr): SAE[row2i[int(gr)]] = enc[j]; got[row2i[int(gr)]] = True
        row0 += nrows
        if si % 20 == 0: print(f"  shard {si}/{len(order)}", flush=True)
print(f"[perfeat] gathered {int(got.sum())} residues x {H} features; ranking...", flush=True)

# per-feature AUROC (vectorized rank-sum) for each structural label
ranks = rankdata(SAE[got].astype(np.float32), axis=0)     # [nR, H]
def auroc_all(y):
    yk = y[keep_idx][got]
    pos = yk == 1; neg = yk == 0; npos, nneg = int(pos.sum()), int(neg.sum())
    if npos < 20 or nneg < 20: return None
    Rpos = ranks[pos].sum(0)                               # [H]
    return (Rpos - npos * (npos + 1) / 2) / (npos * nneg)  # AUROC per feature

results = {}
print(f"\n{'structural label':22} {'npos':>6} {'best feat':>10} {'AUROC':>7}   (top-3 features)")
labels = {"buried_vs_surface": buried}
for d in DOMAINS: labels[f"in_{d}"] = domlab[d]
for name, y in labels.items():
    au = auroc_all(y)
    if au is None: continue
    au = np.where(np.isnan(au), 0.5, au); au = np.maximum(au, 1 - au)  # symmetric (fires-on-or-off)
    top = np.argsort(-au)[:3]
    results[name] = {"npos": int((y[keep_idx][got] == 1).sum()), "best_feature": int(top[0]),
                     "best_auroc": round(float(au[top[0]]), 3),
                     "top3": [(int(f), round(float(au[f]), 3)) for f in top]}
    print(f"{name:22} {results[name]['npos']:>6} {top[0]:>10} {au[top[0]]:>7.3f}   {results[name]['top3']}")

mx = max((r["best_auroc"] for r in results.values()), default=0)
print(f"\nVERDICT: best single-feature structural AUROC = {mx:.3f}")
print(">0.80 => the SAE HAS clean structural-residue features. ~0.5-0.65 => it does NOT isolate them")
print("(structure is in the raw ESM3 residues at ceiling, but not decomposed into interpretable features).")
json.dump(results, open("/data/savithas/phase3_full/per_feature_structural.json", "w"), indent=2)
print("[wrote] per_feature_structural.json")
