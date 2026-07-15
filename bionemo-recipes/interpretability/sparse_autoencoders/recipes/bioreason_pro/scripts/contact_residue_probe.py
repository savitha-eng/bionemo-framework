#!/usr/bin/env python
"""STRUCTURAL contact-residue probe (paper §4.5.5-6 flavor): does the residue representation encode a
residue's 3D structural context — is it BURIED (many 3D contacts, core) or SURFACE (few)?

Contact number per residue = # other residues whose Cβ (Cα for Gly) is < 8Å away (standard contact def).
Label per residue: buried (top tercile of contact number within its protein) vs surface (bottom tercile);
middle dropped for a clean binary. Probe from that residue's activation: SAE-svd256 vs raw vs random.
Non-circular & structural — tests 3D-context encoding beyond sequence/domain identity.

Usage: contact_residue_probe.py <sae.pt> <store> <layer> <af_dir> [out.json] [--shard-stride N] [--max-res N]
"""
import glob, json, sys, argparse, os
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq, pandas as pd
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import biotite.structure.io.pdbx as pdbx
import biotite.structure as struc

p = argparse.ArgumentParser()
p.add_argument("sae"); p.add_argument("store"); p.add_argument("layer", type=int); p.add_argument("af_dir")
p.add_argument("out", nargs="?", default=None)
p.add_argument("--shard-stride", type=int, default=2)
p.add_argument("--max-res", type=int, default=60000)
a = p.parse_args(); dev = "cuda"
out = a.out or f"contact_residue_probe_l{a.layer}.json"


def contact_numbers(acc):
    """per-residue contact number from the AF structure, in sequence order (None if no structure)."""
    fp = os.path.join(a.af_dir, f"{acc}.cif")
    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
        return None
    try:
        arr = pdbx.get_structure(pdbx.CIFFile.read(fp), model=1)
        arr = arr[struc.filter_amino_acids(arr)]
        cb = arr[(arr.atom_name == "CB") | ((arr.res_name == "GLY") & (arr.atom_name == "CA"))]
        xyz = cb.coord
        if len(xyz) < 5: return None
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        return ((d < 8.0) & (d > 0.1)).sum(1)                # contact number per residue (seq order)
    except Exception:
        return None


# ---- residue-band rows + within-protein residue index ----
tl = pq.read_table(f"{a.store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64)
N = len(band); prot = band == "protein"
df = pd.DataFrame({"pid": rpid[prot].astype(str), "tidx": rtidx[prot], "row": np.arange(N)[prot]})
df = df.sort_values(["pid", "tidx"]).reset_index(drop=True)
df["res"] = df.groupby("pid").cumcount()
print(f"[contact] {len(df):,} residue rows across {df['pid'].nunique()} proteins", flush=True)

# ---- per-protein contact numbers -> buried/surface label per residue-band row ----
lab = np.full(len(df), -1, np.int8)                         # 1=buried, 0=surface, -1=drop
have = 0
for pid, g in df.groupby("pid", sort=False):
    cn = contact_numbers(pid)
    if cn is None: continue
    have += 1
    res = g["res"].to_numpy(); ok = res < len(cn)
    c = cn[res[ok]]
    if len(c) < 10: continue
    hi, lo = np.quantile(c, 0.67), np.quantile(c, 0.33)
    rows = g["row"].to_numpy()[ok]
    y = np.where(c >= hi, 1, np.where(c <= lo, 0, -1)).astype(np.int8)
    # map back into lab via df index
    lab[g.index.to_numpy()[ok]] = y
print(f"[contact] {have} proteins had structures; {int((lab==1).sum())} buried, {int((lab==0).sum())} surface", flush=True)

# subsample labeled residues
labeled = df.index[lab != -1].to_numpy()
rng = np.random.default_rng(0)
if len(labeled) > a.max_res:
    labeled = np.sort(rng.choice(labeled, a.max_res, replace=False))
sub = df.loc[labeled]; y_all = lab[labeled]
row_to_i = {int(r): i for i, r in enumerate(sub["row"].to_numpy())}
keep_rows = np.array(sorted(row_to_i), dtype=np.int64); nR = len(sub)
print(f"[contact] probing {nR} residues", flush=True)

# ---- gather activations for the kept residues ----
ck = torch.load(a.sae, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval(); H = sae.hidden_dim
SAE = RAW = RND = None; D = None; row0 = 0; got = np.zeros(nR, bool)
order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si, sp in enumerate(order):
        nrows = pq.read_metadata(sp).num_rows
        if si % a.shard_stride == 0:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]; lo2, hi2 = row0, row0 + n
            lr = keep_rows[(keep_rows >= lo2) & (keep_rows < hi2)]
            if len(lr):
                if SAE is None:
                    D = X.shape[1]; SAE = np.zeros((nR, H), np.float32); RAW = np.zeros((nR, D), np.float32); RND = np.zeros((nR, H), np.float32)
                xb = torch.from_numpy(np.ascontiguousarray(X[lr - lo2])).to(dev)
                enc = sae.encode(xb).cpu().numpy(); rnd = rsae.encode(xb).cpu().numpy(); rawb = xb.cpu().numpy()
                for j, gr in enumerate(lr):
                    i = row_to_i[int(gr)]; SAE[i] = enc[j]; RAW[i] = rawb[j]; RND[i] = rnd[j]; got[i] = True
        row0 += nrows
        if si % 20 == 0: print(f"  shard {si}/{len(order)}", flush=True)
keep = got; idx = np.where(keep)[0]; y = y_all
print(f"[contact] gathered {int(keep.sum())}/{nR} residues", flush=True)

rng.shuffle(idx); h = len(idx) // 2; tr_i, te_i = idx[:h], idx[h:]
svd = TruncatedSVD(n_components=256, random_state=0).fit(SAE[tr_i]); SAE_SVD = svd.transform(SAE)
rawsvd = TruncatedSVD(n_components=256, random_state=0).fit(RAW[tr_i]); RAW_256 = rawsvd.transform(RAW)  # FAIR ctrl
def probe(Xf):
    sc = StandardScaler().fit(Xf[tr_i]); Xs = sc.transform(Xf)
    clf = LogisticRegression(C=1.0, max_iter=300, tol=1e-3).fit(Xs[tr_i], y[tr_i])
    return round(float(roc_auc_score(y[te_i], clf.decision_function(Xs[te_i]))), 3)
res = {"sae_svd256": probe(SAE_SVD), "raw_pca256": probe(RAW_256), "raw_full": probe(RAW), "random": probe(RND),
       "n_buried": int((y[idx] == 1).sum()), "n_surface": int((y[idx] == 0).sum())}
print(f"\n=== CONTACT (buried vs surface) residue probe ===")
print(f"  SAE-svd256 {res['sae_svd256']} | raw-pca256 {res['raw_pca256']} (FAIR, same dim) | "
      f"raw-full2560 {res['raw_full']} | random {res['random']}")
print("  FAIR test = SAE-svd256 vs raw-pca256 (both 256-dim). If tied, the 'SAE loses to raw-2560' was a")
print("  dimensionality artifact. raw-full>>random => 3D signal is real; SAE~=raw@matched-dim => ceiling.")
Path(out).write_text(json.dumps({"layer": a.layer, "level": "residue-contact", **res}, indent=2))
print(f"[wrote] {out}")
