#!/usr/bin/env python
"""SEQUENCE-LEVEL, RESIDUE-RESOLUTION InterPro domain-boundary probe.

For each residue, is it INSIDE a given InterPro domain? Probe from that residue's activation (SAE-svd256
vs raw vs random). Non-circular & sequence-grounded: domain boundaries are a structural/sequence annotation,
and this tests at RESIDUE resolution whether the model's residue representation encodes where domains start/
end along the polypeptide — the mechanistic version of the paper's "domain architecture along the polypeptide".

Residue index within a protein = ordinal of its protein-band token (1 token/residue, sequence order).
Domain spans come from interpro_location {IPRxxxx: [start,end]} (1-indexed residues).

Usage: residue_domain_probe.py <sae.pt> <store> <layer> [out.json] [--shard-stride N] [--max-res N]
"""
import glob, json, sys, argparse
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq, pandas as pd
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import bioreason_pro_sae.data as brp_data

p = argparse.ArgumentParser()
p.add_argument("sae"); p.add_argument("store"); p.add_argument("layer", type=int)
p.add_argument("out", nargs="?", default=None)
p.add_argument("--shard-stride", type=int, default=2)
p.add_argument("--max-res", type=int, default=50000, help="subsample residues for the probe (memory)")
a = p.parse_args(); dev = "cuda"
out = a.out or f"residue_domain_probe_l{a.layer}.json"

# ---- residue-band rows + within-protein residue index ----
tl = pq.read_table(f"{a.store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64)
N = len(band); prot = band == "protein"
df = pd.DataFrame({"pid": rpid[prot].astype(str), "tidx": rtidx[prot], "row": np.arange(N)[prot]})
df = df.sort_values(["pid", "tidx"]).reset_index(drop=True)
df["res"] = df.groupby("pid").cumcount()                    # 0-indexed residue position within protein
print(f"[resdom] {len(df):,} residue rows across {df['pid'].nunique()} proteins")

# ---- interpro_location per protein ----
tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
loc_of = {}
for pid, loc in zip(tr["protein_id"], tr["interpro_location"]):
    if not loc: continue
    try: loc_of[str(pid)] = json.loads(loc) if isinstance(loc, str) else loc
    except Exception: pass
cnt = Counter()
for pid in df["pid"].unique():
    for d in loc_of.get(pid, {}): cnt[d] += 1
DOMAINS = [d for d, c in cnt.most_common(30) if c >= 30]
print(f"[resdom] {len(DOMAINS)} abundant domains")

# ---- subsample residues for the probe ----
if len(df) > a.max_res:
    df = df.sample(n=a.max_res, random_state=0).sort_values("row").reset_index(drop=True)
row_to_i = {int(r): i for i, r in enumerate(df["row"].to_numpy())}
nR = len(df)

def label(pid, res, dom):
    span = loc_of.get(pid, {}).get(dom)
    if not span: return 0
    if isinstance(span[0], list):
        return int(any(s[0] <= res + 1 <= s[1] for s in span))
    return int(span[0] <= res + 1 <= span[1])

# ---- gather per-residue activations for the kept residues (raw + SAE), subsampled shards ----
ck = torch.load(a.sae, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval()
H = sae.hidden_dim
keep_rows = np.array(sorted(row_to_i), dtype=np.int64)
keep_set = set(keep_rows.tolist())
SAE = None; RAW = None; RND = None; D = None; row0 = 0; got = np.zeros(nR, bool)
order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si, sp in enumerate(order):
        nrows = pq.read_metadata(sp).num_rows
        if si % a.shard_stride == 0:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            # which of this shard's global rows are kept?
            lo, hi = row0, row0 + n
            loc_rows = keep_rows[(keep_rows >= lo) & (keep_rows < hi)]
            if len(loc_rows):
                if SAE is None:
                    D = X.shape[1]; SAE = np.zeros((nR, H), np.float32); RAW = np.zeros((nR, D), np.float32)
                    RND = np.zeros((nR, H), np.float32)
                xb = torch.from_numpy(np.ascontiguousarray(X[loc_rows - lo])).to(dev)
                enc = sae.encode(xb).cpu().numpy(); rnd = rsae.encode(xb).cpu().numpy()
                for j, gr in enumerate(loc_rows):
                    i = row_to_i[int(gr)]; SAE[i] = enc[j]; RAW[i] = xb[j].cpu().numpy(); RND[i] = rnd[j]; got[i] = True
        row0 += nrows
        if si % 20 == 0: print(f"  shard {si}/{len(order)}", flush=True)
keep = got
print(f"[resdom] gathered {int(keep.sum())}/{nR} residues (shard-subsampled)")

pids = df["pid"].to_numpy(); ress = df["res"].to_numpy()
idx = np.where(keep)[0]
rng = np.random.default_rng(0); rng.shuffle(idx); h = len(idx) // 2
tr_i, te_i = idx[:h], idx[h:]
svd = TruncatedSVD(n_components=256, random_state=0).fit(SAE[tr_i]); SAE_SVD = svd.transform(SAE)

def probe(Xf, y):
    sc = StandardScaler().fit(Xf[tr_i]); Xs = sc.transform(Xf)
    clf = LogisticRegression(C=1.0, max_iter=300, tol=1e-3).fit(Xs[tr_i], y[tr_i])
    return round(float(roc_auc_score(y[te_i], clf.decision_function(Xs[te_i]))), 3)

results = {}
print(f"\n{'domain':14} {'npos':>6} {'SAE-svd':>8} {'raw':>6} {'rand':>6}")
for d in DOMAINS:
    y = np.array([label(pids[i], ress[i], d) for i in range(nR)])
    if y[tr_i].sum() < 20 or y[te_i].sum() < 20: continue
    a_sae = probe(SAE_SVD, y); a_raw = probe(RAW, y); a_rnd = probe(RND, y)
    results[d] = {"npos_res": int(y[idx].sum()), "sae_svd": a_sae, "raw": a_raw, "random": a_rnd}
    print(f"{d:14} {int(y[idx].sum()):>6} {a_sae:>8} {a_raw:>6} {a_rnd:>6}")

mean = lambda k: round(float(np.mean([r[k] for r in results.values()])), 3) if results else 0.0
print(f"\nMEAN residue-level AUROC ({len(results)} domains): SAE-svd {mean('sae_svd')} | raw {mean('raw')} | random {mean('random')}")
Path(out).write_text(json.dumps({"layer": a.layer, "level": "residue", "n_domains": len(results),
    "results": results, "mean": {k: mean(k) for k in ("sae_svd", "raw", "random")}}, indent=2))
print(f"[wrote] {out}")
