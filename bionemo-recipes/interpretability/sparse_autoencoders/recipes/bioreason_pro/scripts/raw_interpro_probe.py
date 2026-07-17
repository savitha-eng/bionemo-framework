#!/usr/bin/env python
"""RAW InterPro-domain decodability from a residue store (no SAE) — for the ESM3 frozen-encoder control.

Pools raw per-residue embeddings per protein, trains logistic regression -> InterPro domain presence,
reports mean held-out AUROC. Used on the ESM3 layer-37 pre-projection store to answer: is structure
already decodable from the FROZEN encoder (extends the L16->L32 layer curve back to 'layer 0')?

Usage: raw_interpro_probe.py <store> <layer> [out.json]
"""
import glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "src")
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import bioreason_pro_sae.data as brp_data

store, layer = sys.argv[1], int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else f"raw_interpro_l{layer}.json"

tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
prot_mask = band == "protein"
pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)

tr, _, _ = brp_data.load_reasoning_splits(max_length_protein=2000)
ipro_of = {str(pid): set(ips) if ips else set() for pid, ips in zip(tr["protein_id"], tr["interpro_ids"])}
cnt = Counter()
for p in pids:
    for d in ipro_of.get(p, ()):
        cnt[d] += 1
DOMAINS = [d for d, c in cnt.most_common(60) if c >= 50]
print(f"[raw] {len(DOMAINS)} abundant domains among {nP} proteins; store={store} layer={layer}", flush=True)

# mean-pool raw residue embeddings per protein
rpi = np.array([pidx.get(str(p), -1) for p in rpid])
D = None; ss = None; cnt_t = np.zeros(nP); row0 = 0
for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
    X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
    if D is None:
        D = X.shape[1]; ss = np.zeros((nP, D), np.float32)
    m = prot_mask[row0:row0 + n]; idx = rpi[row0:row0 + n][m]
    np.add.at(ss, idx, X[m].astype(np.float32)); np.add.at(cnt_t, idx, 1)
    row0 += n
keep = cnt_t > 0
RAW = ss / np.clip(cnt_t, 1, None)[:, None]
print(f"[raw] pooled {int(keep.sum())}/{nP} proteins, dim {D}", flush=True)

rng = np.random.default_rng(0); perm = rng.permutation(np.where(keep)[0]); h = len(perm) // 2
tr_i, te_i = perm[:h], perm[h:]
sc = StandardScaler().fit(RAW[tr_i]); Xs = sc.transform(RAW)

results = {}
print(f"\n{'domain':14} {'npos':>5} {'raw AUROC':>10}")
for d in DOMAINS:
    y = np.array([1 if d in ipro_of.get(pids[i], ()) else 0 for i in range(nP)])
    if y[tr_i].sum() < 10 or y[te_i].sum() < 10:
        continue
    clf = LogisticRegression(C=1.0, max_iter=300, tol=1e-3).fit(Xs[tr_i], y[tr_i])
    au = float(roc_auc_score(y[te_i], clf.decision_function(Xs[te_i])))
    results[d] = {"npos": int(y[perm].sum()), "raw": round(au, 3)}
    print(f"{d:14} {int(y[perm].sum()):>5} {au:>10.3f}")

mean = round(float(np.mean([r["raw"] for r in results.values()])), 3) if results else 0.0
print(f"\nMEAN raw InterPro AUROC ({len(results)} domains): {mean}")
Path(out).write_text(json.dumps({"store": store, "layer": layer, "n_domains": len(results),
    "mean_raw_auroc": mean, "results": results}, indent=2))
print(f"[wrote] {out}")
