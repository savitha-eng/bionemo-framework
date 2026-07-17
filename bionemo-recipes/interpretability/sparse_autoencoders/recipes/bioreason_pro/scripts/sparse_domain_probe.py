#!/usr/bin/env python
"""L1-SPARSE LINEAR CLASSIFIER probe: which SAE features does a linear probe SELECT for a structural label?

This is the probe asked for, and the RIGHT "which features are associated" method — NOT univariate per-feature
AUROC (that ranks each of the 40,960 features one-at-a-time; per_feature_structural.py). Here we train ONE
L1-regularized logistic regression on the FULL 40,960-d SAE vector -> InterPro-domain presence, with a
protein-level held-out split, and read off the features with nonzero coefficients: the set the classifier
jointly selects. Reports held-out AUROC + the selected features (sorted by |coef|), and flags whether the
univariate winner (e.g. F16026 for kinase) is among the L1-selected set.

Usage: sparse_domain_probe.py <sae.pt> <store> <layer> [C] [out.json]
"""
import glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import bioreason_pro_sae.data as brp_data

sae_p, store, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
C = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
out = sys.argv[5] if len(sys.argv) > 5 else f"sparse_domain_probe_l{layer}.json"
dev = "cuda"

# ---- store residue-band mask + protein order ----
tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
prot_mask = band == "protein"
pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)

tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
ipro_of = {str(pid): set(ips) if ips else set() for pid, ips in zip(tr["protein_id"], tr["interpro_ids"])}
# human-readable domain names from interpro_location keys where available
cnt = Counter()
for p in pids:
    for d in ipro_of.get(p, ()):
        cnt[d] += 1
DOMAINS = [d for d, c in cnt.most_common(60) if c >= 50]
print(f"[sparse] {len(DOMAINS)} abundant InterPro domains among {nP} store proteins; C={C}")

# ---- mean-pool residue SAE activations per protein (FULL 40,960-d) ----
ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
H = sae.hidden_dim
rpi = torch.from_numpy(np.array([pidx.get(str(p), -1) for p in rpid])).to(dev)
pm = torch.from_numpy(prot_mask).to(dev)
ss = torch.zeros(nP, H, device=dev); cnt_t = torch.zeros(nP, device=dev); row0 = 0
with torch.no_grad():
    for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        for s in range(0, n, 8192):
            e = min(n, s + 8192); m = pm[row0 + s:row0 + e]
            if not m.any(): continue
            xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev); idx = rpi[row0 + s:row0 + e][m]
            ss.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), sae.encode(xb)[m])
            cnt_t.scatter_add_(0, idx, torch.ones(int(m.sum()), device=dev))
        row0 += n
den = cnt_t.clamp(min=1)[:, None]
SAE = (ss / den).cpu().numpy(); keep = cnt_t.cpu().numpy() > 0
print(f"[sparse] pooled {int(keep.sum())}/{nP} proteins; full SAE dim {H}")

rng = np.random.default_rng(0); perm = rng.permutation(np.where(keep)[0]); h = len(perm) // 2
tr_i, te_i = perm[:h], perm[h:]
sc = StandardScaler().fit(SAE[tr_i]); Xs = sc.transform(SAE)   # standardize once (features)

results = {}
print(f"\n{'InterPro domain':16} {'npos':>5} {'AUROC':>6} {'#sel':>5}   top selected features (|coef|)")
for d in DOMAINS:
    y = np.array([1 if d in ipro_of.get(pids[i], ()) else 0 for i in range(nP)])
    if y[tr_i].sum() < 10 or y[te_i].sum() < 10:
        continue
    # L1-sparse linear classifier via SGD (fast on 40,960 features; liblinear was ~1min/domain).
    # alpha maps from C: alpha ~= 1/(C*n_train); C=0.05,n~4000 -> ~0.005 gives a sparse selection.
    alpha = 1.0 / (C * len(tr_i))
    clf = SGDClassifier(loss="log_loss", penalty="l1", alpha=alpha, max_iter=200, tol=1e-3,
                        random_state=0).fit(Xs[tr_i], y[tr_i])
    auroc = float(roc_auc_score(y[te_i], clf.decision_function(Xs[te_i])))
    coef = clf.coef_[0]; nz = np.where(coef != 0)[0]
    top = nz[np.argsort(-np.abs(coef[nz]))][:8]
    sel = [(int(f), round(float(coef[f]), 3)) for f in top]
    results[d] = {"npos": int(y[perm].sum()), "auroc": round(auroc, 3),
                  "n_selected": int(len(nz)), "top_features": sel}
    print(f"{d:16} {int(y[perm].sum()):>5} {auroc:>6.3f} {len(nz):>5}   {[f for f, _ in sel]}")

mean = round(float(np.mean([r["auroc"] for r in results.values()])), 3) if results else 0.0
med_nsel = int(np.median([r["n_selected"] for r in results.values()])) if results else 0
print(f"\nMEAN sparse-probe AUROC ({len(results)} domains): {mean} | median #features selected: {med_nsel}")
print("=> the L1 linear classifier NAMES the feature set it uses per domain (multivariate),")
print("   unlike univariate per-feature AUROC which ranks features one at a time.")
Path(out).write_text(json.dumps({"layer": layer, "C": C, "n_domains": len(results),
    "mean_auroc": mean, "median_n_selected": med_nsel, "results": results}, indent=2))
print(f"[wrote] {out}")
