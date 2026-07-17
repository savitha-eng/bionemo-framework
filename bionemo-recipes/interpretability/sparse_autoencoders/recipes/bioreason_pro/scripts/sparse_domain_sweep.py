#!/usr/bin/env python
"""L1-sparse linear probe with a REGULARIZATION SWEEP + pooled-matrix caching (proper probe protocol).

Fixes the single-arbitrary-alpha run: sweeps C from strong->weak regularization and reports, per C,
mean held-out AUROC, median #features selected, and a DISTINCTNESS check (fraction of domains whose
top-weighted feature is unique across domains — a shared top feature = bad attribution). Also flags
whether F16026 is the top-1 selected feature for the kinase domain when forced sparse.

Caching: the expensive step is pooling the 40,960-d SAE per protein over the slow dim_ store (~40 min).
We np.save that matrix once (keyed by store+layer); subsequent runs load it instantly and only re-fit.

Usage: sparse_domain_sweep.py <sae.pt> <store> <layer> [out.json]
"""
import glob, json, sys, hashlib, os
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import bioreason_pro_sae.data as brp_data

sae_p, store, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else f"sparse_domain_sweep_l{layer}.json"
CACHE = f"/data/savithas/phase3_full/pooled_sae_l{layer}_{hashlib.md5(store.encode()).hexdigest()[:8]}.npz"
dev = "cuda"

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

# ---- pooled per-protein SAE matrix (cached) ----
if os.path.exists(CACHE):
    z = np.load(CACHE); SAE = z["SAE"]; keep = z["keep"]
    print(f"[cache] loaded pooled matrix {SAE.shape} from {CACHE}", flush=True)
else:
    ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
    sae = TopKSAE(**cfg).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    H = sae.hidden_dim
    rpi = torch.from_numpy(np.array([pidx.get(str(p), -1) for p in rpid])).to(dev)
    pm = torch.from_numpy(prot_mask).to(dev)
    ss = torch.zeros(nP, H, device=dev); cnt_t = torch.zeros(nP, device=dev); row0 = 0
    with torch.no_grad():
        for si, sp in enumerate(sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))):
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            for s in range(0, n, 8192):
                e = min(n, s + 8192); m = pm[row0 + s:row0 + e]
                if not m.any(): continue
                xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev); idx = rpi[row0 + s:row0 + e][m]
                ss.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), sae.encode(xb)[m])
                cnt_t.scatter_add_(0, idx, torch.ones(int(m.sum()), device=dev))
            row0 += n
            if si % 20 == 0: print(f"  pool shard {si}", flush=True)
    den = cnt_t.clamp(min=1)[:, None]
    SAE = (ss / den).cpu().numpy().astype(np.float32); keep = cnt_t.cpu().numpy() > 0
    np.savez(CACHE, SAE=SAE, keep=keep)
    print(f"[cache] pooled + saved {SAE.shape} -> {CACHE}", flush=True)

rng = np.random.default_rng(0); perm = rng.permutation(np.where(keep)[0]); h = len(perm) // 2
tr_i, te_i = perm[:h], perm[h:]
sc = StandardScaler().fit(SAE[tr_i]); Xs = sc.transform(SAE)

# subset of well-characterized domains for the sweep (fast + interpretable), always include kinase
KEY = ["IPR000719", "IPR011009", "IPR016024", "IPR015943", "IPR009057", "IPR036179",
       "IPR013083", "IPR027417", "IPR013783", "IPR008271"]
sweep_domains = [d for d in KEY if d in DOMAINS] or DOMAINS[:10]

Y = {d: np.array([1 if d in ipro_of.get(pids[i], ()) else 0 for i in range(nP)]) for d in sweep_domains}
sweep = {}
print(f"\n{'C':>7} {'mean AUROC':>11} {'med #sel':>9} {'distinct top-1':>15} {'kinase top-1':>13}")
for C in [0.001, 0.01, 0.1]:   # strong -> weak L1 (bounded: liblinear on 40,960 feats is slow)
    per = {}
    for d in sweep_domains:
        y = Y[d]
        if y[tr_i].sum() < 8 or y[te_i].sum() < 8:
            continue
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=200).fit(Xs[tr_i], y[tr_i])
        au = float(roc_auc_score(y[te_i], clf.decision_function(Xs[te_i])))
        coef = clf.coef_[0]; nz = np.where(coef != 0)[0]
        top = nz[np.argsort(-np.abs(coef[nz]))][:5] if len(nz) else np.array([], int)
        per[d] = {"auroc": round(au, 3), "n_sel": int(len(nz)), "top": [int(f) for f in top]}
    if not per: continue
    top1 = [v["top"][0] for v in per.values() if v["top"]]
    distinct = len(set(top1)) / max(len(top1), 1)
    kin = per.get("IPR000719", {}).get("top", [None])
    sweep[C] = {"mean_auroc": round(float(np.mean([v["auroc"] for v in per.values()])), 3),
                "median_n_sel": int(np.median([v["n_sel"] for v in per.values()])),
                "distinct_top1_frac": round(distinct, 2),
                "kinase_top1": kin[0] if kin else None,
                "kinase_top1_is_F16026": (kin[0] == 16026) if kin else False,
                "per_domain": per}
    print(f"{C:>7} {sweep[C]['mean_auroc']:>11} {sweep[C]['median_n_sel']:>9} "
          f"{sweep[C]['distinct_top1_frac']:>15} {str(sweep[C]['kinase_top1']):>13}")

json.dump({"layer": layer, "store": store, "sweep_domains": sweep_domains, "sweep": sweep},
          open(out, "w"), indent=2)
print(f"\n[wrote] {out}")
print("Read: as C shrinks (stronger L1) -> fewer features, and if attribution is clean the top-1")
print("features become DISTINCT per domain (distinct_top1_frac -> 1.0) and kinase top-1 -> F16026.")
