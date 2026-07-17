#!/usr/bin/env python
"""Jared-faithful probe (the corrected 5-step process), per band. Fixes the audit findings:
  1. OVERLAP metric (single-latent AUROC) — sink-excluded — PLUS specificity (AUPRC, precision@q95).
  2. TRAINED probe — 5-fold CV, DIMENSIONALITY-MATCHED: SAE-SVD-K vs raw-PCA-K vs random-K.
  3. GAP = probe_CV - overlap_best  -> monosemantic / distributed / not-encoded.
  4. NULLS — label-shuffle lift + the random-feature (random-SAE) baseline (critical on reasoning band:
     if random probe ~= real, the concept is LEAKED into the text, not represented).
  5. (enrichment left to a separate GSEA pass.)
Caches the per-band pooled SAE+raw matrices (slow store -> pool once).
Usage: probe_v2.py <sae.pt> <store> <layer> <protein|reasoning> [K] [out.json]
"""
import glob, json, sys, hashlib, os
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

sae_p, store, layer, BAND = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 256
out = sys.argv[6] if len(sys.argv) > 6 else f"probe_v2_{BAND}_l{layer}.json"
CACHE = f"/data/savithas/phase3_full/pooled_{BAND}_l{layer}_{hashlib.md5(store.encode()).hexdigest()[:6]}.npz"
dev = "cuda"

tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
if BAND == "protein":
    tokmask = band == "protein"
else:  # reasoning = response text, accessions excluded (still circular for GO — that's the point of the null)
    rs = pq.read_table(f"{store}/token_labels_with_role.parquet")
    role = np.array(rs.column("role").to_pylist(), dtype=object)
    tokmask = (band == "text") & (role == "response")
    if "is_accession" in rs.column_names:
        tokmask &= ~np.array(rs.column("is_accession").to_pylist(), dtype=bool)
pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
goids = [set(json.loads(g) if g else []) for g in pr.column("go_ids").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)
import bioreason_pro_sae.data as brp
tr_ds, _, _ = brp.load_reasoning_splits(max_length_protein=2000)
ipro_of = {str(p): set(i) if i else set() for p, i in zip(tr_ds["protein_id"], tr_ds["interpro_ids"])}

if os.path.exists(CACHE):
    z = np.load(CACHE); SAE = z["SAE"]; RAW = z["RAW"]; RND = z["RND"]; keep = z["keep"]
    print(f"[cache] loaded {CACHE}", flush=True)
else:
    ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
    sae = TopKSAE(**cfg).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    rsae = TopKSAE(**cfg).to(dev).eval()
    H = sae.hidden_dim
    rpi = torch.from_numpy(np.array([pidx.get(str(p), -1) for p in rpid])).to(dev)
    tm = torch.from_numpy(tokmask).to(dev)
    D = None; ss = rr = rn = cnt = None; row0 = 0
    with torch.no_grad():
        for si, sp in enumerate(sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))):
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            if D is None:
                D = X.shape[1]; ss = torch.zeros(nP, H, device=dev); rr = torch.zeros(nP, D, device=dev); rn = torch.zeros(nP, H, device=dev); cnt = torch.zeros(nP, device=dev)
            for s in range(0, n, 8192):
                e = min(n, s + 8192); m = tm[row0 + s:row0 + e]
                if not m.any(): continue
                xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev); idx = rpi[row0 + s:row0 + e][m]
                ss.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), sae.encode(xb)[m])
                rr.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), xb[m])
                rn.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), rsae.encode(xb)[m])
                cnt.scatter_add_(0, idx, torch.ones(int(m.sum()), device=dev))
            row0 += n
            if si % 20 == 0: print(f"  pool shard {si}", flush=True)
    den = cnt.clamp(min=1)[:, None]
    SAE = (ss / den).cpu().numpy(); RAW = (rr / den).cpu().numpy(); RND = (rn / den).cpu().numpy(); keep = cnt.cpu().numpy() > 0
    np.savez(CACHE, SAE=SAE, RAW=RAW, RND=RND, keep=keep)
    print(f"[cache] pooled + saved {CACHE}", flush=True)

# sink mask (per-token freq) for the OVERLAP metric
freq = np.load("/data/savithas/phase3_full/per_token_freq_l30.npy") if BAND == "protein" else None
sink = (freq > 0.10) if freq is not None else np.zeros(SAE.shape[1], bool)

idx_keep = np.where(keep)[0]
rng = np.random.default_rng(0)
# matched-dim reps
skf = StratifiedKFold(5, shuffle=True, random_state=0)

def overlap(Xf, y):  # single-latent AUROC (correlation), sink-excluded, + specificity of best
    au = np.array([roc_auc_score(y, Xf[:, f]) if Xf[:, f].std() > 0 else 0.5 for f in range(Xf.shape[1])])
    au = np.maximum(au, 1 - au); au[sink[:Xf.shape[1]]] = 0
    bf = int(np.argmax(au)); s = Xf[:, bf]
    ap = average_precision_score(y, s); q = np.quantile(s, 0.95)
    prec = float((y[s >= q].mean())) if (s >= q).sum() else 0.0
    return round(float(au[bf]), 3), bf, round(float(ap), 3), round(prec, 3)

def probe_cv(Xf, y):  # 5-fold CV trained LR
    sc = StandardScaler().fit(Xf); Xs = sc.transform(Xf)
    return round(float(cross_val_score(LogisticRegression(C=1.0, max_iter=300), Xs, y, cv=skf, scoring="roc_auc").mean()), 3)

# concept panel: top GO terms + top InterPro domains (data-derived, diverse — NOT hand-picked)
from collections import Counter
goc = Counter(t for i in idx_keep for t in goids[i]); ipc = Counter(d for i in idx_keep for d in ipro_of.get(pids[i], ()))
CONCEPTS = [("GO", t) for t, c in goc.most_common(25) if c >= 40] + [("IPR", d) for d, c in ipc.most_common(15) if c >= 40]

# fit matched-dim projections ONCE (on kept proteins)
Xk = {"sae": SAE[idx_keep], "raw": RAW[idx_keep], "rnd": RND[idx_keep]}
proj = {"sae": TruncatedSVD(K, random_state=0).fit_transform(Xk["sae"]),
        "raw": TruncatedSVD(min(K, Xk["raw"].shape[1] - 1), random_state=0).fit_transform(Xk["raw"]),
        "rnd": TruncatedSVD(K, random_state=0).fit_transform(Xk["rnd"])}
SAEk = Xk["sae"]

results = {}
print(f"\nband={BAND} layer={layer} K={K}  (n={len(idx_keep)} proteins, {len(CONCEPTS)} concepts)")
print(f"{'concept':16} {'npos':>5} {'ovl':>5} {'ovlP@95':>7} {'probeSAE':>8} {'raw':>5} {'RAND':>5} {'gap':>5} {'shuf':>5}")
for kind, t in CONCEPTS:
    y = np.array([1 if (t in goids[i] if kind == "GO" else t in ipro_of.get(pids[i], ())) else 0 for i in idx_keep])
    if y.sum() < 15 or (1 - y).sum() < 15: continue
    ovl, bf, ap, p95 = overlap(SAEk, y)
    ps = probe_cv(proj["sae"], y); pr_ = probe_cv(proj["raw"], y); pn = probe_cv(proj["rnd"], y)
    yshuf = rng.permutation(y); pshuf = probe_cv(proj["sae"], yshuf)
    gap = round(ps - ovl, 3)
    results[t] = {"kind": kind, "npos": int(y.sum()), "overlap_best_auroc": ovl, "overlap_best_feat": bf,
                  "overlap_ap": ap, "overlap_prec_q95": p95, "probe_sae_cv": ps, "probe_raw_cv": pr_,
                  "probe_random_cv": pn, "gap": gap, "probe_shuffle": pshuf}
    tag = "LEAK" if pn >= ps - 0.02 else ("mono" if abs(gap) < 0.03 else "distrib")
    print(f"{t:16} {int(y.sum()):>5} {ovl:>5} {p95:>7} {ps:>8} {pr_:>5} {pn:>5} {gap:>5} {pshuf:>5}  {tag}")

mean = lambda k: round(float(np.mean([r[k] for r in results.values()])), 3) if results else 0.0
summ = {k: mean(k) for k in ["overlap_best_auroc", "overlap_prec_q95", "probe_sae_cv", "probe_raw_cv", "probe_random_cv", "gap"]}
print(f"\nMEAN: {summ}")
print("KEY: if probe_random_cv ~= probe_sae_cv -> concept is LEAKED into the band (not represented). "
      "gap~0 -> monosemantic; gap large -> distributed.")
json.dump({"band": BAND, "layer": layer, "K": K, "mean": summ, "results": results}, open(out, "w"), indent=2)
print(f"[wrote] {out}")
