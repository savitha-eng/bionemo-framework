#!/usr/bin/env python
"""NON-CIRCULAR InterPro-DOMAIN probe: predict InterPro domain presence from protein RESIDUE activations.

Domain membership is a sequence/structure annotation (a conserved fold/motif), NOT a functional description
that the reasoning restates -> non-circular, unlike GO-label overlap. Directly parallels the paper's
InterPro-based interpretability analysis. Mirrors probe_trained.py: mean-pool residues per protein, train
logistic regression, report held-out AUROC for SAE (SVD-256 dim-matched) vs raw-hidden vs random-SAE.
Headline is NOT "SAE dense > raw" (capacity) — it's whether a compact SAE subspace recovers domain identity
as well as the raw residue representation, and how the protein band compares to the reasoning band.

Usage: interpro_probe.py <sae.pt> <store> <layer> [out.json]
"""
import glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
import bioreason_pro_sae.data as brp_data

sae_p, store, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else f"interpro_probe_l{layer}.json"
dev = "cuda"

# ---- store residue-band mask + protein order ----
tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
prot_mask = band == "protein"
pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)

# ---- InterPro labels per protein (from dataset, matched by protein_id) ----
tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
ipro_of = {str(pid): set(ips) if ips else set() for pid, ips in zip(tr["protein_id"], tr["interpro_ids"])}
cnt = Counter()
for p in pids:
    for d in ipro_of.get(p, ()):
        cnt[d] += 1
DOMAINS = [d for d, c in cnt.most_common(60) if c >= 50]     # abundant enough for a probe
print(f"[ipro] {len(DOMAINS)} abundant InterPro domains among {nP} store proteins")

# ---- mean-pool residue activations per protein (SAE, raw, random) ----
ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval()
H = sae.hidden_dim
rpi = torch.from_numpy(np.array([pidx.get(str(p), -1) for p in rpid])).to(dev)
pm = torch.from_numpy(prot_mask).to(dev)
D = None; ss = rr = rn = cnt_t = None; row0 = 0
with torch.no_grad():
    for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        if D is None:
            D = X.shape[1]; ss = torch.zeros(nP, H, device=dev); rr = torch.zeros(nP, D, device=dev)
            rn = torch.zeros(nP, H, device=dev); cnt_t = torch.zeros(nP, device=dev)
        for s in range(0, n, 8192):
            e = min(n, s + 8192); m = pm[row0 + s:row0 + e]
            if not m.any(): continue
            xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev); idx = rpi[row0 + s:row0 + e][m]
            ss.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), sae.encode(xb)[m])
            rr.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), xb[m])
            rn.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), rsae.encode(xb)[m])
            cnt_t.scatter_add_(0, idx, torch.ones(int(m.sum()), device=dev))
        row0 += n
den = cnt_t.clamp(min=1)[:, None]
SAE = (ss / den).cpu().numpy(); RAW = (rr / den).cpu().numpy(); RND = (rn / den).cpu().numpy()
keep = cnt_t.cpu().numpy() > 0
print(f"[ipro] pooled {int(keep.sum())}/{nP} proteins with residues; SAE dim {H}, raw dim {D}")

rng = np.random.default_rng(0); perm = rng.permutation(np.where(keep)[0]); h = len(perm) // 2
tr_i, te_i = perm[:h], perm[h:]
svd = TruncatedSVD(n_components=256, random_state=0).fit(SAE[tr_i]); SAE_SVD = svd.transform(SAE)  # dim-matched


def probe(Xf, y):
    sc = StandardScaler().fit(Xf[tr_i]); Xs = sc.transform(Xf)
    clf = LogisticRegression(C=1.0, max_iter=300, tol=1e-3).fit(Xs[tr_i], y[tr_i])
    return round(float(roc_auc_score(y[te_i], clf.decision_function(Xs[te_i]))), 3)


results = {}
name_of = {str(pid): None for pid in tr["protein_id"]}
print(f"\n{'InterPro domain':16} {'npos':>5} {'SAE-svd':>8} {'raw':>6} {'rand':>6}")
for d in DOMAINS:
    y = np.array([1 if d in ipro_of.get(pids[i], ()) else 0 for i in range(nP)])
    if y[tr_i].sum() < 10 or y[te_i].sum() < 10:
        continue
    a_sae = probe(SAE_SVD, y); a_raw = probe(RAW, y); a_rnd = probe(RND, y)
    results[d] = {"npos": int(y[perm].sum()), "sae_svd": a_sae, "raw": a_raw, "random": a_rnd}
    print(f"{d:16} {int(y[perm].sum()):>5} {a_sae:>8} {a_raw:>6} {a_rnd:>6}")

mean = lambda k: round(float(np.mean([r[k] for r in results.values()])), 3) if results else 0.0
print(f"\nMEAN AUROC ({len(results)} domains): SAE-svd {mean('sae_svd')} | raw {mean('raw')} | random {mean('random')}")
print("SAE-svd ~ raw => SAE re-expresses the residue rep (protein band is the ceiling, matches GO finding).")
print("SAE-svd >> random => the probe is real, not capacity. raw >> SAE would mean SAE loses domain info.")
Path(out).write_text(json.dumps({"layer": layer, "n_domains": len(results),
    "results": results, "mean": {k: mean(k) for k in ("sae_svd", "raw", "random")}}, indent=2))
print(f"[wrote] {out}")
