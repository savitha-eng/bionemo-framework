#!/usr/bin/env python
"""Proper PROBE (trained classifier), the thing Jared means — distinct from GO-label overlap.

For a chosen functional concept (a GO term), mean-pool the representation per protein over a band
(protein residues OR reasoning tokens), train a logistic-regression classifier to predict the label,
and report held-out AUROC. Run on three representations to make it meaningful:
  - SAE features   (dense L2 probe, and an L1-SPARSE probe -> how few features recover the signal)
  - raw hidden     (dense L2 probe; the reference)
  - random-SAE     (dense L2 probe; baseline)
The headline is not "SAE dense > raw" (SAE has 16x more dims = capacity), it's "a SPARSE handful of
SAE features recovers what the dense raw representation carries" (Jared's tAI §7.2 result).

Band 'reasoning' = text tokens, role=response, accessions excluded (needs token_labels_with_role.parquet).
Usage: probe_trained.py <sae.pt> <store> <layer> <protein|reasoning> [out.json]
"""
import glob, json, sys
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sae_p, store, layer, BAND = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
out = sys.argv[5] if len(sys.argv) > 5 else f"probe_{BAND}_l{layer}.json"
dev = "cuda"

tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
if BAND == "protein":
    tokmask = band == "protein"
else:
    rs = pq.read_table(f"{store}/token_labels_with_role.parquet")
    role = np.array(rs.column("role").to_pylist(), dtype=object)
    tokmask = (band == "text") & (role == "response")
    if "is_accession" in rs.column_names:
        tokmask &= ~np.array(rs.column("is_accession").to_pylist(), dtype=bool)
print(f"[probe] band={BAND}: {int(tokmask.sum()):,} tokens")

pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
goids = [set(json.loads(g) if g else []) for g in pr.column("go_ids").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)
rpi = torch.from_numpy(np.array([pidx.get(p, -1) for p in rpid])).to(dev)
tm = torch.from_numpy(tokmask).to(dev)

ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval()
H = sae.hidden_dim

D = None; ss = rr = rn = cnt = None; row0 = 0   # mean-pool per protein
with torch.no_grad():
    for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        if D is None:
            D = X.shape[1]; ss = torch.zeros(nP, H, device=dev); rr = torch.zeros(nP, D, device=dev)
            rn = torch.zeros(nP, H, device=dev); cnt = torch.zeros(nP, device=dev)
        for s in range(0, n, 8192):
            e = min(n, s + 8192); m = tm[row0 + s:row0 + e]
            if not m.any(): continue
            xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev); idx = rpi[row0 + s:row0 + e][m]
            ss.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), sae.encode(xb)[m])
            rr.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), xb[m])
            rn.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), rsae.encode(xb)[m])
            cnt.scatter_add_(0, idx, torch.ones(int(m.sum()), device=dev))
        row0 += n
den = cnt.clamp(min=1)[:, None]
SAE = (ss / den).cpu().numpy(); RAW = (rr / den).cpu().numpy(); RND = (rn / den).cpu().numpy()
keep = cnt.cpu().numpy() > 0
print(f"[probe] pooled {int(keep.sum())}/{nP} proteins with band tokens; SAE dim {H}, raw dim {D}")

CONCEPTS = {"mitochondrion": "GO:0005739", "nucleus": "GO:0005634", "plasma membrane": "GO:0005886",
    "cytoplasm": "GO:0005737", "kinase activity": "GO:0016301", "transporter activity": "GO:0005215",
    "oxidoreductase activity": "GO:0016491", "DNA binding": "GO:0003677", "transcription regulator": "GO:0140110"}
rng = np.random.default_rng(0); perm = rng.permutation(np.where(keep)[0]); h = len(perm) // 2
tr, te = perm[:h], perm[h:]

def probe(Xf, y, sparse=False):
    sc = StandardScaler().fit(Xf[tr]); Xs = sc.transform(Xf)
    if sparse:
        clf = LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=300, tol=1e-3)
    else:
        clf = LogisticRegression(C=1.0, max_iter=300, tol=1e-3)
    clf.fit(Xs[tr], y[tr])
    auc = roc_auc_score(y[te], clf.decision_function(Xs[te]))
    nnz = int((np.abs(clf.coef_) > 1e-6).sum())
    return round(float(auc), 3), nnz

results = {}
print(f"\n{'concept':22} {'SAE(dense)':11} {'SAE(sparse)':16} {'raw':7} {'random':7}")
for nm, t in CONCEPTS.items():
    y = np.array([1 if t in goids[i] else 0 for i in range(nP)])
    if y[tr].sum() < 5 or y[te].sum() < 5:
        continue
    a_sae, _ = probe(SAE, y); a_sp, nnz = probe(SAE, y, sparse=True)
    a_raw, _ = probe(RAW, y); a_rnd, _ = probe(RND, y)
    results[nm] = {"go": t, "sae_dense": a_sae, "sae_sparse": a_sp, "sae_sparse_nfeat": nnz, "raw": a_raw, "random": a_rnd}
    print(f"{nm:22} {a_sae:<11} {str(a_sp)+' ('+str(nnz)+'f)':16} {a_raw:<7} {a_rnd:<7}")

mean = lambda k: round(float(np.mean([r[k] for r in results.values()])), 3)
print(f"\nMEAN AUROC: SAE-dense {mean('sae_dense')} | SAE-sparse {mean('sae_sparse')} | raw {mean('raw')} | random {mean('random')}")
Path(out).write_text(json.dumps({"band": BAND, "layer": layer, "results": results,
    "mean": {k: mean(k) for k in ["sae_dense", "sae_sparse", "raw", "random"]}}, indent=2))
print(f"[wrote] {out}")
