#!/usr/bin/env python
"""Non-circular reproduction probe: can we learn the reproduction label from the protein RESIDUES?
Plus: do the reasoning feature F4777 (fires on repro reasoning) and the residue probe AGREE on which
proteins are reproductive? (text-side feature vs sequence-side probe — a cross-modal consistency check).

- residue probe: mean-pool the representation over each protein's RESIDUE tokens, train a classifier for
  the reproduction GO label, held-out AUROC. SAE (SVD-matched to raw dim) vs raw vs random. Non-circular
  because residues are the input sequence, not a functional description.
- agreement: pool F4777 over each protein's REASONING tokens (its reproduction-reasoning score), then
  correlate with (a) the label and (b) the residue-probe prediction; report top-K overlap.

Usage: repro_residue_probe.py <sae.pt> <store> <layer> <feature_id> <go_term> [out.json]
"""
import glob, json, sys
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

sae_p, store, layer, FEAT, GO = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
out = sys.argv[6] if len(sys.argv) > 6 else f"repro_residue_f{FEAT}.json"
dev = "cuda"

tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rs = pq.read_table(f"{store}/token_labels_with_role.parquet")
role = np.array(rs.column("role").to_pylist(), dtype=object)
isacc = np.array(rs.column("is_accession").to_pylist(), dtype=bool) if "is_accession" in rs.column_names else np.zeros(len(role), bool)
prot_mask = band == "protein"                                   # residues
reas_mask = (band == "text") & (role == "response") & (~isacc)  # reasoning (for F4777)

pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
goids = [set(json.loads(g) if g else []) for g in pr.column("go_ids").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)
rpi = torch.from_numpy(np.array([pidx.get(p, -1) for p in rpid])).to(dev)
pm = torch.from_numpy(prot_mask).to(dev); rm = torch.from_numpy(reas_mask).to(dev)

ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval()
H = sae.hidden_dim

D = None; ss = rr = rn = cnt = None; f_reas = None; row0 = 0
with torch.no_grad():
    for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        if D is None:
            D = X.shape[1]; ss = torch.zeros(nP, H, device=dev); rr = torch.zeros(nP, D, device=dev)
            rn = torch.zeros(nP, H, device=dev); cnt = torch.zeros(nP, device=dev); f_reas = torch.zeros(nP, device=dev)
        for s in range(0, n, 8192):
            e = min(n, s + 8192)
            xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev)
            mp = pm[row0 + s:row0 + e]
            if mp.any():                                        # residue pooling (mean) for the probe
                idx = rpi[row0 + s:row0 + e][mp]; enc = sae.encode(xb)[mp]
                ss.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), enc)
                rr.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), xb[mp])
                rn.scatter_add_(0, idx.unsqueeze(1).expand(-1, H), rsae.encode(xb)[mp])
                cnt.scatter_add_(0, idx, torch.ones(int(mp.sum()), device=dev))
            mr = rm[row0 + s:row0 + e]
            if mr.any():                                        # F4777 reasoning score (max over reasoning tokens)
                idr = rpi[row0 + s:row0 + e][mr]; fv = sae.encode(xb)[mr][:, FEAT]
                f_reas.scatter_reduce_(0, idr, fv, reduce="amax", include_self=True)
        row0 += n
den = cnt.clamp(min=1)[:, None]
SAE = (ss / den).cpu().numpy(); RAW = (rr / den).cpu().numpy(); RND = (rn / den).cpu().numpy()
F4777 = f_reas.cpu().numpy(); keep = cnt.cpu().numpy() > 0
y = np.array([1 if GO in goids[i] else 0 for i in range(nP)])
print(f"[repro] {GO}: {int(y.sum())} positive / {nP} proteins | {int(keep.sum())} with residues")

rng = np.random.default_rng(0); perm = rng.permutation(np.where(keep)[0]); h = len(perm) // 2
tr, te = perm[:h], perm[h:]
svd = TruncatedSVD(n_components=256, random_state=0).fit(SAE[tr]); SAE_SVD = svd.transform(SAE)  # Jared §7.2 SVD-256
def probe(Xf):
    sc = StandardScaler().fit(Xf[tr]); Xs = sc.transform(Xf)
    clf = LogisticRegression(C=1.0, max_iter=300, tol=1e-3).fit(Xs[tr], y[tr])
    return clf, roc_auc_score(y[te], clf.decision_function(Xs[te]))

clf_sae, auc_sae = probe(SAE_SVD)
_, auc_raw = probe(RAW); _, auc_rnd = probe(RND)
# residue-probe prediction on held-out (using raw probe as the sequence-side predictor)
scaler = StandardScaler().fit(RAW[tr]); clf_raw = LogisticRegression(C=1.0, max_iter=300).fit(scaler.transform(RAW[tr]), y[tr])
resid_pred = clf_raw.decision_function(scaler.transform(RAW))

print(f"\n=== RESIDUE probe for reproduction (non-circular) ===")
print(f"  SAE(svd-matched) {auc_sae:.3f} | raw {auc_raw:.3f} | random {auc_rnd:.3f}")

# --- agreement: F4777 reasoning score vs label + vs residue-probe prediction (held-out) ---
te_keep = te
auc_f_label = roc_auc_score(y[te_keep], F4777[te_keep]) if y[te_keep].sum() >= 3 else float("nan")
rho_f_resid = spearmanr(F4777[te_keep], resid_pred[te_keep]).correlation
# top-K overlap: proteins F4777 fires on most vs proteins the residue probe scores highest
K = int(y.sum())
topf = set(np.argsort(-F4777)[:K]); topr = set(np.argsort(-resid_pred)[:K])
overlap = len(topf & topr) / K
truepos_f = len(topf & set(np.where(y == 1)[0])) / K
print(f"\n=== agreement: F4777 (reasoning) vs residues ===")
print(f"  F4777 -> reproduction label AUROC (held-out): {auc_f_label:.3f}")
print(f"  F4777(reasoning) vs residue-probe prediction, Spearman rho: {rho_f_resid:.3f}")
print(f"  top-{K} overlap  F4777 vs residue-probe: {overlap:.2f}")
print(f"  fraction of F4777's top-{K} proteins that are truly reproduction: {truepos_f:.2f}")
print("\n  high F4777->label AUROC + high overlap/rho => the text-side feature and the sequence-side")
print("  probe agree on which proteins are reproductive (cross-modal consistency, non-circular).")
Path(out).write_text(json.dumps({"go": GO, "feature": FEAT, "npos": int(y.sum()),
    "residue_probe": {"sae_svd": auc_sae, "raw": auc_raw, "random": auc_rnd},
    "agreement": {"f4777_label_auc": auc_f_label, "f4777_resid_rho": rho_f_resid,
                  "topk_overlap": overlap, "f4777_topk_truepos": truepos_f}}, indent=2))
print(f"[wrote] {out}")
