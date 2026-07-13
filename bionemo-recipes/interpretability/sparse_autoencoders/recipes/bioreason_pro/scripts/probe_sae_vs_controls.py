#!/usr/bin/env python
"""Jared-style concept probe WITH controls (CodonFM SAE paper, §3).

For each specific GO term, find the best single-feature detector and its held-out rank-AUROC,
separately among three detector sets:
  - SAE features        (the trained SAE latents)
  - raw hidden dims      (the 2560 model dimensions, signed)
  - random-SAE features  (a randomly-initialised SAE of the same shape — same feature count as SAE)

Detector = mean-pool the value over the chosen band's tokens per protein, rank proteins, AUROC vs
"protein has GO term". Best detector per term is picked on a TRAIN protein split and scored on a
held-out TEST split (this is essential: the SAE has far more features than raw dims, so without
held-out selection it would win purely on more draws). AUROC is symmetric (max(a,1-a)) so signed raw
dims and anti-correlated features get a fair shot.

Band: 'protein' (residues) or 'reasoning' (text tokens with role=response, i.e. prompt excluded to
avoid the go_pred leakage — needs token_labels_with_role.parquet from add_role_sidecar_bystore.py).

Usage: probe_sae_vs_controls.py <sae.pt> <store> <layer> [protein|reasoning]
"""
import glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

sae_p, store, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
BAND = sys.argv[4] if len(sys.argv) > 4 else "protein"
POOL = sys.argv[5] if len(sys.argv) > 5 else "mean"   # mean (Jared) | max (peak magnitude; fair to sparse feats)
dev = "cuda"

tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
if BAND == "protein":
    tokmask = band == "protein"
else:  # reasoning = text tokens, prompt excluded via role sidecar
    rp = Path(f"{store}/token_labels_with_role.parquet")
    role = np.array(pq.read_table(rp).column("role").to_pylist(), dtype=object) if rp.exists() else None
    tokmask = (band == "text") & (role == "response") if role is not None else (band == "text")
print(f"[probe] band={BAND}: {int(tokmask.sum()):,} tokens")

pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
goids = [json.loads(g) if g else [] for g in pr.column("go_ids").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)
rpi = torch.from_numpy(np.array([pidx.get(p, -1) for p in rpid])).to(dev)
tm = torch.from_numpy(tokmask).to(dev)

ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval()  # random-init control (same shape)
H = sae.hidden_dim

D = None; ss = sr = sraw = pcnt = None; row0 = 0
with torch.no_grad():
    for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        if D is None:
            D = X.shape[1]
            ss = torch.zeros(nP, H, device=dev); sr = torch.zeros(nP, H, device=dev)
            sraw = torch.zeros(nP, D, device=dev); pcnt = torch.zeros(nP, device=dev)
        for s in range(0, n, 8192):
            e = min(n, s + 8192); m = tm[row0 + s:row0 + e]
            if not m.any(): continue
            xb = torch.from_numpy(X[s:e]).to(dev); idx = rpi[row0 + s:row0 + e][m]
            ih, id_ = idx.unsqueeze(1).expand(-1, H), idx.unsqueeze(1).expand(-1, D)
            if POOL == "mean":
                ss.scatter_add_(0, ih, sae.encode(xb)[m]); sr.scatter_add_(0, ih, rsae.encode(xb)[m])
                sraw.scatter_add_(0, id_, xb[m]); pcnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
            else:  # max of magnitude: peak detector value per protein (fair to sparse SAE features)
                ss.scatter_reduce_(0, ih, sae.encode(xb)[m], reduce="amax", include_self=True)
                sr.scatter_reduce_(0, ih, rsae.encode(xb)[m], reduce="amax", include_self=True)
                sraw.scatter_reduce_(0, id_, xb[m].abs(), reduce="amax", include_self=True)
        row0 += n
if POOL == "mean":
    den = pcnt.clamp(min=1)[:, None]; ss, sr, sraw = ss / den, sr / den, sraw / den
P = {"SAE": ss.cpu().numpy(), "random-SAE": sr.cpu().numpy(), "raw-hidden": sraw.cpu().numpy()}

# specific GO terms: prevalence-filtered, generic roots dropped
obo = {}; cur = None
for line in open("/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo"):
    line = line.strip()
    if line.startswith("id: GO:"): cur = line[4:]
    elif line.startswith("name:") and cur: obo[cur] = line[6:]; cur = None
GENERIC = ("binding", "process", "intracellular", "organelle", "regulation", "anatomical", "cellular",
           "catalytic activity", "molecular_function", "biological_process", "cellular_component",
           "metabolic process", "biosynthetic process", "macromolecule", "protein-containing")
sets = [set(g) for g in goids]; freq = Counter(t for g in goids for t in g); terms = []
for t, _ in freq.most_common():
    pv = sum(t in s for s in sets) / nP; nm = obo.get(t, "").lower()
    if 0.02 <= pv <= 0.4 and not any(gk in nm for gk in GENERIC): terms.append(t)
    if len(terms) >= 60: break
Y = np.stack([np.array([1 if t in s else 0 for s in sets]) for t in terms]).astype(float)  # (T, nP)
print(f"[probe] {len(terms)} specific GO terms")

rng = np.random.default_rng(0); perm = rng.permutation(nP); h = nP // 2; tr, te = perm[:h], perm[h:]
def auc_mat(A, rows, Ys):  # symmetric rank-AUROC: (T, F)
    sub = A[rows]; m = sub.shape[0]; oi = np.argsort(sub, 0); R = np.empty_like(sub); ar = np.arange(1, m + 1)
    for j in range(sub.shape[1]): R[oi[:, j], j] = ar
    npos = Ys.sum(1); a = (Ys @ R - (npos * (npos + 1) / 2)[:, None]) / (npos[:, None] * (m - npos)[:, None] + 1e-9)
    return np.maximum(a, 1 - a)
valid = (Y[:, tr].sum(1) >= 3) & ((1 - Y[:, tr]).sum(1) >= 3) & (Y[:, te].sum(1) >= 3) & ((1 - Y[:, te]).sum(1) >= 3)

results = {}
for name, A in P.items():
    act = np.where((np.abs(A) > 0).any(0))[0]; Aa = A[:, act]
    atr = auc_mat(Aa, tr, Y[:, tr]); ate = auc_mat(Aa, te, Y[:, te])
    best = np.argmax(atr, 1); held = ate[np.arange(len(terms)), best]  # held-out AUROC of train-best detector per term
    results[name] = held
    print(f"[{name:11}] mean best-concept held-out AUROC = {held[valid].mean():.3f}  (n={int(valid.sum())} terms, {len(act)} detectors)")

sae_h, rnd_h, raw_h = results["SAE"], results["random-SAE"], results["raw-hidden"]
win_raw = (sae_h[valid] > raw_h[valid]).mean(); win_rnd = (sae_h[valid] > rnd_h[valid]).mean()
print(f"\nSAE beats raw-hidden on {win_raw*100:.1f}% of terms | SAE beats random-SAE on {win_rnd*100:.1f}% of terms")
print("\n=== per-term (top 15 by SAE held-out AUROC) ===")
order = np.argsort(-np.where(valid, sae_h, -1))[:15]
for j in order:
    print(f"  {obo.get(terms[j],terms[j])[:34]:34} SAE={sae_h[j]:.2f} raw={raw_h[j]:.2f} rand={rnd_h[j]:.2f}")
