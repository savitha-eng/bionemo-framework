#!/usr/bin/env python
"""Per-feature validation of protein SAE features (BioReason-Pro, blog/paper deliverable).

For each active SAE feature we produce an evidence bundle and keep the ones that pass:
  1. selectivity   -> held-out GO-AUROC (best specific term picked on TRAIN, scored on TEST)
  2. control margin -> feature AUROC minus the best raw-hidden dim AND best random-SAE feature
                       for the SAME term (both baselines also train-selected, test-scored)
  3. coherence     -> fraction of the feature's top-20 proteins that carry the term
The reasoning-text agreement (4th piece) is added from the xmodal_caption metadata column.

Max-pool per protein over the protein band (peak activation; natural for sparse features).
Writes analysis/validated_features_<layer>.csv, sorted by held-out AUROC.

Usage: validate_features.py <sae.pt> <store> <layer> <pub_dir>
"""
import glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

sae_p, store, layer, pub = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
BAND = sys.argv[5] if len(sys.argv) > 5 else "protein"   # protein | reasoning
dev = "cuda"

tl = pq.read_table(f"{store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
goids = [json.loads(g) if g else [] for g in pr.column("go_ids").to_pylist()]
pidx = {p: i for i, p in enumerate(pids)}; nP = len(pids)
rpi = torch.from_numpy(np.array([pidx.get(p, -1) for p in rpid])).to(dev)
if BAND == "protein":
    tokmask = band == "protein"
else:  # reasoning = text tokens, prompt excluded via role sidecar (avoids go_pred leakage)
    role = np.array(pq.read_table(f"{store}/token_labels_with_role.parquet").column("role").to_pylist(), dtype=object)
    tokmask = (band == "text") & (role == "response")
print(f"[validate] band={BAND}: {int(tokmask.sum()):,} tokens")
tm = torch.from_numpy(tokmask).to(dev)

ck = torch.load(sae_p, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
rsae = TopKSAE(**cfg).to(dev).eval()
H = sae.hidden_dim

D = None; ps = pr_ = praw = None; row0 = 0   # max-pool per protein
with torch.no_grad():
    for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1])):
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        if D is None:
            D = X.shape[1]; ps = torch.zeros(nP, H, device=dev); pr_ = torch.zeros(nP, H, device=dev); praw = torch.zeros(nP, D, device=dev)
        for s in range(0, n, 8192):
            e = min(n, s + 8192); m = tm[row0 + s:row0 + e]
            if not m.any(): continue
            xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev); idx = rpi[row0 + s:row0 + e][m]
            ps.scatter_reduce_(0, idx.unsqueeze(1).expand(-1, H), sae.encode(xb)[m], reduce="amax", include_self=True)
            pr_.scatter_reduce_(0, idx.unsqueeze(1).expand(-1, H), rsae.encode(xb)[m], reduce="amax", include_self=True)
            praw.scatter_reduce_(0, idx.unsqueeze(1).expand(-1, D), xb[m].abs(), reduce="amax", include_self=True)
        row0 += n
Asae, Arnd, Araw = ps.cpu().numpy(), pr_.cpu().numpy(), praw.cpu().numpy()

# specific GO terms
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
    if len(terms) >= 80: break
Y = np.stack([np.array([1 if t in s else 0 for s in sets]) for t in terms]).astype(float)  # (T,nP)
rng = np.random.default_rng(0); perm = rng.permutation(nP); h = nP // 2; tr, te = perm[:h], perm[h:]
validT = (Y[:, tr].sum(1) >= 3) & ((1 - Y[:, tr]).sum(1) >= 3) & (Y[:, te].sum(1) >= 3) & ((1 - Y[:, te]).sum(1) >= 3)

def aucs(A, rows, Ys):  # symmetric rank-AUROC (T, F)
    sub = A[rows]; m = sub.shape[0]; oi = np.argsort(sub, 0); R = np.empty_like(sub); ar = np.arange(1, m + 1)
    for j in range(sub.shape[1]): R[oi[:, j], j] = ar
    npos = Ys.sum(1); a = (Ys @ R - (npos * (npos + 1) / 2)[:, None]) / (npos[:, None] * (m - npos)[:, None] + 1e-9)
    return np.maximum(a, 1 - a)

# baseline best per term (train-select dim, test-score) for raw + random
def baseline_best(A):
    act = np.where((np.abs(A) > 0).any(0))[0]; Aa = A[:, act]
    atr, ate = aucs(Aa, tr, Y[:, tr]), aucs(Aa, te, Y[:, te])
    bi = np.argmax(atr, 1); return ate[np.arange(len(terms)), bi]   # (T,) held-out best
raw_best, rnd_best = baseline_best(Araw), baseline_best(Arnd)

# per SAE feature: best term on train, test AUROC, margin, coherence
act = np.where((Asae > 0).any(0))[0]; Aa = Asae[:, act]
atr, ate = aucs(Aa, tr, Y[:, tr]), aucs(Aa, te, Y[:, te])
bt = np.argmax(atr, 0)  # best term per feature (train)
# caption metadata
meta = pq.read_table(f"{pub}/feature_metadata.parquet").to_pandas().set_index("feature_id")
rows = []
for j, f in enumerate(act):
    t = bt[j]
    if not validT[t]:
        continue
    held = ate[t, j]
    # coherence: top-20 proteins by this feature's activation, fraction carrying term t
    top = np.argsort(-Asae[:, f])[:20]; coh = float(Y[t, top].mean())
    margin = held - max(raw_best[t], rnd_best[t])
    cap = meta.loc[f].xmodal_caption if f in meta.index and "xmodal_caption" in meta.columns else ""
    rows.append((int(f), obo.get(terms[t], terms[t]), round(float(held), 3),
                 round(float(raw_best[t]), 3), round(float(rnd_best[t]), 3), round(float(margin), 3),
                 round(coh, 2), cap if isinstance(cap, str) else ""))
import csv
rows.sort(key=lambda r: -r[2])
out = Path(f"{Path(__file__).resolve().parents[1]}/analysis/validated_features_l{layer}_{BAND}.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["feature", "go_term", "held_auc", "raw_best", "rand_best", "margin", "coherence", "reasoning_caption"])
    w.writerows(rows)
passed = [r for r in rows if r[2] >= 0.70 and r[5] > 0 and r[6] >= 0.5]
print(f"wrote {out}: {len(rows)} features scored")
print(f"VALIDATED (held_auc>=0.70 AND beats raw+random AND coherence>=0.5): {len(passed)}")
for r in passed[:20]:
    print(f"  F{r[0]:<6} {r[1][:26]:26} auc={r[2]} (raw {r[3]}/rand {r[4]}, margin {r[5]:+}) coh={r[6]} | {r[7][:40]}")
