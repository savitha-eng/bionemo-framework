"""WINNER'S-CURSE correction for feature_biology_table (Polina's best_single_train_test discipline).

The headline feature->biology AUROCs are optimistically biased: the term is Fisher-SELECTED on all proteins
and the AUROC is scored on the SAME proteins. Honest version: split proteins 50/50, SELECT each feature's
best term on TRAIN (Fisher on train top-k), then report that (feature,term)'s AUROC on held-out TEST. If the
AUROC holds on test the feature is robust; if it collapses it was overfit to the selection.

Outputs feature_biology_traintest.json: per feature {term, auroc_train, auroc_test, drop}. Reuses the cached
pooled_protein matrix (fast, no store streaming)."""
import numpy as np, json, pyarrow.parquet as pq, sys
from scipy.stats import hypergeom, rankdata
from collections import Counter
sys.path.insert(0, "src"); import bioreason_pro_sae.data as brp

z = np.load("/data/savithas/phase3_full/pooled_protein_l30_00e1d8.npz"); SAE = z["SAE"]; keep = z["keep"]
freq = np.load("/data/savithas/phase3_full/per_token_freq_l30.npy")
pr = pq.read_table("/data/savithas/phase3_subset_8k/L30_subset8k/proteins.parquet")
pids = [str(x) for x in pr.column("protein_id").to_pylist()]
tr, _, _ = brp.load_reasoning_splits(max_length_protein=2000)
go = {str(p): set(g if g else []) for p, g in zip(tr["protein_id"], tr["go_ids"])}
ipr = {str(p): set(i if i else []) for p, i in zip(tr["protein_id"], tr["interpro_ids"])}
ipr_name = {}
for p, forms in zip(tr["protein_id"], tr["interpro_formatted"]):
    for line in str(forms).split("\n"):
        if line.startswith("- IPR"):
            k = line[2:].split(":")[0].strip()
            nm = line.split(":", 1)[1].split("(")[0].strip() if ":" in line else k; ipr_name[k] = nm

idx = np.where(keep)[0]; P = len(idx); pset = [pids[i] for i in idx]
X = SAE[idx]
# 50/50 protein split
rng = np.random.default_rng(0); perm = rng.permutation(P); half = P // 2
tr_i, te_i = perm[:half], perm[half:]
terms = {}
for src, d, nm in [("GO", go, {}), ("IPR", ipr, ipr_name)]:
    c = Counter(t for p in pset for t in d.get(p, ()))
    for t, n in c.items():
        if 20 <= n <= P * 0.5:
            terms[(src, t)] = (np.array([t in d.get(p, ()) for p in pset]), nm.get(t, t))
live = [f for f in range(X.shape[1]) if (X[:, f] > 0).sum() >= 10 and freq[f] <= 0.10]
print(f"[traintest] {P} proteins ({half} train / {P-half} test), {len(terms)} terms, {len(live)} live features")

def auroc(col, mem, rows):
    m = mem[rows]; c = col[rows]; npos = int(m.sum()); nneg = len(m) - npos
    if npos < 3 or nneg < 3: return None
    r = rankdata(c); a = (r[m].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return max(a, 1 - a)

# SELECT best term on TRAIN top-10%, then AUROC on train vs test
order_tr = np.argsort(-X[tr_i], axis=0)
rows = []
for f in live:
    topk = set(tr_i[order_tr[:max(4, int(half * 0.1)), f]].tolist()); best = 1.0; bt = None
    for (src, t), (mem, name) in terms.items():
        mi = set(np.where(mem)[0].tolist()); k = len(topk & mi)
        if k < 4: continue
        pval = hypergeom.sf(k - 1, half, len(mi & set(tr_i.tolist())), len(topk))
        if pval < best: best = pval; bt = (src, t, name)
    if not bt or best * len(terms) >= 0.01: continue
    mem = terms[(bt[0], bt[1])][0]
    a_tr = auroc(X[:, f], mem, tr_i); a_te = auroc(X[:, f], mem, te_i)
    if a_tr is None or a_te is None: continue
    rows.append({"feature": int(f), "term": f"{bt[0]}:{bt[1]}", "name": bt[2],
                 "auroc_train": round(a_tr, 3), "auroc_test": round(a_te, 3), "drop": round(a_tr - a_te, 3)})

rows.sort(key=lambda r: -r["auroc_test"])
drops = [r["drop"] for r in rows]
struct = [r for r in rows if r["term"].startswith("IPR")]
func = [r for r in rows if r["term"].startswith("GO")]
print(f"\n[traintest] {len(rows)} features survive train-selection (FDR<0.01 on train)")
print(f"  median train->test drop: {np.median(drops):+.3f}  (small => robust, not winner's-curse)")
print(f"  InterPro (structural): {len(struct)}, mean test AUROC {np.mean([r['auroc_test'] for r in struct]):.3f}")
print(f"  GO (functional):       {len(func)}, mean test AUROC {np.mean([r['auroc_test'] for r in func]):.3f}")
print(f"  features with drop>0.05 (overfit to selection): {sum(1 for d in drops if d > 0.05)}")
print(f"\n{'feat':>7} {'train':>6} {'test':>6} {'drop':>6}  term")
for r in rows[:15]:
    print(f"  F{r['feature']:<6}{r['auroc_train']:>6}{r['auroc_test']:>6}{r['drop']:>6}  {r['name'][:40]}")
json.dump(rows, open("/data/savithas/phase3_full/feature_biology_traintest.json", "w"), indent=2)
print(f"[wrote] feature_biology_traintest.json ({len(rows)} features)")
