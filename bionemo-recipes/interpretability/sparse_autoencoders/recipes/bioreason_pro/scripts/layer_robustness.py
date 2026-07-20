"""LAYER ROBUSTNESS CHECK: re-run structure-monosemanticity + cross-modal pairing at a MIDDLE layer to test
whether the L30 conclusions (structure localized? cross-modal alignment? fusion?) are layer-robust or a
late-layer artifact. Fusion, if it exists, would appear where the modalities mix (middle layers), not at L30.

Self-contained from the pooled per-protein SAE matrices (probe_v2 cache). Sink filter = per-protein firing
rate (no per_token_freq needed). Usage: layer_robustness.py <pooled_protein.npz> <pooled_reasoning.npz> <store> <layer>"""
import sys, json, numpy as np, pyarrow.parquet as pq
from scipy.stats import hypergeom, rankdata
from collections import Counter
sys.path.insert(0, "src"); import bioreason_pro_sae.data as brp

pp, pr_, store, layer = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
SP = np.load(pp)["SAE"]; keepP = np.load(pp)["keep"]
SR = np.load(pr_)["SAE"]; keepR = np.load(pr_)["keep"]
keep = keepP & keepR
SP, SR = SP[keep], SR[keep]
pr = pq.read_table(f"{store}/proteins.parquet"); pids = [str(x) for x in pr.column("protein_id").to_pylist()]
pids = [pids[i] for i in np.where(keep)[0]]
tr, _, _ = brp.load_reasoning_splits(max_length_protein=2000)
ipr = {str(p): set(i if i else []) for p, i in zip(tr["protein_id"], tr["interpro_ids"])}
iprname = {}
for p, forms in zip(tr["protein_id"], tr["interpro_formatted"]):
    for line in str(forms).split("\n"):
        if line.startswith("- IPR"):
            k = line[2:].split(":")[0].strip(); iprname[k] = line.split(":", 1)[1].split("(")[0].strip() if ":" in line else k
P = len(pids)
# sink filter: exclude features firing in >40% of proteins (dense/near-generic), proxy for per-token freq
prot_fire = (SP > 0).mean(0)
clean = prot_fire <= 0.40

# ---- structure monosemanticity: best-feature-per-InterPro-domain AUROC (protein band) ----
# CARRIES L30 FIXES: (P3) winner's-curse train/test -> SELECT best feature on TRAIN, REPORT test AUROC;
# (sink) exclude near-dense features. NOTE: sink filter here is per-PROTEIN firing rate<=0.4 (proxy for the
# L30 per-token freq<=0.10; the true localization metric is domain_f1 at L22, run separately).
cnt = Counter(t for p in pids for t in ipr.get(p, ()))
domains = [(d, iprname.get(d, d)) for d, n in cnt.most_common(40) if 20 <= n <= P * 0.5]
rng2 = np.random.default_rng(1); pm = rng2.permutation(P); half = P // 2
tr_i, te_i = pm[:half], pm[half:]
def auroc_sub(col, mem, rows):
    m = mem[rows]; npos = int(m.sum()); nneg = len(m) - npos
    if npos < 5 or nneg < 5: return None
    r = rankdata(col[rows]); a = (r[m].sum() - npos * (npos + 1) / 2) / (npos * nneg); return max(a, 1 - a)
struct = []
for d, nm in domains:
    mem = np.array([d in ipr.get(p, ()) for p in pids])
    tr_aucs = np.array([(auroc_sub(SP[:, f], mem, tr_i) or 0) if clean[f] else 0 for f in range(SP.shape[1])])
    bf = int(tr_aucs.argmax())                                    # SELECT on train
    te = auroc_sub(SP[:, bf], mem, te_i)                          # REPORT on held-out test
    if te is None: continue
    struct.append({"domain": d, "name": nm, "best_feature": bf,
                   "auroc_train": round(float(tr_aucs[bf]), 3), "auroc_test": round(float(te), 3)})
struct.sort(key=lambda z: -z["auroc_test"])
print(f"[L{layer}] STRUCTURE monosemanticity (best feature per domain, TRAIN-selected / TEST-reported, top 10):")
for s in struct[:10]: print(f"    {s['name'][:32]:32} F{s['best_feature']:<6} test={s['auroc_test']} (train {s['auroc_train']})")
print(f"    mean TEST AUROC over {len(struct)} domains: {np.mean([s['auroc_test'] for s in struct]):.3f}  "
      f"(L30 winner's-curse-corrected structural test AUROC ~0.85 IPR)")

# ---- cross-modal pairing: top structural bio features -> reasoning partner (+ perm null) ----
SRc = SR - SR.mean(0); SRn = np.linalg.norm(SRc, axis=0) + 1e-9
rng = np.random.default_rng(0); perm = rng.permutation(P); SRc_null = SRc[perm]
biofeats = [s["best_feature"] for s in struct[:15]]
pairs = []
for s in struct[:15]:
    bf = s["best_feature"]; sp = SP[:, bf]
    if (sp > 0).sum() < 20: continue
    spc = sp - sp.mean(); den = np.linalg.norm(spc) * SRn
    r = (SRc * spc[:, None]).sum(0) / den; r[bf] = -2
    null_max = float(((SRc_null * spc[:, None]).sum(0) / den).max())
    k = int(np.argsort(-r)[0])
    pairs.append({"bio": bf, "domain": s["name"], "reason": k, "r": round(float(r[k]), 3), "null_max_r": round(null_max, 3)})
print(f"\n[L{layer}] CROSS-MODAL pairing (bio structural feature -> reasoning partner):")
for p in pairs: print(f"    {p['domain'][:30]:30} F{p['bio']:<6} -> F{p['reason']:<6} r={p['r']:+.2f} (null {p['null_max_r']:+.2f})")
n04 = sum(1 for p in pairs if p["r"] >= 0.4)
print(f"    {n04}/{len(pairs)} pairs with r>=0.4 (L30 was 20/20); mean r={np.mean([p['r'] for p in pairs]):.2f}")
json.dump({"layer": layer, "structure": struct, "crossmodal_pairs": pairs,
           "mean_best_feature_auroc_test": float(np.mean([s["auroc_test"] for s in struct])),
           "n_pairs_ge_0.4": n04}, open(f"/data/savithas/phase3_full/layer_robustness_l{layer}.json", "w"), indent=2)
print(f"[wrote] layer_robustness_l{layer}.json")
