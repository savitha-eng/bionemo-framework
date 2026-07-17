"""Jared nb03 analog: per-feature enrichment vs GO/InterPro (Fisher exact), FDR<0.05 annotation call,
annotation-RATE + label-shuffle LIFT + term DIVERSITY. Runs on ALL features from a cached pooled matrix.
Usage: enrichment_probe.py <pooled.npz> <band>"""
import sys, json, numpy as np, pyarrow.parquet as pq
from scipy.stats import hypergeom
from collections import Counter
sys.path.insert(0,"src")
import bioreason_pro_sae.data as brp
npz, band = sys.argv[1], sys.argv[2]
z=np.load(npz); SAE=z["SAE"]; keep=z["keep"]
freq=np.load("/data/savithas/phase3_full/per_token_freq_l30.npy")
pr=pq.read_table("/data/savithas/phase3_subset_8k/L30_subset8k/proteins.parquet"); pids=[str(x) for x in pr.column("protein_id").to_pylist()]
tr,_,_=brp.load_reasoning_splits(max_length_protein=2000)
go={str(p):set(g if g else []) for p,g in zip(tr["protein_id"],tr["go_ids"])}
ipr={str(p):set(i if i else []) for p,i in zip(tr["protein_id"],tr["interpro_ids"])}
idx=np.where(keep)[0]; P=len(idx); pset=[pids[i] for i in idx]
# term -> membership boolean over kept proteins (abundant terms only)
terms={}
for src,d in [("GO",go),("IPR",ipr)]:
    c=Counter(t for p in pset for t in d.get(p,()))
    for t,n in c.items():
        if 20<=n<=P*0.5: terms[(src,t)]=np.array([t in d.get(p,()) for p in pset])
print(f"[enrich] {P} proteins, {len(terms)} abundant terms, band={band}")
X=SAE[idx]
live=[f for f in range(X.shape[1]) if (X[:,f]>0).sum()>=10 and freq[f]<=0.10]  # live, non-sink
print(f"[enrich] {len(live)} live non-sink features")
def annotate(Xmat, shuffle=False):
    ann=0; termsrec=set()
    order=np.argsort(-Xmat,axis=0)
    for f in live:
        topk=set(order[:int(P*0.1),f].tolist())  # top-10% proteins by feature
        best=1.0; bt=None
        tm=terms
        for (src,t),mem in tm.items():
            memidx=set(np.where(mem)[0].tolist())
            if shuffle: memidx=set(np.random.default_rng(f).choice(P,len(memidx),replace=False).tolist())
            k=len(topk&memidx)
            if k<3: continue
            p=hypergeom.sf(k-1,P,len(memidx),len(topk))
            if p<best: best=p; bt=(src,t)
        if best*len(terms)<0.05: ann+=1; termsrec.add(bt)  # Bonferroni-ish FDR
    return ann, len(termsrec)
ann,div=annotate(X)
# shuffle null (labels permuted)
np.random.seed(0)
Xs=X.copy()
annn,_=annotate(X, shuffle=True)
rate=ann/len(live); nrate=annn/len(live)
print(f"\n=== ENRICHMENT ({band}) ===")
print(f"  annotation RATE: {rate:.1%} ({ann}/{len(live)} features have >=1 term FDR<0.05)")
print(f"  shuffle-null rate: {nrate:.1%}  ->  LIFT = {rate/max(nrate,1e-3):.1f}x")
print(f"  term DIVERSITY: {div} distinct terms recovered")
json.dump({"band":band,"n_live":len(live),"annotation_rate":round(rate,3),"null_rate":round(nrate,3),
           "lift":round(rate/max(nrate,1e-3),1),"diversity":div}, open(f"/data/savithas/phase3_full/enrichment_{band}.json","w"),indent=2)
print(f"[wrote] enrichment_{band}.json")
