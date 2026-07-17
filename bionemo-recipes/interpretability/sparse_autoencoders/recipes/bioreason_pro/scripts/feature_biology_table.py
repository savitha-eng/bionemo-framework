"""DEFINITIVE per-feature biology: for each live non-sink protein-band SAE feature, its best-enriched
GO/InterPro term (Fisher exact, FDR) + per-feature AUROC. Outputs the specific feature->biology map."""
import numpy as np, json, pyarrow.parquet as pq, sys
from scipy.stats import hypergeom, rankdata
from collections import Counter
sys.path.insert(0,"src"); import bioreason_pro_sae.data as brp
z=np.load("/data/savithas/phase3_full/pooled_protein_l30_00e1d8.npz"); SAE=z["SAE"]; keep=z["keep"]
freq=np.load("/data/savithas/phase3_full/per_token_freq_l30.npy")
pr=pq.read_table("/data/savithas/phase3_subset_8k/L30_subset8k/proteins.parquet"); pids=[str(x) for x in pr.column("protein_id").to_pylist()]
tr,_,_=brp.load_reasoning_splits(max_length_protein=2000)
go={str(p):set(g if g else []) for p,g in zip(tr["protein_id"],tr["go_ids"])}
ipr={str(p):set(i if i else []) for p,i in zip(tr["protein_id"],tr["interpro_ids"])}
# term names
ipr_name={}
for p,forms in zip(tr["protein_id"],tr["interpro_formatted"]):
    for line in str(forms).split("\n"):
        if line.startswith("- IPR"):
            k=line[2:].split(":")[0].strip(); nm=line.split(":",1)[1].split("(")[0].strip() if ":" in line else k; ipr_name[k]=nm
idx=np.where(keep)[0]; P=len(idx); pset=[pids[i] for i in idx]
terms={}
for src,d,nm in [("GO",go,{}),("IPR",ipr,ipr_name)]:
    c=Counter(t for p in pset for t in d.get(p,()))
    for t,n in c.items():
        if 20<=n<=P*0.5: terms[(src,t)]=(np.array([t in d.get(p,()) for p in pset]), nm.get(t,t))
X=SAE[idx]; live=[f for f in range(X.shape[1]) if (X[:,f]>0).sum()>=10 and freq[f]<=0.10]
print(f"[bio-table] {P} proteins, {len(terms)} terms, {len(live)} live features")
order=np.argsort(-X,axis=0); rows=[]
for f in live:
    topk=set(order[:int(P*0.1),f].tolist()); best=1.0; bt=None
    for (src,t),(mem,name) in terms.items():
        mi=set(np.where(mem)[0].tolist()); k=len(topk&mi)
        if k<4: continue
        p=hypergeom.sf(k-1,P,len(mi),len(topk))
        if p<best: best=p; bt=(src,t,name)
    if bt and best*len(terms)<0.01:
        # per-feature AUROC for that term
        mem=terms[(bt[0],bt[1])][0]; r=rankdata(X[:,f]); npos=mem.sum()
        auc=(r[mem].sum()-npos*(npos+1)/2)/(npos*(P-npos)); auc=max(auc,1-auc)
        rows.append((f,bt[0],bt[1],bt[2],best*len(terms),round(float(auc),3)))
rows.sort(key=lambda r:r[4])
print(f"\n=== DEFINITIVE SAE BIOLOGY FEATURES (feature -> term, FDR, AUROC), top 25 of {len(rows)} ===")
for f,src,t,nm,fdr,auc in rows[:25]: print(f"  F{f} -> {src}:{t} {nm[:34]}  FDR={fdr:.1e}  AUROC={auc}")
json.dump([{"feature":int(f),"term":f"{src}:{t}","name":nm,"fdr":float(fdr),"auroc":auc} for f,src,t,nm,fdr,auc in rows],
          open("/data/savithas/phase3_full/feature_biology_table.json","w"),indent=2)
print(f"[wrote] feature_biology_table.json ({len(rows)} annotated features)")
