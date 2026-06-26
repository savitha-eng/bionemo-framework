# Consolidated descriptive analysis on a store: per-(band,role) coverage + role taxonomy +
# protein-band-ONLY GO-AUC (leakage-free), single streaming encode pass. (cross-modal is separate.)
import argparse, glob, json
from pathlib import Path
from collections import Counter
import numpy as np, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
p=argparse.ArgumentParser()
p.add_argument("--sae",required=True); p.add_argument("--store",required=True); p.add_argument("--layer",type=int,required=True)
p.add_argument("--device",default="cuda"); p.add_argument("--encode-batch",type=int,default=8192); p.add_argument("--out-json",required=True)
a=p.parse_args(); dev=a.device
tl=pq.read_table(f"{a.store}/token_labels_with_role.parquet")
band=np.array(tl.column("position_type").to_pylist(),dtype=object); role=np.array(tl.column("role").to_pylist(),dtype=object)
rpid=np.array(tl.column("protein_id").to_pylist(),dtype=object)
grp=np.full(len(band),-1,dtype=np.int64); G=["protein","go","prompt_text","reasoning_text"]
grp[band=="protein"]=0; grp[band=="go"]=1; grp[(band=="text")&(role=="prompt")]=2; grp[(band=="text")&(role=="response")]=3
gt=torch.from_numpy(grp).to(dev)
pr=pq.read_table(f"{a.store}/proteins.parquet"); pids=[str(x) for x in pr.column("protein_id").to_pylist()]
goids=[json.loads(g) if g else [] for g in pr.column("go_ids").to_pylist()]; pidx={p_:i for i,p_ in enumerate(pids)}; nP=len(pids)
rpi=torch.from_numpy(np.array([pidx.get(p_,-1) for p_ in rpid])).to(dev)
protmask=(gt==0)
ck=torch.load(a.sae,map_location="cpu"); sae=TopKSAE(**ck["model_config"]).to(dev).eval(); sae.load_state_dict(ck["model_state_dict"]); H=sae.hidden_dim
fired=torch.zeros(len(G),H,dtype=torch.bool,device=dev); pmax=torch.zeros(nP,H,device=dev); row0=0
shards=sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"),key=lambda q:int(Path(q).stem.split("_")[1]))
with torch.no_grad():
  for si,sp in enumerate(shards):
    X=shard_table_to_array(pq.read_table(sp)); n=X.shape[0]
    for s in range(0,n,a.encode_batch):
      e=min(n,s+a.encode_batch); c=sae.encode(torch.from_numpy(X[s:e]).to(dev))
      gg=gt[row0+s:row0+e]; act=c>0
      for gi in range(len(G)):
        m=gg==gi
        if m.any(): fired[gi]|=act[m].any(0)
      pm=protmask[row0+s:row0+e]
      if pm.any():
        ii=rpi[row0+s:row0+e][pm].unsqueeze(1).expand(-1,H); pmax.scatter_reduce_(0,ii,c[pm],reduce="amax",include_self=True)
    row0+=n
    if si%20==0: print(f"  shard {si}/{len(shards)}",flush=True)
f=fired.cpu().numpy(); prot,go,pt,rt=f; pmax=pmax.cpu().numpy()
# role taxonomy
cat=np.full(H,"general_text",dtype=object)
cat[(pt)&(~rt)&(~prot)&(~go)]="prompt_annotation"; cat[(rt)&(~pt)&(~prot)&(~go)]="reasoning_only"
cat[prot|go]="bio"; cat[~(prot|go|pt|rt)]="dead"
# protein-band GO-AUC (de-biased internal split)
sets=[set(g) for g in goids]; freq=Counter(t for g in goids for t in g); terms=[]
for t,_ in freq.most_common():
  pv=sum(t in s for s in sets)/nP
  if 0.02<=pv<=0.5: terms.append(t)
  if len(terms)>=80: break
Y=np.stack([np.array([1 if t in s else 0 for s in sets]) for t in terms]).astype(float)
rng=np.random.default_rng(0); perm=rng.permutation(nP); h=nP//2; tr,te=perm[:h],perm[h:]
active=np.where((pmax>0).any(0))[0]; A=pmax[:,active]
def rauc(rows,Ys):
  sub=A[rows]; m=sub.shape[0]; oi=np.argsort(sub,0); R=np.empty_like(sub); ar=np.arange(1,m+1)
  for j in range(sub.shape[1]): R[oi[:,j],j]=ar
  npos=Ys.sum(1); sr=Ys@R; return (sr-(npos*(npos+1)/2)[:,None])/(npos[:,None]*(m-npos)[:,None]+1e-9)
atr=rauc(tr,Y[:,tr]); ate=rauc(te,Y[:,te])
valid=((Y[:,tr].sum(1)>=3)&((1-Y[:,tr]).sum(1)>=3)&(Y[:,te].sum(1)>=3)&((1-Y[:,te]).sum(1)>=3)); atr[~valid]=0.5
best=np.argmax(atr,0); held=ate[best,np.arange(len(active))]
out={"store":a.store,"n_features":H,"n_proteins":nP,
  "coverage":{G[i]:int(f[i].sum()) for i in range(len(G))},
  "role_taxonomy":dict(Counter(cat)),
  "protein_band_AUC":{"active_protein_feats":int(len(active)),
    "n_AUC>0.65":int((held>0.65).sum()),"n_AUC>0.7":int((held>0.7).sum()),"n_AUC>0.8":int((held>0.8).sum()),
    "top":[{"feat":int(active[j]),"auc":round(float(held[j]),3),"term":terms[best[j]]} for j in np.argsort(-held)[:10]]}}
Path(a.out_json).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
