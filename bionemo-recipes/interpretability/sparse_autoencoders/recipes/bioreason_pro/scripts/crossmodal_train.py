# Memory-bounded SAE-V cross-modal (omega vs baseline) on a TRAIN sample, with role split:
# text band = REASONING (response) tokens only — excludes the fixed boilerplate that produced the
# retracted +0.59 artifact. Bands: protein, go, reasoning. (arXiv:2502.17514 Eq.7 + our baseline.)
import argparse, glob, json
from pathlib import Path
from itertools import combinations
import numpy as np, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
p=argparse.ArgumentParser()
p.add_argument("--sae",required=True); p.add_argument("--store",required=True); p.add_argument("--layer",type=int,required=True)
p.add_argument("--max-shards",type=int,default=15); p.add_argument("--top-k",type=int,default=16); p.add_argument("--delta",type=float,default=1.0)
p.add_argument("--device",default="cuda"); p.add_argument("--out-json",required=True)
a=p.parse_args(); dev=a.device; K=a.top_k
tl=pq.read_table(f"{a.store}/token_labels_with_role.parquet")
band=np.array(tl.column("position_type").to_pylist(),dtype=object); role=np.array(tl.column("role").to_pylist(),dtype=object)
# group: 0 protein, 1 go, 2 reasoning(text&response); drop prompt-text/boilerplate (-1)
grp=np.full(len(band),-1,dtype=np.int64); grp[band=="protein"]=0; grp[band=="go"]=1; grp[(band=="text")&(role=="response")]=2
GN=["protein","go","reasoning"]
shards=sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"),key=lambda q:int(Path(q).stem.split("_")[1]))[:a.max_shards]
Xs=[]; gs=[]; row0=0
for sp in shards:
    X=shard_table_to_array(pq.read_table(sp)); n=X.shape[0]
    Xs.append(X); gs.append(grp[row0:row0+n]); row0+=n
X=np.concatenate(Xs); g=np.concatenate(gs); del Xs,gs
keep=g>=0; X=X[keep]; g=g[keep]; N=X.shape[0]
print(f"[xmodal-train] {N:,} tokens after role filter (protein/go/reasoning), {len(shards)} shards")
ck=torch.load(a.sae,map_location="cpu"); sae=TopKSAE(**ck["model_config"]).to(dev).eval(); sae.load_state_dict(ck["model_state_dict"]); H=sae.hidden_dim
gt=torch.from_numpy(g).to(dev)
# encode all (batched), keep codes on CPU sparse? we need top-K per (feature,band). Track running top-K.
NEG=-1e30
topv={b:torch.full((H,K),NEG,device=dev) for b in range(3)}; topi={b:torch.full((H,K),-1,dtype=torch.long,device=dev) for b in range(3)}
Znorm=torch.from_numpy(X).to(dev); Znorm=Znorm/ (Znorm.norm(dim=1,keepdim=True)+1e-8)  # for cosine
with torch.no_grad():
  for s in range(0,N,8192):
    e=min(N,s+8192); c=sae.encode(torch.from_numpy(X[s:e]).to(dev)); gg=gt[s:e]; gidx=torch.arange(s,e,device=dev)
    for b in range(3):
      m=gg==b
      if not m.any(): continue
      cb=c[m].T; ib=gidx[m].unsqueeze(0).expand(H,-1)  # (H, nb)
      allv=torch.cat([topv[b],cb],1); alli=torch.cat([topi[b],ib],1)
      topv[b],o=torch.topk(allv,K,dim=1); topi[b]=torch.gather(alli,1,o)
fires={b:(topv[b][:,-1]>a.delta) for b in range(3)}  # has >=K activations > delta in band b
rng=np.random.default_rng(0)
def omega(feats,ba,bb):
    out={}
    for f in feats:
        ia=topi[ba][f]; ib=topi[bb][f]; cos=(Znorm[ia]*Znorm[ib]).sum(1); out[int(f)]=float(cos.mean())
    return out
def baseline(ba,bb,n=20000):
    ta=torch.where(gt==ba)[0]; tb=torch.where(gt==bb)[0]
    if len(ta)==0 or len(tb)==0: return None
    ia=ta[torch.from_numpy(rng.integers(0,len(ta),n)).to(dev)]; ib=tb[torch.from_numpy(rng.integers(0,len(tb),n)).to(dev)]
    return float((Znorm[ia]*Znorm[ib]).sum(1).mean())
rep={"store":a.store,"n_tokens":int(N),"bands":GN,"K":K,"note":"text=reasoning only (boilerplate excluded)","pairs":{}}
for ba,bb in combinations(range(3),2):
    both=torch.where(fires[ba]&fires[bb])[0]
    name=f"{GN[ba]}-{GN[bb]}"
    if len(both)==0: rep["pairs"][name]={"n_multimodal":0}; print(f"{name}: 0 multimodal feats"); continue
    w=omega(both,ba,bb); base=baseline(ba,bb); vals=np.array(list(w.values()))
    rep["pairs"][name]={"n_multimodal":int(len(both)),"mean_omega":round(float(vals.mean()),3),
      "baseline_cos":round(base,3),"delta_over_baseline":round(float(vals.mean())-base,3),
      "n_omega>baseline":int((vals>base).sum())}
    print(f"{name}: {len(both)} multimodal feats | mean_omega={vals.mean():.3f} baseline={base:.3f} delta={vals.mean()-base:+.3f}")
Path(a.out_json).write_text(json.dumps(rep,indent=2))
