# Per-feature PROTEIN-BAND-ONLY GO-AUC (held-out, de-biased) -> writes go_auc_protein + go_term_protein
# into the dashboard feature_metadata. This is the leakage-free bio metric (vs the all-token go_auc).
import glob, json, sys
from pathlib import Path
from collections import Counter
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
sae_p, store, layer, pub = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
dev="cuda"
tl=pq.read_table(f"{store}/token_labels.parquet")
band=np.array(tl.column("position_type").to_pylist(),dtype=object); rpid=np.array(tl.column("protein_id").to_pylist(),dtype=object)
pr=pq.read_table(f"{store}/proteins.parquet"); pids=[str(x) for x in pr.column("protein_id").to_pylist()]
goids=[json.loads(g) if g else [] for g in pr.column("go_ids").to_pylist()]; pidx={p:i for i,p in enumerate(pids)}; nP=len(pids)
rpi=torch.from_numpy(np.array([pidx.get(p,-1) for p in rpid])).to(dev); pm=torch.from_numpy(band=="protein").to(dev)
ck=torch.load(sae_p,map_location="cpu"); sae=TopKSAE(**ck["model_config"]).to(dev).eval(); sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()}); H=sae.hidden_dim
pmax=torch.zeros(nP,H,device=dev); row0=0
with torch.no_grad():
  for sp in sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"),key=lambda q:int(Path(q).stem.split("_")[1])):
    X=shard_table_to_array(pq.read_table(sp)); n=X.shape[0]
    for s in range(0,n,8192):
      e=min(n,s+8192); c=sae.encode(torch.from_numpy(X[s:e]).to(dev)); m=pm[row0+s:row0+e]
      if m.any(): ii=rpi[row0+s:row0+e][m].unsqueeze(1).expand(-1,H); pmax.scatter_reduce_(0,ii,c[m],reduce="amax",include_self=True)
    row0+=n
pmax=pmax.cpu().numpy()
sets=[set(g) for g in goids]; freq=Counter(t for g in goids for t in g); terms=[]
for t,_ in freq.most_common():
  pv=sum(t in s for s in sets)/nP
  if 0.02<=pv<=0.5: terms.append(t)
  if len(terms)>=80: break
# obo names
obo={}
for line in open("/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo"):
  line=line.strip()
  if line.startswith("id: GO:"): cur=line[4:]
  elif line.startswith("name:"): obo[cur]=line[6:]
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
go_auc_p=np.zeros(H); go_term_p=np.array([""]*H,dtype=object)
for j,f in enumerate(active):
  if held[j]>0.65: go_auc_p[f]=round(float(held[j]),3); go_term_p[f]=obo.get(terms[best[j]],terms[best[j]])
m=pq.read_table(f"{pub}/feature_metadata.parquet"); cols=[c for c in m.column_names if c not in ("go_auc_protein","go_term_protein")]; m=m.select(cols)
ids=m.column("feature_id").to_pylist()
m=m.append_column("go_auc_protein",pa.array([float(go_auc_p[i]) for i in ids],pa.float32()))
m=m.append_column("go_term_protein",pa.array([go_term_p[i] for i in ids],pa.string()))
pq.write_table(m,f"{pub}/feature_metadata.parquet")
print(f"wrote go_auc_protein/go_term_protein. features with protein-band AUC>0.65: {int((go_auc_p>0.65).sum())}")
print("top:",[(int(active[j]),round(float(held[j]),3),obo.get(terms[best[j]],terms[best[j]])) for j in np.argsort(-held)[:6]])
