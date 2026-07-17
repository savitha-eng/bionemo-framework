# Load-bearing proxy DONE RIGHT: per-band mean_active + decoder-norm load-fraction, calibrated on OUR anchors.
import glob, sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0,"src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
store="/data/savithas/phase3_subset_8k/L30_subset8k"; layer=30; STRIDE=8
tl=pq.read_table(f"{store}/token_labels.parquet")
band=np.array(tl.column("position_type").to_pylist(),dtype=object)
ck=torch.load("/data/savithas/phase3_full/sae-l30-exp16-balanced/checkpoint_final.pt",map_location="cpu")
sae=TopKSAE(**ck["model_config"]).to("cuda").eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()},strict=False)
H=sae.hidden_dim
# per-band accumulators
acc={b:{"fires":np.zeros(H),"asum":np.zeros(H)} for b in ["protein","text"]}
row0=0
order=sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"),key=lambda q:int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si,sp in enumerate(order):
        n=pq.read_metadata(sp).num_rows
        if si%STRIDE==0:
            X=shard_table_to_array(pq.read_table(sp)); b=band[row0:row0+n]
            for s0 in range(0,n,8192):
                e0=min(n,s0+8192); bb=b[s0:e0]
                z=sae.encode(torch.from_numpy(np.ascontiguousarray(X[s0:e0])).to("cuda")).cpu().numpy()
                for bn in ["protein","text"]:
                    m=bb==bn
                    if m.any():
                        acc[bn]["fires"]+=(z[m]>0).sum(0); acc[bn]["asum"]+=np.where(z[m]>0,z[m],0).sum(0)
        row0+=n
ma={bn:np.divide(acc[bn]["asum"],acc[bn]["fires"],out=np.zeros(H),where=acc[bn]["fires"]>0) for bn in acc}
# decoder norm per feature (for load-fraction, since normalize_input scale differs)
Wdec=sae.decoder.weight.detach().cpu().numpy() if hasattr(sae,'decoder') else sae.W_dec.detach().cpu().numpy().T
dnorm=np.linalg.norm(Wdec,axis=0) if Wdec.shape[0]==H else np.linalg.norm(Wdec,axis=1)
np.savez("/data/savithas/phase3_full/load_bearing_l30.npz",ma_protein=ma["protein"],ma_text=ma["text"],dnorm=dnorm)
def load(f,bn): return ma[bn][f]*dnorm[f]   # load-fraction proxy (mean_active x decoder norm)
print("band-correct mean_active x decoder-norm (LOAD proxy). Higher=more steerable.")
print("\n-- STEERABLE ANCHOR: synapse reasoning cluster (DID steer in prior work) [text band] --")
for f in [16494,4456,22453,31154,39580,21531]: print(f"  F{f}: ma_text={ma['text'][f]:.2f} dnorm={dnorm[f]:.2f} LOAD={load(f,'text'):.2f}")
print("-- kinase REASONING cluster (did NOT steer) [text band] --")
for f in [34724,10154,29958,13063,19563,35647]: print(f"  F{f}: ma_text={ma['text'][f]:.2f} dnorm={dnorm[f]:.2f} LOAD={load(f,'text'):.2f}")
print("-- kinase PROTEIN cluster (did NOT steer) [protein band] --")
for f in [16026,37746,2062,8444,21967,28718]: print(f"  F{f}: ma_prot={ma['protein'][f]:.2f} dnorm={dnorm[f]:.2f} LOAD={load(f,'protein'):.2f}")
al=ma['text'][ma['text']>0]
print(f"\ntext-band mean_active dist: median={np.median(al):.2f} p90={np.percentile(al,90):.2f} p99={np.percentile(al,99):.2f} max={ma['text'].max():.2f}")
