"""Jared nb01 SAE health: reconstruction FVU (variance explained) + dead/dense fraction, on a store sample."""
import glob,sys,numpy as np,torch,pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0,"src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
store="/data/savithas/phase3_subset_8k/L30_subset8k"; STRIDE=4
ck=torch.load("/data/savithas/phase3_full/sae-l30-exp16-balanced/checkpoint_final.pt",map_location="cpu")
sae=TopKSAE(**ck["model_config"]).to("cuda").eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()},strict=False)
num=0.0; den=0.0; fires=np.zeros(sae.hidden_dim); ntok=0
order=sorted(glob.glob(f"{store}/layer30/shard_*.parquet"),key=lambda q:int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si,sp in enumerate(order):
        if si%STRIDE: continue
        X=shard_table_to_array(pq.read_table(sp))
        for s in range(0,X.shape[0],8192):
            xb=torch.from_numpy(np.ascontiguousarray(X[s:s+8192])).to("cuda").float()
            xn,info=sae._normalize(xb); z=sae.encode(xb); rec=sae.decode(z)   # rec in NORMALIZED space
            num+=((xn-rec)**2).sum().item(); den+=((xn-xn.mean(0))**2).sum().item()
            fires+=(z>0).sum(0).cpu().numpy(); ntok+=xb.shape[0]
        if si%50==0: print(f"  shard {si} ntok={ntok}",flush=True)
fvu=num/den; dead=(fires==0).mean(); dense=(fires/ntok>0.1).mean()
print(f"\n[health] FVU={fvu:.3f} (var explained={1-fvu:.1%}); dead={dead:.1%}; dense(>10%tok)={dense:.1%}; ntok={ntok:,}")
import json; json.dump({"fvu":round(fvu,3),"var_explained":round(1-fvu,3),"dead_frac":round(float(dead),3),"dense_frac":round(float(dense),3)},open("/data/savithas/phase3_full/sae_health_l30.json","w"),indent=2)
