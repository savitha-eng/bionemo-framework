import glob, sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0,"src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
store="/data/savithas/phase3_subset_8k/L30_subset8k"; layer=30; STRIDE=8
tl=pq.read_table(f"{store}/token_labels.parquet")
prot=np.array(tl.column("position_type").to_pylist(),dtype=object)=="protein"
ck=torch.load("/data/savithas/phase3_full/sae-l30-exp16-balanced/checkpoint_final.pt",map_location="cpu")
sae=TopKSAE(**ck["model_config"]).to("cuda").eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()},strict=False)
H=sae.hidden_dim; fires=np.zeros(H); asum=np.zeros(H); row0=0
order=sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"),key=lambda q:int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si,sp in enumerate(order):
        n=pq.read_metadata(sp).num_rows
        if si%STRIDE==0:
            X=shard_table_to_array(pq.read_table(sp)); m=prot[row0:row0+n]
            if m.any():
                z=sae.encode(torch.from_numpy(np.ascontiguousarray(X[m])).to("cuda")).cpu().numpy()
                fires+=(z>0).sum(0); asum+=np.where(z>0,z,0).sum(0)
        row0+=n
mean_active=np.divide(asum,fires,out=np.zeros(H),where=fires>0)
np.save("/data/savithas/phase3_full/mean_active_l30.npy",mean_active)
print(f"mean_active distribution: median={np.median(mean_active[fires>0]):.1f} p90={np.percentile(mean_active[fires>0],90):.1f} max={mean_active.max():.1f}")
print("--- PROTEIN kinase cluster (steering FAILED for single F16026) ---")
for f in [16026,37746,2062,8444,21967,28718,30032,36334,13790,7908]:
    print(f"  F{f}: mean_active={mean_active[f]:.1f}")
print("--- REASONING kinase-mechanism cluster (predicted steerable?) ---")
for f in [34724,10154,29958,13063,19563,35647]:
    print(f"  F{f}: mean_active={mean_active[f]:.1f}")
print("--- known-STEERABLE anchors from prior work (synapse cluster that DID steer) ---")
for f in [16494,4456,22453,31154]:
    print(f"  F{f}: mean_active={mean_active[f]:.1f}")
print("Jared calibration: steerable ~270-290, null ~1.6, threshold mean_active>50")
