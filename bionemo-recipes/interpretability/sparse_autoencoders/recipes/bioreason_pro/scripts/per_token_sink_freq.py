#!/usr/bin/env python
"""Canonical per-TOKEN feature firing frequency (the proper way to identify sink features).
For each SAE feature: fraction of RESIDUE tokens on which it is active. Samples shards (stride) for speed.
TopK k=128/40960 => mean freq ~0.31%; sinks fire far above that. Compares to the per-protein density proxy.
"""
import glob, sys, numpy as np, torch, pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
store = "/data/savithas/phase3_subset_8k/L30_subset8k"; layer = 30; STRIDE = 8
tl = pq.read_table(f"{store}/token_labels.parquet")
prot = np.array(tl.column("position_type").to_pylist(), dtype=object) == "protein"
ck = torch.load("/data/savithas/phase3_full/sae-l30-exp16-balanced/checkpoint_final.pt", map_location="cpu")
sae = TopKSAE(**ck["model_config"]).to("cuda").eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k,v in ck["model_state_dict"].items()}, strict=False)
H = sae.hidden_dim
fires = np.zeros(H, np.int64); ntok = 0; row0 = 0
order = sorted(glob.glob(f"{store}/layer{layer}/shard_*.parquet"), key=lambda q:int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si, sp in enumerate(order):
        n = pq.read_metadata(sp).num_rows
        if si % STRIDE == 0:
            X = shard_table_to_array(pq.read_table(sp)); m = prot[row0:row0+n]
            if m.any():
                z = sae.encode(torch.from_numpy(np.ascontiguousarray(X[m])).to("cuda"))
                fires += (z > 0).sum(0).cpu().numpy(); ntok += int(m.sum())
            print(f"  shard {si}/{len(order)} ntok={ntok}", flush=True)
        row0 += n
freq = fires / max(ntok, 1)
np.save("/data/savithas/phase3_full/per_token_freq_l30.npy", freq)
print(f"\n[per-token] sampled {ntok:,} residue tokens; mean feat freq={freq.mean()*100:.3f}% (expect ~{128/H*100:.2f}%)")
for thr in [0.5, 0.25, 0.1, 0.05]:
    print(f"  features firing on >{thr*100:.0f}% of tokens (SINKS): {(freq>thr).sum()}")
dens = np.load("/data/savithas/phase3_full/sink_density_l30.npy")
print(f"\ntop-10 per-TOKEN sinks: {[int(i) for i in np.argsort(-freq)[:10]]}")
print(f"  their per-token freq: {[round(float(freq[i]),3) for i in np.argsort(-freq)[:10]]}")
print(f"  their per-protein density: {[round(float(dens[i]),2) for i in np.argsort(-freq)[:10]]}")
# how many of my 381 per-protein 'sinks' (dens>0.8) are real per-token sinks (freq>0.1)?
proxy = dens > 0.8; real = freq > 0.1
print(f"\nper-protein proxy (dens>0.8): {proxy.sum()} features; of those, real per-token sinks (freq>0.1): {(proxy&real).sum()}")
print(f"once-per-protein (proxy but freq<0.05): {(proxy&(freq<0.05)).sum()}  <- flagged by proxy but NOT real sinks")
