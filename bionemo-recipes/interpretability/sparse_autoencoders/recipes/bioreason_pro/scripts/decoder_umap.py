import numpy as np, torch, json, sys
sys.path.insert(0,"src"); from sae.architectures import TopKSAE
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ck=torch.load("/data/savithas/phase3_full/sae-l30-exp16-balanced/checkpoint_final.pt",map_location="cpu")
sae=TopKSAE(**ck["model_config"]); W=sae.decoder.weight.detach().numpy().T  # [40960, 2560]
freq=np.load("/data/savithas/phase3_full/per_token_freq_l30.npy")
live=np.where((freq>0.0005)&(freq<0.10))[0]  # live non-sink
Wl=W[live]/ (np.linalg.norm(W[live],axis=1,keepdims=True)+1e-8)
try:
    import umap, hdbscan
    emb=umap.UMAP(n_neighbors=15,min_dist=0.1,metric="cosine",random_state=0).fit_transform(Wl)
    cl=hdbscan.HDBSCAN(min_cluster_size=20).fit_predict(emb)
    nclust=len(set(cl))-(1 if -1 in cl else 0)
    fig,ax=plt.subplots(figsize=(8,7)); ax.scatter(emb[:,0],emb[:,1],s=2,c=cl,cmap="tab20",alpha=0.4)
    ax.set_title(f"SAE decoder-weight UMAP + HDBSCAN: {nclust} feature families ({len(live)} live features)")
    plt.tight_layout(); plt.savefig("results/charts/decoder_umap_families.png",dpi=120)
    from collections import Counter; sizes=Counter(cl); sizes.pop(-1,None)
    print(f"[umap] {nclust} feature families; largest: {sizes.most_common(8)}")
    json.dump({"n_families":nclust,"n_live":len(live),"largest_sizes":sizes.most_common(15)},open("/data/savithas/phase3_full/decoder_umap.json","w"),indent=2)
except ImportError as e:
    print("umap/hdbscan not installed:",e)
