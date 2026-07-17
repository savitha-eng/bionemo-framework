"""Jared nb01 logit-lens: project SAE feature decoder directions through the LLM output head to see which
TEXT tokens each feature promotes = free, data-independent naming prior. Names a VARIETY of features
(bio detectors + steerable clusters + top synthesis features + a frequency-stratified sample)."""
import sys, json, numpy as np, torch
sys.path.insert(0,"src")
from bioreason_pro_sae.model_loader import _install_unsloth_stub, load_bioreason_pro_sft
_install_unsloth_stub()
from sae.architectures import TopKSAE
DEFAULT_CKPT=("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
              "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")
dev="cuda"
ck=torch.load("/data/savithas/phase3_full/sae-l30-exp16-balanced/checkpoint_final.pt",map_location="cpu")
sae=TopKSAE(**ck["model_config"]).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()},strict=False)
Wdec=sae.decoder.weight.detach()   # [2560, 40960] cols = feature directions
model=load_bioreason_pro_sft(ckpt_dir=DEFAULT_CKPT, bioreason_pro_root="/data/savithas/bioreason-pro", device=dev).eval()
tok=model.text_tokenizer
lm=model.text_model.lm_head if hasattr(model.text_model,"lm_head") else model.text_model.get_output_embeddings()
norm=model.text_model.model.norm  # final RMSNorm
Wlm=lm.weight.detach()             # [vocab, 2560]
# curated variety: bio detectors, steerable clusters, top synthesis feats, + freq-stratified sample
bio=[16026,679,11009,5540,17703,8306,7079]
steer=[16494,4456,22453,31154,39580,21531, 30713,10643,20498,36378,15839,24389, 4306,6569,4939,29338,3606,11608]
syn=json.load(open("/data/savithas/phase3_full/synthesis_word_l30.json"))["features"]
syn_top=[int(f) for f,v in sorted(syn.items(),key=lambda x:-x[1].get("novelty_word",0) if x[1].get("interpretable") else -1)[:60]]
freq=np.load("/data/savithas/phase3_full/per_token_freq_l30.npy")
rng=np.random.default_rng(0); samp=rng.choice(np.where((freq>0.001)&(freq<0.05))[0],120,replace=False).tolist()
feats=sorted(set(bio+steer+syn_top+samp))
print(f"[logit-lens] naming {len(feats)} features")
res={}
with torch.no_grad():
    for f in feats:
        d=Wdec[:,f]
        dn=norm(d.unsqueeze(0).float()).to(Wlm.dtype)      # apply final norm
        logits=(dn@Wlm.T).squeeze(0)
        top=torch.topk(logits,12).indices.tolist()
        toks=[tok.decode([t]).strip() for t in top]
        res[int(f)]={"promotes":[t for t in toks if t]}
        print(f"  F{f}: {res[int(f)]['promotes']}")
json.dump(res,open("/data/savithas/phase3_full/logit_lens_l30.json","w"),indent=2)
print("[wrote] logit_lens_l30.json")
