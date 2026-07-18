"""Formalize the CLEAN synthesis-leaning reasoning features from the cached echo-synthesis activations
(no store re-streaming). Applies the frequency filter (drops near-dense always-on features that contaminated
the AUROC ranking), then labels the survivors with their biology from the dashboard feature_examples.

Output echo_synthesis_clean_l30.json: sparse reasoning features ranked by how much MORE they fire on synthesis
(novel elaboration) than echo (restatement). AUROC is the RANK ('more synthesis than others'), NOT a claim any
feature 'is synthesis'."""
import json, numpy as np, pyarrow.parquet as pq
from scipy.stats import rankdata

DFULL = "/data/savithas/phase3_full"
PUB = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/"
       "bioreason_pro/multimodal_dashboard/public/l30_balanced/feature_examples.parquet")
FREQ_MAX = 0.02

d = np.load(f"{DFULL}/echo_synthesis_l30_acts.npz"); Zg = d["Zg"].astype(np.float32); yg = d["yg"]
freq = np.load(f"{DFULL}/per_token_freq_l30.npy")
npos, nneg = int(yg.sum()), int((1 - yg).sum())
r = rankdata(Zg, axis=0); auc = (r[yg == 1].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
active = (Zg > 0).sum(0)
clean = (freq <= FREQ_MAX) & (active >= 30)
print(f"[formalize] {int(clean.sum())} clean features (freq<={FREQ_MAX:.0%}); "
      f"{int((freq > FREQ_MAX).sum())} near-dense excluded (e.g. F39979 freq={freq[39979]*100:.0f}%)")

df = pq.read_table(PUB).to_pandas(); df = df[df.band == "reasoning"]
def phrases(fid, n=4):
    sub = df[df.feature_id == fid].nlargest(n, "max_activation"); out = []
    for _, x in sub.iterrows():
        toks = str(x["sequence"]).split(); act = np.asarray(x["activations"], float)
        m = min(len(toks), len(act))
        if m == 0: continue
        pk = act[:m].argmax()
        out.append(" ".join(toks[max(0, pk - 4):pk + 4]))
    return out

auc_masked = np.where(clean, auc, 0.5)
synth = np.argsort(-auc_masked)[:15]; echo = np.argsort(auc_masked)[:15]
def rec(f, want):
    return {"feature": int(f), "synth_vs_echo_auroc": round(float(auc[f]), 3),
            "freq_pct": round(float(freq[f] * 100), 3), "top_phrases": phrases(f)}
out = {"null_auroc_max": 0.506, "freq_max_pct": FREQ_MAX * 100,
       "framing": "sparse reasoning features ranked by synthesis-vs-echo lean; AUROC is the RANK, not a "
                  "claim any feature 'is synthesis'. Each carries a specific biology it ELABORATES beyond the prompt.",
       "synthesis_features": [rec(f, 1) for f in synth],
       "echo_features": [rec(f, 0) for f in echo]}
json.dump(out, open(f"{DFULL}/echo_synthesis_clean_l30.json", "w"), indent=2)
print(f"[wrote] echo_synthesis_clean_l30.json")
print(f"\n{'feat':>7} {'AUROC':>6} {'freq':>6}  biology (top firing phrase)")
for f in synth:
    ph = phrases(f, 1)
    print(f"  F{f:<6} {auc[f]:>6.3f} {freq[f]*100:>5.2f}%  {ph[0][:60] if ph else '(no examples)'}")
