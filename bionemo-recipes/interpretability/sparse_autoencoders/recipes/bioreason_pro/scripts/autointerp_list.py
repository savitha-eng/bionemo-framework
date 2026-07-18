"""General auto-interp: name an arbitrary feature list from its top dashboard firing examples (a given band).
Used for the CLEAN synthesis-leaning reasoning features (freq-filtered) so each has a validated biology label."""
import json, os, time, sys
import numpy as np, pyarrow.parquet as pq
from openai import OpenAI

PUB = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/"
       "bioreason_pro/multimodal_dashboard/public/l30_balanced/feature_examples.parquet")
feats = [int(x) for x in sys.argv[1].split(",")]
band = sys.argv[2] if len(sys.argv) > 2 else "reasoning"
out_path = sys.argv[3] if len(sys.argv) > 3 else "/data/savithas/phase3_full/autointerp_list_l30.json"
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NIM_API_KEY"])
df = pq.read_table(PUB).to_pandas(); df = df[df.band == band]

def snippets(fid, n=4):
    sub = df[df.feature_id == fid].nlargest(n, "max_activation"); out = []
    for _, x in sub.iterrows():
        toks = str(x["sequence"]).split(); act = np.asarray(x["activations"], float)
        m = min(len(toks), len(act))
        if m == 0: continue
        pk = act[:m].argmax(); lo, hi = max(0, pk - 12), min(m, pk + 13)
        out.append(" ".join(("**" + toks[j] + "**" if act[j] >= 0.6 * act[pk] and act[j] > 0 else toks[j]) for j in range(lo, hi)))
    return out

PROMPT = ("SAE feature of BioReason-Pro (protein GO-function reasoning model), {band} band. Snippets where it "
          "fires most (**bold**=peak). Name the single biological concept it detects.\n{ev}\n\n"
          "Answer EXACTLY:\nLabel: <=6 words\nConfidence: 0.00-1.00")
out = {}
for fid in feats:
    ev = snippets(fid)
    if not ev: out[fid] = {"error": "no examples"}; continue
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": PROMPT.format(band=band, ev="\n".join(f"- {s}" for s in ev))}],
                max_tokens=100, temperature=0)
            txt = resp.choices[0].message.content
            lab = next((l.split(":", 1)[1].strip() for l in txt.splitlines() if l.lower().startswith("label")), txt[:50])
            out[fid] = {"label": lab, "raw": txt}; print(f"  F{fid}: {lab}", flush=True); break
        except Exception as e:
            if attempt == 4: out[fid] = {"error": str(e)[:80]}
            time.sleep(6 * (attempt + 1))
    time.sleep(1)
json.dump(out, open(out_path, "w"), indent=2)
print(f"[wrote] {out_path} ({len([v for v in out.values() if 'label' in v])} named)")
