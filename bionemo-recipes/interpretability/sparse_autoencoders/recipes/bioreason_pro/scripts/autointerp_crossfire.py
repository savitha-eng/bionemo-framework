"""A1: auto-interpret the CROSS-FIRE reasoning partners (F3184, F13384, F10235, ...) that co-fire with the
bio/structure features. These were identified only by co-firing correlation (crossmodal_pairing) and are
UNNAMED -- so we can't say what they 'say'. This names them from their top reasoning-band firing examples.

For each feature: pull top reasoning-band examples from the L30 dashboard feature_examples.parquet, highlight
the peak-activation tokens, feed the snippets to an LLM -> Label/Description/Confidence. Grounds the cross-modal
story (does the reasoning partner actually discuss the same biology as its bio partner?)."""
import json, os, time, sys
import numpy as np, pyarrow.parquet as pq
from openai import OpenAI

DFULL = "/data/savithas/phase3_full"
PUB = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/"
       "bioreason_pro/multimodal_dashboard/public/l30_balanced/feature_examples.parquet")
N_EX = 4                     # top reasoning examples per feature
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NIM_API_KEY"])

# target features = the reasoning partners from crossmodal_pairing + the two co-firing clusters
pairs = json.load(open(f"{DFULL}/crossmodal_pairing_l30.json"))["results"]
partner_of = {}             # reasoning_feature -> (bio_name, r)
for r in pairs:
    if "partners" not in r: continue
    p = r["partners"][0]
    partner_of.setdefault(p["reasoning_feature"], (r["bio_name"], p["r"]))
clusters = {3184: "GPCR", 38862: "GPCR", 313: "GPCR", 25291: "GPCR", 10781: "GPCR",
            13384: "P450", 35665: "P450", 27901: "P450", 27308: "P450", 25846: "P450"}
for f, c in clusters.items():
    partner_of.setdefault(f, (c + " (cluster)", None))
targets = sorted(partner_of)
print(f"[A1] naming {len(targets)} cross-fire reasoning partners", flush=True)

df = pq.read_table(PUB).to_pandas()
df = df[df.band == "reasoning"]

def snippets(fid):
    sub = df[df.feature_id == fid].nlargest(N_EX, "max_activation")
    out = []
    for _, x in sub.iterrows():
        toks = str(x["sequence"]).split(); act = np.asarray(x["activations"], float)
        n = min(len(toks), len(act))
        if n == 0: continue
        toks, act = toks[:n], act[:n]
        pk = act.argmax()
        lo, hi = max(0, pk - 12), min(n, pk + 13)
        s = " ".join(("**" + toks[j] + "**" if act[j] >= 0.6 * act[pk] and act[j] > 0 else toks[j]) for j in range(lo, hi))
        out.append(s)
    return out

PROMPT = ("You are interpreting an SAE feature of BioReason-Pro (protein GO-function reasoning model). "
          "Below are reasoning-text snippets where this feature fires most; **bold** tokens are the peak "
          "activations. Identify the single concept the feature detects.\nSnippets:\n{ev}\n\n"
          "Answer EXACTLY:\nLabel: <=6 words\nDescription: 1 sentence\nConfidence: 0.00-1.00")
out = {}
for fid in targets:
    ev = snippets(fid)
    if not ev:
        out[fid] = {"error": "no reasoning examples"}; continue
    evtxt = "\n".join(f"- {s}" for s in ev)
    bio, r = partner_of[fid]
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": PROMPT.format(ev=evtxt)}], max_tokens=120, temperature=0)
            txt = resp.choices[0].message.content
            label = next((l.split(":", 1)[1].strip() for l in txt.splitlines() if l.lower().startswith("label")), txt[:60])
            out[fid] = {"label": label, "raw": txt, "cofire_bio_partner": bio, "cofire_r": r}
            print(f"  F{fid} (co-fires {bio} r={r}): {label}", flush=True); break
        except Exception as e:
            if attempt == 4: out[fid] = {"error": str(e)[:100]}
            time.sleep(6 * (attempt + 1))
    time.sleep(1)
json.dump(out, open(f"{DFULL}/autointerp_crossfire_l30.json", "w"), indent=2)
print(f"[wrote] autointerp_crossfire_l30.json ({len([v for v in out.values() if 'label' in v])} named)")
