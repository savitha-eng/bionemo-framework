import json,time,os
from openai import OpenAI
c=OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NIM_API_KEY"])
syn=json.load(open("/data/savithas/phase3_full/echo_synthesis_l30.json"))["synthesis_features"]
ll=json.load(open("/data/savithas/phase3_full/logit_lens_l30.json"))
pf=json.load(open("/data/savithas/phase3_full/per_feature_structural.json"))
bio_lab={16026:"IPR000719 Protein kinase",679:"IPR016024 ARM fold",11009:"IPR015943 WD40",5540:"IPR009057 homeodomain",17703:"IPR036179 Ig-like"}
jobs=[]
for f,v in list(syn.items())[:10]: jobs.append(("reasoning-synthesis",int(f),{"top_words":v["words"],"synthesis_auroc":v["auroc"]}))
for f,lab in bio_lab.items(): jobs.append(("protein-detector",int(f),{"detects_domain":lab,"logit_lens":ll.get(str(f),{}).get("promotes",[])}))
P="You are interpreting an SAE feature of BioReason-Pro (a protein+reasoning model). Band: {band}. Evidence: {ev}. Give exactly:\nLabel: <=6 words\nDescription: 1 sentence\nConfidence: 0.00-1.00"
out={}
for band,f,ev in jobs:
    for attempt in range(6):
        try:
            r=c.chat.completions.create(model="meta/llama-3.1-70b-instruct",messages=[{"role":"user","content":P.format(band=band,ev=json.dumps(ev))}],max_tokens=120,temperature=0)
            out[f]={"band":band,"raw":r.choices[0].message.content}; print(f"F{f} ({band}): {r.choices[0].message.content.splitlines()[0] if r.choices[0].message.content else ''}",flush=True); break
        except Exception as e:
            time.sleep(8*(attempt+1))
    time.sleep(2)
json.dump(out,open("/data/savithas/phase3_full/autointerp_l30.json","w"),indent=2); print("[wrote] autointerp_l30.json")
