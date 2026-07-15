#!/usr/bin/env python
"""LLM-judge for steering generations (hardens the lexical coherence metric).

For each steered generation, an independent LLM (NIM Llama-3-70B-Instruct, OpenAI-compatible) rates two
things the lexical metric CANNOT: (1) is the target concept GENUINELY present as part of the reasoning
(not just sprinkled words), and (2) is it coherent, well-formed protein reasoning? Aggregates per alpha.

Endpoint via env (NIM local or hosted build.nvidia.com):
  NIM_BASE_URL  (default http://localhost:8000/v1)
  NIM_MODEL     (default meta/llama-3.1-70b-instruct)
  NIM_API_KEY   (required for hosted integrate.api.nvidia.com; unneeded for a local NIM)

Usage: llm_judge_steering.py --gens <jsonl> --concept "synapse / nervous-system" [--out judge.json] [--max N]
"""
import argparse, json, os, sys, time
from collections import defaultdict
from openai import OpenAI

p = argparse.ArgumentParser()
p.add_argument("--gens", required=True)
p.add_argument("--concept", required=True)
p.add_argument("--out", default="")
p.add_argument("--max", type=int, default=0, help="cap #generations judged (0=all)")
a = p.parse_args()

BASE = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")
KEY = os.environ.get("NIM_API_KEY", "")
out = a.out or a.gens.replace(".jsonl", "") + "_judge.json"
client = OpenAI(base_url=BASE, api_key=KEY, timeout=30, max_retries=2)

SYS = ("You are a strict evaluator of protein-function reasoning text produced by a language model whose "
       "internal features were artificially clamped to try to inject a target concept. Judge ONLY what is "
       "written. Respond with a compact JSON object, nothing else.")
PROMPT = ("Target concept: {concept}\n\nReasoning text:\n\"\"\"\n{text}\n\"\"\"\n\n"
          "Rate on two axes:\n"
          "- concept_present: 0 = the target concept is absent; 1 = a few target words appear but are NOT "
          "integrated into the reasoning (superficial sprinkling); 2 = the model genuinely reasons about the "
          "target concept.\n"
          "- coherent: 0 = gibberish/repetition-loop; 1 = grammatical but confused/contradictory; 2 = coherent, "
          "well-formed protein reasoning.\n"
          'Return exactly: {{"concept_present": <0|1|2>, "coherent": <0|1|2>}}')


def judge(text):
    for attempt in range(7):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=40,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": PROMPT.format(concept=a.concept, text=text[:3000])}])
            txt = r.choices[0].message.content
            s = txt[txt.find("{"): txt.rfind("}") + 1]
            d = json.loads(s)
            return int(d["concept_present"]), int(d["coherent"])
        except Exception as e:
            if ("429" in str(e) or "RateLimit" in type(e).__name__) and attempt < 6:
                time.sleep(min(45, 3 * (2 ** attempt)))       # backoff on rate limit
                continue
            return None
    return None


recs = [json.loads(l) for l in open(a.gens)]
if a.max: recs = recs[:a.max]
print(f"[judge] {len(recs)} generations via {MODEL} @ {BASE}", flush=True)
per = defaultdict(lambda: {"n": 0, "present2": 0, "coh2": 0, "both": 0, "cp": [], "ch": []})
for i, r in enumerate(recs):
    v = judge(r["text"]); time.sleep(1.2)                     # base spacing to respect rate limit
    if v is None: continue
    cp, ch = v; d = per[r["alpha"]]
    d["n"] += 1; d["cp"].append(cp); d["ch"].append(ch)
    d["present2"] += (cp == 2); d["coh2"] += (ch == 2); d["both"] += (cp == 2 and ch == 2)
    if i % 20 == 0: print(f"  {i}/{len(recs)}", flush=True)

res = {}
print(f"\n{'alpha':>6} {'n':>4} {'%genuine-concept':>16} {'%coherent':>10} {'%BOTH':>7}  (LLM-judge)")
for al in sorted(per):
    d = per[al]; n = max(1, d["n"])
    res[al] = {"n": d["n"], "pct_concept_genuine": round(100 * d["present2"] / n),
               "pct_coherent": round(100 * d["coh2"] / n), "pct_both": round(100 * d["both"] / n),
               "mean_concept": round(sum(d["cp"]) / n, 2), "mean_coherent": round(sum(d["ch"]) / n, 2)}
    print(f"{al:>6} {d['n']:>4} {res[al]['pct_concept_genuine']:>15}% {res[al]['pct_coherent']:>9}% "
          f"{res[al]['pct_both']:>6}%")
print("\n%BOTH = genuine concept AND coherent — the HONEST steering rate (vs lexical coh_c1 which overstates).")
json.dump({"concept": a.concept, "model": MODEL, "per_alpha": res}, open(out, "w"), indent=2)
print(f"[wrote] {out}")
