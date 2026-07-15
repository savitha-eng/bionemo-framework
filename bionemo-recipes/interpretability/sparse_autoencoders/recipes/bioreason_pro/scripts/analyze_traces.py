#!/usr/bin/env python
"""Honest steering validation from saved traces — fixes the two flaws the lexical metric had:
 1. CHAR-LEVEL coherence (catches pseudo-word loops like 'synaptotereguliaritys...' that word-level
    distinct3 missed): char_distinct8 = distinct 8-char-grams/total; longest12 = frac covered by the
    most-repeated 12-gram. Coherent = char_distinct8>=0.75 AND longest12<=0.20 AND nonascii<=0.02.
 2. PER-PROTEIN BASELINE removal: only count a protein as a GENUINE injection if it had ZERO concept
    words at alpha=0 (so proteins naturally about the concept don't confound the cluster verdict).

Genuine-coherent-injection rate = frac of zero-baseline proteins that, at a given alpha, gain concept
words AND stay coherent by the char metric. This is the honest steering signal (no LLM needed).

Usage: analyze_traces.py <traces_dir> [out.json]
"""
import json, glob, re, sys
from pathlib import Path
from collections import defaultdict

tdir = sys.argv[1] if len(sys.argv) > 1 else "/data/savithas/phase3_full"
out = sys.argv[2] if len(sys.argv) > 2 else "/data/savithas/phase3_full/steering_validation.json"


def char_distinct(t, k=8):
    t = re.sub(r"\s+", " ", t); g = [t[i:i+k] for i in range(len(t)-k+1)]
    return len(set(g))/max(1, len(g))


def longest12(t):
    from collections import Counter
    t = re.sub(r"\s+", " ", t); g = Counter(t[i:i+12] for i in range(len(t)-11))
    return (g.most_common(1)[0][1]*12/max(1, len(t))) if g else 0.0


def coherent(r):
    t = r["text"]
    na = sum(1 for c in t if not c.isspace() and ord(c) > 127)/max(1, len(t))
    return char_distinct(t) >= 0.75 and longest12(t) <= 0.20 and na <= 0.02


results = {}
files = sorted(glob.glob(f"{tdir}/traces_*.jsonl"))
print(f"{'cluster':15} {'alpha':>5} {'n0base':>7} {'inject':>7} {'coh&inj':>8}  {'gen-rate':>8}  meanCharD")
for f in files:
    cl = Path(f).stem.replace("traces_", "")
    recs = [json.loads(l) for l in open(f)]
    by = defaultdict(dict)
    for r in recs:
        by[r["protein"]][r["alpha"]] = r
    alphas = sorted({r["alpha"] for r in recs})
    base0 = {p: (by[p].get(0.0, by[p].get(0, {})).get("c1", 0)) for p in by}
    zerob = [p for p in by if base0[p] == 0]                 # proteins with NO baseline concept words
    per = {}
    for a in alphas:
        if a == 0:
            continue
        inj = coh_inj = 0; cds = []
        for p in zerob:
            r = by[p].get(a)
            if not r:
                continue
            cds.append(char_distinct(r["text"]))
            if r["c1"] > 0:
                inj += 1
                if coherent(r):
                    coh_inj += 1
        n0 = len(zerob)
        per[a] = {"n_zerobase": n0, "n_inject": inj, "n_coherent_inject": coh_inj,
                  "genuine_rate": round(coh_inj/max(1, n0), 2), "mean_char_distinct": round(sum(cds)/max(1, len(cds)), 3)}
        print(f"{cl:15} {a:>5.0f} {n0:>7} {inj:>7} {coh_inj:>8}  {per[a]['genuine_rate']:>8}  {per[a]['mean_char_distinct']}")
    results[cl] = {"n_proteins": len(by), "n_zero_baseline": len(zerob), "per_alpha": per}
    print()

json.dump(results, open(out, "w"), indent=2)
print(f"[wrote] {out}")
print("gen-rate = frac of ZERO-BASELINE proteins with coherent concept injection (the honest steering signal).")
