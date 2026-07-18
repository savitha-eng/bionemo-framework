"""AUTO-INTERP VALIDATION: are the LLM feature labels grounded in data, or plausible-but-wrong hallucinations?
For each labeled feature, take its top-activating proteins (reasoning band) and Fisher-enrich their GO/InterPro
annotations. If the data-driven enriched term matches the LLM label, the label is validated; if not, flag it.

Usage: autointerp_validate.py <labels.json> [--out validate.json]. labels.json = {feature_id: {label: ...}}."""
import json, sys, numpy as np, pyarrow.parquet as pq
from scipy.stats import hypergeom
from collections import Counter
sys.path.insert(0, "src"); import bioreason_pro_sae.data as brp

labels_path = sys.argv[1]
out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "/data/savithas/phase3_full/autointerp_validate.json"
labels = json.load(open(labels_path))
labeled = {int(k): v.get("label", "") for k, v in labels.items() if isinstance(v, dict) and v.get("label")}

R = np.load("/data/savithas/phase3_full/pooled_reasoning_l30_00e1d8.npz"); SR = R["SAE"]; keep = R["keep"]
pr = pq.read_table("/data/savithas/phase3_subset_8k/L30_subset8k/proteins.parquet")
pids = [str(x) for x in pr.column("protein_id").to_pylist()]
tr, _, _ = brp.load_reasoning_splits(max_length_protein=2000)
go = {str(p): set(g if g else []) for p, g in zip(tr["protein_id"], tr["go_ids"])}
ipr = {str(p): set(i if i else []) for p, i in zip(tr["protein_id"], tr["interpro_ids"])}
goname, iprname = {}, {}
for p, forms in zip(tr["protein_id"], tr["interpro_formatted"]):
    for line in str(forms).split("\n"):
        if line.startswith("- IPR"):
            k = line[2:].split(":")[0].strip()
            iprname[k] = line.split(":", 1)[1].split("(")[0].strip() if ":" in line else k
# GO names from go_pred formatting if available (else raw id)
idx = np.where(keep)[0]; P = len(idx); pset = [pids[i] for i in idx]; SRk = SR[idx]
terms = {}
for src, d, nm in [("GO", go, {}), ("IPR", ipr, iprname)]:
    c = Counter(t for p in pset for t in d.get(p, ()))
    for t, n in c.items():
        if 20 <= n <= P * 0.5: terms[(src, t)] = (np.array([t in d.get(p, ()) for p in pset]), nm.get(t, t))

rows = []
for f, lab in labeled.items():
    col = SRk[:, f]
    if (col > 0).sum() < 20: rows.append({"feature": f, "label": lab, "enriched": None, "note": "too sparse"}); continue
    topk = set(np.argsort(-col)[:max(20, int(P * 0.05))].tolist()); best = 1.0; bt = None
    for (src, t), (mem, name) in terms.items():
        mi = set(np.where(mem)[0].tolist()); k = len(topk & mi)
        if k < 4: continue
        pv = hypergeom.sf(k - 1, P, len(mi), len(topk))
        if pv < best: best = pv; bt = (src, t, name)
    fdr = best * len(terms)
    rows.append({"feature": f, "label": lab, "enriched_term": (f"{bt[0]}:{bt[1]}" if bt else None),
                 "enriched_name": (bt[2] if bt else None), "fdr": (float(fdr) if bt else None)})

sig = [r for r in rows if r.get("fdr") is not None and r["fdr"] < 0.05]
print(f"[validate] {len(labeled)} labeled features; {len(sig)} have a significant enriched term (FDR<0.05)")
print(f"\n{'feat':>7}  {'LLM label':32} -> data-enriched term (FDR)")
for r in rows:
    en = r.get("enriched_name") or "(none sig)"; fdr = r.get("fdr")
    fs = f"FDR={fdr:.0e}" if fdr is not None else ""
    print(f"  F{r['feature']:<6}{r['label'][:32]:32} -> {en[:34]} {fs}")
json.dump(rows, open(out_path, "w"), indent=2)
print(f"[wrote] {out_path}")
