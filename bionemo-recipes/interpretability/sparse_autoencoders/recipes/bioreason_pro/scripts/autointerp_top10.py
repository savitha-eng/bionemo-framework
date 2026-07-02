#!/usr/bin/env python
"""Autointerp (LABEL + DETECTION SCORE) for an explicit feature list, from a dashboard pub dir.

For each feature:
  1. LABEL  - LLM reads the top-N activating windows (with go_terms context) -> one-sentence explanation.
  2. SCORE  - detection test (Bills/EleutherAI style): show the LLM a shuffled mix of this feature's
              held-out windows + random OTHER-feature windows, ask which match the label, compute F1.
              A high F1 means the label genuinely predicts where the feature fires.
Protein-side features additionally carry their GO label + AUC from feature_metadata (selective GO
prediction), since raw residue windows aren't human-readable. Output: JSON per feature.
Usage: autointerp_top10.py --pub <dir> --features 18403,30664,... --n-examples 50 --out <json>
"""
import json
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pyarrow.parquet as pq
from openai import OpenAI

SYS_LABEL = ("You are interpreting a sparse-autoencoder feature from a multimodal protein-reasoning LLM. "
             "Given its top-activating text windows (the [[token]] is where it fires hardest), state in ONE "
             "sentence the precise concept the feature detects. Be specific and concrete; if it is a protein "
             "(amino-acid) feature, use the provided GO context to name the protein property.")
SYS_SCORE = ("You are testing a feature explanation. Given the explanation and a numbered list of text windows, "
             "reply with ONLY the comma-separated numbers of windows where the described feature would activate "
             "(the [[token]] marks the candidate firing site). Reply with numbers only, e.g. '1,4,5'.")


def win(e, ntok=22):
    s = e["sequence"].split(" "); ac = np.array(json.loads(e["activations"]) if isinstance(e["activations"], str) else e["activations"]); t = int(ac.argmax())
    if t < len(s): s[t] = f"[[{s[t]}]]"
    lo = max(0, t - 12)
    return " ".join(s[lo: lo + ntok])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pub", required=True); p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="meta/llama-3.3-70b-instruct")
    p.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    p.add_argument("--n-examples", type=int, default=50)
    p.add_argument("--n-score", type=int, default=10, help="# positive windows in the detection test")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    rng = random.Random(0)
    client = OpenAI(base_url=args.base_url, api_key=os.environ["NIM_API_KEY"])
    feats = [int(x) for x in args.features.split(",")]

    meta = {r["feature_id"]: r for r in pq.read_table(f"{args.pub}/feature_metadata.parquet").to_pylist()}
    ex = {}
    for r in pq.read_table(f"{args.pub}/feature_examples.parquet").to_pylist():
        ex.setdefault(r["feature_id"], []).append(r)
    for f in ex: ex[f].sort(key=lambda r: -r["max_activation"])
    all_other = [(f, e) for f, lst in ex.items() for e in lst[:8]]  # negative pool

    def label_one(fid):
        egs = ex.get(fid, [])[:args.n_examples]
        if not egs: return "(no examples)"
        m = meta.get(fid, {})
        go_ctx = ""
        if (m.get("protein_frac") or 0) > 0.5:
            gl = m.get("go_label") or ""; auc = m.get("go_auc")
            gos = "; ".join(sorted({e["go_terms"] for e in egs if e.get("go_terms")}))[:400]
            go_ctx = f"\nThis is a PROTEIN feature. GO label: {gl} (AUC {auc}). GO terms on firing proteins: {gos}\n"
        body = "\n".join("  - " + win(e) for e in egs[:20])
        prompt = f"Top activating windows for feature {fid}:\n{body}\n{go_ctx}\nOne sentence:"
        for att in range(5):
            try:
                r = client.chat.completions.create(model=args.model,
                    messages=[{"role": "system", "content": SYS_LABEL}, {"role": "user", "content": prompt}],
                    temperature=0.2, top_p=0.7, max_tokens=90)
                return r.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001
                if att == 4: return f"(error:{type(e).__name__})"
                time.sleep(2 * (att + 1))

    def score_one(fid, label):
        pos = ex.get(fid, [])[:args.n_examples]
        if len(pos) < 3 or label.startswith("("): return None
        # held-out positives = ranks beyond what labeling saw; fall back to a tail slice
        pos_test = pos[max(0, len(pos) - args.n_score):]
        negs = [e for (g, e) in all_other if g != fid]
        neg_test = rng.sample(negs, min(args.n_score, len(negs)))
        items = [(1, e) for e in pos_test] + [(0, e) for e in neg_test]
        rng.shuffle(items)
        listing = "\n".join(f"{i+1}. {win(e)}" for i, (_, e) in enumerate(items))
        prompt = f"Explanation: {label}\n\nWindows:\n{listing}\n\nWhich numbers activate?"
        for att in range(5):
            try:
                r = client.chat.completions.create(model=args.model,
                    messages=[{"role": "system", "content": SYS_SCORE}, {"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=60)
                picked = {int(x) for x in __import__("re").findall(r"\d+", r.choices[0].message.content)}
                tp = sum(1 for i, (y, _) in enumerate(items) if y == 1 and (i + 1) in picked)
                fp = sum(1 for i, (y, _) in enumerate(items) if y == 0 and (i + 1) in picked)
                fn = sum(1 for i, (y, _) in enumerate(items) if y == 1 and (i + 1) not in picked)
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                return {"f1": round(f1, 2), "precision": round(prec, 2), "recall": round(rec, 2),
                        "n_pos": len(pos_test), "n_neg": len(neg_test)}
            except Exception as e:  # noqa: BLE001
                if att == 4: return {"f1": None, "error": type(e).__name__}
                time.sleep(2 * (att + 1))

    def work(fid):
        lab = label_one(fid)
        sc = score_one(fid, lab)
        m = meta.get(fid, {})
        return str(fid), {"feature_id": fid, "label": lab, "detection": sc,
                          "dashboard_label": m.get("label"), "go_auc": m.get("go_auc"),
                          "protein_frac": m.get("protein_frac"), "text_frac": m.get("text_frac"),
                          "max_activation": m.get("max_activation"),
                          "n_examples_used": len(ex.get(fid, [])[:args.n_examples])}

    out = {}
    with ThreadPoolExecutor(max_workers=args.workers) as exr:
        for fut in as_completed([exr.submit(work, f) for f in feats]):
            k, v = fut.result(); out[k] = v
            print(f"  F{k}: {str(v['label'])[:70]}  | det F1={v['detection'].get('f1') if v['detection'] else 'na'}", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {len(out)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
