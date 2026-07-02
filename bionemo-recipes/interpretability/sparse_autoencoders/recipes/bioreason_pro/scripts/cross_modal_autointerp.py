#!/usr/bin/env python
"""Cross-modal autointerp: for co-firing features, label the PROTEIN-side and TEXT-side firing
SEPARATELY, then judge whether they're the SAME concept. This is the semantic test of whether a
feature is genuinely cross-modal (fusion) vs generic (fires on both but means different things).

Uses the dashboard's band-tagged feature_examples. TEXT = prompt/reasoning/answer/text bands.
Serial (workers=1) with backoff for the shared NIM key. Model default = llama-3.1-70b (3.3 stalls).
"""
import argparse, json, os, time
import pyarrow.parquet as pq
from openai import OpenAI

TEXT_BANDS = {"prompt", "reasoning", "answer", "text"}


def win(seq, acts, ctx=8):
    toks = seq.split(" ") if isinstance(seq, str) else list(seq)
    a = (json.loads(acts) if isinstance(acts, str) else list(acts))[:len(toks)]
    if not a:
        return seq[:160]
    pk = max(range(len(a)), key=lambda i: a[i])
    lo, hi = max(0, pk - ctx), min(len(toks), pk + ctx + 1)
    out = toks[lo:hi]; pkr = pk - lo
    if 0 <= pkr < len(out):
        out[pkr] = f"«{out[pkr]}»"
    return " ".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pub", required=True); p.add_argument("--features", required=True)
    p.add_argument("--n", type=int, default=8, help="windows per side shown to the LLM")
    p.add_argument("--model", default="meta/llama-3.1-70b-instruct")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NIM_API_KEY"], timeout=60)
    feats = [int(x) for x in args.features.split(",")]

    ex = pq.read_table(f"{args.pub}/feature_examples.parquet").to_pandas()

    def call(sys, usr, mx=120, temp=0.2):
        for att in range(5):
            try:
                r = client.chat.completions.create(model=args.model, temperature=temp, max_tokens=mx,
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}])
                return r.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001
                if att == 4:
                    return f"(error:{type(e).__name__})"
                time.sleep(3 * (att + 1))

    SYS = "You are a mechanistic-interpretability analyst. Be concise and specific."
    out = {}
    for fid in feats:
        sub = ex[ex.feature_id == fid]
        pro = sub[sub.band == "protein"].nlargest(args.n, "max_activation")
        txt = sub[sub.band.isin(TEXT_BANDS)].nlargest(args.n, "max_activation")
        if len(pro) == 0 or len(txt) == 0:
            out[str(fid)] = {"skip": "missing one side"}; print(f"F{fid}: skip", flush=True); continue
        pw = "\n".join("  - " + win(r.sequence, r.activations) for _, r in pro.iterrows())
        tw = "\n".join("  - " + win(r.sequence, r.activations) for _, r in txt.iterrows())
        plab = call(SYS, f"Top PROTEIN-token windows where feature {fid} fires (« » = peak):\n{pw}\nIn one sentence, what does it detect?")
        tlab = call(SYS, f"Top TEXT windows where feature {fid} fires (« » = peak):\n{tw}\nIn one sentence, what does it detect?")
        judge = call(SYS,
            f"A single SAE feature fires on BOTH modalities of a protein model.\n"
            f"PROTEIN-side meaning: {plab}\nTEXT-side meaning: {tlab}\n"
            f"Are these the SAME underlying concept (genuine cross-modal feature), or unrelated/generic? "
            f"Answer strictly as: VERDICT: SAME|RELATED|DIFFERENT|GENERIC — then one clause why.", mx=60, temp=0.0)
        v = "?"
        for cand in ("SAME", "RELATED", "DIFFERENT", "GENERIC"):
            if cand in (judge or "").upper():
                v = cand; break
        out[str(fid)] = {"protein_label": plab, "text_label": tlab, "judge": judge, "verdict": v,
                         "n_protein": int(len(pro)), "n_text": int(len(txt))}
        print(f"F{fid}: [{v}] P='{(plab or '')[:55]}' | T='{(tlab or '')[:55]}'", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    # summary
    from collections import Counter
    c = Counter(v.get("verdict") for v in out.values() if isinstance(v, dict) and v.get("verdict"))
    print("\n=== cross-modal verdicts ===")
    for k, n in c.most_common():
        print(f"  {k}: {n}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
