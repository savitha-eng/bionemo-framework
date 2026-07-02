#!/usr/bin/env python
"""Band-split autointerp: label a feature's REASONING-side and ANSWER-side firing SEPARATELY (and note
PROMPT), then judge. Standard autointerp labels from top windows mixed across bands, so it can't tell a
genuine reasoning feature from an output/answer feature or a prompt-leakage feature. This separates them.

reasoning = model's OWN chain-of-thought (genuine) ; answer = final output ; prompt = handed-in input
(prompt-dominant => likely LEAKAGE, echoing fed-in GO/InterPro terms rather than computing).
Serial (workers=1) + backoff for the shared NIM key. Default model llama-3.1-70b (3.3 stalls).
"""
import argparse, json, os, time
import pyarrow.parquet as pq
from openai import OpenAI


def win(seq, acts, ctx=8):
    toks = seq.split(" ") if isinstance(seq, str) else list(seq)
    a = (json.loads(acts) if isinstance(acts, str) else list(acts))[:len(toks)]
    if not a:
        return (seq or "")[:160]
    pk = max(range(len(a)), key=lambda i: a[i])
    lo, hi = max(0, pk - ctx), min(len(toks), pk + ctx + 1)
    out = toks[lo:hi]; pkr = pk - lo
    if 0 <= pkr < len(out):
        out[pkr] = f"«{out[pkr]}»"
    return " ".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pub", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8); p.add_argument("--top-feats", type=int, default=15)
    p.add_argument("--model", default="meta/llama-3.1-70b-instruct")
    args = p.parse_args()
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NIM_API_KEY"], timeout=60)

    ex = pq.read_table(f"{args.pub}/feature_examples.parquet").to_pandas()
    meta = pq.read_table(f"{args.pub}/feature_metadata.parquet").to_pandas().set_index("feature_id")
    # HQ TEXT features (strong on a text band by PEAK, wide span, rare) — where reasoning/answer lives
    TEXT = {"prompt", "reasoning", "answer", "text"}
    ex["mod"] = ex["band"].map(lambda b: "protein" if b == "protein" else ("text" if b in TEXT else "go"))
    pk = ex.groupby(["feature_id", "mod"])["max_activation"].max().unstack(fill_value=0)
    tpk = pk["text"] if "text" in pk else pk.iloc[:, 0] * 0
    ppk = pk["protein"] if "protein" in pk else tpk * 0
    af = meta["activation_frequency"].reindex(pk.index).fillna(1.0)
    sp = meta["max_span"].reindex(pk.index).fillna(0)
    hq = pk.index[(tpk > 8) & (tpk > ppk) & (sp >= 5) & (af < 0.005)]
    feats = list(tpk[hq].sort_values(ascending=False).index[: args.top_feats])
    print(f"{len(feats)} HQ text features to band-split", flush=True)

    def call(sys, usr, mx=110, temp=0.2):
        for att in range(5):
            try:
                r = client.chat.completions.create(model=args.model, temperature=temp, max_tokens=mx,
                    messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}])
                return r.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001
                if att == 4:
                    return f"(error:{type(e).__name__})"
                time.sleep(3 * (att + 1))

    SYS = "You interpret sparse-autoencoder features of a protein-reasoning LLM. Be concise and specific."
    out = {}
    for fid in feats:
        sub = ex[ex.feature_id == fid]
        labs, peaks = {}, {}
        for band in ("reasoning", "answer", "prompt"):
            b = sub[sub.band == band].nlargest(args.n, "max_activation")
            peaks[band] = float(b.max_activation.max()) if len(b) else 0.0
            if len(b) == 0 or peaks[band] < 0.5:
                labs[band] = "(does not fire here)"
                continue
            w = "\n".join("  - " + win(r.sequence, r.activations) for _, r in b.iterrows())
            labs[band] = call(SYS, f"Top {band.upper()}-band windows where feature {fid} fires (« »=peak):\n{w}\nIn one sentence, what does it detect?")
        # verdict: genuine reasoning? consistent across bands? prompt-leakage?
        judge = call(SYS,
            f"An SAE feature fires with these per-band peaks: reasoning={peaks['reasoning']:.1f} "
            f"answer={peaks['answer']:.1f} prompt={peaks['prompt']:.1f}.\n"
            f"REASONING meaning: {labs['reasoning']}\nANSWER meaning: {labs['answer']}\nPROMPT meaning: {labs['prompt']}\n"
            f"Classify strictly as one of: VERDICT: REASONING (genuine chain-of-thought feature) | "
            f"ANSWER (output-generation) | PROMPT-LEAKAGE (mostly echoes fed-in prompt terms) | "
            f"CONSISTENT (same concept across reasoning+answer). Then one clause why.", mx=60, temp=0.0)
        v = next((c for c in ("PROMPT-LEAKAGE", "CONSISTENT", "REASONING", "ANSWER") if c in (judge or "").upper()), "?")
        out[str(fid)] = {"feature_id": int(fid), "peaks": peaks, "reasoning_label": labs["reasoning"],
                         "answer_label": labs["answer"], "prompt_label": labs["prompt"], "judge": judge, "verdict": v}
        print(f"F{fid} [{v}] R={peaks['reasoning']:.0f}/A={peaks['answer']:.0f}/P={peaks['prompt']:.0f} | R:'{labs['reasoning'][:45]}' A:'{labs['answer'][:45]}'", flush=True)
    json.dump(out, open(args.out, "w"), indent=2)
    from collections import Counter
    c = Counter(v["verdict"] for v in out.values() if isinstance(v, dict))
    print("\n=== band verdicts ===")
    for k, n in c.most_common():
        print(f"  {k}: {n}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
