#!/usr/bin/env python
"""GENERATION-mode steering: clamp a reasoning feature during the model's OWN generation and read
whether the reasoning TEXT shifts toward the concept. This is the real causal test (vs teacher-forced
logit steering): we truncate the input at <think>, let BioReason-Pro generate its reasoning, and add
alpha*(feature decoder direction) to the residual at layer L on each GENERATED token.

Strongest demo: on proteins the feature does NOT normally fire on, does clamping *induce* the concept?

Usage: steer_generation.py --feature 4777 --layer 16 --sae <ckpt> --concept-words sperm,meiosis,gamete,... \
   --num-proteins 8 --alphas 0,40 --max-new 180
"""
import argparse, sys, re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--feature", type=int, default=-1)
    p.add_argument("--features", default="", help="comma-sep list -> GROUP/subspace steering "
                   "(clamp the whole co-firing cluster at once). Overrides --feature.")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--concept-words", default="sperm,meiosis,gamete,oocyte,fertiliz,reproduct,germ,testis,ovar")
    p.add_argument("--concept-words-2", default="synap,neuron,axon,dendrit,nervous,neurit",
                   help="SECOND concept (specificity control): if the repro cluster boosts repro words but NOT "
                        "these, and the neuron cluster does the reverse, that's a double dissociation = real "
                        "feature-specific causality (vs generic degradation).")
    p.add_argument("--random-dirs", action="store_true",
                   help="CONTROL: replace the feature decoder dirs with random matched-norm dirs. If this also "
                        "injects the concept words, the effect is NOT feature-specific.")
    p.add_argument("--alphas", default="0,40")
    p.add_argument("--clamp-mode", default="add", choices=["add", "set"],
                   help="add: h += alpha*decoder_dir (crude). set: clamp feature activation to alpha via "
                        "encode->override->contribution (Jared-style, accounts for current activation).")
    p.add_argument("--save-gens", default="", help="JSONL path: dump full generated text + coherence per "
                   "(protein,alpha) for the LLM-judge (concept-present + real-reasoning rating)")
    p.add_argument("--num-proteins", type=int, default=8)
    p.add_argument("--steer-after", type=int, default=0,
                   help="DECISION POINT: start clamping only after N generated tokens (Goodfire: token-0 steering fails)")
    p.add_argument("--scale-file", default="", help="npz w/ per-feature scale (mean_active) -> alphas become p95-style multipliers")
    p.add_argument("--scale-key", default="ma_text", help="npz key for per-feature scale (ma_text | ma_protein)")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-new", type=int, default=180)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default="/data/savithas/bioreason-pro")
    args = p.parse_args()
    dev = "cuda"
    alphas = [float(x) for x in args.alphas.split(",")]
    words = [w.lower() for w in args.concept_words.split(",")]

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft
    from sae.architectures import TopKSAE

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    sae = sae.to(dev).eval()
    W = sae.decoder.weight.detach()
    col_layout = (W.shape[1] != W.shape[0] and W.shape[0] == 2560)   # decoder cols are 2560-dim directions
    def dir_of(f):
        return (W[:, f] if col_layout else W[f]).float().to(dev)
    feats = [int(x) for x in args.features.split(",") if x.strip() != ""] if args.features else [args.feature]
    feats_t = torch.tensor(feats, device=dev)
    if args.scale_file:                                              # p95-style per-feature dose (Jared): alpha := mult * scale[f]
        _sc = np.load(args.scale_file)[args.scale_key]
        scale_vec = torch.tensor([float(_sc[f]) for f in feats], device=dev)
        print(f"[gen-steer] p95-style dosing: per-feature scale({args.scale_key})={[round(float(x),2) for x in scale_vec]}")
    else:
        scale_vec = torch.ones(len(feats), device=dev)
    D_mat = torch.stack([dir_of(f) for f in feats], 0)               # [n_feat, 2560]
    if args.random_dirs:                                             # matched-norm random control
        g = torch.Generator(device=dev).manual_seed(0)
        norms = D_mat.norm(dim=1, keepdim=True)
        R = torch.randn(D_mat.shape, generator=g, device=dev)
        D_mat = R / R.norm(dim=1, keepdim=True) * norms
        print("[gen-steer] RANDOM matched-norm control directions in use")
    D_sum = D_mat.sum(0)                                             # subspace push for add-mode
    print(f"[gen-steer] {'GROUP' if len(feats) > 1 else 'single'} steering feats={feats}, "
          f"per-dir norms={[round(float(D_mat[i].norm()), 2) for i in range(len(feats))]}, mode={args.clamp_mode}")

    model = load_bioreason_pro_sft(ckpt_dir=args.ckpt_dir, bioreason_pro_root=args.bioreason_root,
                                   device=dev, max_length_text=10000, max_length_protein=2000).eval()
    tok = model.text_tokenizer
    THINK_O = tok.convert_tokens_to_ids("<think>")

    layer = model.text_model.model.layers[args.layer]
    state = {"alpha": 0.0, "gen_step": 0}
    def hook(module, inp, out):
        if state["alpha"] == 0.0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] == 1:                                  # generation step (not the prompt prefill)
            state["gen_step"] += 1
            if state["gen_step"] <= args.steer_after:        # DECISION POINT: don't clamp the first N tokens
                return out
            hf = h.float()
            # SAE has normalize_input=True: activations z and decoder dirs d_f live in NORMALIZED (per-token
            # zero-mean/unit-var) space. So the injected delta must be DENORMALIZED (x std) before adding to the
            # raw residual, else the magnitude is wrong by each token's std (direction is right, scale was not).
            _, info = sae._normalize(hf); std = info["std"]              # [.., 1] per-token std
            if args.clamp_mode == "set":                     # clamp EACH feature's activation to alpha (Jared-style)
                z = sae.encode(hf)[..., feats_t]                         # [.., n_feat] normalized-space activations
                target = state["alpha"] * scale_vec                      # per-feature p95-style target (or scalar if scale=1)
                delta = ((target - z).unsqueeze(-1) * D_mat).sum(-2)     # normalized-space delta
                h = h + (delta * std).to(h.dtype)                        # denormalize -> raw residual space
            else:                                            # crude activation addition along the subspace
                h = h + (state["alpha"] * D_sum * std).to(h.dtype)
        return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h
    layer.register_forward_hook(hook)

    _, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=2000)
    ds = {"validation": val_ds, "test": test_ds}[args.split].select(range(args.num_proteins))
    collate = brp_data.make_collate_fn(tok, 10000, 2000)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    words2 = [w.lower() for w in args.concept_words_2.split(",")]
    def count_words(txt):
        t = txt.lower()
        return sum(t.count(w) for w in words)
    def count_words2(txt):
        t = txt.lower()
        return sum(t.count(w) for w in words2)

    _vow = re.compile(r"[aeiouy]")
    def coherence(txt):
        """QUANTITATIVE coherence, 3 orthogonal degeneracy signals (higher distinct3 / lower others = better):
        distinct3 = unique word-trigrams / total  (catches REPETITION LOOPS; ppl would miss these);
        nonascii  = frac non-space chars that are non-ASCII (catches foreign-token injection 发育/卵巢);
        gibber    = frac alphabetic words that are non-wordish (no vowel / len<2; catches SALAD IQNDQ/ndwton)."""
        w = txt.split()
        tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
        d3 = len(set(tri)) / len(tri) if tri else 1.0
        chars = [c for c in txt if not c.isspace()]
        nonascii = sum(1 for c in chars if ord(c) > 127) / max(1, len(chars))
        aw = [x for x in re.findall(r"[A-Za-z]+", txt) if len(x) >= 1]
        gib = sum(1 for x in aw if len(x) < 2 or not _vow.search(x.lower())) / max(1, len(aw))
        return d3, nonascii, gib
    def is_coherent(d3, na, gib):   # thresholds: intact-ish English reasoning, not loop/foreign/salad
        return d3 >= 0.55 and na <= 0.02 and gib <= 0.20

    passthru = ("protein_sequences", "batch_idx_map", "go_aspects", "structure_coords")
    totals = {a: 0 for a in alphas}; totals2 = {a: 0 for a in alphas}
    coh_c1 = {a: 0 for a in alphas}                          # concept words ONLY from coherent generations
    d3s = {a: [] for a in alphas}; nas = {a: [] for a in alphas}; gibs = {a: [] for a in alphas}
    ncoh = {a: 0 for a in alphas}; ntot = 0; nshown = 0; gen_records = []
    for bi, batch in enumerate(loader):
        ids = batch["input_ids"][0]
        idl = ids.tolist()
        if THINK_O not in idl:
            continue
        cut = idl.index(THINK_O) + 1                         # keep prompt + <think>, generate reasoning
        inp = ids[:cut].unsqueeze(0).to(dev)
        am = torch.ones_like(inp)
        kw = {k: batch[k] for k in passthru if k in batch}
        gen = {}; ntot += 1
        for a in alphas:
            state["alpha"] = a; state["gen_step"] = 0            # reset decision-point counter each generation
            with torch.no_grad():
                out = model.generate(input_ids=inp, attention_mask=am, max_new_tokens=args.max_new,
                                     do_sample=False, **kw)
            new = out[0][inp.shape[1]:] if out.shape[1] > inp.shape[1] else out[0]
            txt = tok.decode(new, skip_special_tokens=True)
            gen[a] = txt; totals[a] += count_words(txt); totals2[a] += count_words2(txt)
            d3, na, gib = coherence(txt); d3s[a].append(d3); nas[a].append(na); gibs[a].append(gib)
            if is_coherent(d3, na, gib):                      # count concept words ONLY when text is coherent
                ncoh[a] += 1; coh_c1[a] += count_words(txt)
            if args.save_gens:
                gen_records.append({"protein": bi, "alpha": a, "text": txt, "c1": count_words(txt),
                                    "distinct3": round(d3, 3), "nonascii": round(na, 3), "gibber": round(gib, 3),
                                    "lex_coherent": bool(is_coherent(d3, na, gib))})
        state["alpha"] = 0.0
        if nshown < 3:
            print(f"\n===== protein {bi} =====")
            for a in alphas:
                d3, na, gib = coherence(gen[a])
                print(f"  [a={a:.0f}] c1={count_words(gen[a])} d3={d3:.2f} na={na:.2f} gib={gib:.2f} "
                      f"{'COH' if is_coherent(d3,na,gib) else 'deg'}: {gen[a][:200]}")
            nshown += 1
        if bi % 2 == 0:
            print(f"  ...{bi}/{len(ds)}  raw_c1={{{', '.join(f'{a:.0f}:{totals[a]}' for a in alphas)}}}", flush=True)

    mean = lambda L: sum(L) / len(L) if L else 0.0
    print(f"\n=== steering vs COHERENCE ({ntot} proteins) — the honest table ===")
    print(f"  {'alpha':>5} {'raw_c1':>7} {'raw_c2':>7} | {'distinct3':>9} {'nonascii':>8} {'gibber':>7} "
          f"{'%coherent':>9} | {'coh_c1':>7}")
    print(f"  {'':>5} {'(target)':>7} {'(ctrl)':>7} | {'↑good':>9} {'↓good':>8} {'↓good':>7} {'of gens':>9} | {'coherent-only':>7}")
    for a in alphas:
        print(f"  {a:>5.0f} {totals[a]:>7} {totals2[a]:>7} | {mean(d3s[a]):>9.2f} {mean(nas[a]):>8.3f} "
              f"{mean(gibs[a]):>7.2f} {100*ncoh[a]/max(1,ntot):>8.0f}% | {coh_c1[a]:>7}")
    print("\ncoh_c1 = target concept words counted ONLY in generations that pass the coherence bar")
    print("(distinct3>=0.55, nonascii<=0.02, gibber<=0.20). If coh_c1 stays ~0 while raw_c1 is high,")
    print("the 'steering' is repetition-loop / salad injection, NOT coherent reasoning.")
    if args.save_gens:
        import json as _json
        with open(args.save_gens, "w") as _f:
            for r in gen_records:
                _f.write(_json.dumps(r) + "\n")
        print(f"[saved] {len(gen_records)} full generations -> {args.save_gens}")


if __name__ == "__main__":
    main()
