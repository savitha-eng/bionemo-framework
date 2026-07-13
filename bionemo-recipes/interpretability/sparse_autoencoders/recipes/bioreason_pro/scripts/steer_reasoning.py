#!/usr/bin/env python
"""Causal steering of a REASONING SAE feature in BioReason-Pro (Jared-style clamp sweep).

We clamp a single SAE feature's decoder direction into the residual stream at layer L on the response
(reasoning+answer) positions, teacher-forcing held-out proteins, and measure how the model's
probability of emitting the feature's CONCEPT tokens (e.g. "mitochond"/"mitochondrion") shifts as a
function of clamp strength alpha. A dose-dependent rise that a matched-norm RANDOM direction does not
produce => the feature causally promotes that concept in the model's reasoning.

This is the generation-relevant test the extraction note calls for (probes ran on teacher-forced
distilled targets; this intervenes on the model's own next-token distribution).

Usage:
  steer_reasoning.py --sae <ckpt> --feature 27761 --layer 16 --concept "mitochond,mitochondrion" \
     --num-proteins 60 --alphas 0,4,8,16,32 --out steer_f27761.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--feature", type=int, required=True)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--concept", required=True, help="comma-sep concept token strings, e.g. 'mitochond,mitochondrion'")
    p.add_argument("--alphas", default="0,4,8,16,32")
    p.add_argument("--num-proteins", type=int, default=60)
    p.add_argument("--split", default="validation")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default="/data/savithas/bioreason-pro")
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    p.add_argument("--out", default="steer_result.json")
    args = p.parse_args()
    dev = "cuda"
    alphas = [float(x) for x in args.alphas.split(",")]

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft
    from sae.architectures import TopKSAE

    # ---- SAE decoder direction for the feature ----
    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    W = sae.decoder.weight.detach()               # (input_dim, hidden_dim); column f = what feature f writes
    d_f = (W[:, args.feature] if W.shape[1] != W.shape[0] and W.shape[0] == 2560 else W[args.feature]).float()
    # RAW decoder column: adding alpha*d_f == clamping feature f to activation alpha (interpretable units,
    # feature's natural max is ~30). Control = random direction scaled to the SAME norm (fair specificity test).
    g = torch.Generator().manual_seed(args.feature)
    d_rand = torch.randn(d_f.shape, generator=g); d_rand = d_rand / (d_rand.norm() + 1e-8) * d_f.norm()
    print(f"[steer] feature {args.feature}, decoder-col norm={d_f.norm():.3f}, dim={d_f.numel()}, layer {args.layer}")

    # ---- model + data ----
    model = load_bioreason_pro_sft(ckpt_dir=args.ckpt_dir, bioreason_pro_root=args.bioreason_root,
                                   device=dev, max_length_text=args.max_length_text,
                                   max_length_protein=args.max_length_protein).eval()
    tok = model.text_tokenizer
    THINK_O = tok.convert_tokens_to_ids("<think>")
    # --concept = concept groups separated by ';'; each group is comma-sep synonyms. First group = the
    # feature's TARGET concept; the rest are DISTRACTORS. Selectivity = does the clamp raise the target
    # more than the distractors (a generic perturbation would raise all of them).
    groups = {}
    for grp in args.concept.split(";"):
        words = grp.split(","); ids = set()
        for w in words:
            ids.update(tok.encode(w, add_special_tokens=False))
            ids.update(tok.encode(" " + w, add_special_tokens=False))
        groups[words[0]] = torch.tensor(sorted(ids), device=dev)
    print(f"[steer] concept groups: {{ {', '.join(f'{k}:{len(v)}' for k,v in groups.items())} }}")

    _, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=args.max_length_protein)
    ds = {"validation": val_ds, "test": test_ds}[args.split].select(range(args.num_proteins))
    collate = brp_data.make_collate_fn(tok, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    layer = model.text_model.model.layers[args.layer]
    state = {"vec": None}   # (H,) direction*alpha to add on response positions; None = off
    resp_mask = {"m": None}

    def hook(module, inp, out):
        if state["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        add = state["vec"].to(h.dtype)                      # stay in residual dtype (bf16) — else downstream matmuls break
        m = resp_mask["m"]
        h = h + (m.to(h.dtype).unsqueeze(-1) * add if m is not None else add)
        return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h
    layer.register_forward_hook(hook)

    def concept_logprob(batch):                                  # -> {group: mean logP(any token in group)}
        with torch.no_grad():
            out = model(**{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()})
        logits = out.logits[0]
        ids = batch["input_ids"][0].to(dev)
        idl = ids.tolist(); oi = idl.index(THINK_O) if THINK_O in idl else 0
        rmask = (torch.arange(logits.shape[0], device=dev) >= oi)
        lp = F.log_softmax(logits[rmask], dim=-1)
        return {name: float(lp[:, gids].logsumexp(-1).mean()) for name, gids in groups.items()}

    gnames = list(groups)
    results = {"feature": args.feature, "layer": args.layer, "concept": args.concept, "groups": gnames,
               "alphas": alphas,
               "steered": {n: {a: [] for a in alphas} for n in gnames},
               "control": {n: {a: [] for a in alphas} for n in gnames}}
    for bi, batch in enumerate(loader):
        try:
            ids = batch["input_ids"][0]
            idl = ids.tolist(); oi = idl.index(THINK_O) if THINK_O in idl else 0
            resp_mask["m"] = (torch.arange(len(ids), device=dev) >= oi).float()
            for a in alphas:
                state["vec"] = (a * d_f).to(dev);   sp = concept_logprob(batch)
                state["vec"] = (a * d_rand).to(dev); cp = concept_logprob(batch)
                for n in gnames:
                    results["steered"][n][a].append(sp[n]); results["control"][n][a].append(cp[n])
            state["vec"] = None
        except Exception as e:
            print(f"  protein {bi} skipped: {e}")
        if bi % 10 == 0:
            print(f"  {bi}/{len(ds)}", flush=True)

    summ = {}
    for n in gnames:
        print(f"\n=== [{n}] dose-response: mean logP({n}) on response positions ===")
        print(f"{'alpha':>6} {'STEERED':>10} {'control':>10} {'steer-ctrl':>11}")
        summ[n] = {}
        for a in alphas:
            s = float(np.mean(results["steered"][n][a])); c = float(np.mean(results["control"][n][a]))
            summ[n][a] = {"steered": s, "control": c, "diff": s - c}
            print(f"{a:>6.0f} {s:>10.3f} {c:>10.3f} {s-c:>+11.3f}")
    tgt = gnames[0]
    print(f"\n=== SELECTIVITY at alpha={alphas[-1]} (steer-ctrl; target='{tgt}' should be >> distractors) ===")
    for n in gnames:
        print(f"  {n:16} steer-ctrl = {summ[n][alphas[-1]]['diff']:+.3f}")
    print("target >> distractors => the clamp promotes THIS concept specifically, not a generic lift.")
    results["summary"] = summ
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
