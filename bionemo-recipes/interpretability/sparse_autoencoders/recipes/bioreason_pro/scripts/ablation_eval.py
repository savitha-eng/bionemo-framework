#!/usr/bin/env python
"""Causal input-ablation: does the protein/GO EMBEDDING channel shape the model's text output?

Teacher-forces held-out proteins and measures next-token cross-entropy on the model's REASONING and
ANSWER tokens, under: intact / protein-embeds-zeroed / go-embeds-zeroed / both-zeroed. dCE = CE(ablated)
- CE(intact). dCE > 0 => the bio channel causally improves the model's own text => fusion (deeper than
a tool-caller, whose bio info lives only in the text prompt and would give dCE ~ 0).
Usage: ablation_eval.py --num-proteins 200 [--split validation]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default="/data/savithas/bioreason-pro")
    p.add_argument("--split", default="validation")
    p.add_argument("--num-proteins", type=int, default=200)
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    args = p.parse_args()
    dev = "cuda"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft

    model = load_bioreason_pro_sft(ckpt_dir=args.ckpt_dir, bioreason_pro_root=args.bioreason_root,
                                   device=dev, max_length_text=args.max_length_text,
                                   max_length_protein=args.max_length_protein)
    model.eval()
    tok = model.text_tokenizer
    THINK_O = tok.convert_tokens_to_ids("<think>"); THINK_C = tok.convert_tokens_to_ids("</think>")
    pad_tok = tok.pad_token_id

    _, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=args.max_length_protein)
    ds = {"validation": val_ds, "test": test_ds}[args.split].select(range(args.num_proteins))
    collate = brp_data.make_collate_fn(tok, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    orig_prot = model.process_protein_embeddings
    orig_go = model.process_go_aspects
    def zero_list(fn):
        def w(*a, **k):
            r = fn(*a, **k)
            return [torch.zeros_like(e) for e in r] if r is not None else r
        return w

    CONDS = {"intact": (False, False), "no_protein": (True, False),
             "no_go": (False, True), "no_both": (True, True)}
    acc = {c: {"reasoning": [0.0, 0], "answer": [0.0, 0]} for c in CONDS}
    per = {role: [] for role in ("reasoning", "answer")}  # per-protein dCE(no_protein - intact) for significance

    def per_token_ce(batch, zero_p, zero_g):
        model.process_protein_embeddings = zero_list(orig_prot) if zero_p else orig_prot
        model.process_go_aspects = zero_list(orig_go) if zero_g else orig_go
        with torch.no_grad():
            out = model(**{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()})
        logits = out.logits[0]                                  # (L, V)
        ids = batch["input_ids"][0].to(dev)
        ce = F.cross_entropy(logits[:-1], ids[1:], reduction="none")  # next-token CE, len L-1
        idl = ids.tolist()
        oi = idl.index(THINK_O) if THINK_O in idl else -1
        ci = idl.index(THINK_C) if THINK_C in idl else -1
        out_ce = {}
        if oi >= 0 and ci >= 0:
            # positions predicting reasoning tokens (oi+1..ci) and answer tokens (ci+1..end, non-pad)
            r = ce[oi:ci - 1]
            a_end = len(ids) - 1
            a = ce[ci:a_end]
            out_ce["reasoning"] = float(r.mean()) if r.numel() else None
            out_ce["answer"] = float(a.mean()) if a.numel() else None
        return out_ce

    for bi, batch in enumerate(loader):
        try:
            ces = {}
            for c, (zp, zg) in CONDS.items():
                ce = per_token_ce(batch, zp, zg)
                ces[c] = ce
                for role in ("reasoning", "answer"):
                    if ce.get(role) is not None:
                        acc[c][role][0] += ce[role]; acc[c][role][1] += 1
            for role in ("reasoning", "answer"):  # paired per-protein dCE for significance
                if ces["intact"].get(role) is not None and ces["no_protein"].get(role) is not None:
                    per[role].append(ces["no_protein"][role] - ces["intact"][role])
        except Exception as e:
            print(f"  protein {bi} skipped: {e}")
        if bi % 25 == 0:
            print(f"  {bi}/{len(ds)}", flush=True)

    print("\n=== mean next-token CE by condition (lower = predicts its own text better) ===")
    base = {role: acc["intact"][role][0] / max(acc["intact"][role][1], 1) for role in ("reasoning", "answer")}
    for c in CONDS:
        for role in ("reasoning", "answer"):
            s, n = acc[c][role]
            m = s / max(n, 1)
            d = m - base[role]
            print(f"  {c:11} {role:9} CE={m:.4f}  dCE_vs_intact={d:+.4f}  (n={n})")
    print("\n=== protein-ablation significance (paired per-protein dCE = no_protein - intact) ===")
    for role in ("reasoning", "answer"):
        d = np.array(per[role])
        if len(d):
            mean, std = d.mean(), d.std(ddof=1)
            se = std / np.sqrt(len(d)); t = mean / se if se > 0 else float("nan")
            print(f"  {role:9} mean dCE={mean:+.4f} ± {std:.4f} (sd) | t={t:.1f} | "
                  f"%proteins dCE>0: {100*(d>0).mean():.1f}% (n={len(d)})")
    print("\ndCE > 0 under ablation => the bio embedding causally helps the model generate that text (fusion).")


if __name__ == "__main__":
    main()
