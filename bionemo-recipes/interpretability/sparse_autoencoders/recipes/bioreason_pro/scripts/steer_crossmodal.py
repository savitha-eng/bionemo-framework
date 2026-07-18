#!/usr/bin/env python
"""CROSS-MODAL 2x2 factorial steering: does a BIO (residue-band) feature causally drive the reasoning?

A paired feature = (bio feature that fires on protein residues) + (reasoning feature that co-fires across
proteins, r>=0.8). We steer them in a 2x2:  reasoning-feature {off,on} x residue-feature {off,on}
  ctrl   : no steering
  reason : clamp the reasoning feature on GENERATED tokens        (Arm 1)
  residue: clamp the bio feature on the protein-band PREFILL slots (Arm 2)  <- the novel injection
  both   : residue on prefill + reasoning on generation           (Arm 3)
Residue-feature causal effect = main effect of residue injection = (residue+both) - (ctrl+reason).
Interaction (both-reason) - (residue-ctrl) = is the coupling context-dependent.

Residue injection: protein slots are input_ids==model.protein_token_id; on prefill we set-clamp the bio
feature's activation there to dose*ma_protein[f] (normalized-space delta, denormalized to the residual).
Reasoning clamp reuses the validated set-mode + p95 dose (dose*ma_text[f]) after --steer-after tokens.

Usage: steer_crossmodal.py --sae <l30.pt> --layer 30 --reason-features 3184 --residue-features 7369 \
   --concept-words gpcr,rhodopsin,transmembrane,receptor,ligand --scale-file load_bearing_l30.npz \
   --reason-dose 6 --residue-dose 6 --num-proteins 8 --save-gens D10_gpcr_2x2.jsonl
"""
import argparse, sys, re, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")
CONDS = ["ctrl", "reason", "residue", "both"]   # (reason_on, residue_on): (0,0)(1,0)(0,1)(1,1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--layer", type=int, default=30)
    p.add_argument("--reason-features", required=True, help="comma-sep reasoning-band feature(s)")
    p.add_argument("--residue-features", required=True, help="comma-sep bio/residue-band feature(s)")
    p.add_argument("--concept-words", required=True, help="comma-sep concept words to count in the reasoning")
    p.add_argument("--scale-file", required=True);
    p.add_argument("--reason-scale-key", default="ma_text"); p.add_argument("--residue-scale-key", default="ma_protein")
    p.add_argument("--reason-dose", type=float, default=6.0, help="reasoning clamp = dose*ma_text[f]")
    p.add_argument("--residue-dose", type=float, default=6.0, help="residue clamp = dose*ma_protein[f]")
    p.add_argument("--steer-after", type=int, default=25)
    p.add_argument("--num-proteins", type=int, default=8); p.add_argument("--max-new", type=int, default=180)
    p.add_argument("--split", default="validation")
    p.add_argument("--exclude-domain", default="", help="Q2: keep only proteins whose interpro_ids LACK this "
                   "IPR id (clean concept-absent injection test)")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT); p.add_argument("--bioreason-root", default="/data/savithas/bioreason-pro")
    p.add_argument("--save-gens", default="")
    args = p.parse_args(); dev = "cuda"
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
    col_layout = (W.shape[1] != W.shape[0] and W.shape[0] == 2560)
    dir_of = lambda f: (W[:, f] if col_layout else W[f]).float().to(dev)
    sc = np.load(args.scale_file)

    def build(feat_str, scale_key, dose):
        feats = [int(x) for x in feat_str.split(",") if x.strip() != ""]
        ft = torch.tensor(feats, device=dev)
        D = torch.stack([dir_of(f) for f in feats], 0)                 # [n,2560]
        target = torch.tensor([dose * float(sc[scale_key][f]) for f in feats], device=dev)  # per-feature clamp target
        return feats, ft, D, target
    r_feats, r_ft, r_D, r_target = build(args.reason_features, args.reason_scale_key, args.reason_dose)
    d_feats, d_ft, d_D, d_target = build(args.residue_features, args.residue_scale_key, args.residue_dose)
    print(f"[xmodal] reason {r_feats} clamp->{[round(float(x),1) for x in r_target]} (gen tokens); "
          f"residue {d_feats} clamp->{[round(float(x),1) for x in d_target]} (protein prefill)", flush=True)

    model = load_bioreason_pro_sft(ckpt_dir=args.ckpt_dir, bioreason_pro_root=args.bioreason_root,
                                   device=dev, max_length_text=10000, max_length_protein=2000).eval()
    tok = model.text_tokenizer
    THINK_O = tok.convert_tokens_to_ids("<think>")
    PROT_ID = model.protein_token_id
    layer = model.text_model.model.layers[args.layer]

    st = {"reason_on": False, "residue_on": False, "gen_step": 0, "prot_mask": None}
    def clamp(h, ft, D, target, pos_mask):
        """set-clamp features ft to `target` at positions where pos_mask (or all if None); normalized->raw."""
        hf = h.float(); _, info = sae._normalize(hf); std = info["std"]
        z = sae.encode(hf)[..., ft]                                     # [b,s,n]
        delta = ((target - z).unsqueeze(-1) * D).sum(-2)               # [b,s,2560] normalized-space
        add = (delta * std)
        if pos_mask is not None:
            add = add * pos_mask.unsqueeze(-1).to(add.dtype)
        return h + add.to(h.dtype)
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:                                             # PREFILL: residue injection at protein slots
            if st["residue_on"] and st["prot_mask"] is not None and st["prot_mask"].any():
                h = clamp(h, d_ft, d_D, d_target, st["prot_mask"])
        else:                                                          # GENERATION: reasoning clamp after decision pt
            st["gen_step"] += 1
            if st["reason_on"] and st["gen_step"] > args.steer_after:
                h = clamp(h, r_ft, r_D, r_target, None)
        return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h
    layer.register_forward_hook(hook)

    _, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=2000)
    ds = {"validation": val_ds, "test": test_ds}[args.split]
    if args.exclude_domain:                                        # Q2: concept-absent proteins only
        def lacks(row):
            ipr = row["interpro_ids"] or []
            return args.exclude_domain not in (ipr if isinstance(ipr, (list, tuple)) else [ipr])
        idx = [i for i in range(len(ds)) if lacks(ds[i])][: args.num_proteins]
        ds = ds.select(idx)
        print(f"[xmodal] Q2: {len(ds)} proteins LACKING {args.exclude_domain} (clean injection test)", flush=True)
    else:
        ds = ds.select(range(args.num_proteins))
    collate = brp_data.make_collate_fn(tok, 10000, 2000)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    _vow = re.compile(r"[aeiouy]")
    def coherence(txt):
        w = txt.split(); tri = [tuple(w[i:i+3]) for i in range(len(w)-2)]
        d3 = len(set(tri))/len(tri) if tri else 1.0
        chars = [c for c in txt if not c.isspace()]
        na = sum(1 for c in chars if ord(c) > 127)/max(1, len(chars))
        aw = re.findall(r"[A-Za-z]+", txt)
        gib = sum(1 for x in aw if len(x) < 2 or not _vow.search(x.lower()))/max(1, len(aw))
        return d3, na, gib
    coh_ok = lambda d3, na, gib: d3 >= 0.55 and na <= 0.02 and gib <= 0.20
    cw = lambda t: sum(t.lower().count(w) for w in words)

    passthru = ("protein_sequences", "batch_idx_map", "go_aspects", "structure_coords")
    agg = {c: {"c1": 0, "coh_c1": 0, "ncoh": 0} for c in CONDS}; ntot = 0; recs = []
    for bi, batch in enumerate(loader):
        ids = batch["input_ids"][0]; idl = ids.tolist()
        if THINK_O not in idl: continue
        cut = idl.index(THINK_O) + 1
        inp = ids[:cut].unsqueeze(0).to(dev); am = torch.ones_like(inp)
        prot_mask = (inp[0] == PROT_ID)
        st["prot_mask"] = prot_mask
        kw = {k: batch[k] for k in passthru if k in batch}; ntot += 1
        n_prot = int(prot_mask.sum())
        for c in CONDS:
            st["reason_on"] = c in ("reason", "both"); st["residue_on"] = c in ("residue", "both"); st["gen_step"] = 0
            with torch.no_grad():
                out = model.generate(input_ids=inp, attention_mask=am, max_new_tokens=args.max_new, do_sample=False, **kw)
            new = out[0][inp.shape[1]:] if out.shape[1] > inp.shape[1] else out[0]
            txt = tok.decode(new, skip_special_tokens=True)
            d3, na, gib = coherence(txt); ok = coh_ok(d3, na, gib); n1 = cw(txt)
            agg[c]["c1"] += n1;
            if ok: agg[c]["ncoh"] += 1; agg[c]["coh_c1"] += n1
            if args.save_gens:
                recs.append({"protein": bi, "cond": c, "n_prot_slots": n_prot, "text": txt, "c1": n1,
                             "distinct3": round(d3, 3), "nonascii": round(na, 3), "gibber": round(gib, 3),
                             "coherent": bool(ok)})
        st["reason_on"] = st["residue_on"] = False
        if bi < 2:
            print(f"\n== protein {bi} ({n_prot} protein slots) ==")
            for c in CONDS:
                r = [x for x in recs if x["protein"] == bi and x["cond"] == c][0]
                print(f"  {c:8} c1={r['c1']} d3={r['distinct3']} {'COH' if r['coherent'] else 'deg'}: {r['text'][:150]}")
        if bi % 2 == 0: print(f"  ...{bi}/{len(ds)}", flush=True)

    print(f"\n=== CROSS-MODAL 2x2 ({ntot} proteins): concept='{args.concept_words}' ===")
    print(f"  {'cond':8} {'raw_c1':>7} {'coh_c1':>7} {'%coh':>6}")
    for c in CONDS:
        print(f"  {c:8} {agg[c]['c1']:>7} {agg[c]['coh_c1']:>7} {100*agg[c]['ncoh']/max(1,ntot):>5.0f}%")
    mainR = (agg["residue"]["coh_c1"] + agg["both"]["coh_c1"]) - (agg["ctrl"]["coh_c1"] + agg["reason"]["coh_c1"])
    mainT = (agg["reason"]["coh_c1"] + agg["both"]["coh_c1"]) - (agg["ctrl"]["coh_c1"] + agg["residue"]["coh_c1"])
    inter = (agg["both"]["coh_c1"] - agg["reason"]["coh_c1"]) - (agg["residue"]["coh_c1"] - agg["ctrl"]["coh_c1"])
    print(f"\n  MAIN EFFECT residue (structure->reasoning, coherent concept words): {mainR:+d}")
    print(f"  MAIN EFFECT reason  (text feature writes concept)                 : {mainT:+d}")
    print(f"  INTERACTION (residue effect bigger when reasoning on)             : {inter:+d}")
    print("  -> residue main-effect ~0 => bio feature causally inert for output (read-only, prompt-mediated).")
    if args.save_gens:
        with open(args.save_gens, "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        print(f"[saved] {len(recs)} gens -> {args.save_gens}")


if __name__ == "__main__":
    main()
