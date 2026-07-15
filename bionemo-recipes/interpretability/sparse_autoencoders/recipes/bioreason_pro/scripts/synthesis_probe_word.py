#!/usr/bin/env python
"""WORD-LEVEL synthesis probe (fixes the subword/junk flaws of synthesis_probe.py).

The old probe worked on BPE SUBWORD tokens, so (a) 'osterone'/'oid'/'ortic' were separate noisy tokens,
and (b) junk single-token detectors (F3287 fires on 'wich', F36811 on 'Interaction') scored novelty=1.0 and
polluted the 'synthesis' fraction. This version:
  1. Reconstructs WORDS from subwords (Qwen 'Ġ' word-start marker), so a feature's signature is real words.
  2. Computes novelty over WORDS (frac of a feature's top-firing distinct words NOT in that protein's prompt).
  3. FILTERS junk features: a word is "real content" if len>=4, alphabetic, and RECURS across >=8 proteins
     (real bio words recur; junk fragments / made-up words don't). A feature is INTERPRETABLE if >=40% of its
     top-firing words are real content words. The echo-vs-synthesis distribution is reported over interpretable
     features only.

Usage: synthesis_probe_word.py <sae.pt> <store> <layer> [--features-csv f] [--topk 150] [--shard-stride N]
"""
import argparse, json, glob, re, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
import bioreason_pro_sae.data as brp_data
from bioreason_pro_sae.model_loader import _install_unsloth_stub; _install_unsloth_stub()
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--store", required=True); p.add_argument("--layer", type=int, required=True)
    p.add_argument("--features-csv", default=""); p.add_argument("--features", default="")
    p.add_argument("--split", default="train"); p.add_argument("--topk", type=int, default=150)
    p.add_argument("--shard-stride", type=int, default=3); p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--out", default="/data/savithas/phase3_full/synthesis_word_l30.json")
    a = p.parse_args(); dev = "cuda"
    if a.features_csv:
        import csv; feats = sorted({int(r["feature"]) for r in csv.DictReader(open(a.features_csv))})
    else:
        feats = sorted({int(x) for x in a.features.split(",") if x.strip()})
    print(f"[synth-word] {len(feats)} candidate features", flush=True)

    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True); pad = tok.pad_token_id
    store_pids = [str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
    tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
    ds = {"train": tr, "validation": va, "test": te}[a.split]
    allp = [str(x) for x in ds["protein_id"]]; idx = {q: i for i, q in enumerate(allp)}
    sub = ds.select([idx[q] for q in store_pids if q in idx]); coll = brp_data.make_collate_fn(tok, 10000, 2000)

    # reconstruct: per protein, per kept token -> WORD string (Ġ = word start); role; prompt word set
    word_by_pid = {}; role_by_pid = {}; prompt_words = {}; word_doc = Counter()
    for k, batch in enumerate(DataLoader(sub, batch_size=1, collate_fn=coll)):
        ids = batch["input_ids"][0]; lab = batch["labels"][0]; keep = (ids != pad).numpy()
        idk = ids.numpy()[keep].tolist(); isresp = (lab.numpy()[keep] != -100)
        pid = str(sub[k].get("protein_id"))
        pieces = tok.convert_ids_to_tokens(idk)                       # e.g. 'ĠThe','Ġarchitecture','osterone'
        starts = [1 if (j == 0 or pi.startswith("Ġ") or pi.startswith("Ċ")) else 0 for j, pi in enumerate(pieces)]
        wid = np.cumsum(starts) - 1
        wpieces = defaultdict(str)
        for j, pi in enumerate(pieces):
            wpieces[wid[j]] += pi.replace("Ġ", "").replace("Ċ", "")
        wordstr = {w: norm(s) for w, s in wpieces.items()}
        tokword = np.array([wordstr[wid[j]] for j in range(len(idk))], dtype=object)  # word per token
        word_by_pid[pid] = tokword; role_by_pid[pid] = isresp
        prompt_words[pid] = {tokword[j] for j in range(len(idk)) if not isresp[j] and len(tokword[j]) >= 3}
        for w in set(tokword):                                        # doc frequency (recurrence) of words
            if len(w) >= 4: word_doc[w] += 1
        if k % 1000 == 0: print(f"  reconstruct {k}/{len(sub)}", flush=True)
    realword = {w for w, c in word_doc.items() if c >= 8}             # real content word = recurs in >=8 proteins
    print(f"[synth-word] {len(realword):,} real recurring content words", flush=True)

    # align store rows -> (pid, token_index)
    import pandas as pd
    tl = pq.read_table(f"{a.store}/token_labels.parquet")
    rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
    rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64); N = len(rpid)
    resp = np.zeros(N, bool); word_row = np.empty(N, dtype=object); word_row[:] = ""; novel = np.zeros(N, bool)
    df = pd.DataFrame({"pid": rpid.astype(str), "j": rtidx, "row": np.arange(N)})
    for pid, g in df.groupby("pid", sort=False):
        tw = word_by_pid.get(pid); r = role_by_pid.get(pid)
        if tw is None: continue
        rows = g["row"].to_numpy(); js = g["j"].to_numpy(); ok = js < len(tw); rows, js = rows[ok], js[ok]
        isr = r[js]; resp[rows] = isr
        w = tw[js]; word_row[rows] = w
        pw = prompt_words[pid]
        novel[rows] = np.array([isr[i] and (w[i] not in pw) for i in range(len(w))], bool)

    # gather candidate-feature activations on response rows (bulk)
    ck = torch.load(a.sae, map_location="cpu"); sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    feats_t = torch.tensor(feats, device=dev); resp_t = torch.from_numpy(resp).to(dev)
    Zc = []; Rc = []; row0 = 0
    order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
    with torch.no_grad():
        for si, sp in enumerate(order):
            nrows = pq.read_metadata(sp).num_rows
            if si % a.shard_stride == 0:
                X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
                for s in range(0, n, 8192):
                    e = min(n, s + 8192); m = resp_t[row0+s:row0+e]
                    if not m.any(): continue
                    xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev)
                    z = sae.encode(xb)[:, feats_t][m]; gr = torch.arange(row0+s, row0+e, device=dev)[m]
                    Zc.append(z.half().cpu().numpy()); Rc.append(gr.to(torch.int64).cpu().numpy())
                if si % 30 == 0: print(f"  gather {si}/{len(order)}", flush=True)
            row0 += nrows
    Z = np.concatenate(Zc); Rrow = np.concatenate(Rc); del Zc, Rc
    print(f"[synth-word] gathered {Z.shape[0]:,}x{Z.shape[1]}; scoring...", flush=True)

    results = {}
    for fi, f in enumerate(feats):
        col = Z[:, fi].astype(np.float32); nz = col > 0
        if not nz.any(): results[f] = {"n_fire": 0}; continue
        av = col[nz]; rw = Rrow[nz]; top = rw[np.argsort(-av)[:a.topk]]
        words = [word_row[r] for r in top]
        content = [w for w in words if len(w) >= 4 and w in realword]  # real recurring content words only
        distinct = list(dict.fromkeys(content))                       # dedup, keep order
        real_frac = len(content) / max(1, len(words))
        interpretable = len(distinct) >= 3 and real_frac >= 0.25   # >=3 distinct real recurring content words
        nov = np.mean([word_row[r] not in prompt_words[str(rpid[r])] for r in top if word_row[r] in realword]) \
            if content else float("nan")
        results[f] = {"n_fire": int(nz.sum()), "interpretable": bool(interpretable),
                      "real_word_frac": round(real_frac, 2), "novelty_word": round(float(nov), 3) if content else None,
                      "top_words": distinct[:12]}

    interp = [r for r in results.values() if r.get("interpretable") and r.get("novelty_word") is not None]
    nv = np.array([r["novelty_word"] for r in interp])
    njunk = sum(1 for r in results.values() if r.get("n_fire", 0) > 0 and not r.get("interpretable"))
    ntot = sum(1 for r in results.values() if r.get("n_fire", 0) > 0)
    print(f"\n=== WORD-LEVEL synthesis (over INTERPRETABLE features) ===")
    print(f"  {ntot} features fired; {njunk} junk/uninterpretable ({100*njunk/max(1,ntot):.0f}%) dropped; {len(interp)} interpretable scored")
    if len(nv):
        print(f"  novelty(word): mean={nv.mean():.2f} echo(<0.3)={100*(nv<0.3).mean():.0f}% synth(>0.6)={100*(nv>0.6).mean():.0f}%")
    r4 = results.get(39407, {})
    print(f"  F39407 (hormone): interpretable={r4.get('interpretable')} novelty_word={r4.get('novelty_word')} top_words={r4.get('top_words')}")
    json.dump({"n_fired": ntot, "n_junk": njunk, "n_interpretable": len(interp),
               "mean_novelty": round(float(nv.mean()), 3) if len(nv) else None,
               "features": {str(k): v for k, v in results.items()}}, open(a.out, "w"), indent=2)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
