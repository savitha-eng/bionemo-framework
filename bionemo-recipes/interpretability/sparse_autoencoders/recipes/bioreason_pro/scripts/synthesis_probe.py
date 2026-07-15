#!/usr/bin/env python
"""GOODFIRE-STYLE SYNTHESIS PROBE — separate prompt-ECHO reasoning features from SYNTHESIS features.

The circularity worry: a reasoning feature with high GO-AUROC (e.g. "response to hormone") might just
fire when the model RE-READS the hormone GO term already present in the prompt (go_pred/InterPro/UniProt
summary) — not when it SYNTHESIZES novel reasoning. GO-AUROC alone can't tell these apart.

Two non-circular signals per feature, computed over its top-activating RESPONSE (reasoning) tokens:
  1. NOVELTY  = fraction of top-firing content tokens whose word is NOT present in that example's prompt.
                low  -> echo detector (fires on restated prompt terms; circular / uninteresting)
                high -> synthesis (fires on words the model generated that the prompt didn't contain)
  2. CONNECTIVE = fraction of top-firing tokens that are inference markers (therefore/thus/suggests/
                implies/indicates/consistent/because...). Content-independent reasoning-BEHAVIOR signal
                (Goodfire R1: backtracking/verifying/committing features). High -> "the model is reasoning".

Token strings are reconstructed by re-running the exact collate (as add_role_sidecar_bystore.py) so
(protein_id, token_index) -> token id -> string, aligned to the store's activation rows.

Usage: synthesis_probe.py --sae <ckpt> --store <store> --layer 30 [--features 39407,17006,... | --features-csv <val.csv>] \
        [--topk 150] [--out synthesis_l30.json]
"""
import argparse, json, glob, re, sys
from pathlib import Path
from collections import defaultdict
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
CONNECTIVES = {  # inference / synthesis markers (single-token, lowercased, stripped)
    "therefore", "thus", "hence", "because", "since", "consequently", "accordingly", "so",
    "suggest", "suggests", "suggesting", "suggested", "imply", "implies", "implying", "implied",
    "indicate", "indicates", "indicating", "indicated", "consistent", "support", "supports",
    "supporting", "reflect", "reflects", "reflecting", "given", "thereby", "means", "likely",
    "probably", "underlie", "underlies", "drives", "enables", "allows", "contributes", "whereas",
    "however", "moreover", "furthermore", "overall", "conclude", "reasonable", "expect", "expected",
    "predict", "predicts", "suggestive", "presumably", "infer", "inferred", "consequence"}


def norm(s):
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s.strip().lower())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--features", default="")
    p.add_argument("--features-csv", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--topk", type=int, default=150)
    p.add_argument("--shard-stride", type=int, default=1, help="gather every Nth shard (subsample for speed; "
                   "top-K over a token subsample is fine for the novelty metric)")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--out", default="")
    a = p.parse_args()
    dev = "cuda"
    out = a.out or f"synthesis_l{a.layer}.json"

    if a.features_csv:
        import csv
        feats = [int(r["feature"]) for r in csv.DictReader(open(a.features_csv))]
    else:
        feats = [int(x) for x in a.features.split(",") if x.strip()]
    feats = sorted(set(feats))
    print(f"[synth] {len(feats)} candidate features")

    # ---- reconstruct token strings + prompt content-word set per protein (exact collate) ----
    tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True); pad = tok.pad_token_id
    store_pids = [str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
    tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
    ds = {"train": tr, "validation": va, "test": te}[a.split]
    allp = [str(x) for x in ds["protein_id"]]; idx = {q: i for i, q in enumerate(allp)}
    sel = [idx[q] for q in store_pids if q in idx]
    print(f"[synth] matched {len(sel)}/{len(store_pids)} store proteins in {a.split}")
    sub = ds.select(sel); coll = brp_data.make_collate_fn(tok, 10000, 2000)
    tid_by_pid = {}                                   # pid -> np.array of kept token ids (index == token_index)
    role_by_pid = {}                                  # pid -> np.array bool is_response
    prompt_words = {}                                 # pid -> set of prompt content words
    strcache = {}                                     # tid -> normalized string (decode once)
    def nstr(t):
        if t not in strcache: strcache[t] = norm(tok.decode([int(t)]))
        return strcache[t]
    for k, batch in enumerate(DataLoader(sub, batch_size=1, collate_fn=coll)):
        ids = batch["input_ids"][0]; lab = batch["labels"][0]; keep = (ids != pad).numpy()
        idk = ids.numpy()[keep]; lk = lab.numpy()[keep]
        pid = str(sub[k].get("protein_id"))
        isresp = lk != -100
        tid_by_pid[pid] = idk.astype(np.int32); role_by_pid[pid] = isresp
        pw = set()
        for t in idk[~isresp]:
            w = nstr(t)
            if len(w) >= 3 and w.isalpha(): pw.add(w)
        prompt_words[pid] = pw
        if k % 1000 == 0: print(f"  reconstruct {k}/{len(sub)}", flush=True)

    # ---- classify ONLY token ids that actually appear (batch-decode; avoids 150k full-vocab per-token decode) ----
    V = len(tok.get_vocab())
    appearing = np.unique(np.concatenate(list(tid_by_pid.values()))).astype(np.int64)
    print(f"[synth] batch-decoding {len(appearing):,} appearing token ids...", flush=True)
    decoded = tok.batch_decode([[int(t)] for t in appearing])
    conn_v = np.zeros(V, bool); content_v = np.zeros(V, bool); wid_v = np.full(V, -1, np.int64)
    word2id = {}
    for t, d in zip(appearing, decoded):
        w = norm(d)
        if w in CONNECTIVES: conn_v[t] = True
        if len(w) >= 3 and w.isalpha():
            content_v[t] = True
            wid_v[t] = word2id.setdefault(w, len(word2id))
    prompt_wids = {pid: np.fromiter((word2id[w] for w in pw if w in word2id), np.int64)
                   for pid, pw in prompt_words.items()}   # per-protein prompt word-id sets

    # ---- align store rows -> response tokens with novel/connective flags (vectorized numpy; np.isin) ----
    import pandas as pd
    tl = pq.read_table(f"{a.store}/token_labels.parquet")
    rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
    rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64)
    N = len(rpid)
    resp = np.zeros(N, bool); novel = np.zeros(N, bool); conn = np.zeros(N, bool)
    is_content = np.zeros(N, bool); tid_row = np.zeros(N, np.int32)
    df = pd.DataFrame({"pid": rpid.astype(str), "j": rtidx, "row": np.arange(N)})
    for pid, g in df.groupby("pid", sort=False):
        tids = tid_by_pid.get(pid); r = role_by_pid.get(pid)
        if tids is None: continue
        rows = g["row"].to_numpy(); js = g["j"].to_numpy()
        ok = js < len(tids); rows = rows[ok]; js = js[ok]
        t = tids[js]; tid_row[rows] = t
        isr = r[js]; resp[rows] = isr
        conn[rows] = conn_v[t] & isr
        cont = content_v[t] & isr; is_content[rows] = cont
        if cont.any():                                   # novel = content resp token whose word-id not in prompt
            pw = prompt_wids.get(pid)
            wid = wid_v[t[cont]]
            nov = np.ones(len(wid), bool) if pw is None or len(pw) == 0 else ~np.isin(wid, pw)
            novel[rows[cont]] = nov
    print(f"[synth] {int(resp.sum()):,} response tokens ({int(is_content.sum()):,} content, "
          f"{int(conn.sum()):,} connective)")

    # ---- gather candidate-feature activations on response rows, per-feature top-K ----
    ck = torch.load(a.sae, map_location="cpu"); cfg = ck["model_config"]
    sae = TopKSAE(**cfg).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
    feats_t = torch.tensor(feats, device=dev)
    resp_t = torch.from_numpy(resp).to(dev)
    Zc = []; Rc = []                                         # bulk-collect [nresp,F] acts + store rows (ONE transfer/batch)
    row0 = 0
    order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
    with torch.no_grad():
        for si, sp in enumerate(order):
            nrows = pq.read_metadata(sp).num_rows                      # cheap: advance row0 without reading data
            if si % a.shard_stride == 0:                               # only read+encode subsampled shards
                X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
                for s in range(0, n, 8192):
                    e = min(n, s + 8192); m = resp_t[row0 + s:row0 + e]
                    if not m.any(): continue
                    xb = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(dev)
                    z = sae.encode(xb)[:, feats_t][m]                  # [nresp, F]
                    grows = torch.arange(row0 + s, row0 + e, device=dev)[m]
                    Zc.append(z.half().cpu().numpy())                  # ONE transfer/batch (not per-feature)
                    Rc.append(grows.to(torch.int64).cpu().numpy())
                if si % 20 == 0: print(f"  gather shard {si}/{len(order)}", flush=True)
            row0 += nrows
    Z = np.concatenate(Zc); Rrow = np.concatenate(Rc); del Zc, Rc     # Z:[Nr,F] f16, Rrow:[Nr]
    print(f"[synth] gathered {Z.shape[0]:,}x{Z.shape[1]}; scoring...", flush=True)

    # ---- per-feature synthesis metrics over top-K activating response tokens ----
    results = {}
    for fi, f in enumerate(feats):
        col = Z[:, fi].astype(np.float32); nz = col > 0
        if not nz.any():
            results[f] = {"n_fire": 0}; continue
        av = col[nz]; rw = Rrow[nz]
        k = min(a.topk, len(av)); top = rw[np.argsort(-av)[:k]]
        cm = is_content[top]; ncontent = int(cm.sum())
        nov = float(novel[top][cm].mean()) if ncontent else float("nan")
        cf = float(conn[top].mean())
        toks = [(tok.decode([int(tid_row[r])]).strip(), bool(novel[r]), bool(conn[r])) for r in top[:20]]
        results[f] = {"n_fire": int(len(av)), "topk": k, "n_content": ncontent,
                      "novelty": round(nov, 3), "connective": round(cf, 3),
                      "top_tokens": toks}

    ranked = sorted([f for f in feats if results[f].get("n_fire", 0) > 0],
                    key=lambda f: (-(results[f]["novelty"] if results[f]["novelty"] == results[f]["novelty"] else -1)))
    print(f"\n{'feat':>7} {'nfire':>7} {'novelty':>8} {'connect':>8}  top tokens (●=novel ▸=connective)")
    for f in ranked:
        r = results[f]
        tt = " ".join(("●" if n else "") + ("▸" if c else "") + t for t, n, c in r["top_tokens"][:12])
        print(f"{f:>7} {r['n_fire']:>7} {r['novelty']:>8} {r['connective']:>8}  {tt[:90]}")
    Path(out).write_text(json.dumps({"layer": a.layer, "features": results}, indent=2))
    print(f"\n[wrote] {out}")
    print("HIGH novelty = fires on words the model synthesized (not in prompt) = non-circular signal.")
    print("LOW novelty  = echo detector (fires on restated prompt GO terms) = circular.")


if __name__ == "__main__":
    main()
