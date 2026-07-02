#!/usr/bin/env python
"""Rebuild dna_l16_vep feature_examples.parquet with REAL, PER-BAND top-N windows.

build_dashboard.py stored only top-K examples per feature over ANY band, so cross-modal features
(e.g. F15966: text_frac=0.90 by rate, yet its highest-magnitude tokens are DNA) ended up with only
DNA examples — their text side was invisible. This mirrors the protein dashboard.py contract instead:
for every feature, the top-N activating examples PER BAND (dna, text), each with a real ±window token
window, per-token activations, band, max_activation, sequence_id. Content is re-derived without a model
forward (tokenizer + dna_collate_fn), exactly like reconstruct_token_labels.py; activations come from
encoding the (intact-token_labels-aligned) layer16 store rows with the ep3 SAE.

    /data/savithas/repos/bioreason-nemotron/.venv/bin/python dna_dashboard_perband.py \
        --num-examples 2000 --window 8 --n-examples 8 --device cuda:1
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

DNA_STACK = "/data/savithas/phase3_full/dna_extract_wt"
CKPT = "/data/savithas/converted/ev-4b-vep"
EMB = "/data/savithas/dna_embeddings/evo2_1b/vep_non_snv"
DATASET = "wanglab/variant_effect_non_snv"
STORE = "/data/savithas/dna_sae/activations/vep_non_snv_L16"
SAE = "/data/savithas/dna_sae/sae-dna-l16-vep-ep3/checkpoint_final.pt"
OUT_DIR = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/"
           "recipes/bioreason_pro/multimodal_dashboard/public/dna_l16_vep")
BANDS = ["dna", "text"]


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=2000)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--n-examples", type=int, default=8, help="top examples per feature PER band")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length-text", type=int, default=8192)
    p.add_argument("--truncate-dna-per-side", type=int, default=1024)
    p.add_argument("--encode-batch", type=int, default=16384)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--store", default=STORE)
    p.add_argument("--sae", default=SAE)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--dna-stack", default=DNA_STACK)
    p.add_argument("--ckpt-dir", default=CKPT)
    p.add_argument("--embedding-base", default=EMB)
    p.add_argument("--dataset-path", default=DATASET)
    return p.parse_args()


def build_token_content(args):
    """sid -> list of per-KEPT-token content strings, aligned 1:1 with store token_index."""
    sys.path.insert(0, args.dna_stack)
    from bioreason_dna.dataset.load import load_chat_dataset
    from bioreason_dna.dataset.collate import dna_collate_fn
    from bioreason_dna.dataset.embedding import EmbeddingStore
    from bioreason_dna.models.constants import DNA_START_TOKEN, DNA_PAD_TOKEN, DNA_END_TOKEN
    from transformers import AutoTokenizer
    from datasets import concatenate_datasets

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    S = tok.convert_tokens_to_ids(DNA_START_TOKEN)
    P = tok.convert_tokens_to_ids(DNA_PAD_TOKEN)
    E = tok.convert_tokens_to_ids(DNA_END_TOKEN)

    tr, va, te = load_chat_dataset(dataset_path=args.dataset_path, dataset_config=None, task="vep",
                                   truncate_dna_per_side=args.truncate_dna_per_side, num_proc=4)
    ds = concatenate_datasets([d for d in (tr, va, te) if d is not None])
    n = min(args.num_examples, len(ds))
    ds = ds.select(range(n))
    store = EmbeddingStore(base_path=args.embedding_base, roles=["reference", "variant"],
                           block="blocks.20.mlp.l3", prefix="evo2_1b")

    content, order = {}, []
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        batch = [ds[i] for i in range(start, min(n, start + args.batch_size))]
        b = dna_collate_fn(batch, tokenizer=tok, max_length=args.max_length_text,
                           embedding_store=store, train_mode=True)
        ii, am = b["input_ids"], b["attention_mask"]
        for j in range(len(batch)):
            gi = start + j
            kept = ii[j][am[j].bool()].tolist()
            sid = str(ds[gi].get("sequence_id", f"row{gi}"))
            refseq = str(ds[gi].get("reference_sequence") or "")
            varseq = str(ds[gi].get("variant_sequence") or "")
            n_ref = store.get("reference", sid).shape[0]
            pieces, pad_i = [], 0
            for tid in kept:
                if tid == S:
                    pieces.append("⟦S⟧")
                elif tid == E:
                    pieces.append("⟦E⟧")
                elif tid == P:
                    if pad_i < n_ref:
                        pieces.append(refseq[pad_i] if pad_i < len(refseq) else "?")
                    else:
                        vi = pad_i - n_ref
                        pieces.append(varseq[vi] if vi < len(varseq) else "?")
                    pad_i += 1
                else:
                    pieces.append(tok.decode([tid]))
            content[sid] = pieces
            order.append(sid)
        if start % (args.batch_size * 25) == 0:
            print(f"[content] {start + len(batch)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[content] derived per-token content for {len(content)} examples")
    return content, order


def main():  # noqa: D103
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    t_start = time.time()

    content, ex_order = build_token_content(args)

    # example -> store row start + band array, from the intact token_labels.parquet
    print("[dna] loading token_labels.parquet (sequence_id + position_type)")
    tl = pq.read_table(Path(args.store) / "token_labels.parquet",
                       columns=["sequence_id", "position_type"])
    sid_all = tl.column("sequence_id").to_numpy(zero_copy_only=False)
    pos_all = tl.column("position_type").to_numpy(zero_copy_only=False)
    change = np.ones(len(sid_all), dtype=bool)
    change[1:] = sid_all[1:] != sid_all[:-1]
    starts = np.flatnonzero(change)
    ex_start, ex_len = {}, {}
    for k, s in enumerate(starts):
        e = int(starts[k + 1]) if k + 1 < len(starts) else len(sid_all)
        ex_start[str(sid_all[s])] = int(s)
        ex_len[str(sid_all[s])] = e - int(s)

    # rows spanning the first --num-examples examples (contiguous, leading)
    have = [s for s in ex_order if s in ex_start]
    R = max(ex_start[s] + ex_len[s] for s in have)
    band_row = pos_all[:R].astype(str)
    sid_row = sid_all[:R]
    print(f"[dna] first {len(have)} examples span {R:,} store rows")

    # load store rows [0:R), encode with SAE
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sae" / "src"))
    from sae.activation_store import shard_table_to_array
    from sae.architectures.topk import TopKSAE
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    shard_rows = int(pq.read_metadata(shards[0]).num_rows)
    n_shards = (R // shard_rows) + 1
    print(f"[dna] reading {n_shards}/{len(shards)} shards")
    X = np.concatenate([shard_table_to_array(pq.read_table(sp)) for sp in shards[:n_shards]], axis=0)[:R]

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                         for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim
    N = args.n_examples

    # per (band, feature) top-N EXAMPLES: track best per-example activation + its store row.
    # We reduce to per-example maxima first (an example contributes ONE window), so top-N over examples.
    band_code = {b: i for i, b in enumerate(BANDS)}
    row_bcode = np.array([band_code.get(b, -1) for b in band_row], dtype=np.int64)
    row_exidx = np.searchsorted(starts, np.arange(R), side="right") - 1  # example index per store row
    row_bcode_t = torch.from_numpy(row_bcode).to(dev)
    row_exidx_t = torch.from_numpy(row_exidx).to(dev)
    n_ex = len(have)

    NEG = -1e30
    # per band: (n_ex, H) best activation of that example's band-tokens, and the store row achieving it
    best_act = {b: torch.full((n_ex, H), NEG, device=dev) for b in BANDS}
    best_row = {b: torch.full((n_ex, H), -1, dtype=torch.long, device=dev) for b in BANDS}
    print(f"[dna] encoding {R:,} rows x {H:,} features for per-band per-example maxima")
    with torch.no_grad():
        for s in range(0, R, args.encode_batch):
            e = min(R, s + args.encode_batch)
            codes = sae.encode(torch.from_numpy(X[s:e]).to(dev))  # (nb, H)
            ex_b = row_exidx_t[s:e]
            bc_b = row_bcode_t[s:e]
            rows_b = torch.arange(s, e, device=dev)
            for bi, b in enumerate(BANDS):
                m = bc_b == bi
                if not m.any():
                    continue
                cm = codes[m]                     # (nm, H)
                exm = ex_b[m]                     # (nm,)
                rowm = rows_b[m]                  # (nm,)
                # scatter-max into best_act[b][exm], tracking row
                cur = best_act[b].index_select(0, exm)          # (nm, H)
                upd = cm > cur
                # new max per (example-row, feature)
                # use scatter_reduce for the value, then recover rows in a 2nd pass below
                best_act[b].index_reduce_(0, exm, cm, "amax", include_self=True)
            if s % (args.encode_batch * 10) == 0:
                print(f"  {e:,}/{R:,}", flush=True)

    # recover the peak store row per (band, example, feature): 2nd streaming pass matching best_act
    print("[dna] pass 2: locating peak store row per (band, example, feature)")
    with torch.no_grad():
        for s in range(0, R, args.encode_batch):
            e = min(R, s + args.encode_batch)
            codes = sae.encode(torch.from_numpy(X[s:e]).to(dev))
            ex_b = row_exidx_t[s:e]
            bc_b = row_bcode_t[s:e]
            rows_b = torch.arange(s, e, device=dev)
            for bi, b in enumerate(BANDS):
                m = bc_b == bi
                if not m.any():
                    continue
                cm = codes[m]; exm = ex_b[m]; rowm = rows_b[m]
                target = best_act[b].index_select(0, exm)       # (nm, H) the known max
                hit = (cm >= target - 1e-6) & (target > 0)      # this row achieves the example max
                if hit.any():
                    hi, hj = torch.nonzero(hit, as_tuple=True)  # row-in-batch, feature
                    flat_ex = exm[hi]
                    best_row[b].index_put_((flat_ex, hj), rowm[hi], accumulate=False)
    del X

    # per (band, feature): top-N examples by best_act
    def char_expand(pieces, acts):
        chars, cacts = [], []
        for pc, a in zip(pieces, acts):
            for ch in pc:
                chars.append(ch); cacts.append(a)
        return "".join(chars), cacts

    # we need per-window per-token activations -> re-encode just the window rows at the end.
    # First collect (feature, band, example, peak_row, peak_act) for the top-N.
    print(f"[dna] selecting top-{N} examples per (feature, band) and gathering window rows")
    picks = []   # (feature, band, sid, peak_global_row, peak_act, rank)
    for bi, b in enumerate(BANDS):
        ba = best_act[b].cpu().numpy()     # (n_ex, H)
        br = best_row[b].cpu().numpy()
        # top-N examples per feature
        # argpartition per column is expensive at H=40960; do torch topk on GPU instead
    # GPU topk per band per feature over examples
    for bi, b in enumerate(BANDS):
        vals, idx = torch.topk(best_act[b], k=min(N, n_ex), dim=0)  # (N, H)
        vals = vals.cpu().numpy(); idx = idx.cpu().numpy()
        br = best_row[b].cpu().numpy()
        for f in range(H):
            for rk in range(vals.shape[0]):
                a = float(vals[rk, f])
                if a <= 0:
                    continue
                ex_i = int(idx[rk, f])
                sid = have[ex_i]
                prow = int(br[ex_i, f])
                if prow < 0:
                    continue
                picks.append((f, b, sid, prow, a, rk))
        print(f"  band {b}: cumulative picks {len(picks):,}")

    # gather unique window store-rows, re-encode for per-token window activations
    W = args.window
    win = {}  # pick_index -> (lo_g, hi_g)
    needset = set()
    for pi, (f, b, sid, prow, a, rk) in enumerate(picks):
        base = ex_start[sid]
        t = prow - base
        L = ex_len[sid]
        lo = max(0, t - W); hi = min(L, t + W + 1)
        win[pi] = (base + lo, base + hi)
        needset.update(range(base + lo, base + hi))
    needed = np.array(sorted(needset), dtype=np.int64)
    print(f"[dna] {len(picks):,} picks, {len(needed):,} unique window rows to re-encode")

    n_shards2 = (int(needed.max()) // shard_rows) + 1
    X2 = np.concatenate([shard_table_to_array(pq.read_table(sp)) for sp in shards[:n_shards2]], axis=0)
    Xn = X2[needed]; del X2
    feats_needed = np.sort(np.unique([p[0] for p in picks]))
    feat_pos = {int(f): i for i, f in enumerate(feats_needed)}
    row_pos = {int(g): i for i, g in enumerate(needed)}
    codesW = np.zeros((len(needed), len(feats_needed)), dtype=np.float32)
    fidx = torch.from_numpy(feats_needed.astype(np.int64)).to(dev)
    with torch.no_grad():
        for s in range(0, len(needed), args.encode_batch):
            e = min(len(needed), s + args.encode_batch)
            c = sae.encode(torch.from_numpy(Xn[s:e]).to(dev))
            codesW[s:e] = c.index_select(1, fidx).cpu().numpy()
    del Xn

    # build output rows
    rows = {k: [] for k in ("feature_id", "protein_id", "band", "activation_value", "example_rank",
                            "token_index", "residue_idx", "window_start", "sequence_window",
                            "highlight_values", "sequence", "activations", "max_activation")}
    for pi, (f, b, sid, prow, a, rk) in enumerate(picks):
        lo_g, hi_g = win[pi]
        base = ex_start[sid]
        lo_t = lo_g - base
        pieces = content[sid][lo_t: lo_t + (hi_g - lo_g)]
        fpi = feat_pos[int(f)]
        tok_acts = [float(codesW[row_pos[g], fpi]) for g in range(lo_g, hi_g)]
        seq, char_acts = char_expand(pieces, tok_acts)
        rows["feature_id"].append(int(f)); rows["protein_id"].append(sid); rows["band"].append(b)
        rows["activation_value"].append(a); rows["example_rank"].append(int(rk))
        rows["token_index"].append(prow - base); rows["residue_idx"].append(0)
        rows["window_start"].append(lo_t)
        rows["sequence_window"].append(seq); rows["highlight_values"].append(tok_acts)
        rows["sequence"].append(seq); rows["activations"].append(char_acts)
        rows["max_activation"].append(a)

    # re-rank example_rank WITHIN (feature, band) by activation so the SPA orders correctly
    import pandas as pd
    df = pd.DataFrame(rows)
    df = df.sort_values(["feature_id", "band", "activation_value"], ascending=[True, True, False])
    df["example_rank"] = df.groupby(["feature_id", "band"]).cumcount()

    tbl = pa.table({
        "feature_id": pa.array(df.feature_id.tolist(), pa.int64()),
        "protein_id": pa.array(df.protein_id.tolist(), pa.string()),
        "band": pa.array(df.band.tolist(), pa.string()),
        "activation_value": pa.array(df.activation_value.tolist(), pa.float64()),
        "example_rank": pa.array(df.example_rank.tolist(), pa.int64()),
        "token_index": pa.array(df.token_index.tolist(), pa.int64()),
        "residue_idx": pa.array(df.residue_idx.tolist(), pa.int64()),
        "window_start": pa.array(df.window_start.tolist(), pa.int64()),
        "sequence_window": pa.array(df.sequence_window.tolist(), pa.string()),
        "highlight_values": pa.array(df.highlight_values.tolist(), pa.list_(pa.float64())),
        "sequence": pa.array(df.sequence.tolist(), pa.string()),
        "activations": pa.array(df.activations.tolist(), pa.list_(pa.float64())),
        "max_activation": pa.array(df.max_activation.tolist(), pa.float64()),
    })
    fe_path = out / "feature_examples.parquet"
    pq.write_table(tbl, str(fe_path), compression="snappy")
    nboth = df.groupby("feature_id")["band"].agg(lambda x: len(set(x)) > 1).sum()
    print(f"\n[dna] wrote {fe_path}: {len(df):,} rows over {df.feature_id.nunique():,} features "
          f"({nboth:,} with BOTH bands) in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
