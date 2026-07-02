#!/usr/bin/env python
"""Rebuild feature_examples.parquet for the DNA (dna_l16_vep) dashboard with REAL token windows.

The DNA store's feature_examples (from build_dashboard.py) carries the correct top-activating
examples per feature per band (feature_id, protein_id=sequence_id, band, token_index, max_activation)
but its ``sequence`` was a placeholder ("[dna] <sid> @tok<i>"). This script fills in real windows:

  * per-token CONTENT is re-derived WITHOUT a model forward, exactly like reconstruct_token_labels.py:
    run the tokenizer + dna_collate_fn over the first --num-examples examples. The kept-token stream is
        <|im_start|> user \n  <|DNA_START|> <|dna_pad|>*n_ref <|DNA_END|> \n
                              <|DNA_START|> <|dna_pad|>*n_var <|DNA_END|> \n  <question text> ... <answer>
    n_ref == len(reference_sequence) and n_var == len(variant_sequence) (Evo2 = 1 token / nucleotide,
    verified 1:1), so pad slot i<n_ref -> reference_sequence[i], slot i>=n_ref -> variant_sequence[i-n_ref].
    DNA markers render as ⟦S⟧/⟦E⟧; text tokens decode to their Qwen subword text.

  * store rows are located via the (intact) token_labels.parquet: row(example k, token_index t) =
    example_start[k] + t, row-aligned 1:1 with the layer16 shards. We encode only the store rows that
    fall inside some example window (±--window tokens around each example's peak token) with the ep3 SAE.

  * output matches the schema App.jsx reads: feature_id, protein_id, band, sequence, activations,
    max_activation, example_rank (+ the legacy build_dashboard columns kept for compatibility). Because
    ProteinSequence.jsx does sequence.split('') (one char per activation value), the ``sequence`` is the
    per-token content concatenated char-by-char and ``activations`` is the per-token activation REPEATED
    across that token's characters, so highlighting stays token-aligned. Text tokens are space-separated
    (the space char gets activation 0); DNA nucleotides are contiguous.

    /data/savithas/repos/bioreason-nemotron/.venv/bin/python dna_dashboard_windows.py \
        --num-examples 2000 --window 8 --device cuda:1
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


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--num-examples", type=int, default=2000)
    p.add_argument("--window", type=int, default=8, help="tokens of context each side of the peak token")
    p.add_argument("--batch-size", type=int, default=8, help="collate batch size")
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
    """Return sid -> list of per-KEPT-token content strings (same order/length as store token_index)."""
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

    content = {}   # sid -> list[str]
    mism = 0
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        batch = [ds[i] for i in range(start, min(n, start + args.batch_size))]
        b = dna_collate_fn(batch, tokenizer=tok, max_length=args.max_length_text,
                           embedding_store=store, train_mode=True)
        ii = b["input_ids"]
        am = b["attention_mask"]
        for j in range(len(batch)):
            gi = start + j
            row = ii[j]
            keep = am[j].bool()
            kept = row[keep].tolist()
            sid = str(ds[gi].get("sequence_id", f"row{gi}"))
            refseq = str(ds[gi].get("reference_sequence") or "")
            varseq = str(ds[gi].get("variant_sequence") or "")
            n_ref = store.get("reference", sid).shape[0]
            # walk kept tokens; assign nucleotides to dna_pad slots in order (ref then var)
            pieces = []
            pad_i = 0
            for tid in kept:
                if tid == S:
                    pieces.append("⟦S⟧")
                elif tid == E:
                    pieces.append("⟦E⟧")
                elif tid == P:
                    if pad_i < n_ref:
                        nt = refseq[pad_i] if pad_i < len(refseq) else "?"
                    else:
                        vi = pad_i - n_ref
                        nt = varseq[vi] if vi < len(varseq) else "?"
                    pieces.append(nt)
                    pad_i += 1
                else:
                    pieces.append(tok.decode([tid]))
            # sanity: pad slots consumed both sequences fully
            if pad_i != n_ref + varseq.__len__() and pad_i != n_ref + store.get("variant", sid).shape[0]:
                mism += 1
                if mism <= 5:
                    print(f"[content] slot/seq mismatch ex{gi} sid={sid} pad_i={pad_i} "
                          f"n_ref={n_ref} len(var)={len(varseq)}")
            content[sid] = pieces
        if start % (args.batch_size * 25) == 0:
            print(f"[content] {start + len(batch)}/{n} examples ({time.time()-t0:.0f}s)", flush=True)
    print(f"[content] derived per-token content for {len(content)} examples, {mism} mismatches")
    return content


def main():  # noqa: D103
    args = parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    t_start = time.time()

    # ---- 1. per-token content (tokenizer/collator only, no forward) ----
    content = build_token_content(args)

    # ---- 2. example -> store-row start (from intact token_labels.parquet) ----
    print("[dna] loading token_labels.parquet for example row offsets")
    tl = pq.read_table(Path(args.store) / "token_labels.parquet", columns=["sequence_id"])
    sid_all = tl.column("sequence_id").to_numpy(zero_copy_only=False)
    change = np.ones(len(sid_all), dtype=bool)
    change[1:] = sid_all[1:] != sid_all[:-1]
    starts = np.flatnonzero(change)
    ex_start = {}   # sid -> first store row
    ex_len = {}
    for k, s in enumerate(starts):
        e = int(starts[k + 1]) if k + 1 < len(starts) else len(sid_all)
        ex_start[str(sid_all[s])] = int(s)
        ex_len[str(sid_all[s])] = e - int(s)
    del sid_all

    # ---- 3. existing feature_examples: the top examples we must fill with real windows ----
    fe_path = out / "feature_examples.parquet"
    bak = out / "feature_examples.parquet.pre_windows.bak"
    if not bak.exists():
        import shutil
        shutil.copy2(fe_path, bak)
        print(f"[dna] backed up existing feature_examples -> {bak}")
    fe = pq.read_table(fe_path).to_pandas()
    print(f"[dna] {len(fe):,} example rows over {fe['feature_id'].nunique():,} features; "
          f"bands={fe['band'].value_counts().to_dict()}")

    # ---- 4. gather the unique store rows needed by all windows ----
    W = args.window
    row_lo = np.empty(len(fe), dtype=np.int64)   # window lo store-row (inclusive) per example
    row_hi = np.empty(len(fe), dtype=np.int64)   # window hi store-row (exclusive)
    keep_row = np.zeros(len(fe), dtype=bool)
    for r in range(len(fe)):
        sid = fe.protein_id.iat[r]
        ti = int(fe.token_index.iat[r])
        if sid not in ex_start or sid not in content:
            continue
        L = ex_len[sid]
        lo = max(0, ti - W)
        hi = min(L, ti + W + 1)
        row_lo[r] = ex_start[sid] + lo
        row_hi[r] = ex_start[sid] + hi
        keep_row[r] = True
    print(f"[dna] {keep_row.sum():,}/{len(fe):,} example rows resolvable to windows")

    needed = np.zeros(0, dtype=np.int64)
    if keep_row.any():
        segs = [np.arange(row_lo[r], row_hi[r]) for r in np.nonzero(keep_row)[0]]
        needed = np.unique(np.concatenate(segs))
    print(f"[dna] {len(needed):,} unique store rows to encode")

    # ---- 5. load those store rows from shards, encode with the SAE ----
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sae" / "src"))
    from sae.activation_store import shard_table_to_array
    from sae.architectures.topk import TopKSAE
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    shard_rows = int(pq.read_metadata(shards[0]).num_rows)
    max_row = int(needed.max()) + 1 if len(needed) else 0
    n_shards_needed = (max_row // shard_rows) + 1
    print(f"[dna] reading {n_shards_needed}/{len(shards)} shards (shard_rows={shard_rows})")
    X = np.concatenate([shard_table_to_array(pq.read_table(sp)) for sp in shards[:n_shards_needed]], axis=0)
    Xneed = X[needed]
    del X

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v
                         for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim

    row_pos = {int(g): i for i, g in enumerate(needed)}   # global store row -> index into Xneed/codes
    # encode only the features that actually appear (memory: full H per row is 40960 -> too big for 1M rows)
    feats = np.sort(fe.feature_id.unique().astype(np.int64))
    feat_pos = {int(f): i for i, f in enumerate(feats)}
    codes = np.zeros((len(needed), len(feats)), dtype=np.float32)
    fidx = torch.from_numpy(feats).to(dev)
    print(f"[dna] encoding {len(needed):,} rows x {len(feats):,} needed features")
    with torch.no_grad():
        for s in range(0, len(needed), args.encode_batch):
            e = min(len(needed), s + args.encode_batch)
            c = sae.encode(torch.from_numpy(Xneed[s:e]).to(dev))   # (nb, H)
            codes[s:e] = c.index_select(1, fidx).cpu().numpy()
            if s % (args.encode_batch * 10) == 0:
                print(f"  {e:,}/{len(needed):,}", flush=True)
    del Xneed

    # ---- 6. build the real window rows ----
    def char_expand(pieces, acts):
        """One char per activation value; text tokens space-separated, DNA contiguous."""
        chars, cacts = [], []
        for pc, a in zip(pieces, acts):
            for ch in pc:
                chars.append(ch)
                cacts.append(a)
        return "".join(chars), cacts

    rows = {k: [] for k in ("feature_id", "protein_id", "band", "activation_value", "example_rank",
                            "token_index", "residue_idx", "window_start", "sequence_window",
                            "highlight_values", "sequence", "activations", "max_activation")}
    n_ok = 0
    for r in range(len(fe)):
        f = int(fe.feature_id.iat[r])
        sid = fe.protein_id.iat[r]
        band = fe.band.iat[r]
        ti = int(fe.token_index.iat[r])
        maxa = float(fe.max_activation.iat[r])
        rank = int(fe.example_rank.iat[r])
        if not keep_row[r]:
            # keep the row but with placeholder (rare: sid outside first N examples)
            seq = f"[{band}] {sid} @tok{ti}"
            rows["feature_id"].append(f); rows["protein_id"].append(sid); rows["band"].append(band)
            rows["activation_value"].append(maxa); rows["example_rank"].append(rank)
            rows["token_index"].append(ti); rows["residue_idx"].append(0); rows["window_start"].append(ti)
            rows["sequence_window"].append(seq); rows["highlight_values"].append([maxa])
            rows["sequence"].append(seq); rows["activations"].append([maxa]); rows["max_activation"].append(maxa)
            continue
        lo_g, hi_g = int(row_lo[r]), int(row_hi[r])
        lo_t = lo_g - ex_start[sid]
        pieces = content[sid][lo_t: lo_t + (hi_g - lo_g)]
        fi = feat_pos[f]
        tok_acts = [float(codes[row_pos[g], fi]) for g in range(lo_g, hi_g)]
        seq, char_acts = char_expand(pieces, tok_acts)
        wmax = max(tok_acts) if tok_acts else maxa
        rows["feature_id"].append(f); rows["protein_id"].append(sid); rows["band"].append(band)
        rows["activation_value"].append(maxa); rows["example_rank"].append(rank)
        rows["token_index"].append(ti); rows["residue_idx"].append(0); rows["window_start"].append(lo_t)
        rows["sequence_window"].append(seq); rows["highlight_values"].append(tok_acts)
        rows["sequence"].append(seq); rows["activations"].append(char_acts)
        rows["max_activation"].append(max(maxa, wmax))
        n_ok += 1
        if r % 20000 == 0:
            print(f"  built {r:,}/{len(fe):,}", flush=True)

    tbl = pa.table({
        "feature_id": pa.array(rows["feature_id"], pa.int64()),
        "protein_id": pa.array(rows["protein_id"], pa.string()),
        "band": pa.array(rows["band"], pa.string()),
        "activation_value": pa.array(rows["activation_value"], pa.float64()),
        "example_rank": pa.array(rows["example_rank"], pa.int64()),
        "token_index": pa.array(rows["token_index"], pa.int64()),
        "residue_idx": pa.array(rows["residue_idx"], pa.int64()),
        "window_start": pa.array(rows["window_start"], pa.int64()),
        "sequence_window": pa.array(rows["sequence_window"], pa.string()),
        "highlight_values": pa.array(rows["highlight_values"], pa.list_(pa.float64())),
        "sequence": pa.array(rows["sequence"], pa.string()),
        "activations": pa.array(rows["activations"], pa.list_(pa.float64())),
        "max_activation": pa.array(rows["max_activation"], pa.float64()),
    })
    pq.write_table(tbl, str(fe_path), compression="snappy")
    print(f"\n[dna] wrote {fe_path}: {len(fe):,} rows ({n_ok:,} with real windows) "
          f"in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
