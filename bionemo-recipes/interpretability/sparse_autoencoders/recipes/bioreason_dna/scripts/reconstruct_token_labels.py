#!/usr/bin/env python
"""Reconstruct a row-aligned ``token_labels.parquet`` for the DNA activation store.

The store's ``token_labels.parquet`` was clobbered mid-write by a later same-store extraction
(layers 8/24 append). The layer16 shards themselves are intact. This script re-runs ONLY the
tokenizer+collator (no model forward, no GPU) for the first ``--num-examples`` examples and emits
the per-token band sidecar the dashboard / cross-modal scripts expect:

    sequence_id, token_index, position_type  ("dna" | "text")

Rows are produced in the SAME example order and the SAME per-example token order the extractor used
(``extract_dna.py``: keep = non-pad, band = dna if token in {DNA_START,DNA_PAD,DNA_END} else text),
so they line up 1:1 with the layer16 shard rows. A per-example checksum against ``examples.parquet``
(n_dna / n_text) guards the alignment. dna_subsample was 0 for this store, so no rows are dropped.

    /data/savithas/repos/bioreason-nemotron/.venv/bin/python reconstruct_token_labels.py \
        --store /data/savithas/dna_sae/activations/vep_non_snv_L16 --num-examples 2000 \
        --out /data/savithas/dna_sae/activations/vep_non_snv_L16/token_labels_recon.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DNA_STACK = "/data/savithas/phase3_full/dna_extract_wt"
CKPT = "/data/savithas/converted/ev-4b-vep"
EMB = "/data/savithas/dna_embeddings/evo2_1b/vep_non_snv"
DATASET = "wanglab/variant_effect_non_snv"


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-examples", type=int, default=2000, help="reconstruct labels for the first N examples")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length-text", type=int, default=8192)
    p.add_argument("--truncate-dna-per-side", type=int, default=1024)
    p.add_argument("--dna-stack", default=DNA_STACK)
    p.add_argument("--ckpt-dir", default=CKPT)
    p.add_argument("--embedding-base", default=EMB)
    p.add_argument("--dataset-path", default=DATASET)
    args = p.parse_args()

    sys.path.insert(0, args.dna_stack)
    from bioreason_dna.dataset.load import load_chat_dataset
    from bioreason_dna.dataset.collate import dna_collate_fn
    from bioreason_dna.dataset.embedding import EmbeddingStore
    from bioreason_dna.models.constants import DNA_START_TOKEN, DNA_PAD_TOKEN, DNA_END_TOKEN
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    S = tok.convert_tokens_to_ids(DNA_START_TOKEN)
    P = tok.convert_tokens_to_ids(DNA_PAD_TOKEN)
    E = tok.convert_tokens_to_ids(DNA_END_TOKEN)

    tr, va, te = load_chat_dataset(dataset_path=args.dataset_path, dataset_config=None, task="vep",
                                   truncate_dna_per_side=args.truncate_dna_per_side, num_proc=4)
    from datasets import concatenate_datasets
    ds = concatenate_datasets([d for d in (tr, va, te) if d is not None])
    n = min(args.num_examples, len(ds))
    ds = ds.select(range(n))

    store = EmbeddingStore(base_path=args.embedding_base, roles=["reference", "variant"],
                           block="blocks.20.mlp.l3", prefix="evo2_1b")

    ex = pq.read_table(Path(args.store) / "examples.parquet")
    ck_nd = ex.column("n_dna").to_numpy()
    ck_nt = ex.column("n_text").to_numpy()
    ck_sid = ex.column("sequence_id").to_pylist()

    sids_out, tidx_out, band_out = [], [], []
    mism = 0
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
            keep_np = keep.cpu().numpy()
            abs_index = np.nonzero(keep_np)[0]
            token_index = (abs_index - abs_index.min()) if abs_index.size else abs_index
            is_dna = ((row == S) | (row == P) | (row == E))[keep].cpu().numpy()
            bands = np.where(is_dna, "dna", "text")
            sid = str(ds[gi].get("sequence_id", f"row{gi}"))
            # checksum vs the store's examples.parquet
            if int(is_dna.sum()) != int(ck_nd[gi]) or int((~is_dna).sum()) != int(ck_nt[gi]) or sid != str(ck_sid[gi]):
                mism += 1
                if mism <= 5:
                    print(f"[recon] MISMATCH ex{gi} sid={sid}/{ck_sid[gi]} "
                          f"dna={int(is_dna.sum())}/{ck_nd[gi]} text={int((~is_dna).sum())}/{ck_nt[gi]}")
            sids_out.extend([sid] * len(bands))
            tidx_out.extend(int(t) for t in token_index)
            band_out.extend(bands.tolist())
        if start % (args.batch_size * 25) == 0:
            print(f"[recon] {start + len(batch)}/{n} examples, {len(band_out):,} rows", flush=True)

    if mism:
        raise SystemExit(f"[recon] FATAL: {mism} example mismatches vs examples.parquet — alignment unsafe")

    tbl = pa.table({
        "sequence_id": pa.array(sids_out, pa.string()),
        "token_index": pa.array(tidx_out, pa.int32()),
        "position_type": pa.array(band_out, pa.string()),
    })
    pq.write_table(tbl, args.out)
    print(f"[recon] wrote {args.out}: {len(band_out):,} rows over {n} examples "
          f"(dna={band_out.count('dna'):,} text={band_out.count('text'):,})")


if __name__ == "__main__":
    main()
