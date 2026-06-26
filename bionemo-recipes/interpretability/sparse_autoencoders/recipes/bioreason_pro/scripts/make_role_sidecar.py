#!/usr/bin/env python
"""Write a per-token ROLE column (protein/go/prompt/reasoning/answer) aligned to a store's token_labels.

The store's token_labels has position_type (protein/go/text); this refines 'text' into prompt/reasoning/
answer using the assistant <think>...</think> markers (prompt = before <think>, reasoning = between,
answer = after </think>). Writes token_labels_with_role.parquet so downstream (crossmodal_cooccur) can
test protein<->reasoning or protein<->answer fusion specifically, stripping out prompt boilerplate.
Usage: make_role_sidecar.py --store <store> [--max-length-text 10000 --max-length-protein 2000]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", required=True)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import _install_unsloth_stub
    _install_unsloth_stub()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    pad_tok = tok.pad_token_id
    THINK_OPEN = tok.convert_tokens_to_ids("<think>")
    THINK_CLOSE = tok.convert_tokens_to_ids("</think>")

    train_ds, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=args.max_length_protein)
    splits = {"train": train_ds, "validation": val_ds, "test": test_ds}
    store_pids = [str(x) for x in pq.read_table(Path(args.store) / "proteins.parquet").column("protein_id").to_pylist()]
    best, best_sel, best_name = -1, None, "train"
    for name, sds in splits.items():
        idx = {str(p): i for i, p in enumerate(sds["protein_id"])}
        sel = [idx[p] for p in store_pids if p in idx]
        if len(sel) > best:
            best, best_sel, best_name = len(sel), sel, name
    ds = splits[best_name].select(best_sel)
    print(f"[role] matched {best}/{len(store_pids)} store proteins in split '{best_name}'")
    collate_fn = brp_data.make_collate_fn(tok, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    pid_marks = {}  # pid -> (think_open_idx, think_close_idx) in unpadded token order
    for b_i, batch in enumerate(loader):
        row = batch["input_ids"][0]
        ids = row[row != pad_tok].tolist()
        pid = str(ds[b_i].get("protein_id", f"row{b_i}"))
        pid_marks[pid] = (ids.index(THINK_OPEN) if THINK_OPEN in ids else -1,
                          ids.index(THINK_CLOSE) if THINK_CLOSE in ids else -1)
        if b_i % 1000 == 0:
            print(f"  reproduced {b_i}/{len(ds)}", flush=True)

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    pid_col = [str(x) for x in tl.column("protein_id").to_pylist()]
    tidx = tl.column("token_index").to_pylist()
    ptype = tl.column("position_type").to_pylist()
    role = []
    for pid, ti, pt in zip(pid_col, tidx, ptype):
        if pt != "text":
            role.append(pt)
            continue
        o, c = pid_marks.get(pid, (-1, -1))
        if c < 0:
            role.append("text")
        else:
            role.append("prompt" if (o >= 0 and ti < o) else ("reasoning" if ti < c else "answer"))
    out = tl.append_column("role", pa.array(role, pa.string()))
    pq.write_table(out, Path(args.store) / "token_labels_with_role.parquet")
    import collections
    print(f"[role] wrote token_labels_with_role.parquet: {dict(collections.Counter(role))}")


if __name__ == "__main__":
    main()
