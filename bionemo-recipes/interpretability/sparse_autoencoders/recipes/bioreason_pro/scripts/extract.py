# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Extract BioReason-Pro residual-stream activations for SAE training (faithful fused forward).

Runs the authors' real multimodal teacher-forced forward (ESM3 protein embeds + trained GO band
spliced into placeholder slots), captures the residual stream at one or more LLM layers via a
forward hook on ``text_model.model.layers[L]``, drops pad tokens, and writes one SAE
``ActivationStore`` per layer plus a row-aligned label sidecar (position_type per token) and a
per-protein metadata table (go_ids, token counts).

Same code at every scale — only ``--num-proteins`` grows.

Smoke (Step A), 1 GPU:
    CUDA_VISIBLE_DEVICES=3 python scripts/extract.py --num-proteins 500 \
        --layers 24 28 32 35 --output /data/.../layer_smoke500 --verify-hidden-states

Subset (Step C):
    CUDA_VISIBLE_DEVICES=3 python scripts/extract.py --num-proteins 20000 --shuffle \
        --layers 28 --output /data/.../layer28_subset20k
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


_RECIPE_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_RECIPE_SRC))

DEFAULT_CKPT = (
    "/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
    "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b"
)
DEFAULT_BIOREASON_ROOT = "/data/savithas/bioreason-pro"
FULL_TRAIN_SIZE = 117002  # for full-split token/disk projection


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Extract BioReason-Pro activations for SAE training")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default=DEFAULT_BIOREASON_ROOT)
    p.add_argument("--output", required=True, help="Output dir; per-layer stores go under layer<L>/")
    p.add_argument("--layers", type=int, nargs="+", default=[24, 28, 32, 35],
                   help="Decoder layer indices to hook (output = residual stream after the block)")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--num-proteins", type=int, default=500)
    p.add_argument("--shuffle", action="store_true", help="Shuffle rows before selecting (subset sampling)")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    p.add_argument("--shard-size", type=int, default=200_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--verify-hidden-states", action="store_true",
                   help="On the first batch, confirm hook output == output_hidden_states[L+1]")
    p.add_argument("--no-go-pred", action="store_true",
                   help="Fusion control: drop GO-GPT predictions (go_speculations) from the prompt")
    p.add_argument("--inference-mode", action="store_true",
                   help="Fusion control: truncate after assistant-start (no reasoning/answer tokens)")
    p.add_argument("--report-json", default=None, help="Write the tokens/disk report to this JSON path")
    return p.parse_args()


class ShardAccumulator:
    """Buffer activations locally and call ``store.append`` only with full shards.

    ``ActivationStore.append`` re-concatenates its whole internal buffer on every call, which is
    O(n^2) when fed many small chunks before a large ``shard_size`` flush. We accumulate a python
    list and hand the store exactly one ``shard_size`` block at a time, so each ``append`` is O(1)
    concatenations of a single full shard.
    """

    def __init__(self, store, shard_size: int):
        self.store = store
        self.shard_size = shard_size
        self._chunks = []
        self._rows = 0

    def add(self, arr: np.ndarray):
        self._chunks.append(arr)
        self._rows += arr.shape[0]
        if self._rows >= self.shard_size:
            self._drain()

    def _drain(self):
        buf = np.concatenate(self._chunks, axis=0)
        n_full = (buf.shape[0] // self.shard_size) * self.shard_size
        if n_full:
            self.store.append(buf[:n_full])  # whole shards flush immediately inside the store
        rem = buf[n_full:]
        self._chunks = [rem] if rem.shape[0] else []
        self._rows = rem.shape[0]

    def close(self):
        if self._chunks:
            self.store.append(np.concatenate(self._chunks, axis=0))
        self._chunks = []
        self._rows = 0


class LabelSidecar:
    """Row-aligned per-token labels, flushed incrementally to one parquet (memory-safe at scale)."""

    _SCHEMA = pa.schema([
        ("protein_id", pa.string()),
        ("token_index", pa.int32()),   # position in the (unpadded) sequence
        ("position_type", pa.string()),
    ])

    def __init__(self, path: Path, flush_rows: int = 1_000_000):
        self.writer = pq.ParquetWriter(str(path), self._SCHEMA, compression="snappy")
        self.flush_rows = flush_rows
        self._pid, self._idx, self._pt = [], [], []

    def append(self, protein_id, token_indices, position_types):
        self._pid.extend([protein_id] * len(token_indices))
        self._idx.extend(int(i) for i in token_indices)
        self._pt.extend(str(t) for t in position_types)
        if len(self._pid) >= self.flush_rows:
            self._flush()

    def _flush(self):
        if not self._pid:
            return
        batch = pa.record_batch(
            [pa.array(self._pid, pa.string()), pa.array(self._idx, pa.int32()), pa.array(self._pt, pa.string())],
            schema=self._SCHEMA,
        )
        self.writer.write_batch(batch)
        self._pid, self._idx, self._pt = [], [], []

    def close(self):
        self._flush()
        self.writer.close()


def main():  # noqa: D103
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from sae.activation_store import ActivationStore, ActivationStoreConfig

    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft

    # --- Load faithful, hookable model ---
    print(f"[load] model from {args.ckpt_dir}")
    t0 = time.time()
    model = load_bioreason_pro_sft(
        ckpt_dir=args.ckpt_dir,
        bioreason_pro_root=args.bioreason_root,
        device=args.device,
        max_length_text=args.max_length_text,
        max_length_protein=args.max_length_protein,
    )
    pid_tok = model.protein_token_id
    gid_tok = model.go_token_id
    pad_tok = model.text_tokenizer.pad_token_id
    n_layers = model.text_config.num_hidden_layers
    hidden = model.text_config.hidden_size
    print(f"[load] done in {time.time()-t0:.1f}s | layers={n_layers} hidden={hidden} "
          f"protein_id={pid_tok} go_id={gid_tok} pad_id={pad_tok}")
    for L in args.layers:
        if not (0 <= L < n_layers):
            raise ValueError(f"layer {L} out of range [0,{n_layers})")

    # --- Data ---
    print(f"[data] loading reasoning splits (split={args.split})")
    train_ds, val_ds, test_ds = brp_data.load_reasoning_splits(
        max_length_protein=args.max_length_protein, include_go_pred=not args.no_go_pred)
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[args.split]
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    n = min(args.num_proteins, len(ds))
    ds = ds.select(range(n))
    print(f"[data] using {n} proteins from '{args.split}' (full split has {len(train_ds)} train rows) "
          f"| go_pred={'OFF' if args.no_go_pred else 'on'} inference_mode={args.inference_mode}")

    collate_fn = brp_data.make_collate_fn(
        model.text_tokenizer, max_length_text=args.max_length_text, max_length_protein=args.max_length_protein,
        inference_mode=args.inference_mode
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Hooks (capture all layers in one forward) ---
    captured = {}

    def mk_hook(L):
        def hook(module, inp, out):
            captured[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [model.text_model.model.layers[L].register_forward_hook(mk_hook(L)) for L in args.layers]

    # --- Per-layer stores + sidecar ---
    stores = {L: ActivationStore(out / f"layer{L}", ActivationStoreConfig(shard_size=args.shard_size))
              for L in args.layers}
    accum = {L: ShardAccumulator(stores[L], args.shard_size) for L in args.layers}
    sidecar = LabelSidecar(out / "token_labels.parquet")
    protein_meta = []  # per-protein: id, go_ids, counts

    counts = Counter()
    n_tokens_kept = 0
    fwd_time = 0.0
    verified = False
    global_row = 0  # dataset row index in DataLoader order (shuffle=False preserves it)
    t_start = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="extract"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            tf = time.time()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                protein_sequences=batch.get("protein_sequences"),
                batch_idx_map=batch.get("batch_idx_map"),
                structure_coords=batch.get("structure_coords"),
                labels=labels,
                go_aspects=batch.get("batch_go_aspects"),
                output_hidden_states=args.verify_hidden_states and not verified,
            )
            fwd_time += time.time() - tf

            # One-time hidden-states off-by-one verification.
            if args.verify_hidden_states and not verified:
                hs = outputs.hidden_states  # tuple len n_layers+1 (embeddings + each block)
                ok = {}
                for L in args.layers:
                    ok[L] = bool(torch.allclose(captured[L].float(), hs[L + 1].float(), atol=1e-3))
                print(f"[verify] len(hidden_states)={len(hs)} (==n_layers+1={n_layers + 1}); "
                      f"hook[L]==hidden_states[L+1]: {ok}; teacher-forced loss={outputs.loss.item():.4f}")
                verified = True

            bsz = input_ids.shape[0]
            for i in range(bsz):
                row = input_ids[i]
                tags, keep = brp_data.tag_and_keep_mask(row, pid_tok, gid_tok, pad_tok)
                keep_t = torch.from_numpy(keep).to(args.device)
                abs_index = np.nonzero(keep)[0]  # positions in the (left-padded) row
                # left-padded => kept tokens are a contiguous right block; renumber 0..n_kept-1
                token_index = abs_index - abs_index.min() if abs_index.size else abs_index

                for L in args.layers:
                    acts = captured[L][i][keep_t].float().cpu().numpy()  # (n_kept, hidden)
                    accum[L].add(acts)

                drow = ds[global_row]
                pid_val = str(drow.get("protein_id", f"row{global_row}"))
                tags_list = tags.tolist()
                c = Counter(tags_list)
                counts.update(tags_list)
                n_tokens_kept += len(tags_list)
                sidecar.append(pid_val, token_index, tags_list)
                go_ids = drow.get("go_ids") or []
                if isinstance(go_ids, str):
                    go_ids = [go_ids]
                protein_meta.append({
                    "protein_id": pid_val,
                    "go_ids": json.dumps(list(go_ids)),
                    "n_protein": int(c.get(brp_data.TAG_PROTEIN, 0)),
                    "n_go": int(c.get(brp_data.TAG_GO, 0)),
                    "n_text": int(c.get(brp_data.TAG_TEXT, 0)),
                })
                global_row += 1

    for h in handles:
        h.remove()
    for L in args.layers:
        accum[L].close()
        stores[L].finalize(metadata={
            "model": "bioreason-pro-sft", "layer": L, "hidden_dim": hidden,
            "n_proteins": n, "split": args.split, "position_types": "protein|go|text",
            "hook": f"text_model.model.layers[{L}] output[0] (residual after block {L})",
        })
    sidecar.close()
    pq.write_table(pa.Table.from_pylist(protein_meta), str(out / "proteins.parquet"))

    # --- Report ---
    elapsed = time.time() - t_start
    per_protein = {k: counts[k] / n for k in (brp_data.TAG_PROTEIN, brp_data.TAG_GO, brp_data.TAG_TEXT)}
    tot_per_protein = n_tokens_kept / n
    proj_full_tokens = tot_per_protein * FULL_TRAIN_SIZE
    bytes_per_tok_layer = hidden * 4  # float32 in store (raw upper bound)
    report = {
        "n_proteins": n, "split": args.split, "layers": args.layers,
        "tokens_kept_total": n_tokens_kept, "tokens_per_protein": tot_per_protein,
        "tokens_per_protein_by_type": per_protein,
        "counts_by_type": dict(counts),
        "forward_time_s": round(fwd_time, 1), "wall_time_s": round(elapsed, 1),
        "throughput_tok_per_s": round(n_tokens_kept / fwd_time) if fwd_time else None,
        "projected_full_train_tokens": int(proj_full_tokens),
        "projected_full_disk_gb_per_layer": round(proj_full_tokens * bytes_per_tok_layer / 1e9, 1),
    }
    print("\n=== EXTRACTION REPORT ===")
    print(json.dumps(report, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
