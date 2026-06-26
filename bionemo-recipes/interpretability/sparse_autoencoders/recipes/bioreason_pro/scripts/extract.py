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

Same code at every scale — only ``--num-proteins`` grows. Multi-GPU follows the codonfm/esm2
recipe pattern (DDP via ``torchrun``): each rank extracts a contiguous slice of the dataset into
``<output>/.tmp_rank_<r>/``; rank 0 merges the per-rank stores + sidecar + proteins table into
``<output>/`` and writes ``extract_metadata.json`` (the cache-skip sentinel). Re-running a completed
output dir is a no-op.

Storage: write into the recipe's local cache on the shared FS, mirroring Jared's
``.cache/activations/<name>`` convention — e.g. ``cache_dir/activations/<name>`` (gitignored).

Single-GPU smoke (Step A):
    CUDA_VISIBLE_DEVICES=3 python scripts/extract.py --num-proteins 500 \
        --layers 24 28 32 35 --output cache_dir/activations/smoke500 --verify-hidden-states

Multi-GPU full extraction (4 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/extract.py \
        --num-proteins 117002 --layers 28 32 \
        --output cache_dir/activations/train_full_L28_L32
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    p.add_argument("--output", required=True,
                   help="Output dir (local cache, e.g. cache_dir/activations/<name>); per-layer stores go under layer<L>/")
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
    """Buffer activations locally and persist exactly one ``shard_size`` block at a time.

    ``ActivationStore.append`` re-concatenates its whole internal buffer on every call, which is
    O(n^2) when fed many small chunks before a large ``shard_size`` flush. We accumulate a python
    list and hand the store exactly one ``shard_size`` block at a time, so each ``append`` is O(1)
    concatenations of a single full shard.

    Writing a 200k x 2560 parquet shard takes ~10-17s; with one accumulator per layer those writes
    would block the model forward. If an ``executor`` (a single-worker thread per layer) is given,
    each full shard is written in the background so the next forward proceeds, and the 5 layers'
    writes run concurrently. ``max_pending`` bounds outstanding shards (memory backpressure).
    """

    def __init__(self, store, shard_size: int, executor=None, max_pending: int = 2):
        self.store = store
        self.shard_size = shard_size
        self.executor = executor
        self.max_pending = max_pending
        self._chunks = []
        self._rows = 0
        self._futures = []

    def add(self, arr: np.ndarray):
        self._chunks.append(arr)
        self._rows += arr.shape[0]
        if self._rows >= self.shard_size:
            self._drain()

    def _persist(self, block: np.ndarray):
        if self.executor is None:
            self.store.append(block)
            return
        while len(self._futures) >= self.max_pending:
            self._futures.pop(0).result()  # backpressure + surfaces writer exceptions
        self._futures.append(self.executor.submit(self.store.append, block))

    def _drain(self):
        buf = np.concatenate(self._chunks, axis=0)
        n_full = (buf.shape[0] // self.shard_size) * self.shard_size
        if n_full:
            self._persist(np.ascontiguousarray(buf[:n_full]))  # own contiguous block for the writer
        rem = buf[n_full:]
        self._chunks = [rem.copy()] if rem.shape[0] else []
        self._rows = rem.shape[0]

    def close(self):
        if self._chunks:
            self._persist(np.ascontiguousarray(np.concatenate(self._chunks, axis=0)))
        self._chunks = []
        self._rows = 0
        for f in self._futures:
            f.result()
        self._futures = []


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


def _merge_layer(out: Path, world_size: int, L: int, base_meta: dict, hidden: int) -> int:
    """Move per-rank layer shards into out/layer<L>/, renumbered + concatenated in rank order."""
    final = out / f"layer{L}"
    final.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    total = 0
    for r in range(world_size):
        tmp = out / f".tmp_rank_{r}" / f"layer{L}"
        meta_path = tmp / "metadata.json"
        if not meta_path.exists():
            print(f"  WARNING: rank {r} layer {L} did not finalize — skipping its shards.")
            continue
        m = json.loads(meta_path.read_text())
        for i in range(m["n_shards"]):
            shutil.move(str(tmp / f"shard_{i:05d}.parquet"), str(final / f"shard_{shard_idx:05d}.parquet"))
            shard_idx += 1
        total += m["n_samples"]
    meta = dict(base_meta)
    meta.update(layer=L, hidden_dim=hidden, n_samples=total, n_shards=shard_idx,
                hook=f"text_model.model.layers[{L}] output[0] (residual after block {L})")
    (final / "metadata.json").write_text(json.dumps(meta, indent=2))
    return total


def _merge_sidecar(out: Path, world_size: int) -> None:
    """Stream per-rank token_labels.parquet into one (row-group copy, memory-safe at 400M+ rows)."""
    writer = None
    for r in range(world_size):
        f = out / f".tmp_rank_{r}" / "token_labels.parquet"
        if not f.exists():
            continue
        pf = pq.ParquetFile(str(f))
        if writer is None:
            writer = pq.ParquetWriter(str(out / "token_labels.parquet"), pf.schema_arrow, compression="snappy")
        for i in range(pf.num_row_groups):
            writer.write_table(pf.read_row_group(i))
    if writer is not None:
        writer.close()


def _merge_proteins(out: Path, world_size: int) -> None:
    """Concatenate per-rank proteins.parquet in rank order (small: ~117k rows)."""
    tables = []
    for r in range(world_size):
        f = out / f".tmp_rank_{r}" / "proteins.parquet"
        if f.exists():
            tables.append(pq.read_table(f))
    if tables:
        pq.write_table(pa.concat_tables(tables), str(out / "proteins.parquet"))


def _merge_all(out: Path, world_size: int, layers, base_meta: dict, hidden: int) -> dict:
    """Rank-0 merge of per-rank temp stores → final layout. Returns per-layer token totals."""
    totals = {L: _merge_layer(out, world_size, L, base_meta, hidden) for L in layers}
    _merge_sidecar(out, world_size)
    _merge_proteins(out, world_size)
    for r in range(world_size):
        shutil.rmtree(out / f".tmp_rank_{r}", ignore_errors=True)
    return totals


def _measured_bytes_per_token(out: Path, layers, total_tokens: int):
    """Actual on-disk (snappy-compressed) bytes/token, measured from the written shards.

    Returns (compressed_bytes_per_token_per_layer, raw_float32_bytes_per_token). The compressed
    figure is what disk projections should use; raw float32 is the uncompressed upper bound.
    """
    import glob

    by_layer = []
    for L in layers:
        b = sum(os.path.getsize(f) for f in glob.glob(str(out / f"layer{L}" / "shard_*.parquet")))
        by_layer.append(b)
    avg = (sum(by_layer) / len(by_layer)) / total_tokens if (by_layer and total_tokens) else 0.0
    return avg, None


def _corpus_report(out: Path, args, hidden: int, wall_s: float, fwd_s: float, world_size: int) -> dict:
    """Build the tokens/disk report from the merged proteins.parquet (corpus-level)."""
    t = pq.read_table(out / "proteins.parquet")
    cnt = {b: int(np.asarray(t.column(f"n_{b}").to_numpy(zero_copy_only=False)).sum())
           for b in ("protein", "go", "text")}
    n = t.num_rows
    total = sum(cnt.values())
    per_protein = {b: cnt[b] / n for b in cnt}
    tot_per_protein = total / n
    proj_full_tokens = tot_per_protein * FULL_TRAIN_SIZE
    raw_bytes_per_tok = hidden * 4  # float32, uncompressed upper bound
    comp_bytes_per_tok, _ = _measured_bytes_per_token(out, args.layers, total)
    return {
        "n_proteins": n, "split": args.split, "layers": args.layers, "world_size": world_size,
        "tokens_kept_total": total, "tokens_per_protein": tot_per_protein,
        "tokens_per_protein_by_type": per_protein,
        "counts_by_type": cnt,
        "forward_time_s_rank0": round(fwd_s, 1), "wall_time_s_rank0": round(wall_s, 1),
        "measured_compressed_bytes_per_token": round(comp_bytes_per_tok, 1),
        "raw_float32_bytes_per_token": raw_bytes_per_tok,
        "projected_full_train_tokens": int(proj_full_tokens),
        # disk projections use the measured snappy-compressed size (what actually lands on disk)
        "projected_full_disk_gb_per_layer": round(proj_full_tokens * comp_bytes_per_tok / 1e9, 1),
        "projected_full_disk_gb_all_layers": round(
            proj_full_tokens * comp_bytes_per_tok / 1e9 * len(args.layers), 1),
        "projected_full_disk_gb_per_layer_raw_upper_bound": round(
            proj_full_tokens * raw_bytes_per_tok / 1e9, 1),
    }


def main():  # noqa: D103
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Distributed setup (codonfm/esm2 pattern: torchrun --nproc_per_node=N) ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        from datetime import timedelta

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=48))
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = args.device if torch.cuda.is_available() else "cpu"
    args.device = device
    is_main = rank == 0
    print(f"[rank {rank}/{world_size}] device={device}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # --- Cache skip (Jared's behavior: a completed output dir is a no-op) ---
    sentinel = out / "extract_metadata.json"
    if sentinel.exists():
        if is_main:
            meta = json.loads(sentinel.read_text())
            print(f"[cache] {out} already complete: {meta.get('tokens_kept_total', '?'):,} tokens. Skipping.")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    # rank 0 clears stale temp dirs before any rank writes its own
    if is_main:
        for tmp in out.glob(".tmp_rank_*"):
            shutil.rmtree(tmp, ignore_errors=True)
    if world_size > 1:
        dist.barrier()

    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft
    from sae.activation_store import ActivationStore, ActivationStoreConfig

    # --- Load faithful, hookable model (one copy per rank/GPU) ---
    if is_main:
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
    if is_main:
        print(f"[load] done in {time.time()-t0:.1f}s | layers={n_layers} hidden={hidden} "
              f"protein_id={pid_tok} go_id={gid_tok} pad_id={pad_tok}")
    for L in args.layers:
        if not (0 <= L < n_layers):
            raise ValueError(f"layer {L} out of range [0,{n_layers})")

    # --- Data ---
    if is_main:
        print(f"[data] loading reasoning splits (split={args.split})")
    train_ds, val_ds, test_ds = brp_data.load_reasoning_splits(
        max_length_protein=args.max_length_protein, include_go_pred=not args.no_go_pred)
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[args.split]
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    n = min(args.num_proteins, len(ds))
    ds = ds.select(range(n))

    # Shard the selected proteins across ranks (contiguous slices, like codonfm/esm2)
    if world_size > 1:
        chunk = n // world_size
        start = rank * chunk
        end = n if rank == world_size - 1 else (rank + 1) * chunk
        ds = ds.select(range(start, end))
        print(f"[rank {rank}] proteins {start}-{end} ({len(ds)})")
    n_local = len(ds)
    if is_main:
        print(f"[data] {n} proteins total from '{args.split}' (full split has {len(train_ds)} train rows) "
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

    # --- Per-rank work dir: temp under output for multi-GPU, output itself for single-GPU ---
    work = out / f".tmp_rank_{rank}" if world_size > 1 else out
    work.mkdir(parents=True, exist_ok=True)

    # one background writer thread per layer so the 5 parquet writes (~10s each) overlap each
    # other and the model forward instead of serially blocking it
    writer_pools = {L: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"wL{L}") for L in args.layers}
    stores = {L: ActivationStore(work / f"layer{L}", ActivationStoreConfig(shard_size=args.shard_size))
              for L in args.layers}
    accum = {L: ShardAccumulator(stores[L], args.shard_size, executor=writer_pools[L]) for L in args.layers}
    sidecar = LabelSidecar(work / "token_labels.parquet")
    protein_meta = []  # per-protein: id, go_ids, counts

    counts = Counter()
    n_tokens_kept = 0
    fwd_time = 0.0
    do_verify = args.verify_hidden_states and is_main
    verified = False
    global_row = 0  # row index within this rank's ds slice (shuffle=False preserves order)
    t_start = time.time()

    iterator = tqdm(loader, desc="extract") if is_main else loader
    with torch.no_grad():
        for batch in iterator:
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
                output_hidden_states=do_verify and not verified,
            )
            fwd_time += time.time() - tf

            # One-time hidden-states off-by-one verification (rank 0 only).
            if do_verify and not verified:
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

            del outputs  # free logits (~GBs) so GPU memory doesn't creep up over the run
            if global_row % 200 == 0:
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()
    base_meta = {
        "model": "bioreason-pro-sft", "split": args.split, "position_types": "protein|go|text",
        "n_proteins": n,
    }
    for L in args.layers:
        accum[L].close()
        stores[L].finalize(metadata={
            "model": "bioreason-pro-sft", "layer": L, "hidden_dim": hidden,
            "n_proteins": n_local, "split": args.split, "position_types": "protein|go|text",
            "hook": f"text_model.model.layers[{L}] output[0] (residual after block {L})",
        })
    for p in writer_pools.values():
        p.shutdown(wait=True)
    sidecar.close()
    pq.write_table(pa.Table.from_pylist(protein_meta), str(work / "proteins.parquet"))
    print(f"[rank {rank}] {n_tokens_kept:,} tokens from {n_local} proteins in {time.time()-t_start:.1f}s", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Multi-GPU merge (rank 0) ---
    if world_size > 1:
        dist.barrier()
        if is_main:
            _merge_all(out, world_size, args.layers, base_meta, hidden)
        dist.barrier()
        dist.destroy_process_group()

    # --- Report + sentinel (rank 0 / single-GPU) ---
    if is_main:
        elapsed = time.time() - t_start
        report = _corpus_report(out, args, hidden, elapsed, fwd_time, world_size)
        sentinel.write_text(json.dumps(report, indent=2))
        print("\n=== EXTRACTION REPORT ===")
        print(json.dumps(report, indent=2))
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"Output: {out}")


if __name__ == "__main__":
    main()
