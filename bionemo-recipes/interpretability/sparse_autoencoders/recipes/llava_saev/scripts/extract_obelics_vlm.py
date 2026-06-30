#!/usr/bin/env python
"""Extract VLM residual activations from OBELICS into an SAE ActivationStore.

This is the SAE-V LLaVA recipe extractor. It uses the same store contract as the
BioReason-Pro SAE recipe:

  <store>/layer<L>/shard_*.parquet
  <store>/layer<L>/metadata.json
  <store>/token_labels.parquet

The sidecar is row-aligned with the activation rows and uses the existing
cross-modal metric column names: protein_id, token_index, position_type.

OBELICS rows contain interleaved image URLs and text spans. We fetch one image
URL per document and join non-empty text spans into the text turn.

Distributed extraction:
  torchrun --nproc_per_node=8 scripts/extract_obelics_vlm.py ...

Each rank writes <output>/.tmp_rank_<r>/, then rank 0 merges shards and sidecars
into <output>/ after all ranks finish.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image

from sae.activation_store import ActivationStore, ActivationStoreConfig


def find_decoder_layers(model):
    cands = [
        "model.language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in cands:
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return obj, path
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[0]):
                best = (mod, name)
    if best is None:
        raise RuntimeError("Could not locate decoder layers; inspect model with print(model).")
    return best[0], best[1]


def get_image_token_id(model):
    cfg = model.config
    for attr in ("image_token_id", "image_token_index"):
        if hasattr(cfg, attr) and getattr(cfg, attr) is not None:
            return int(getattr(cfg, attr))
    raise RuntimeError("No image_token_id/image_token_index on config.")


def distributed_context():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return rank, world, local_rank


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(str(text).split()[:max_words])


def obelics_text(ex) -> str | None:
    texts = ex.get("texts") or []
    spans = [str(t).strip() for t in texts if t and str(t).strip()]
    if not spans:
        return None
    return " ".join(spans)


def obelics_image_urls(ex) -> list[str]:
    urls = []
    for u in ex.get("images") or []:
        if isinstance(u, str) and u.startswith(("http://", "https://")):
            urls.append(u)
    return urls


def fetch_image(session: requests.Session, urls: list[str], timeout: float, min_size: int) -> Image.Image | None:
    for url in urls:
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
            if image.width >= min_size and image.height >= min_size:
                return image
        except Exception:
            continue
    return None


def write_rank_store(args, rank: int, world: int, local_rank: int) -> dict:
    dev = f"cuda:{local_rank}"
    tdtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoVLM

    print(f"[rank {rank}] load {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoVLM.from_pretrained(args.model, torch_dtype=tdtype, low_cpu_mem_usage=True).to(dev).eval()

    layers, layer_path = find_decoder_layers(model)
    hidden = (
        model.config.get_text_config().hidden_size
        if hasattr(model.config, "get_text_config")
        else getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    )
    img_id = get_image_token_id(model)
    if not (0 <= args.layer < len(layers)):
        raise ValueError(f"layer {args.layer} out of range [0,{len(layers)}) at {layer_path}")
    print(
        f"[rank {rank}] layers='{layer_path}' n_layers={len(layers)} hidden={hidden} image_token_id={img_id}",
        flush=True,
    )

    captured = {}

    def hook(module, inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = layers[args.layer].register_forward_hook(hook)

    final_root = Path(args.output)
    rank_root = final_root if world == 1 else final_root / f".tmp_rank_{rank}"
    if rank_root.exists() and args.overwrite:
        shutil.rmtree(rank_root)
    rank_root.mkdir(parents=True, exist_ok=True)

    layer_dir = rank_root / f"layer{args.layer}"
    store = ActivationStore(layer_dir, ActivationStoreConfig(shard_size=args.shard_size))

    schema = pa.schema([("protein_id", pa.string()), ("token_index", pa.int32()), ("position_type", pa.string())])
    sidecar = pq.ParquetWriter(str(rank_root / "token_labels.parquet"), schema, compression="snappy")

    target = args.num_samples // world + (1 if rank < (args.num_samples % world) else 0)
    scan_limit = args.scan_limit or args.num_samples * args.scan_multiplier
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "bionemo-saev-llava-extractor/0.1"})

    buf = []
    buf_rows = 0
    n_done = 0
    n_seen = 0
    n_fetch_fail = 0
    n_no_text = 0
    n_preprocess_fail = 0
    n_tokens = 0
    n_img_tokens = 0

    def flush(force=False):
        nonlocal buf, buf_rows
        while buf_rows >= args.shard_size or (force and buf_rows > 0):
            block = np.concatenate(buf, axis=0)
            take = block.shape[0] if force else (block.shape[0] // args.shard_size) * args.shard_size
            if take == 0:
                break
            store.append(np.ascontiguousarray(block[:take]))
            rem = block[take:]
            buf = [rem] if rem.shape[0] else []
            buf_rows = rem.shape[0]
            if force:
                break

    with torch.no_grad():
        for idx, ex in enumerate(ds):
            if idx >= scan_limit or n_done >= target:
                break
            if idx % world != rank:
                continue
            n_seen += 1
            text_body = obelics_text(ex)
            if not text_body:
                n_no_text += 1
                continue
            image = fetch_image(session, obelics_image_urls(ex), args.image_timeout, args.min_image_size)
            if image is None:
                n_fetch_fail += 1
                continue
            caption = truncate_words(text_body, args.max_text_words)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": caption}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            try:
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(dev)
            except Exception as exc:
                n_preprocess_fail += 1
                if n_preprocess_fail <= 5:
                    print(
                        f"[rank {rank}] skip preprocess_fail={n_preprocess_fail} "
                        f"idx={idx} mode={getattr(image, 'mode', None)} size={getattr(image, 'size', None)} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                continue

            captured.clear()
            _ = model(**inputs)
            h = captured["h"]
            ids = inputs["input_ids"][0]
            S = ids.shape[0]
            assert h.shape[1] == S, f"hidden seq {h.shape[1]} != input_ids {S}; image-token expansion mismatch"

            acts = h[0].float().cpu().numpy()
            bands = np.where(ids.detach().cpu().numpy() == img_id, "image", "text")
            pid = f"obelics_r{rank:03d}_{n_done:07d}"

            buf.append(acts)
            buf_rows += acts.shape[0]
            sidecar.write_batch(
                pa.record_batch(
                    [
                        pa.array([pid] * S, pa.string()),
                        pa.array(np.arange(S, dtype=np.int32)),
                        pa.array(bands.tolist(), pa.string()),
                    ],
                    schema=schema,
                )
            )
            n_done += 1
            n_tokens += S
            n_img_tokens += int((bands == "image").sum())
            flush(force=False)
            if n_done % args.log_every == 0:
                print(
                    f"[rank {rank}] {n_done}/{target} docs, {n_tokens} tokens "
                    f"({100*n_img_tokens/max(n_tokens,1):.1f}% image), "
                    f"seen={n_seen} fetch_fail={n_fetch_fail} no_text={n_no_text} "
                    f"preprocess_fail={n_preprocess_fail}",
                    flush=True,
                )

    handle.remove()
    flush(force=True)
    sidecar.close()
    store.finalize(
        metadata={
            "model": args.model,
            "layer": args.layer,
            "hidden_dim": int(hidden),
            "n_documents": n_done,
            "dataset": args.dataset,
            "split": args.split,
            "modalities": "image|text",
            "hook": f"{layer_path}[{args.layer}] output[0] (residual after block {args.layer})",
            "image_token_id": img_id,
            "rank": rank,
            "world_size": world,
            "preprocess_fail": n_preprocess_fail,
        }
    )
    print(
        f"[rank {rank}] done docs={n_done} tokens={n_tokens} image={n_img_tokens} "
        f"fetch_fail={n_fetch_fail} no_text={n_no_text} preprocess_fail={n_preprocess_fail} -> {rank_root}",
        flush=True,
    )
    return {
        "rank": rank,
        "docs": n_done,
        "tokens": n_tokens,
        "image_tokens": n_img_tokens,
        "fetch_fail": n_fetch_fail,
        "no_text": n_no_text,
        "preprocess_fail": n_preprocess_fail,
        "hidden": int(hidden),
        "layer_path": layer_path,
        "image_token_id": img_id,
    }


def merge_rank_stores(args, world: int):
    root = Path(args.output)
    final_layer = root / f"layer{args.layer}"
    if final_layer.exists() and args.overwrite:
        shutil.rmtree(final_layer)
    final_layer.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([("protein_id", pa.string()), ("token_index", pa.int32()), ("position_type", pa.string())])
    sidecar_writer = pq.ParquetWriter(str(root / "token_labels.parquet"), schema, compression="snappy")

    shard_idx = 0
    total_rows = 0
    total_docs = 0
    total_image = 0
    total_text = 0
    total_preprocess_fail = 0
    merged_meta = None
    rank_summaries = []

    for rank in range(world):
        rank_root = root / f".tmp_rank_{rank}"
        rank_layer = rank_root / f"layer{args.layer}"
        meta_path = rank_layer / "metadata.json"
        if not meta_path.exists():
            print(f"[merge] missing rank {rank} metadata, skipping", flush=True)
            continue
        meta = json.loads(meta_path.read_text())
        merged_meta = merged_meta or meta
        total_docs += int(meta.get("n_documents", 0))
        total_preprocess_fail += int(meta.get("preprocess_fail", 0))
        rank_summaries.append(meta)

        pf = pq.ParquetFile(rank_root / "token_labels.parquet")
        for batch in pf.iter_batches(batch_size=200_000):
            table = pa.Table.from_batches([batch], schema=schema)
            sidecar_writer.write_table(table)
            band = table.column("position_type").to_numpy(zero_copy_only=False)
            total_image += int((band == "image").sum())
            total_text += int((band == "text").sum())

        for sp in sorted(rank_layer.glob("shard_*.parquet"), key=lambda p: int(p.stem.split("_")[1])):
            rows = pq.read_metadata(sp).num_rows
            total_rows += rows
            shutil.copy2(sp, final_layer / f"shard_{shard_idx:05d}.parquet")
            shard_idx += 1

    sidecar_writer.close()
    if merged_meta is None:
        raise RuntimeError("No rank stores were merged.")
    metadata = {
        "n_samples": total_rows,
        "hidden_dim": int(merged_meta["hidden_dim"]),
        "n_shards": shard_idx,
        "shard_size": args.shard_size,
        "model": args.model,
        "layer": args.layer,
        "n_documents": total_docs,
        "dataset": args.dataset,
        "split": args.split,
        "modalities": "image|text",
        "hook": merged_meta["hook"],
        "image_token_id": merged_meta.get("image_token_id"),
        "world_size": world,
        "image_tokens": total_image,
        "text_tokens": total_text,
        "preprocess_fail": total_preprocess_fail,
        "rank_summaries": rank_summaries,
    }
    (final_layer / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (root / "extract_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"[merge] docs={total_docs} rows={total_rows} image={total_image} text={total_text} "
        f"shards={shard_idx} -> {root}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="HuggingFaceM4/OBELICS")
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=100_000, help="Target successful documents across all ranks")
    p.add_argument("--scan-limit", type=int, default=None, help="Maximum raw dataset rows to scan across all ranks")
    p.add_argument("--scan-multiplier", type=int, default=5)
    p.add_argument("--max-text-words", type=int, default=128)
    p.add_argument("--image-timeout", type=float, default=8.0)
    p.add_argument("--min-image-size", type=int, default=16)
    p.add_argument("--shard-size", type=int, default=200_000)
    p.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    rank, world, local_rank = distributed_context()
    write_rank_store(args, rank, world, local_rank)
    if world > 1:
        dist.barrier()
        if rank == 0:
            merge_rank_stores(args, world)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
