#!/usr/bin/env python
"""Run SAE-V paper Algorithm 1 metrics on a cached VLM activation store.

This is intentionally separate from ``crossmodal_cooccur.py``.  The original
control script asks whether a feature co-fires within many paired samples.  The
SAE-V paper's cosine-ranking metric instead:

1. samples a fixed number of data points,
2. collects activated hidden-state tokens for each feature,
3. computes a feature weight from the feature's global top-K text tokens and
   global top-K vision tokens, and
4. ranks data by the sum of weights of features activated in that data point.

The defaults mirror the paper appendix Table 5: top-K=5, activation_bound=1,
sample_data_size=1000.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from sae.activation_store import shard_table_to_array
from sae.architectures import TopKSAE


IMAGE = 0
TEXT = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sae", required=True, help="Path to checkpoint_final.pt")
    p.add_argument("--store", required=True, help="Activation store root")
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--sample-data-size", type=int, default=1000)
    p.add_argument("--activation-bound", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump-json", required=True)
    p.add_argument("--max-features-json", type=int, default=5000)
    p.add_argument("--max-samples-json", type=int, default=1000)
    return p.parse_args()


def load_sample_window(labels_path: Path, sample_data_size: int):
    """Load the complete row window for the first ``sample_data_size`` samples."""
    pf = pq.ParquetFile(labels_path)
    sample_ids: list[str] = []
    row_bands: list[str] = []
    row_sample: list[int] = []
    sample_order: list[str] = []

    current = None
    current_idx = -1
    stop = False
    for batch in pf.iter_batches(
        batch_size=200_000,
        columns=["protein_id", "position_type"],
    ):
        ids = batch.column(0).to_numpy(zero_copy_only=False)
        bands = batch.column(1).to_numpy(zero_copy_only=False)
        for pid, band in zip(ids, bands, strict=True):
            pid = str(pid)
            if current is None:
                current = pid
                current_idx = 0
                sample_order.append(pid)
            elif pid != current:
                if len(sample_order) >= sample_data_size:
                    stop = True
                    break
                current = pid
                current_idx += 1
                sample_order.append(pid)

            sample_ids.append(pid)
            row_bands.append(str(band))
            row_sample.append(current_idx)
        if stop:
            break

    if len(sample_order) != sample_data_size:
        raise RuntimeError(f"Requested {sample_data_size} samples, found {len(sample_order)}")

    return (
        np.asarray(sample_ids, dtype=object),
        np.asarray(row_bands, dtype=object),
        np.asarray(row_sample, dtype=np.int32),
        sample_order,
    )


def load_sae(path: Path, device: str) -> TopKSAE:
    ckpt = torch.load(path, map_location="cpu")
    sae = TopKSAE(**ckpt["model_config"]).to(device).eval()
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt["model_state_dict"].items()}
    sae.load_state_dict(state)
    return sae


def sparse_topk(sae: TopKSAE, x: torch.Tensor):
    pre_act, _ = sae.encode_pre_act(x)
    codes = torch.relu(pre_act)
    return torch.topk(codes, sae.top_k, dim=-1)


def update_feature_tops(top_vals, top_rows, modality: int, feats, vals, rows, k: int) -> None:
    if feats.size == 0:
        return
    order = np.lexsort((-vals, feats))
    feats = feats[order]
    vals = vals[order]
    rows = rows[order]
    uniq, starts = np.unique(feats, return_index=True)
    ends = np.r_[starts[1:], feats.size]
    for feat, start, end in zip(uniq, starts, ends, strict=True):
        # The feature group is already value-descending.  Only the first k can
        # possibly survive the merge with the existing top-k for this feature.
        end = min(end, start + k)
        merged_vals = np.concatenate([top_vals[modality, feat], vals[start:end]])
        merged_rows = np.concatenate([top_rows[modality, feat], rows[start:end]])
        keep = np.argsort(-merged_vals)[:k]
        top_vals[modality, feat] = merged_vals[keep]
        top_rows[modality, feat] = merged_rows[keep]


def iter_sample_shards(store: Path, layer: int, n_rows: int):
    row0 = 0
    shards = sorted(
        glob.glob(str(store / f"layer{layer}" / "shard_*.parquet")),
        key=lambda q: int(Path(q).stem.split("_")[1]),
    )
    for shard in shards:
        rows = pq.read_metadata(shard).num_rows
        if row0 >= n_rows:
            break
        take = min(rows, n_rows - row0)
        table = pq.read_table(shard)
        x = shard_table_to_array(table)
        if take < x.shape[0]:
            x = x[:take]
        yield row0, x
        row0 += rows


def collect_algorithm1_state(args, sae: TopKSAE, row_band, row_sample):
    n_rows = len(row_band)
    n_samples = int(row_sample.max()) + 1
    n_features = sae.hidden_dim
    k = args.top_k

    top_vals = np.full((2, n_features, k), -np.inf, dtype=np.float32)
    top_rows = np.full((2, n_features, k), -1, dtype=np.int64)
    active_any = np.zeros((n_samples, n_features), dtype=bool)
    active_image = np.zeros((n_samples, n_features), dtype=bool)
    active_text = np.zeros((n_samples, n_features), dtype=bool)

    store = Path(args.store)
    with torch.no_grad():
        for row0, x in iter_sample_shards(store, args.layer, n_rows):
            for start in range(0, x.shape[0], args.batch_size):
                end = min(start + args.batch_size, x.shape[0])
                global_rows = np.arange(row0 + start, row0 + end, dtype=np.int64)
                xb = torch.from_numpy(np.ascontiguousarray(x[start:end]).copy()).to(args.device)
                vals_t, feats_t = sparse_topk(sae, xb)
                vals = vals_t.float().cpu().numpy()
                feats = feats_t.cpu().numpy().astype(np.int64, copy=False)
                mask = vals > args.activation_bound
                if not mask.any():
                    continue

                repeated_rows = np.broadcast_to(global_rows[:, None], feats.shape)
                sample_idx = np.broadcast_to(row_sample[row0 + start : row0 + end, None], feats.shape)
                active_samples = sample_idx[mask]
                active_feats = feats[mask]
                active_any[active_samples, active_feats] = True

                bands = row_band[row0 + start : row0 + end]
                image_row_mask = bands == "image"
                text_row_mask = bands == "text"

                image_mask = mask & image_row_mask[:, None]
                if image_mask.any():
                    image_samples = sample_idx[image_mask]
                    image_feats = feats[image_mask]
                    active_image[image_samples, image_feats] = True
                    update_feature_tops(
                        top_vals,
                        top_rows,
                        IMAGE,
                        image_feats,
                        vals[image_mask].astype(np.float32, copy=False),
                        repeated_rows[image_mask],
                        k,
                    )

                text_mask = mask & text_row_mask[:, None]
                if text_mask.any():
                    text_samples = sample_idx[text_mask]
                    text_feats = feats[text_mask]
                    active_text[text_samples, text_feats] = True
                    update_feature_tops(
                        top_vals,
                        top_rows,
                        TEXT,
                        text_feats,
                        vals[text_mask].astype(np.float32, copy=False),
                        repeated_rows[text_mask],
                        k,
                    )

            print(f"[paper-metric] processed rows {min(row0 + x.shape[0], n_rows):,}/{n_rows:,}", flush=True)

    return top_vals, top_rows, active_any, active_image, active_text


def gather_vectors(args, wanted_rows: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    wanted_rows = np.asarray(sorted(set(int(r) for r in wanted_rows if int(r) >= 0)), dtype=np.int64)
    row_to_pos = {int(row): i for i, row in enumerate(wanted_rows)}
    if wanted_rows.size == 0:
        return np.empty((0, 0), dtype=np.float32), row_to_pos

    meta = json.loads((Path(args.store) / f"layer{args.layer}" / "metadata.json").read_text())
    vectors = np.empty((wanted_rows.size, int(meta["hidden_dim"])), dtype=np.float32)
    ptr = 0
    n_rows = int(wanted_rows[-1]) + 1
    for row0, x in iter_sample_shards(Path(args.store), args.layer, n_rows):
        row1 = row0 + x.shape[0]
        start = ptr
        while ptr < wanted_rows.size and wanted_rows[ptr] < row1:
            ptr += 1
        if ptr > start:
            local = wanted_rows[start:ptr] - row0
            vectors[start:ptr] = x[local]
            print(f"[paper-metric] gathered vectors {ptr:,}/{wanted_rows.size:,}", flush=True)
        if ptr >= wanted_rows.size:
            break
    return vectors, row_to_pos


def compute_feature_weights(top_rows, vectors, row_to_pos, top_k: int, chunk_size: int = 512):
    valid = np.where((top_rows[IMAGE] >= 0).all(axis=1) & (top_rows[TEXT] >= 0).all(axis=1))[0]
    weights = np.zeros(top_rows.shape[1], dtype=np.float32)
    if valid.size == 0:
        return weights, valid

    image_pos = np.asarray([[row_to_pos[int(r)] for r in top_rows[IMAGE, f]] for f in valid], dtype=np.int64)
    text_pos = np.asarray([[row_to_pos[int(r)] for r in top_rows[TEXT, f]] for f in valid], dtype=np.int64)

    for start in range(0, valid.size, chunk_size):
        end = min(start + chunk_size, valid.size)
        image_vecs = vectors[image_pos[start:end]]
        text_vecs = vectors[text_pos[start:end]]
        image_norm = image_vecs / (np.linalg.norm(image_vecs, axis=-1, keepdims=True) + 1e-8)
        text_norm = text_vecs / (np.linalg.norm(text_vecs, axis=-1, keepdims=True) + 1e-8)
        weights[valid[start:end]] = (image_norm * text_norm).sum(axis=-1).mean(axis=-1)
        print(f"[paper-metric] weighted features {end:,}/{valid.size:,}", flush=True)
    return weights, valid


def main() -> None:
    args = parse_args()
    store = Path(args.store)
    labels = store / "token_labels.parquet"
    if not labels.exists():
        raise FileNotFoundError(labels)

    row_ids, row_band, row_sample, sample_order = load_sample_window(labels, args.sample_data_size)
    print(
        f"[paper-metric] sample_data_size={len(sample_order)} rows={len(row_band):,} "
        f"image_rows={(row_band == 'image').sum():,} text_rows={(row_band == 'text').sum():,}",
        flush=True,
    )

    sae = load_sae(Path(args.sae), args.device)
    top_vals, top_rows, active_any, active_image, active_text = collect_algorithm1_state(
        args,
        sae,
        row_band,
        row_sample,
    )

    wanted_rows = top_rows[top_rows >= 0]
    vectors, row_to_pos = gather_vectors(args, wanted_rows)
    weights, valid = compute_feature_weights(top_rows, vectors, row_to_pos, args.top_k)

    weighted = np.where(weights != 0)[0]
    sample_cosine = active_any[:, weighted].astype(np.float32) @ weights[weighted]
    sample_l0 = active_any.sum(axis=1)
    sample_image_l0 = active_image.sum(axis=1)
    sample_text_l0 = active_text.sum(axis=1)
    sample_cooccur = (active_image & active_text).sum(axis=1)

    feature_order = valid[np.argsort(-weights[valid])]
    sample_order_idx = np.argsort(-sample_cosine)

    features = []
    for feat in feature_order[: args.max_features_json]:
        features.append(
            {
                "feature_id": int(feat),
                "weight": float(weights[feat]),
                "top_image_rows": [int(r) for r in top_rows[IMAGE, feat]],
                "top_image_activations": [float(v) for v in top_vals[IMAGE, feat]],
                "top_text_rows": [int(r) for r in top_rows[TEXT, feat]],
                "top_text_activations": [float(v) for v in top_vals[TEXT, feat]],
                "active_samples": int(active_any[:, feat].sum()),
                "image_active_samples": int(active_image[:, feat].sum()),
                "text_active_samples": int(active_text[:, feat].sum()),
                "cooccur_samples": int((active_image[:, feat] & active_text[:, feat]).sum()),
            }
        )

    samples = []
    for idx in sample_order_idx[: args.max_samples_json]:
        samples.append(
            {
                "sample_id": sample_order[int(idx)],
                "cosine_similarity_score": float(sample_cosine[idx]),
                "l0": int(sample_l0[idx]),
                "image_l0": int(sample_image_l0[idx]),
                "text_l0": int(sample_text_l0[idx]),
                "cooccurrence": int(sample_cooccur[idx]),
            }
        )

    out = {
        "metric": "sae-v-paper-algorithm1",
        "pair": "image-text",
        "layer": args.layer,
        "sample_data_size": args.sample_data_size,
        "activation_bound": args.activation_bound,
        "top_k": args.top_k,
        "paper_text_token_vocabulary_size": 32000,
        "paper_vision_token_vocabulary_size": 64,
        "n_rows": int(len(row_band)),
        "n_image_rows": int((row_band == "image").sum()),
        "n_text_rows": int((row_band == "text").sum()),
        "n_latents": int(sae.hidden_dim),
        "n_weighted_features": int(valid.size),
        "n_positive_weight_features": int((weights > 0).sum()),
        "n_weight_gt_0_3": int((weights > 0.3).sum()),
        "n_weight_gt_0_5": int((weights > 0.5).sum()),
        "mean_nonzero_weight": float(weights[weighted].mean()) if weighted.size else 0.0,
        "mean_l0": float(sample_l0.mean()),
        "mean_image_l0": float(sample_image_l0.mean()),
        "mean_text_l0": float(sample_text_l0.mean()),
        "mean_cooccurrence": float(sample_cooccur.mean()),
        "mean_cosine_similarity_score": float(sample_cosine.mean()),
        "features": features,
        "samples": samples,
    }

    Path(args.dump_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dump_json).write_text(json.dumps(out, indent=2))
    print(f"[paper-metric] wrote {args.dump_json}", flush=True)
    print(
        "[paper-metric] "
        f"weighted_features={valid.size} positive={int((weights > 0).sum())} "
        f">0.3={int((weights > 0.3).sum())} >0.5={int((weights > 0.5).sum())} "
        f"mean_l0={sample_l0.mean():.3f} mean_cooccur={sample_cooccur.mean():.3f} "
        f"mean_score={sample_cosine.mean():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
