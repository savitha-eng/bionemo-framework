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

"""Step 2: Train a TopK SAE from cached BioReason-Pro residual-stream activations.

Reads a single-layer ActivationStore produced by ``extract.py`` (the model is never loaded here,
so this is GPU-light and pure-``sae`` / Env B) and trains a Sparse Autoencoder. Adapted from the
evo2 recipe's train.py (identical ``sae.training.Trainer`` wrapper + the #1619 training-quality
flags); only the cache-metadata validation differs (BioReason-Pro stores ``model``/``layer``/
``n_proteins``).

Subset SAE (Step C), single layer, with the validated #1619 flags:
    python scripts/train.py --cache-dir /data/.../layer28_subset20k --layer 28 \
        --expansion-factor 8 --top-k 32 --batch-size 4096 --n-epochs 1 \
        --aggregate-loss --dead-count-global --mix-shards 8 --presample-shards 8 --init-pre-bias \
        --checkpoint-dir /data/.../sae_layer28_subset20k

Opt-in #1619 training-quality fixes (default to previous behavior):
    --aggregate-loss      batch-level FVU + AuxK loss instead of the per-token ratio
    --dead-count-global   count dead-latent inactivity in total tokens (x world_size) under DDP
    --mix-shards N        shuffle + blend N shards per batch (N>1)
    --presample-shards N  spread the pre-bias-init sample across N shards (N>1)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sae.activation_store import load_activations
from sae.architectures import ReLUSAE, TopKSAE
from sae.perf_logger import PerfLogger
from sae.training import ParallelConfig, Trainer, TrainingConfig, WandbConfig
from sae.utils import get_device, set_seed


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(
        description="Train a TopK SAE from cached BioReason-Pro activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-dir", type=str, required=True, help="Path to activation cache (from extract.py)")
    p.add_argument("--balance-modality", action="store_true",
                   help="modality-balance at load time: drop <go>, downsample text to ~=protein (no data rewrite)")
    p.add_argument("--balance-protein-frac", type=float, default=0.5,
                   help="target bio:text token ratio when --balance-modality (0.5=50/50, 0.3=30/70)")
    p.add_argument("--bio-band", type=str, default="protein",
                   help="name of the bio-modality band in token_labels: 'protein' (protein store) or 'dna' (DNA store)")
    p.add_argument("--layer", type=int, required=True, help="Layer index (validated against cache metadata)")

    sae_group = p.add_argument_group("SAE model")
    sae_group.add_argument("--model-type", type=str, default="topk", choices=["topk", "relu"])
    sae_group.add_argument("--expansion-factor", type=int, default=8)
    sae_group.add_argument("--matryoshka-groups", type=str, default=None,
                           help="comma-separated group fractions for a Matryoshka TopK SAE, e.g. "
                                "'0.5,0.25,0.125,0.0625,0.0625' (MAIRA-2 scheme). Omit for vanilla TopK.")
    sae_group.add_argument("--batch-topk", action="store_true",
                           help="BatchTopK sparsity (top k*batch across the batch, EMA inference threshold) "
                                "instead of per-token TopK — the MAIRA-2 multimodal recipe.")
    sae_group.add_argument("--matryoshka-stochastic", action="store_true",
                           help="sample Matryoshka prefix bounds each step (Pareto) instead of fixed.")
    sae_group.add_argument("--matryoshka-modality-weights", type=str, default=None,
                           help="per-prefix protein weight alpha (comma list, one per group), e.g. "
                                "'0.8,0.65,0.5,0.5,0.5' (protein up-weighted in coarse prefixes). Each prefix "
                                "loss = alpha*FVU_protein + (1-alpha)*FVU_text; tokens self-classified via "
                                "protein_dir calibrated from a presample. Requires --balance-modality.")
    sae_group.add_argument("--top-k", type=int, default=32)
    sae_group.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--auxk", type=int, default=None)
    sae_group.add_argument("--auxk-coef", type=float, default=1 / 32)
    sae_group.add_argument("--dead-tokens-threshold", type=int, default=10_000_000)
    sae_group.add_argument("--init-pre-bias", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--l1-coeff", type=float, default=1e-2, help="L1 coefficient (relu only)")
    sae_group.add_argument("--aggregate-loss", action=argparse.BooleanOptionalAction, default=False,
                           help="Batch-level FVU + AuxK loss instead of the per-token ratio (topk only).")
    sae_group.add_argument("--dead-count-global", action=argparse.BooleanOptionalAction, default=False,
                           help="Count dead-latent inactivity in total tokens (x world_size) under DDP (topk only).")
    sae_group.add_argument("--normalize-loss", action=argparse.BooleanOptionalAction, default=False,
                           help="Compute the FVU loss in normalized space (equal per-token weight, "
                                "matches the honest var_explained metric; needs --normalize-input).")

    train_group = p.add_argument_group("Training")
    train_group.add_argument("--lr", type=float, default=3e-4)
    train_group.add_argument("--n-epochs", type=int, default=1)
    train_group.add_argument("--batch-size", type=int, default=4096)
    train_group.add_argument("--log-interval", type=int, default=50)
    train_group.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    train_group.add_argument("--num-workers", type=int, default=0)
    train_group.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument("--max-grad-norm", type=float, default=None)
    train_group.add_argument("--lr-scale-with-latents", action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument("--lr-reference-hidden-dim", type=int, default=2560)
    train_group.add_argument("--warmup-steps", type=int, default=0)
    train_group.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine", "linear"])
    train_group.add_argument("--lr-min", type=float, default=0.0)
    train_group.add_argument("--lr-decay-steps", type=int, default=None)
    train_group.add_argument("--mix-shards", type=int, default=1,
                             help="Shuffle + blend this many shards per batch (>1).")
    train_group.add_argument("--presample-shards", type=int, default=1,
                             help="Spread the pre-bias-init sample across this many shards (>1; needs --init-pre-bias).")

    wb_group = p.add_argument_group("Weights & Biases")
    wb_group.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, dest="wandb_enabled")
    wb_group.add_argument("--wandb-project", type=str, default="bioreason-pro-sae")
    wb_group.add_argument("--wandb-run-name", type=str, default=None)
    wb_group.add_argument("--wandb-group", type=str, default=None)
    wb_group.add_argument("--wandb-job-type", type=str, default=None)

    ckpt_group = p.add_argument_group("Checkpointing")
    ckpt_group.add_argument("--checkpoint-dir", type=str, default=None)
    ckpt_group.add_argument("--checkpoint-steps", type=int, default=None)
    ckpt_group.add_argument("--resume-from", type=str, default=None)

    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-proteins", type=int, default=None,
                   help="Subset cached activations to this many proteins' worth of shards")
    return p.parse_args()


def build_sae(args, input_dim: int) -> torch.nn.Module:  # noqa: D103
    hidden_dim = input_dim * args.expansion_factor
    if args.model_type == "topk":
        return TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            top_k=args.top_k,
            normalize_input=args.normalize_input,
            auxk=args.auxk,
            auxk_coef=args.auxk_coef,
            dead_tokens_threshold=args.dead_tokens_threshold,
            aggregate_loss=args.aggregate_loss,
            dead_count_global=args.dead_count_global,
            normalize_loss=args.normalize_loss,
            matryoshka_groups=([float(g) for g in args.matryoshka_groups.split(",")]
                               if getattr(args, "matryoshka_groups", None) else None),
            batch_topk=getattr(args, "batch_topk", False),
            matryoshka_stochastic=getattr(args, "matryoshka_stochastic", False),
            matryoshka_modality_weights=([float(a) for a in args.matryoshka_modality_weights.split(",")]
                                         if getattr(args, "matryoshka_modality_weights", None) else None),
        )
    elif args.model_type == "relu":
        return ReLUSAE(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=args.l1_coeff)
    raise ValueError(f"Unknown model type: {args.model_type}")


def build_training_config(args, device: str) -> TrainingConfig:  # noqa: D103
    return TrainingConfig(
        lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size, device=device,
        log_interval=args.log_interval, shuffle=args.shuffle, num_workers=args.num_workers,
        pin_memory=args.pin_memory, checkpoint_dir=args.checkpoint_dir, checkpoint_steps=args.checkpoint_steps,
        lr_scale_with_latents=args.lr_scale_with_latents, lr_reference_hidden_dim=args.lr_reference_hidden_dim,
        warmup_steps=args.warmup_steps, max_grad_norm=args.max_grad_norm, lr_schedule=args.lr_schedule,
        lr_min=args.lr_min, lr_decay_steps=args.lr_decay_steps,
    )


def main():  # noqa: D103
    args = parse_args()
    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")
    print(f"Config: {vars(args)}")

    cache_path = Path(args.cache_dir)
    if not (cache_path / "metadata.json").exists():
        raise FileNotFoundError(f"No cache found at {cache_path}. Run extract.py first.")
    store = load_activations(cache_path)
    meta = store.metadata

    # Opt-in MODALITY BALANCING (no data rewrite): drop <go> + downsample text to ~=bio, at load time.
    # bio_band is "protein" for the protein store, "dna" for the DNA store (whose bio tokens dominate).
    if getattr(args, "balance_modality", False):
        import pyarrow.parquet as _pq
        bio = getattr(args, "bio_band", "protein")
        tl = _pq.read_table(cache_path.parent / "token_labels.parquet")
        band = tl.column("position_type").to_numpy(zero_copy_only=False)  # to_numpy >> to_pylist on the 421M-row full store
        n_p = int((band == bio).sum()); n_t = int((band == "text").sum())
        frac = args.balance_protein_frac  # target bio fraction of kept tokens
        # Downsample whichever modality is OVER its target share (keep the scarcer one whole). The protein
        # store has text as the majority (downsample text); the DNA store has bio as the majority (~98% DNA,
        # so we must downsample DNA instead — the old text-only formula would leave it ~98% bio).
        cur_bio_frac = n_p / max(n_p + n_t, 1)
        if cur_bio_frac >= frac:  # bio too dominant -> downsample bio, keep all text
            bio_keep = min(1.0, (frac / (1.0 - frac)) * (n_t / max(n_p, 1))); text_keep = 1.0
        else:                     # text too dominant -> downsample text, keep all bio
            bio_keep = 1.0; text_keep = min(1.0, ((1.0 - frac) / frac) * (n_p / max(n_t, 1)))
        keep_probs = {"go": 0.0, bio: bio_keep, "text": text_keep}
        store.set_modality_balance(band, keep_probs)
        print(f"[balance] modality-balanced loader: drop go, target {bio}-frac={frac} -> "
              f"{bio}-keep={bio_keep:.4f} text-keep={text_keep:.4f} ({bio}={n_p:,} text={n_t:,})", flush=True)

    # Cache validation (BioReason-Pro metadata keys).
    if meta.get("layer") != args.layer:
        raise ValueError(f"Cache layer mismatch: {meta.get('layer')} vs {args.layer}")
    print(f"Cache: model={meta.get('model')} layer={meta.get('layer')} "
          f"hidden_dim={meta.get('hidden_dim')} n_proteins={meta.get('n_proteins')} n_shards={meta.get('n_shards')}")

    # Optional protein-count subsetting (by shard fraction).
    cached_proteins = meta.get("n_proteins", None)
    max_shards = None
    if args.num_proteins and cached_proteins and args.num_proteins < cached_proteins:
        keep = args.num_proteins / cached_proteins
        max_shards = max(1, int(np.ceil(keep * meta["n_shards"])))
        print(f"Subsetting {args.num_proteins}/{cached_proteins} proteins -> {max_shards}/{meta['n_shards']} shards")

    n_shards_to_use = max_shards or meta["n_shards"]
    shard_size = meta.get("shard_size", 200_000)
    est_gb = n_shards_to_use * shard_size * meta["hidden_dim"] * 4 / (1024**3)
    use_streaming = est_gb > 50

    input_dim = meta["hidden_dim"]
    sae = build_sae(args, input_dim)
    print(f"SAE: {args.model_type} input_dim={input_dim} hidden_dim={sae.hidden_dim} "
          f"(expansion={args.expansion_factor}, top_k={args.top_k})")

    # Calibrate the protein/text classifier for per-prefix modality-weighted loss (opt-in). Reads RAW
    # shard-0 (before the balance filter scrambles row alignment), computes the protein-minus-text
    # direction in per-token-normalized space, sets sae.protein_dir/protein_thresh, and VALIDATES
    # classification accuracy against the true band labels — only trustworthy because protein/text are
    # near-orthogonal (RAW_PT~0). If accuracy is low, the modality weighting would be noise.
    if getattr(args, "matryoshka_modality_weights", None):
        import pyarrow.parquet as _pq
        from sae.activation_store import shard_table_to_array
        _tl = _pq.read_table(cache_path.parent / "token_labels.parquet")
        _cb = _tl.column("position_type").to_numpy(zero_copy_only=False)
        _s0 = sorted(cache_path.glob("shard_*.parquet"))[0]
        _X = torch.from_numpy(shard_table_to_array(_pq.read_table(_s0)).astype("float32"))
        _b = _cb[: _X.shape[0]]
        _xn = (_X - _X.mean(-1, keepdim=True)) / (_X.std(-1, keepdim=True) + 1e-8)
        _pm = torch.from_numpy(_b == "protein"); _tm = torch.from_numpy(_b == "text")
        if int(_pm.sum()) < 100 or int(_tm.sum()) < 100:
            raise ValueError(f"modality calibration: too few protein/text tokens in shard 0 ({int(_pm.sum())}/{int(_tm.sum())})")
        _dir = _xn[_pm].mean(0) - _xn[_tm].mean(0); _dir = _dir / (_dir.norm() + 1e-8)
        _proj = _xn @ _dir
        _thr = 0.5 * (_proj[_pm].median() + _proj[_tm].median())
        _sel = _pm | _tm
        _acc = ((_proj > _thr)[_sel] == _pm[_sel]).float().mean().item()
        sae.protein_dir.copy_(_dir); sae.protein_thresh.fill_(float(_thr))
        print(f"[modality-weights] alpha schedule={args.matryoshka_modality_weights} | "
              f"protein/text classifier accuracy={_acc:.3f} (protein={int(_pm.sum())} text={int(_tm.sum())})", flush=True)
        if _acc < 0.9:
            print(f"[modality-weights] WARNING: classifier accuracy {_acc:.3f} < 0.90 — modality weighting will be noisy!", flush=True)

    if args.init_pre_bias and hasattr(sae, "init_pre_bias_from_data"):
        print("Initializing pre_bias from geometric median of data...")
        if args.presample_shards > 1:
            sample = store.sample(32768, seed=args.seed, num_shards=args.presample_shards).float()
        else:
            first_shard = torch.from_numpy(store._load_shard(0)).float()
            sample = first_shard[: min(32768, len(first_shard))]
        sae.init_pre_bias_from_data(sample)
        print(f"  pre_bias initialized (mean={sae.pre_bias.mean().item():.4f})")
        del sample

    training_config = build_training_config(args, device)
    wandb_config = WandbConfig(enabled=args.wandb_enabled, project=args.wandb_project,
                               run_name=args.wandb_run_name, group=args.wandb_group,
                               job_type=args.wandb_job_type, config=vars(args))
    parallel_config = ParallelConfig(dp_size=args.dp_size)
    perf_logger = PerfLogger(log_interval=args.log_interval, use_wandb=args.wandb_enabled,
                             print_logs=True, device=device)
    trainer = Trainer(sae, training_config, wandb_config=wandb_config,
                      perf_logger=perf_logger, parallel_config=parallel_config)

    if use_streaming:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # Init the process group HERE, before the DDP batch-cap all_reduce below. The trainer's own
        # _setup_distributed is guarded by `if not dist.is_initialized()`, so this is safe + idempotent.
        # Bug it fixes: without this, dist.is_initialized() is False below -> the batch-cap is silently
        # skipped -> ranks run UNEVEN batch counts -> last-step all_reduce desync -> teardown SIGABRT
        # (wandb shows "crashed"). With it, all ranks cap to the global-min batch count and finish clean.
        if world_size > 1 and dist.is_available() and not dist.is_initialized():
            from datetime import timedelta
            # 2h collective timeout: a slow NFS shard read on one rank must not trip the 600s default
            # NCCL watchdog (which killed the balanced run mid-epoch at step ~1950).
            dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
            torch.cuda.set_device(rank)
            print(f"[rank {rank}] initialized process group early for DDP batch-cap (2h NCCL timeout)")
        print(f"Streaming from disk (~{est_gb:.0f}GB). "
              f"Peak RAM: ~{args.mix_shards * shard_size * meta['hidden_dim'] * 4 / (1024**3):.1f}GB/process")
        dataloader = store.get_streaming_dataloader(
            batch_size=args.batch_size, shuffle=args.shuffle, seed=args.seed,
            rank=rank, world_size=world_size, max_shards=max_shards, mix_shards=args.mix_shards,
        )
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            import pyarrow.parquet as pq_meta

            dataset = dataloader.dataset
            my_rows = sum(pq_meta.read_metadata(store.path / f"shard_{idx:05d}.parquet").num_rows
                          for idx in dataset.shard_indices)
            t = torch.tensor([my_rows // args.batch_size], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            dataset.max_batches = int(t.item())
            print(f"[rank {rank}] capped to {dataset.max_batches} batches/epoch for DDP sync")
        trainer.fit(dataloader, resume_from=args.resume_from, data_sharded=True)
    else:
        shards = []
        for i, shard in enumerate(store.iter_shards(shuffle_shards=False)):
            if max_shards is not None and i >= max_shards:
                break
            shards.append(torch.from_numpy(shard).float())
        activations_flat = torch.cat(shards)
        print(f"Loaded {activations_flat.shape[0]:,} cached activations into memory")
        trainer.fit(activations_flat, resume_from=args.resume_from)

    print("Training complete.")


if __name__ == "__main__":
    main()
