#!/usr/bin/env bash
set -euo pipefail

ROOT="${SAEV_ROOT:-/data/savithas/saev-llava}"
MODEL="${SAEV_MODEL:-llava-hf/llava-v1.6-mistral-7b-hf}"
DATASET="${SAEV_DATASET:-HuggingFaceM4/OBELICS}"
SAMPLES="${SAEV_NUM_SAMPLES:-100000}"
MAX_STEPS="${SAEV_MAX_STEPS:-30000}"
GPUS="${SAEV_GPUS:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}"

export HF_HOME="${HF_HOME:-$ROOT/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-sae-v-replication}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$ROOT/stores" "$ROOT/ckpts" "$ROOT/out"

STORE="$ROOT/stores/llava_next_mistral_7b_obelics${SAMPLES}"
CKPT="$ROOT/ckpts/llava_next_mistral_7b_L16_exp16_k128_obelics${SAMPLES}"
OUT="$ROOT/out/llava_next_mistral_7b_L16_obelics${SAMPLES}_image-text.json"

torchrun --standalone --nproc_per_node="$GPUS" scripts/extract_obelics_vlm.py \
  --model "$MODEL" --layer 16 \
  --dataset "$DATASET" --split train \
  --output "$STORE" \
  --num-samples "$SAMPLES" --scan-multiplier 20 \
  --max-text-words 128 --shard-size 200000 --dtype float16 \
  --overwrite

torchrun --standalone --nproc_per_node="$GPUS" scripts/train.py \
  --cache-dir "$STORE/layer16" --layer 16 \
  --model-type topk --expansion-factor 16 --top-k 128 \
  --normalize-input --normalize-loss \
  --auxk 2048 --auxk-coef 0.03125 --dead-tokens-threshold 5000000 \
  --init-pre-bias --aggregate-loss --dead-count-global --mix-shards 8 --presample-shards 8 \
  --lr 5e-5 --lr-schedule constant --warmup-steps 0 --batch-size 4096 \
  --n-epochs 1000000 --max-steps "$MAX_STEPS" \
  --checkpoint-dir "$CKPT" \
  --dp-size "$GPUS" \
  --wandb --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "llava_next_mistral_7b_L16_exp16_k128_obelics${SAMPLES}_steps${MAX_STEPS}"

python scripts/crossmodal_cooccur.py \
  --sae "$CKPT/checkpoint_final.pt" \
  --store "$STORE" --layer 16 \
  --pair image-text --shards 999999 --tau 2.0 --min-proteins 5 --topk 8 \
  --dump-json "$OUT"
