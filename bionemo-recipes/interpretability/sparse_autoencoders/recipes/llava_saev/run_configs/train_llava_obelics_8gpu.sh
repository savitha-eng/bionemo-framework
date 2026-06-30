#!/usr/bin/env bash
set -euo pipefail

# Train only. This assumes extraction has already produced the activation store.

ROOT="${SAEV_ROOT:-/data/savithas/saev-llava}"
MODEL="${SAEV_MODEL:-llava-hf/llava-v1.6-mistral-7b-hf}"
SAMPLES="${SAEV_NUM_SAMPLES:-100000}"
MAX_STEPS="${SAEV_MAX_STEPS:-30000}"
GPUS="${SAEV_GPUS:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}"
BATCH_SIZE="${SAEV_BATCH_SIZE:-4096}"
CHECKPOINT_STEPS="${SAEV_CHECKPOINT_STEPS:-5000}"
RUN_METRIC="${SAEV_RUN_METRIC:-1}"

export HF_HOME="${HF_HOME:-$ROOT/hf-cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-sae-v-replication}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/ckpts" "$ROOT/out"

STORE="$ROOT/stores/llava_next_mistral_7b_obelics${SAMPLES}"
CKPT="$ROOT/ckpts/llava_next_mistral_7b_L16_exp16_k128_obelics${SAMPLES}"
OUT="$ROOT/out/llava_next_mistral_7b_L16_obelics${SAMPLES}_image-text.json"

if [ ! -f "$STORE/layer16/metadata.json" ]; then
  echo "Missing activation metadata: $STORE/layer16/metadata.json" >&2
  echo "Run extraction first, or set SAEV_ROOT/SAEV_NUM_SAMPLES to the completed store." >&2
  exit 2
fi

if [ ! -f "$STORE/token_labels.parquet" ]; then
  echo "Missing token labels: $STORE/token_labels.parquet" >&2
  echo "The cross-modal metric needs this row-aligned sidecar." >&2
  exit 2
fi

python - <<PY
import json
from pathlib import Path
store = Path("$STORE")
meta = json.loads((store / "layer16" / "metadata.json").read_text())
print("[train-only] store", store)
print("[train-only] model", meta.get("model"))
print("[train-only] docs", meta.get("n_documents"), "rows", meta.get("n_samples"), "shards", meta.get("n_shards"))
print("[train-only] image_tokens", meta.get("image_tokens"), "text_tokens", meta.get("text_tokens"))
PY

python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" scripts/train.py \
  --cache-dir "$STORE/layer16" --layer 16 \
  --model-type topk --expansion-factor 16 --top-k 128 \
  --normalize-input --normalize-loss \
  --auxk 2048 --auxk-coef 0.03125 --dead-tokens-threshold 5000000 \
  --init-pre-bias --aggregate-loss --dead-count-global --mix-shards 8 --presample-shards 8 \
  --lr 5e-5 --lr-schedule constant --warmup-steps 0 --batch-size "$BATCH_SIZE" \
  --n-epochs 1000000 --max-steps "$MAX_STEPS" \
  --checkpoint-dir "$CKPT" --checkpoint-steps "$CHECKPOINT_STEPS" \
  --dp-size "$GPUS" \
  --wandb --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "llava_next_mistral_7b_L16_exp16_k128_obelics${SAMPLES}_steps${MAX_STEPS}"

if [ "$RUN_METRIC" = "1" ]; then
  python scripts/crossmodal_cooccur.py \
    --sae "$CKPT/checkpoint_final.pt" \
    --store "$STORE" --layer 16 \
    --pair image-text --shards 999999 --tau 2.0 --min-proteins 5 --topk 8 \
    --dump-json "$OUT"
fi
