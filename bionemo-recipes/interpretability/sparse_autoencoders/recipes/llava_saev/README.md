# LLaVA SAE-V Recipe

BioNeMo SAE-framework recipe for SAE-V-style experiments on LLaVA-NeXT/Mistral.

This recipe trains **our** `TopKSAE` implementation on residual-stream activations
from `llava-hf/llava-v1.6-mistral-7b-hf`, then runs the existing cross-modal
co-firing / Eq.7-style metric on the trained checkpoint.

## Paper-Matching Target

SAE-V reports the following LLaVA-NeXT/Mistral SAE-V configuration:

- model family: LLaVA-NeXT/Mistral 7B
- hook layer: `16`
- input dimension: `4096`
- expansion factor: `16`
- feature count: `65536`
- train dataset: Obelics
- train sample count: `100K`
- train steps: `30000`
- batch size: `4096`
- learning rate: `5e-5`
- scheduler: constant

The batch config in this recipe targets that shape and scale while keeping the
BioNeMo TopK/AuxK training flags used by the BioReason-Pro SAE work.

## Files

- `scripts/extract_obelics_vlm.py`: distributed Obelics URL-image extractor.
- `scripts/train.py`: BioReason-Pro SAE training entrypoint with `--max-steps` exposed.
- `scripts/crossmodal_cooccur.py`: BioReason-Pro per-sample cross-modal metric with `image-text` enabled.
- `run_configs/paper_llava_obelics_8gpu.sh`: extraction + training + metric launcher.
- `run_configs/train_llava_obelics_8gpu.sh`: training + metric only; use after extraction has already produced the store.
- `ci/lepton/model_convergence/configs/recipes/saev_llava_obelics_8gpu.yaml`: Lepton launcher config.

## Local 8-GPU Run

From this recipe directory:

```bash
export HF_HOME=/data/savithas/saev-llava/hf-cache
export WANDB_PROJECT=sae-v-replication
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_configs/paper_llava_obelics_8gpu.sh
```

If extraction was already completed separately, run only SAE training and the
cross-modal metric:

```bash
export SAEV_ROOT=/data/savithas/saev-llava
export SAEV_NUM_SAMPLES=100000
export SAEV_MAX_STEPS=30000
export WANDB_PROJECT=sae-v-replication
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash run_configs/train_llava_obelics_8gpu.sh
```

## Outputs

By default the script writes large artifacts to `/data/savithas/saev-llava/`:

- `stores/llava_next_mistral_7b_obelics100000/layer16/`
- `ckpts/llava_next_mistral_7b_L16_exp16_k128_obelics100000/checkpoint_final.pt`
- `out/llava_next_mistral_7b_L16_obelics100000_image-text.json`

These are intentionally not tracked by Git.
