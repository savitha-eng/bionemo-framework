# BioReason-Pro SAE recipe

Sparse Autoencoders for **BioReason-Pro** — a multimodal protein-reasoning model (Qwen3-4B LLM +
ESM3 protein embeddings + a GO graph encoder). This is the first **decoder-LLM** SAE recipe in
`bionemo-framework`: we hook the residual stream of the Qwen3 decoder and train a TopK SAE on the
activations produced by the model's *real* fused forward.

The recipe is **self-contained on the SAE side** and imports the BioReason-Pro package
(`bioreason2`, ESM3) unchanged — we never fork their dataset/model code. Two non-invasive shims
live in `model_loader.py`/`data.py` (a `sys.modules` `unsloth` stub so the `use_unsloth=False`
path imports, and a `load_dataset` redirect from the gated `wanglab/cafa5` to the cached standalone
reasoning repo).

## Environments

Extraction needs **both** the BioReason-Pro model env *and* the `sae` package; train/eval are
pure-`sae`:

* **Env A (extract + loss-recovered):** the BioReason-Pro venv (`bioreason-pro/.venv`) with the
  `sae` package installed editable (`uv pip install -e sae --no-deps`). Has `bioreason2` + ESM3.
* **Env B (train + eval):** a fresh pure-`sae` venv (`recipes/bioreason_pro/.venv`) with
  `pip install -e sae` plus `scikit-learn umap-learn hdbscan pandas matplotlib`.

## Faithful fused forward

`model_loader.load_bioreason_pro_sft` builds a **hookable** model: the text model is a plain
`AutoModelForCausalLM` (so `model.text_model.model.layers[L]` is hookable), with the SFT LLM
weights, ESM3, the trained `protein_projection`, and the **cached GO band**
(`go_embedding.pt` → `go_projection`) — identical to production inference (`predict.py`), but
without vLLM. The forward (`ProteinLLMModel.forward`) splices ESM3 protein embeddings and the GO
band into the `<|protein_pad|>` / `<|go_graph_pad|>` placeholder slots, then runs the LLM
teacher-forced. We hook the residual stream and tag each token `protein` / `go` / `text`.

Resolved hidden-states mapping: the hook on `layers[L]` captures the **raw residual after block L**
(`= output_hidden_states[L+1]`; layer 35 is the pre-final-norm residual).

## Pipeline

```bash
# --- Step A/B: smoke + tokens + layer diagnostic (Env A extract, Env B diagnostic) ---
CUDA_VISIBLE_DEVICES=3 python scripts/extract.py --num-proteins 500 --layers 24 28 32 35 \
    --output $STORE/diag500 --verify-hidden-states          # Env A
python scripts/diagnostic.py --store $STORE/diag500 --layers 24 28 32 35 \
    --out-json $STORE/diag500/diag_report.json --fig-dir $STORE/diag500/figs --wandb   # Env B

# --- Step C: subset SAE at the chosen layer ---
CUDA_VISIBLE_DEVICES=3 python scripts/extract.py --num-proteins 20000 --shuffle --layers 24 \
    --output $NVME/layer24_subset20k                        # Env A; extract to local NVMe
python scripts/train.py --cache-dir $NVME/layer24_subset20k --layer 24 \
    --expansion-factor 8 --top-k 32 --batch-size 4096 --n-epochs 1 \
    --aggregate-loss --dead-count-global --mix-shards 8 --presample-shards 8 --init-pre-bias \
    --checkpoint-dir $OUT/sae_layer24 --wandb               # Env B
python scripts/eval.py --sae $OUT/sae_layer24/checkpoint.pt --store $STORE/layer24_eval_val300 \
    --layer 24 --out-json $OUT/eval_layer24.json            # Env B (R^2/sparsity/%dead/GO-F1)
CUDA_VISIBLE_DEVICES=3 python scripts/eval_loss_recovered.py --sae $OUT/sae_layer24/checkpoint.pt \
    --layer 24 --split validation --num-proteins 150        # Env A (CE loss-recovered)
```

## Storage

Activations are float32 (`hidden=2560` → 10,240 B/token/layer). Measured ~3,600 tokens/protein
(protein ~535 / go 200 / text ~2,878). Write the durable copy to the shared FS; stage to local
NVMe for fast extraction writes and training reads. Reserve the `validation` split for eval — never
extract it into the training store.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `scripts/extract.py` | A | Faithful fused forward → per-layer `ActivationStore` + label sidecar |
| `scripts/diagnostic.py` | B | Layer pick: Tier1 stats + GO-probe F1 + layer-35 Fig-1H |
| `scripts/train.py` | B | TopK SAE training (`sae.training.Trainer`; #1619 flags) |
| `scripts/eval.py` | B | R²/sparsity/%dead + per-protein GO-feature F1 + cross-modal count |
| `scripts/eval_loss_recovered.py` | A | CE loss-recovered via layer-L SAE substitution |
