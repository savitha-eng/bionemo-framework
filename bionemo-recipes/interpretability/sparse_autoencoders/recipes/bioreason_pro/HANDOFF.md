# BioReason-Pro SAE — handoff / continuation doc

> **Quick status:** see **[STATUS.md](STATUS.md)** (updated 2026-06-29) for the current "where are we" review —
> what's accomplished, what's running, what's viewable. This doc holds the deep conventions/methods.

**Purpose:** a self-contained brief so a *new* Claude session (e.g. on a fresh dev pod) or a teammate can
pick up this work. The chat history does NOT transfer between pods, but (a) this doc lives on shared NFS
(`/data/savithas`, readable from any pod) and (b) the memory dir is also on shared NFS
(`/root/.claude → /data/savithas/claude`), so persisted memories survive too. Read this first, then
verify live state with the commands in §"Verify current state".

_Last updated: 2026-06-23 by the session that built the rigorous interpretability framework
(Swiss-Prot residue F1, inspectable SAE-V fusion, train-set AUC) and pivoted from the expansion sweep._

> **⚠️ NEW-POD FIRST STEPS (read §2.1 + §2.2):** much of the 8k analysis lived on `/scratch` (pod-local).
> A copy job mirrored it to `/data/savithas/phase3_subset_8k/` (persists). On a new pod, use the
> `/data` paths, NOT `/scratch`. Verify the copy finished: `tail /data/savithas/phase3_train_interp/preserve.log`.

---

## 0. LATEST STATE, CONVENTIONS & METHODS (2026-06-26) — read this first

### 0.1 What exists now
- **Depth sweep — 14 SAEs, L12 through L35** (exp16, top_k=128), full-data trained. Dashboards for all:
  `multimodal_dashboard/public/l##_exp16/`, viewable as `?model=l##_exp16` (vite on :5174).
- **k-sweep** (L28): `l28_k32`, `l28_k64` — lower k gives FEWER multi-token features (k=128 is best).
- **Modality-balanced SAE** (L24, 50/50 protein:text): `l24_balanced`; a 30/70 variant is the better follow-up.
- **Activation stores:** `cache_dir/activations/train_full_L24_L26_L28_L30_L32` (+ `_L16_L18_L20_L22`,
  `_L12_L14`, `_L33_L34_L35`), ~1.6 TB/layer, 117,002 proteins / 421M tokens. `token_labels.parquet`
  (position_type protein|go|text) at each store root, row-aligned to shard concatenation order.

### 0.2 INTERPRETATION CONVENTIONS (easy to get wrong — be strict)
- **Variance explained: ALWAYS report `var_exp_norm` (normalized FVU), NEVER raw `var_exp`.** Raw is ~0.99 for
  every healthy SAE (a few high-variance residual dims dominate) → meaningless; quoting it is a red flag. The
  honest, loss-aligned metric is **normalized**: healthy range here **~0.81–0.91**.
- **Dead latents — TWO different numbers, don't conflate:** (1) **Training `dead_latents (%)`** (wandb) on the
  *training* distribution — healthy ~0.25–11%, and it STARTS high (e.g. 78% @ step ~2000) and falls via AuxK,
  so judge the END. (2) **`dead-everywhere`** (`per_band_coverage.py`) on the *natural eval* store — a
  *balanced* SAE reads ~77% here from distribution shift, not breakage. Report both, labeled.
- **Reasoning traces: train on the FULL teacher-forced sequence — prompt + `<think>`reasoning`</think>` +
  answer (ALL tokens)** (Jared/Goodfire practice — don't special-case reasoning). Caveat: reasoning =
  GPT-5-distilled SFT targets (imitation), an on-distribution proxy, not the model's rollouts → generation-mode
  is the clean version for *reasoning* claims. Biology unaffected (causal masking; inputs). `<think>`/`</think>`
  (ids 151667/151668) split the text band into prompt/reasoning/answer.
- **Cross-modal: `crossmodal_omega` is ARTIFACT-PRONE — don't trust it** (spikes for unimodal/dense features;
  L32 "541 cross-modal" and balanced-L24 "32" were artifacts). Use **band-mass** (protein_frac>0.2 AND
  text_frac>0.2) or **per-protein co-occurrence** (`crossmodal_cooccur.py`, SAE-V Eq.7 within-sample). By those,
  genuine protein↔text fusion is **rare at every layer** (~80 band-mass features, ~0 cosine alignment).
- **`<go>` graph tokens are a fixed ontology reduction, NOT per-protein GO terms** — don't interpret them;
  GO info the model uses comes from TEXT (`go_pred`). Ablation: GO-graph channel is unused (ΔCE=0).

### 0.3 KEY RESULTS
- **Depth:** pure-protein features peak at the EXTREMES — input L12 (145) and final L35 (195) — thin in the
  deep middle (L30–34 ≈ 62–85). Reasoning-specialized features peak sharply at **L24**.
- **Causal fusion (headline)** — `analysis/INPUT_ABLATION.md`: zeroing the protein embedding raises CE on the
  model's OWN reasoning by **+0.16 (≈15%, t=31, 93% of 300 proteins)**, answer +0.07. The model fuses protein
  structure into its reasoning — **deeper than tool-calling**. GO-graph embed unused (ΔCE=0).
- **SAE-vs-causal tension:** SAE *features* show NO protein↔text fusion (≈0 cosine, all layers), yet the model
  causally fuses → standard SAEs decompose the cross-modal direction into unimodal features.
- **Modality balancing (50/50):** healthy (~11% train-dead, var_exp_norm 0.906); protein-firing coverage
  3.6%→7.9% (8k eval); 13× protein-dominant (555 eval); did NOT recover fusion (band-mass 80→84). 50/50 is
  aggressive (77% dead-everywhere on natural data); 30/70 is the gentler tradeoff.

### 0.4 HOW WE TRAIN
- **Extract** (`run_extract.sh` → `extract.py`, Env A, 8×H100 DDP): SFT `use_unsloth=False`, hook
  `text_model.model.layers[L]`, drop pads → ActivationStore + token_labels.
- **Train** (`run_sae.sh <L> <exp> <auxk> [topk]` → `train.py`, Env B, torchrun 8 GPU): TopK exp16 k128,
  `--normalize-input --normalize-loss --aggregate-loss --dead-count-global --mix-shards 10 --presample-shards 8
  --init-pre-bias --auxk 2048`. **NCCL inits with a 2h timeout** (slow NFS read else trips the 600s watchdog →
  mid-epoch SIGABRT; this killed the balanced run @ step ~1950). End-of-epoch SIGABRT is benign — `run_*.sh`
  promotes the latest step-ckpt to `checkpoint_final.pt`; **clean stale checkpoints before re-running**.
- **Modality balancing:** prefer **resave** (`rebalance_store.py --drop-go --text-keep <p>` →
  `run_prebalanced.sh <store> <L> <epochs> <tag>`) — reads the full store ONCE, trains fast. (In-loader
  `train.py --balance-modality` works but re-reads the full store every epoch → slow.) text-keep: 0.188=50/50,
  0.44=30/70.
- **Lepton:** `lep job create -n <UNIQUE-name> --node-group yo-bom-lepton-001 --resource-shape gpu.8xh100-sxm
  --container-image nvcr.io/nvidia/pytorch:26.02-py3 --image-pull-secrets lepton-nvidia-pstjohn
  --mount /BioNeMo:/data:node-nfs:fs1 --command "bash <script>"`. **Unique names** (two same-named runs →
  wandb confusion — happened with `sae-l24-exp16-balanced`).

### 0.5 SCRIPTS & DASHBOARD QUIRKS
- `per_band_coverage.py` (coverage+dead; **report eval size**: 8k store=7,999 prot, dashboard subset=555),
  `crossmodal_cooccur.py` (streams to full 117k), `ablation_eval.py` (fusion), `rebalance_store.py`,
  `make_role_sidecar.py`, `add_{span,go_text,go_terms,role}_metric.py`, `dashboard.py`/`build_dashboard.py`.
- Dashboard examples = **±48-token windows centered on the PEAK** → a single window is bio OR text, never both;
  for cross-modal you need a FULL-sequence view. `--drop-go` excludes go-centered examples. UMAP is laid out by
  modality (reasoning vs answer don't spatially separate — both text). **Hard-refresh/incognito** after a
  rebuild (DuckDB caches the parquet).

---

## 1. Project goal (Phase 3)
Build the **first decoder-LLM SAE recipe** in bionemo-framework, applied to **BioReason-Pro-SFT**
(Qwen3-4B multimodal: ESM3 protein embeds + GO graph memory + Qwen3 LLM; 36 layers, hidden 2560).
Train TopK SAEs on the residual stream and interpret features — **especially whether the model
represents protein biology, or mostly carries it in the reasoning text.** The recipe mirrors Jared's
codonfm/esm2 SAE recipes (`/data/jwilber/...`) + the shared `sae` package.

## 2. Environments & key paths
- **Recipe dir (the repo, git worktree on /data):**
  `/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro`
- **Shared `sae` package:** `…/sparse_autoencoders/sae/src/sae/` (ActivationStore, architectures, training).
- **node:** `/opt/node/bin` (for the dashboard).  **lep CLI:** `/usr/local/bin/lep` (DGX Cloud Lepton).
- **Pod:** `nvidia-lepton005` — **shared 8×H100 node**, other tenants present.

### 2.1 The two environments + dependencies (what a new pod needs)
Both venvs live on **/data (persist across pods)**, BUT they were built `--system-site-packages` against the
**NVIDIA PyTorch container** (torch + TransformerEngine from the system image). If the new pod uses the **same
image**, they work as-is. If not, rebuild each `.venv` and re-add the extra deps below.
- **Env A** = `/data/savithas/bioreason-pro/.venv` — has the **model + tokenizer (bioreason2, ESM3, transformers)**
  AND the `sae` package installed. Used by: `extract.py`, `dashboard.py`, `inspect_fusion.py`, `fetch_swissprot.py`.
  Extra deps installed this project: **`requests`** (UniProt fetch), **`umap-learn`** (atlas), **`openai`** (auto-interp).
  NOTE: scripts importing `bioreason_pro_sae.*` must `sys.path.insert(0, <recipe>/src)` first (Env A doesn't pip-install it).
- **Env B** = `<recipe>/.venv` (torch 2.12 cu130) — **pure `sae`** (no model). Used by: `train.py`, `swissprot_f1.py`,
  `protein_band_auc_to_meta.py`, `full_feature_analysis.py`, `build_simple_browser.py`, `merge_autointerp.py`,
  `autointerp_run.py`. Extra dep installed: **`openai`**.
- **NIM key** (auto-interp LLM, Llama-3.3-70B): `source /data/savithas/.nim_env` (chmod-600, never committed).
  ROTATE at build.nvidia.com when done — it was pasted in chat.

### 2.2 Data paths — PERSISTED (/data) vs EPHEMERAL (/scratch, gone on new pod)
| What | Path | Persists? |
|---|---|---|
| Full store (117k, all 5 layers, 8.0TB) | `<recipe>/cache_dir/activations/train_full_L24_L26_L28_L30_L32/` | ✅ /data |
| Full L28 SAE (exp8) | `/data/savithas/phase3_full/sae_l28_exp8_normloss_full/` | ✅ /data |
| 8k expansion sweep (L28 exp16/exp32) | `/data/savithas/phase3_8k_sweep/` | ✅ /data |
| Analysis outputs (swissprot, fusion, logs) | `/data/savithas/phase3_train_interp/` | ✅ /data |
| **8k stores + matched SAEs (L28/L30/L32)** | **`/data/savithas/phase3_subset_8k/`** (copied 06-23) | ✅ /data |
| **MATCHED 5-layer 8k store (expansion sweep)** | **`/data/savithas/phase3_subset_8k_5layer/layer{24,26,28,30,32}`** — 145 shards/29M tokens each, SAME 7,999 proteins across all layers, representative (verified). Train exp8/16/32 here for a clean cross-layer comparison. | ✅ /data |
| ~~8k stores + SAEs (original)~~ | ~~`/scratch/savithas/phase3_subset/`~~ | ❌ **pod-local — GONE on new pod** |
| Swiss-Prot annotations (7871 proteins) | `/data/savithas/phase3_train_interp/swissprot.tsv.gz` | ✅ /data |

## 3. Constraints (do not violate)
- **GPUs are shared** — pin to free GPUs (`nvidia-smi`), **never kill other tenants'** processes.
- **`env -u WANDB_API_KEY`** before any wandb run — a leaked key in the env overrides `~/.netrc` and
  would log under someone else. With it unset, netrc auths correctly as **savithas (clara-discovery)**.
- **Write big data to `/data/savithas` (durable NFS), checkpoints too.** `/scratch` is node-local/ephemeral
  (good only as a fast read cache / staging).
- **Git:** never commit to `main`; push to the `savitha-eng` fork; no PR without the user.
- **Do not fork/edit** the BioReason-Pro dataset/processing code (import + call it unchanged).

## 4. Current state (2026-06-22)
### Done
- **Full-train activation extraction — COMPLETE & verified.**
  Path: `<recipe>/cache_dir/activations/train_full_L24_L26_L28_L30_L32/`  (**8.0 TB**)
  - 117,002 proteins · **421,329,433 tokens** (protein 62,960,406 / go 23,400,400 / text 334,968,627)
  - 5 layers (24/26/28/30/32), 1.6 TB & 2,108 shards each; sidecar `token_labels.parquet`,
    `proteins.parquet`, `extract_metadata.json`. **Format: FixedSizeList `act` column** (fast reads).
  - All layers' `n_samples` == sidecar rows (alignment verified).
- **First full-data SAE — COMPLETE: L28, expansion 8.**
  `/data/savithas/phase3_full/sae_l28_exp8_normloss_full/checkpoint_final.pt` (1.2 GB)
  - var_exp_norm **0.890**, dead **1.48%**, FVU 0.0049 (BETTER than 8k: 0.863 / 5.1%).
  - wandb: clara-discovery/bioreason-pro-sae run `zvykuvqg`.
- **Bio eval on full L28/exp8** (`per_band_coverage.py`, val300):
  protein **482** / go **311** / text **20,005** features fire (bio total **666**) — **DOWN from 8k's
  1,222/670/1,658.** Headline: **scaling data alone (fixed dict) shrank bio coverage** (text ate the
  capacity) → motivates the expansion sweep. (Caveat: coverage ≠ quality; eval is only 300 proteins.)

### Interpretability framework (the 06-23 pivot — see §5)
The expansion sweep ran (8k exp16/exp32 exist in `/data/savithas/phase3_8k_sweep/`). We then pivoted to
**solidifying a rigorous, label-grounded interpretability framework** because we were churning on the same
L28/8k analysis. New scripts (all in `scripts/`):
- **`swissprot_f1.py`** — residue-level Swiss-Prot feature F1 (Jared's esm2/codonfm method). THE mechanistic
  bio metric: maps UniProt annotation at residue *r* → protein-band token (per-protein offset = band_min+1,
  since band = ESM3-BOS + residues + EOS at absolute idx 7). Distinguishes a real active-site feature from a
  composition (e.g. valine) counter. Reuses `swissprot.tsv.gz` (fetched for the 8k proteins).
- **`fetch_swissprot.py`** — pulls UniProt per-residue annotations (ACT_SITE/BINDING/DOMAIN/...) for a store's
  exact accessions (network-only). Done: 7871/8000.
- **`inspect_fusion.py`** — inspectable SAE-V: finds features firing on BOTH protein band AND reasoning text
  (response, NOT prompt boilerplate), dumps side-by-side decoded evidence (residues vs words) to judge real
  fusion vs boundary artifact. (Uses the `role` sidecar `token_labels_with_role.parquet`.)
- existing: `full_feature_analysis.py` (coverage + role taxonomy), `protein_band_auc_to_meta.py` (leakage-free
  GO-AUC, now run on TRAIN), `autointerp_run.py` (LLM labels), `merge_autointerp.py`, `build_simple_browser.py`.

### Key findings (L28, 8k train)
- **Bio is scarce & weak.** Protein-band GO-AUC on TRAIN: **90 features >0.65** (val300 gave 52), all
  "catalytic activity", max ~0.78. Top "bio-AUC" features are largely **protein→GO-graph boundary artifacts**.
- **Fusion is scarce.** `inspect_fusion` finds only **6** features firing on both protein + reasoning (of 20,480).
  Consistent with the earlier SAE-V ω≈0 (no shared-concept fusion at L28 once boilerplate excluded).
- **Auto-interp labels are unvalidated** (window-bias mislabels boundary features; no faithfulness check).
  → de-prioritized; rebuild later WITH a detection/faithfulness score, not as a daily activity.
- Token mix (8k): protein 15% / go 5.6% / prompt-text 35.5% / response-text 43.9% → only **~21% biology**
  (a structural reason bio features are scarce — SAE sees 4× more text).

### Pending / not done
- **Answer-segment split** — `role` sidecar lumps reasoning+final-answer as `response`; split out the answer
  (locate `final_answer` span) so segments = {protein, go, prompt, reasoning, answer}.
- **Consolidate** the metrics into ONE `(sae, store, layer) → per-feature record` driver (the "framework").
- **Layer sweep** L28/L30/L32 (8k, matched exp8 SAEs) with the framework → pick best layer.
- **Higher-exp retrain** L30/L32 at exp16/exp32 (L28 exists) → does bigger dict find more pure multimodal feats?

## 5. Active plan (locked 2026-06-23): a → b → c
**a. Solidify the framework** — one driver emitting per-feature: segment profile {protein, go, prompt,
   reasoning, answer} + GO-AUC (leakage-free, protein-band) + **Swiss-Prot residue F1** (mechanistic) +
   phrase profile + role_category + fusion flag. Validate on L28. (swissprot_f1 + inspect_fusion done; need
   the answer-split + consolidation.)
**b. Analyze all layers** — run the driver on **L28/L30/L32** (8k, matched config). Comparison table →
   pick the layer where bio is most mechanistic / fusion clearest. SAEs+stores at `/data/.../phase3_subset_8k/`.
**c. Higher-exp retrain + interp** — train L30/L32 @ exp16/exp32 (~30–60min each on 8k), re-run framework,
   compare exp8/16/32 per layer for *pure multimodal* features.
**Why:** stop re-running the same L28 analysis on dataset variants; build a reusable rigorous engine, then
   just RUN it across layers/expansions. All analysis on the **8k TRAIN set** (SAE is unsupervised → label
   metrics don't leak); re-confirm the winning layer on held-out val300 only for a generalization claim.
**Fixed train config** (only `expansion-factor`+`auxk` change): top_k 128, `--normalize-input --normalize-loss
--init-pre-bias --aggregate-loss --dead-count-global --mix-shards 10 --presample-shards 8`, lr 1e-4 cosine,
lr-min 1e-5, warmup 1000, max-grad-norm 1.0, batch 4096, 1 epoch. auxk: exp8→1024, exp16→2048, exp32→4096.

## 6. Gotchas learned (read before launching runs)
- **NFS-read is the training bottleneck.** The full-store runs are ~70% I/O-stalled reading 1.6 TB/layer
  off NFS (exp8: 2.2 hr compute but 7.7 hr wall). **Stage the layer store to `/scratch` first** (≈45 min
  for 1.6 TB) → GPU-bound ~2.2 hr, and enables concurrent runs.
- **Don't stage while training off `/scratch`** — the `cp` saturates local md0 and **starves/wedges**
  running jobs reading from `/scratch` (this happened to the 8k runs). Stage first, then train.
- **Two trainings reading the same NFS store contend** — both slow to a crawl. Either stage to /scratch
  (then concurrent is fine) or run sequentially.
- **Legacy 8k stores are `dim_` format** (slow 2560-col reconstruction) → multi-minute first-batch fill.
  The full store is FixedSizeList (`act`) → fast. (`sae.activation_store.shard_table_to_array` reads both.)
- **`dead_tokens_threshold=10M`** was tuned for the 28.8M-token (8k) epoch (~35% of epoch). At full scale
  (421M) it's ~2.4% of an epoch, so dead% spikes early (saw 29%) before AuxK revives it (ended 1.5%).
  Consider scaling it (~100–150M) for full-data runs; not strictly required (exp8 ended fine at 1.5%).

## 7. Runbook (concrete, copy-paste)
```bash
# --- paths ---
RECIPE=/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro
VENV=$RECIPE/.venv                                                            # Env B (train/eval)
FULLSTORE=$RECIPE/cache_dir/activations/train_full_L24_L26_L28_L30_L32        # 8.0TB, FixedSizeList
S8K=/scratch/savithas/phase3_subset/L28-32_subset8k/layer28                   # 8k store (legacy dim_)
VAL=/scratch/savithas/phase3_subset/val300_L28                               # held-out eval (L28)
```

**Train one SAE (single GPU).** `STORE` = `<...>/layer<L>` (full) or `$S8K` (8k). Set `EF`+`AUXK`
together (8→1024, 16→2048, 32→4096). `PYTHONUNBUFFERED=1` so logs flush.
```bash
EF=16; AUXK=2048; L=28; GPU=3; NAME=sae-l${L}-exp${EF}-normloss-full
cd $RECIPE
tmux new-session -d -s train-$NAME bash -lc "
env -u WANDB_API_KEY PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU $VENV/bin/python scripts/train.py \
  --cache-dir $FULLSTORE/layer$L --layer $L --model-type topk --expansion-factor $EF --top-k 128 \
  --normalize-input --normalize-loss --auxk $AUXK --auxk-coef 0.03125 --dead-tokens-threshold 10000000 \
  --init-pre-bias --aggregate-loss --dead-count-global --mix-shards 10 --presample-shards 8 \
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --warmup-steps 1000 --max-grad-norm 1.0 \
  --batch-size 4096 --n-epochs 1 --wandb --wandb-project bioreason-pro-sae \
  --wandb-run-name $NAME --wandb-group phase3-expansion-sweep \
  --checkpoint-dir /data/savithas/phase3_full/$NAME > /data/savithas/phase3_full/$NAME.log 2>&1
echo DONE_\$? >> /data/savithas/phase3_full/$NAME.log"
```

**Expansion sweep (8k, fast pilot — both at once, GPUs 3 & 6):**
```bash
# from $S8K (already local); exp16->GPU3, exp32->GPU6 ; checkpoints in /data/savithas/phase3_8k_sweep/
# (exp8 baseline already exists: /scratch/savithas/phase3_subset/sae_l28_exp8_normloss/)
```
**Expansion sweep (full data):** STAGE each layer to /scratch FIRST (NFS reads are the bottleneck;
do NOT stage while a job reads /scratch — it wedges the job):
```bash
cp -r $FULLSTORE/layer28 /scratch/savithas/stage/layer28        # ~45 min, GPUs idle
# then train with --cache-dir /scratch/savithas/stage/layer28  (GPU-bound ~2.2h vs ~8h off NFS)
```

**Monitor any run:**
```bash
tmux ls
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader   # 90%+ = training
tr '\r' '\n' < /data/savithas/phase3_full/<name>.log | grep '^Step' | tail -3    # metrics (or watch wandb)
```

**Bio eval (per-band feature coverage) on a checkpoint:**
```bash
$VENV/bin/python scripts/per_band_coverage.py --sae <ckpt>/checkpoint_final.pt \
  --store $VAL --layer 28 --device cuda --out-json <out>.json
```

**Interpretability framework (the 06-23 rigor metrics). Use /data paths (post-migration):**
```bash
ENVA=/data/savithas/bioreason-pro/.venv
S8K=/data/savithas/phase3_subset_8k/L28-32_subset8k      # layers 28,32 ; L30 in L30_subset8k
SAE=/data/savithas/phase3_subset_8k/sae_l28_exp8_normloss/checkpoint_final.pt
OUT=/data/savithas/phase3_train_interp

# 1) Swiss-Prot residue F1 (mechanistic bio) — Env B; needs swissprot.tsv.gz (fetch_swissprot.py if missing)
CUDA_VISIBLE_DEVICES=3 $VENV/bin/python scripts/swissprot_f1.py --sae $SAE --store $S8K --layer 28 \
  --swissprot $OUT/swissprot.tsv.gz --offset 0 --out $OUT/swissprot_f1_l28.json
# 2) Inspectable SAE-V fusion (Env A; needs sys.path src insert, handled in-script)
CUDA_VISIBLE_DEVICES=6 $ENVA/bin/python scripts/inspect_fusion.py --sae $SAE --store $S8K --layer 28 \
  --max-shards 40 --n-cand 40 --topk 8 --out $OUT/fusion_inspect.json
# 3) Leakage-free GO-AUC (protein-band) -> writes into dashboard feature_metadata
CUDA_VISIBLE_DEVICES=3 $VENV/bin/python scripts/protein_band_auc_to_meta.py $SAE $S8K 28 multimodal_dashboard/public
# fetch Swiss-Prot annotations for a store's accessions (network, no GPU):
$ENVA/bin/python scripts/fetch_swissprot.py --store $S8K --out $OUT/swissprot.tsv.gz
```

**Dashboard:** Env A `scripts/dashboard.py` (highlights) + `build_dashboard.py` (metadata/atlas/GO-AUC),
out-dir `multimodal_dashboard/public`; serve `cd multimodal_dashboard && /opt/node/bin/npm run dev -- --port 5174 --host`; then `ssh -L 5174:localhost:5174 <pod>`.

**Durable Lepton batch job — VALIDATED 2026-06-24 (works end-to-end):**
The recipe `.venv` is self-contained (own torch 2.12+cu130) BUT its `bin/python` symlinks to
`/usr/bin/python3.12`, so the job image MUST be the **26.02 / Ubuntu-24.04 base** (matches the dev pod
where the venv was built). pstjohn's bionemo image puts python elsewhere → venv breaks.
```bash
VENV=/data/savithas/phase3-wt/.../recipes/bioreason_pro/.venv   # self-contained; runs as-is
lep job create -n sae-l28-exp16-full \
  --node-group yo-bom-lepton-001 \
  --resource-shape gpu.8xh100-sxm \
  --container-image nvcr.io/nvidia/pytorch:26.02-py3 \        # MUST be 26.02 (has /usr/bin/python3.12)
  --image-pull-secrets lepton-nvidia-pstjohn \                # works for nvcr.io/nvidia too
  --mount /BioNeMo:/data:node-nfs:fs1 \                       # -> /data, READ-WRITE (checkpoints ok)
  --command "cd <recipe> && env -u WANDB_API_KEY WANDB_ENTITY=clara-discovery \
    \$VENV/bin/torchrun --nproc_per_node 8 scripts/train.py --cache-dir \
    <FULLSTORE>/layer28 --layer 28 --dp-size 8 --model-type topk --expansion-factor 16 --top-k 128 \
    --normalize-input --normalize-loss --auxk 2048 --auxk-coef 0.03125 --dead-tokens-threshold 100000000 \
    --init-pre-bias --aggregate-loss --dead-count-global --mix-shards 10 --presample-shards 8 \
    --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --warmup-steps 1000 --max-grad-norm 1.0 \
    --batch-size 4096 --n-epochs 1 --wandb --wandb-project bioreason-pro-sae \
    --checkpoint-dir /data/savithas/phase3_full/sae-l28-exp16-full"
lep job get -n <name> ; lep job log -i <full-id>   # NOTE: logs don't persist post-completion;
# for durable output, have the job TEE to a /data file (logs stream only while running).
```
- Validated: mount RW + `/data/savithas` visible + `.venv/bin/python` imports torch/sae/pyarrow on 26.02.
- Tradeoff: batch reads the store over NFS (no /scratch staging) → I/O-bound (~8h vs ~2h). Fine per Jared.
- `dead-tokens-threshold` scaled to 100M for the 421M-token full epoch (10M was tuned for the 28.8M 8k epoch).

## 8. Verify current state (run these)
```bash
tmux ls                                   # what's running
nvidia-smi                                # GPU usage (ours + tenants)
cat <full-store>/extract_metadata.json    # dataset receipt: 117,002 proteins / 421,329,433 tokens
ls -la /data/savithas/phase3_full /data/savithas/phase3_8k_sweep   # trained SAEs + logs
```

## 9. Open questions / next steps
1. **Finish the expansion sweep** (8k now; full after staging) → `per_band_coverage` per point → does
   bigger dict raise bio feature count? Plot exp8/16/32 × {8k,full}.
2. **The decisive bio test:** per-band **rank** at scale (sample protein tokens from the full store) +
   **GO-AUC** — is protein still ~rank-3 with 14.6× proteins (intrinsic) or higher (small-N was the cause)?
3. **Methodology note for Jared (pending his reply):** we train on the FULL teacher-forced sequence
   incl. reasoning trace + final answer (all tokens; standard per SAE-V/Anthropic which take all tokens).
   Reasoning traces are **GPT-5-distilled SFT targets** (model trained to imitate), not its own rollouts —
   an on-distribution-ish proxy; a generation-mode run is the clean version for *reasoning* claims.
   The biology (protein/GO) is unaffected (causal masking; those are inputs).
4. Consider running heavy full-data trainings as **`lep job` batch jobs** (durable across pod issues).

## 10. Pointers
- Memory index: `/data/savithas/claude/projects/-data-savithas/memory/MEMORY.md`
  (notably `activationstore-2560col-slow.md`, `lepton-pod-shared-gpus.md`, `bioreason-pro-phase-plan.md`).
- Analysis docs: `<recipe>/analysis/` (SAE_RESULTS, DATASET_AND_MODALITY_STATUS, MODALITY_BALANCING,
  INTERPRETATION_FINDINGS, LAYER_ANALYSIS).
- Jared's reference recipes: `/data/jwilber/bionemo-framework-nemotron_sae/bionemo-recipes/interpretability/sparse_autoencoders/recipes/{codonfm,esm2,nemotron}`.
- wandb: https://wandb.ai/clara-discovery/bioreason-pro-sae

## Multi-GPU DDP end-of-epoch crash (2026-06-24) — IMPORTANT for batch sweeps
8-GPU `torchrun` runs train the full epoch fine (var-exp 0.84–0.87) but **SIGABRT at teardown**: the
dataloader splits SHARDS evenly per rank but the one partial shard (#2107, 18,362 rows vs 200,000) lands
in a single rank → that rank has ~1 fewer batch → uneven step count → the per-step
`all_reduce(stats_last_nonzero)` (from `--dead-count-global`) desyncs at the final step → 10-min NCCL
timeout → crash BEFORE the rank-0-only `checkpoint_final.pt` save (`training.py:762`). Single-GPU is immune.
**Workaround (in `run_sae.sh`):** `--checkpoint-steps 4000` (saves mid-loop, before the desync) + after
torchrun, promote the latest `checkpoint_step_*.pt` → `checkpoint_final.pt`. **Real fix (TODO):** make the
dataloader drop the partial shard / cap all ranks to the min step count, or save final on a barrier-free path.
