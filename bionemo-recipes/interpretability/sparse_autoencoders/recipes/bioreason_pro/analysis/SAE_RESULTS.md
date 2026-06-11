# BioReason-Pro SAE — training results & the variance-explained investigation

Companion to [`LAYER_ANALYSIS.md`](LAYER_ANALYSIS.md) (which chose the layer). This doc covers what
happened **after** we picked the layer: training TopK SAEs, debugging a suspiciously-high
variance-explained number, and selecting the final config.

This is the **first decoder-LLM SAE** in bionemo-framework, on BioReason-Pro (Qwen3-4B + ESM3 protein
embeddings + a GO graph-memory encoder; 36 layers, hidden 2560). All numbers below are on a **subset**
(~8k proteins → 28.8M tokens for training; 300 held-out validation proteins → 1.07M tokens for eval),
the gate before any full-scale run.

---

## TL;DR

- We trained TopK SAEs and saw **variance-explained ≈ 0.97**, which is a **red flag** — that high
  usually means the SAE is acting like a near-identity *copy machine*, not learning interpretable
  features (Anthropic's production SAEs sit ~0.5–0.7).
- **Root cause (two parts, both verified with controls):**
  1. **The metric was computed in the wrong space.** With `normalize-input`, the SAE standardizes each
     token, then *un-standardizes* the reconstruction — which hands back each token's original
     magnitude **for free**. Variance-explained measured on that un-standardized output is inflated.
     Measured honestly (in normalized space), it's **0.76, not 0.97**.
  2. **The dictionary/sparsity knobs were off.** A 16× dictionary and very dense codes also push the
     number up.
- **The "sink token" hypothesis was tested and ruled out as the *cause*.** Sinks *do* exist (4 stable
  outlier channels, ~0.1% of tokens at 7× norm) — but removing them changes var-explained by **0.0001**,
  because `normalize-input` already neutralizes them. Real, but not why the number was high.
- **Winner: layer 32, 8× dictionary, top_k 128, `normalize-input` + AuxK.** Honest var-explained 0.76,
  **12.8% dead latents**, **GO-AUC 0.83**, and **loss-recovered 0.94** — a genuine, faithful feature model.

---

## 1. The variance-explained investigation

### The symptom
The TopK SAE at L32 logged `var_explained ≈ 0.975`. We expected ~0.5–0.7 for an interpretable SAE, so
this looked like the SAE was just copying its input.

### Hypothesis A — the metric is computed on the wrong quantity
`normalize-input` works like RMSNorm-for-the-SAE: each token `x` is standardized over its 2560 dims
(`x_norm = (x − μ)/σ`, with μ/σ computed **per token**), the SAE encodes/decodes in that normalized
space, and the reconstruction is then **de-normalized** back (`× σ + μ`). The catch: de-normalization
re-inserts each token's *exact* mean and magnitude — the SAE gets that part **for free**. Our token
norms vary wildly (p50 = 347, max = 14,214), so that free magnitude is a huge slice of the raw variance.

Measuring var-explained in both spaces on the same SAE:

| space | var-explained | what it means |
|---|---|---|
| **raw** (de-normalized) | **0.975** | inflated — credits the free per-token magnitude |
| **normalized** (honest) | **0.747** | the SAE's *actual* feature-reconstruction quality |

→ **Confirmed.** The headline number should be the normalized one. (We also found a related bug in
`eval.py`: it called `encode()` then `decode()` *without* the normalization `info`, skipping
de-normalization → a garbage R² of −0.11. Fixed.)

**Training-time logging fixed too.** The W&B curves previously showed only the inflated raw
`variance_explained` (~0.97). We added `variance_explained_normalized` to the SAE loss (`topk.py`, both
dense + Triton paths) so training now logs **both** — confirmed on run `sae-l32-exp8-honest-varexp`:
`train/variance_explained = 0.976` (raw) vs `train/variance_explained_normalized = 0.760` (honest,
matching the eval). All recipes get this metric automatically.

### Hypothesis B — sink tokens (your suggestion)
Sink tokens are a small fraction of positions with huge magnitude on a few "wired" outlier channels —
architectural (optimizer-induced), input-independent, and known to dominate SAE loss if not removed.
We ran the SVD screen on 1.07M L32 tokens:

```
sink channels (|v1|>0.1): seed0=[0,4,19,21]  seed1=[0,4,19,21]   ← stable across seeds (architectural)
sink tokens (top 0.10%): mean norm 4659 vs normal 706  (7×)
```

So **sinks are real and present**. But the decisive control — recompute var-explained with all sink
tokens removed:

```
RAW var-explained  ALL tokens     = 0.9750
RAW var-explained  sinks REMOVED  = 0.9749    ← identical
```

→ **Sinks are NOT the cause of the high var-explained.** `normalize-input` already divides out each
token's magnitude before the SAE sees it, so the sinks are neutralized. They are worth screening for
*cleaner features* in a full run, but they don't explain this number. (Note: sink screening is **not**
in the shared `sae` package; PR #1619 is the dead-latent/FVU **training-flags** PR, which we *do* use.)

---

## 2. Config comparison (layer 32)

All trained on the same 28.8M-token store, evaluated on the same 300 held-out proteins with the fixed
eval. **Honest var-exp** = normalized space; **raw** shown for reference.

| config | honest var-exp | raw var-exp | dead % | GO-AUC (text) | cross-modal feats | loss-recovered | verdict |
|---|---|---|---|---|---|---|---|
| **exp8 — 8× dict, top_k 128** | **0.762** | 0.977 | **12.8** | 0.831 | 492 | **0.939** | ✅ **WINNER** |
| exp16 — 16× dict, top_k 128 | 0.746 | 0.975 | 47.9 | 0.828 | 165 | — | dict too big |
| exp8_k32 — 8× dict, top_k 32 | 0.621 | 0.963 | 88.9 | 0.827 | — | — | too sparse |

**What the knobs do:**
- **Dictionary size (expansion factor).** 8× = 20,480 features; 16× = 40,960. Doubling it didn't improve
  reconstruction — it just created dead features (12.8% → 47.9% dead). Bigger ≠ better here.
- **Sparsity (top_k).** Dropping from 128 → 32 features-per-token forced var-exp down (0.76 → 0.62, the
  interpretable direction) but **collapsed 89% of features to dead** — too aggressive with this AuxK.
- **GO-AUC ≈ 0.83 in every config** — biology is robustly captured regardless; the deciding factors are
  dead-latent health and reconstruction fidelity.

---

## 3. The winner — L32, 8× dictionary, top_k 128

| metric | value | target | pass | meaning |
|---|---|---|---|---|
| honest variance-explained | **0.76** | >0.8\* | ~ | feature-reconstruction quality (Anthropic range) |
| **loss-recovered (CE)** | **0.939** | >0.8 | ✅ | model keeps 94% of its fidelity when the SAE recon is substituted at L32 (CE 0.80 → 1.48; zero-ablation = 11.93) |
| dead latents | **12.8%** | <20% | ✅ | healthy feature usage |
| GO-feature AUC (text band) | **0.83** | high | ✅ | features track real biological function |
| cross-modal features | 492 | — | — | fire across >1 modality (for SAE-V analysis) |

\* *The 0.8 target was set against the inflated raw number. 0.76 honest is squarely in the interpretable
regime, and loss-recovered 0.94 independently confirms the SAE is faithful — not a copy machine.*

### Feature usage (the "how many features" question)
- **Dictionary: 20,480 features** (8 × 2,560).
- **17,865 are used** (ever fire on the eval set); ~2,615 dead.
- **Each token activates exactly 128** of them (top_k).

**Verdict: the subset SAE gate is a GO.** L32 / 8× / top_k 128 reconstructs faithfully (loss-recovered
0.94), keeps features alive (12.8% dead), and stays biologically meaningful (GO-AUC 0.83).

---

## 4. Layer 28 vs layer 32 (8× config)

`LAYER_ANALYSIS.md` narrowed the choice to L28–L32; L32 had higher effective rank. Here we train **L28
at the same winning 8× config** for a direct comparison.

Both trained identically (8× dict, top_k 128), each evaluated on its own held-out 300-protein store
(`val300_L28` / `val300_L32`, same proteins, 1.07M tokens each).

| layer (8×, top_k 128) | honest var-exp | dead % | GO-AUC (text) | active features | loss-recovered |
|---|---|---|---|---|---|
| **L32** | 0.762 | **12.8** ✓ | 0.831 | 17,865 | 0.94 |
| L28 | 0.758 | **54.9** ✗ | 0.843 | 9,244 | — (disqualified) |

**L28 does not improve on L32.** Reconstruction quality is a tie (0.76), and L28's GO-AUC is marginally
*higher* (0.843) — but **over half of L28's dictionary dies (54.9% dead)** vs 12.8% at L32, and it uses
only 9,244 of 20,480 features. This is exactly what `LAYER_ANALYSIS.md` predicted from effective rank
(L28 PR ≈ 253 vs L32 ≈ 670): the lower-rank layer can't support a wide dictionary without collapsing.
The 8× config + corrected eval did not rescue L28 — the layer's intrinsic rank is the limiter.
**This empirically confirms the layer-selection study's choice of L32.**

---

## 5. Feature interpretation (do the features mean anything?)

Healthy metrics don't prove the features are *interpretable concepts* — that's the point of an SAE. So
before scaling we ran a **biology-grounded autointerp** on the L32 winner: characterise each feature by
*what the proteins it fires on have in common* (per-protein ROC-AUC against informative GO terms, names
from `go-basic.obo`). Feature → concept direction; complements eval.py's concept → feature.
*(Script: `scripts/autointerp_biology.py`.)*

**Headline: the SAE learned sparse, monosemantic biological features.**

- **Selectivity, not frequency, finds interpretable features.** The highest-*frequency* features fire on
  56–78% of all tokens and are non-selective (AUC ≈ 0.51) — "always-on" / polysemantic. The interpretable
  features are **sparse** (fire on <1% of tokens) and map to a single concept.
- **~28% of active features (4,931 / 17,865) are biologically selective** (best-term AUC above the
  random best-of-60-terms baseline of 0.597 by ≥0.05).
- **Top features map cleanly to concrete GO concepts, and hold up on held-out proteins** (AUC selected on
  a train split, scored on a disjoint test split — kills winner's-curse):

  | feature | fire % | AUC (in-sample) | **AUC (held-out)** | enrichment | concept |
  |---|---|---|---|---|---|
  | 12409 | 0.1 | 0.98 | **0.98** | 3.0× | positive regulation of biological process |
  | 9982 | 0.2 | 0.96 | **0.98** | 1.8× | plasma membrane |
  | 1275 | 0.1 | 0.96 | **0.98** | 3.1× | cytosol |
  | 2048 | 0.1 | 0.92 | **0.93** | 4.1× | catalytic activity |
  | 13476 | 0.1 | 0.92 | **0.94** | 2.7× | protein-containing complex |
  | 16902 | 0.1 | 0.91 | — | 3.7× | nucleic acid binding |

  Held-out AUC tracks in-sample (0.90–0.98), so these are real, not artifacts.
- **Concepts span localization** (cytosol, plasma membrane, membrane, protein complex), **molecular
  function** (catalytic activity, nucleic-acid / protein binding), and **biological process** (regulation,
  development, response to stimulus).
- **Feature splitting is present** — ~8 distinct features all encode "cytosol" (12941, 17901, 1275, 570,
  11897, 14743, 897, 11315). Multiple fine-grained features for one concept is a known healthy-SAE trait,
  not a bug.

**Interpretation verdict: GO.** Combined with var-exp 0.76 / dead 12.8% / loss-recovered 0.94, the SAE is
not just faithful — its sparse features correspond to human-readable biology. This is the validation we
wanted *before* committing to the full extraction.

### 5b. Text-context view — the linguistic trigger (Part B)

The biology view labels features by *which proteins* they fire on. We also decoded the actual
reasoning-**text** windows around each feature's top-activating tokens (`scripts/interp_text_contexts.py`,
reproducing each protein's token sequence and matching by `protein_id`). The features are cleanly
**monosemantic** — each fires on the exact token of its concept (⟦⟧ = the firing token):

| feature | biology label (GO-AUC) | top-activating text contexts | concept |
|---|---|---|---|
| 12409 | positive regulation (0.98) | "…⟦positive⟧ regulation of ruffle assembly…", "…⟦positive⟧ regulation of cyclin-dependent kinase…", "…⟦positive⟧ regulation of cell growth…" | the concept **positive regulation** across many distinct GO terms |
| 2048 | catalytic activity (0.93) | "…transferase activity…", "…hydrolase activity…", "…lyase activity…", "…nucleotidyltransferase activity…" | **enzyme catalytic activity** |
| 16902 | nucleic acid binding (0.91) | "…⟦nucleic⟧ acid binding…" (×5, near RNA/DNA binding) | **nucleic-acid binding** |
| 1275 | cytosol (0.96) | "…GO:0005829 ⟦cytosol⟧…" (×5) | **cytosol** localization |
| 13476 | protein complex (0.94) | "…SPOTS ⟦complex⟧…", "…holoenzyme ⟦complex⟧…", "…SCF ubiquitin ligase ⟦complex⟧…" | **protein complex** |

The two views **agree**: feature 12409 scores GO-AUC 0.98 for "positive regulation of biological
process" *and* fires on the literal word "positive" in that phrase across many different child terms —
it has abstracted the concept, not memorized one term. *(Honest note: these text features partly detect
the GO-term phrasing present in the reasoning prompt — a read-back component, consistent with the fusion
control in `LAYER_ANALYSIS`. The SAE-V protein↔text fusion below shows some of this concept is also
grounded in the protein embedding, not pure text read-back.)*

*Caveats:* on the 300-protein subset; the full run will sharpen rare-concept features.

## 6. Multimodal feature fusion (SAE-V)

BioReason-Pro is multimodal (protein embeddings + GO memory + reasoning text in one stream), so a key
question is whether the SAE finds **features shared across modalities** — the same concept represented
in both the protein embedding and the text. We applied **SAE-V** (arXiv:2502.17514, Eq. 7): for each
feature, take its top-K activating tokens *within each band* and measure the mean rank-paired cosine
(`omega`) of their residual vectors across band pairs. omega → 1 = a fused, modality-agnostic feature.
*(Script: `scripts/crossmodal.py`.)*

**The baseline is essential.** The residual stream has shared directions (DC component, the sink
channels from §1) that inflate cosine between *any* two tokens, so raw omega is meaningless alone. We
compare against the mean cosine of **random cross-band token pairs**:

| band pair | omega (features) | random baseline | **lift over baseline** | reading |
|---|---|---|---|---|
| **protein ↔ text** | 0.50 | **−0.08** | **+0.59** | **strong, genuine fusion** |
| go ↔ text | 0.48 | 0.15 | **+0.33** | moderate fusion |
| protein ↔ go | 0.55 | 0.52 | +0.02 | *not* special — shared injected-embedding structure |

**Reading (the baseline flips the naive interpretation):**
- **protein↔text is the most fused** (+0.59) even though its raw omega looked weakest. Random
  protein/text token pairs are near-orthogonal (−0.08), but feature-selected pairs sit at 0.50 — the SAE
  finds features where **the protein embedding and the reasoning text point the same direction**. These
  are genuine multimodal concepts (the model's shared "protein function" representation), consistent with
  §`LAYER_ANALYSIS` (function decodes from both the protein embedding ~0.73 and the text ~0.83).
- **go↔text** is moderately fused (+0.33).
- **protein↔go** looked most aligned (0.55) but is the *least* fused (+0.02): both are high-norm injected
  embeddings with high background cosine; the features add almost nothing beyond that.
- **Partial, not perfect:** no feature exceeds omega 0.7 absolute, so fusion is real but incomplete —
  the shared cross-modal direction coexists with modality-specific structure.

Only 128 / 200 features fire strongly in the protein / go bands (vs 11,044 in text), echoing the
low-rank injected bands — so the SAE's multimodal features are relatively few, but the protein↔text ones
are clearly real.

## 7. What's next (your call)
1. **Finalize + push to GitHub** for review (currently blocked on git credentials).
2. **Full-scale run on Lepton** — the gate is GREEN; awaiting your one-line OK.
3. **Autointerp labeling** of the top features (what concepts they encode).

## Reproduce
```bash
# train the winner (Env B)
python scripts/train.py --cache-dir <store>/layer32 --layer 32 \
  --model-type topk --expansion-factor 8 --top-k 128 --normalize-input --auxk 1024 \
  --aggregate-loss --dead-count-global --mix-shards 10 --presample-shards 8 --init-pre-bias \
  --n-epochs 1 --batch-size 4096 --lr 1e-4 --lr-schedule cosine --warmup-steps 1000
# eval: honest (normalized) + raw var-exp, dead%, GO-feature AUC, cross-modal (Env B)
python scripts/eval.py --sae sae_l32_exp8/checkpoint_final.pt --store val300_L32 --layer 32
# CE loss-recovered: substitutes SAE recon at the layer hook (Env A — loads the model)
python scripts/eval_loss_recovered.py --sae sae_l32_exp8/checkpoint_final.pt --layer 32 \
  --split validation --num-proteins 150
```
