# BioReason-Pro SAE — dataset, modality results, and open questions

Status doc for team review. Covers (1) what data we're training on, (2) the modality-specific results
(coverage, rank, feature quality), (3) the rank analysis in detail, (4) decisions to make next.
Companions: `SAE_RESULTS.md` (full results), `MODALITY_BALANCING.md` (experiment plan), `LAYER_ANALYSIS.md`.

---

## 1. Dataset — what we're actually using

The model is BioReason-Pro (Qwen3-4B + ESM3 protein embeddings + GO graph memory). Each example is one
**protein** (a reasoning prompt); tokens within an example are tagged **protein** (ESM3 slots) / **go**
(graph memory) / **text** (prompt + reasoning). A subset of "**8,000**" means **8,000 proteins/examples**
(not 8k tokens) — at ~**3,601 tokens/example** that's **28.8M tokens**.

| split / subset | **proteins (examples)** | tokens | used for |
|---|---|---|---|
| **current subset (L28/30/32 + all recent analysis)** | **8,000** | **28.8M** | SAE training |
| earlier subset (layer 24, abandoned) | 20,000 | ~72M | original Step-C (the "~70M" figure) |
| held-out eval | 300 (validation) | 1.07M | all eval / feature analysis |
| **full train split** | **117,002** | **~420M** (projected) | the full run (gated) |
| full validation split | 7,365 | ~27M | — |

**So everything recent runs on 8k proteins → ~28.8M tokens (~6.8% of train proteins).** We *downsized*
from the 20k/72M layer-24 extraction when we switched to the probe-selected layers, for fast cycles.

**Token distribution within the 8k train subset** (`scripts/token_distribution.py`):

| band | corpus share | per-protein: mean / median / range |
|---|---|---|
| **protein** | **15.0%** | 541 / 425 / **16–2,002** (varies by protein length, capped at `max_length_protein`) |
| **go** | **5.6%** | **200 / 200 / fixed** — always exactly 200 GO-memory slots |
| **text** | **79.4%** | 2,860 / 2,654 / 1,163–11,218 |

(val300 is nearly identical: protein 15.4% / go 5.6% / text 79.0%.) Note the **protein-token count
varies hugely** (16→2,002, std 415) — driven by protein length — so the 15% corpus share is dominated by
*long* proteins; the median example is only ~425 protein tokens (~13%). **GO is fixed at exactly 200.**

**Sampling:** `extract.py` takes the **first N** proteins (`select(range(n))`, no `--shuffle`). But the
dataset's native order is **already mixed** (IDs span organisms/prefixes, not sorted) and the subset is
diverse (**15,344 distinct GO terms, 39 GO/protein**), so it's **effectively representative**.

---

## 2. SAE quality (all layers, 8× / top_k 128 / normalize_input + normalize_loss, held-out)

| layer | honest var-exp | dead % | GO-AUC (text) | loss-recovered |
|---|---|---|---|---|
| **L28** | 0.863 | 5.1 | 0.846 | 0.988 |
| L30 | 0.841 | 3.8 | 0.834 | 0.983 |
| L32 | 0.811 | 5.3 | 0.844 | 0.977 |

Faithful, healthy SAEs. (Reconstruction/fidelity rise toward earlier layers; see SAE_RESULTS §1c.)

---

## 3. Modality-specific results — the SAE under-serves biology

### 3a. Feature coverage per band (`per_band_coverage.py`, held-out)
How many of the 20,480 features ever fire on each band's tokens:

| layer | protein feats | GO feats | text feats | **bio total** |
|---|---|---|---|---|
| **L28** | 1,222 (6.0%) | 670 (3.3%) | 19,321 (94.3%) | **1,658** |
| L30 | 210 (1.0%) | 258 (1.3%) | 19,654 (96.0%) | 299 |
| L32 | 309 (1.5%) | 284 (1.4%) | 19,296 (94.2%) | 382 |

**Bio tokens are 21% of the corpus but get 1.5–8% of the features.** ~98% of features at L32 never fire
on a protein token. **L28 fits ~4× more bio features than L32** (earlier layer keeps more bio-specific
structure before the model folds it into text).

### 3b. Per-band feature quality (held-out de-biased GO-AUC)
| band | best-feature AUC |
|---|---|
| text | ~0.84 |
| **protein** | **~0.59** (≈ random baseline 0.59) |
| go | ~0.57 |

Text features are strong; **protein/GO features are weak** — consistent with there being few of them and
little structure to carve.

---

## 4. Rank analysis (why protein features are scarce + weak)

**What "rank" means:** stack a band's tokens (each a 2,560-vector) and ask how many *independent
directions* they span — the effective rank. Few directions = "low rank" = little independent structure =
few distinct concepts an SAE can learn. **Participation ratio** `PR = (Σλ)²/Σλ²` (effective # dims);
`k90` = # components for 90% of variance.

**Two measurements (the space matters):**

| band | RAW (mean-centered) PR / k90 | **NORMALIZED** (per-token, magnitude removed) PR / k90 |
|---|---|---|
| protein | 1.8 / 4 | **3.0 / 10** |
| go | 4.2 / 11 | **4.2 / 12** |
| text | 2.3 / 528 | **103 / 1508** |

- **Raw** is magnitude-confounded — one PC (the magnitude direction) is ~80% of variance, so PR ≈ 1–2 for
  every band. (This is the same magnitude artifact behind the var-explained story.)
- **Normalized** (what the SAE actually models) is the real structural rank: **protein ≈ 3–10, text ≈
  100–1,500.** Text is ~30–150× higher rank → far more concepts to learn → far more features.

**Interpretation:** protein-token activations occupy a tiny subspace → the SAE can only find ~10-ish
distinct protein concepts → few + weak protein features. Three independent measurements agree (coverage
~1.5%, AUC ~0.59, rank ~3–10).

### Rank by layer (per-token normalized PR / k90)

| layer | protein | go | text |
|---|---|---|---|
| L28 | 2.9 / 9 | 4.5 / 19 | 127 / 1186 |
| L30 | 2.9 / 9 | 4.4 / 14 | 113 / 1357 |
| L32 | 3.0 / 10 | 4.2 / 12 | 103 / 1508 |

- **Protein rank is flat (~3) across all three layers** — protein is intrinsically low-rank at every
  depth in this range, *not* a layer artifact. (So going earlier, L24/L26, is **unlikely** to add protein
  structure — worth testing, but the L28→L32 trend says don't expect much.)
- text/go vary only mildly with depth; text is high-rank everywhere.
- **Key reconciliation:** L28 fits 4× more protein *features* (1,222 vs L32's 309) over the **same ~3–10
  protein dimensions** → that's **more redundancy (feature-splitting), not more distinct concepts.** So
  "L28 is the better *bio* layer" should be softened: it gives protein tokens more (redundant) coverage,
  but the underlying protein *structure* is the same low rank at L28 and L32. L28's real edge is fidelity
  + being earlier (steering), not richer biology.

### What do the dead latents look like? (L32, the 1,091 dead-everywhere = 5.3%)
- **Not redundant duplicates:** median decoder-cosine to the nearest *live* feature is **0.094** (near-
  orthogonal); 0/1,091 exceed 0.9. They're genuinely distinct directions, not copies.
- **Not inert/random:** their median max *pre-activation* (before top-k selection) is **4.4** (vs live
  features' median activation ~13); only 1/1,091 is truly ~0. **99% "almost fire."**
- So dead latents are **distinct directions that consistently lose the top-128 competition** — top_k is
  the binding constraint, and AuxK keeps them almost-alive. (Implication: a larger `top_k` or more
  capacity would revive many; they're not wasted random directions.)

**Caveats:**
- The number is **method-sensitive** (`LAYER_ANALYSIS.md` reported protein PR ~85 on a 1,500-protein
  sample with a different normalization). The *ordering* (protein ≪ text) is robust; the absolute value
  isn't.
- **It's measured on only 300 proteins.** Effective rank is bounded by sample diversity, so part of the
  low protein rank is small-N, not intrinsic. More proteins should raise it — **untested**.

---

## 5. Open questions / decisions moving forward

**Data scale**
- **Is the low protein rank/coverage intrinsic, or a small-subset artifact?** We're on 8k proteins (28.8M
  tokens) / 300-protein eval. *Proposed test:* bump the subset to 20k (~72M) or go full (117k / ~420M)
  and re-measure protein rank + coverage. This is the single biggest unknown.
- Should we re-extract with `--shuffle` for rigor (low risk given the data is pre-shuffled, but clean)?

**Layer**
- **L28 fits ~4× more bio features than L32** and is more faithful — but per §4 the protein *rank* is the
  same (~3) at all three layers, so L28's extra protein features are likely **redundant, not new
  concepts**. So L28's edge is fidelity + earliness (steering), not richer biology. *Proposed:* still
  worth testing L24/L26 with the fixed loss, but the flat protein-rank trend says **don't expect earlier
  layers to add protein structure** — the bottleneck looks like the protein representation itself, not
  the layer. (Caveat: rank is sample-limited; scale could change this.)

**Modality balancing** (full plan in `MODALITY_BALANCING.md`)
- **Inverse-frequency loss reweighting** to pull capacity toward protein/GO — does protein-band AUC beat
  0.59, or is the low rank a hard ceiling?
- **Normalized AuxK** (the dead-latent revival is still raw-space).
- **Per-modality `pre_bias`** (the global geometric median is text-dominated).

**The causal question**
- **Does the model actually use the protein/GO input, or is text carrying the biology?** *Proposed:*
  ablate the protein/GO embeddings and measure GO-prediction degradation. (Probing says function decodes
  from text > protein embedding — but that's correlational.)

**Suggested first move:** scale the subset up (20k or full) and re-run the per-band coverage + rank,
because nearly every open question ("is bio under-served intrinsically?", "would reweighting help?",
"which layer?") is confounded by the current small protein count.

---

## 6. Reproduce
```bash
python scripts/per_band_coverage.py --sae <ckpt> --store val300_L<L> --layer L --out-json out.json
# rank: see the per-token-normalized SVD in this doc's §4 (ad-hoc; can fold into rank_analysis.py)
```
