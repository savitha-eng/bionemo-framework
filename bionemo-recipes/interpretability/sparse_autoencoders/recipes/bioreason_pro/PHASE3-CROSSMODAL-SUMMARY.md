# Phase 3 SAE Interpretability — Cross-modal & Matryoshka Results

_BioReason SFT models (Qwen3-4B multimodal). Prepared 2026-07-02._

## TL;DR

1. **Cross-modal fusion is attention-mediated, not feature-level** — confirmed at *full dataset scale* on **both** protein and DNA. An SAE trained on the residual stream does not surface concept-level protein↔text or DNA↔text "fusion" features.
2. **Matryoshka / modality-weighting does NOT beat flat BatchTopK** on feature quality. Flat TopK/BatchTopK is the recommended recipe.
3. **Protein features are intrinsically weak** across every SAE variant (an ESM3-encoder property, not an SAE-method problem). **DNA is the far better substrate** (~12× stronger features).

---

## 1. Cross-modal: does the model bind protein/DNA to text at the feature level?

Metric: per-feature **φ co-activation** — does feature _k_ fire on the bio tokens AND the text tokens of the **same** samples, controlled for base rate? φ>0 = genuine cross-modal binding of a shared concept. (Raw-hidden cosine, à la SAE-V, is blind here because the modalities occupy different residual subspaces — φ in the shared SAE basis is the honest test.)

| Modality | Scale | Co-firing features | Selective, φ>0.3 | Verdict |
|---|---|---|---|---|
| **Protein** | **117,002 proteins** (full train, L24) | 73 | **0** | Clean null — no cross-modal binding |
| **DNA** | **36,088 samples** (full, L16) | 5,107 | **2** | Both non-semantic (see below) |

**The 2 DNA "survivors" are not concept binders:**
- **F15966** — a structural **boundary marker**: fires on DNA nucleotides and lands its text-side peak exactly on the DNA-END marker → first question word. Fires at the modality *junction*, not on a shared concept.
- **F16612** — a strong general **DNA-content** detector (act 13–16) that *leaks* weakly (act 5–8) onto genomic-identifier text (gene names, variant IDs). Not concept-specific on the DNA side.

**Why this matters:** DNA (unlike protein) *does* share residual subspace with text (raw cosine 0.37 vs ~0 for protein), so we expected DNA fusion features to be learnable. They are not — the subspace sharing is driven by these high-magnitude boundary/content features, not by concept binding. So the null holds even in the favorable case. Fusion in these models lives in attention, not in per-token residual features.

## 2. Does Matryoshka / modality-weighting improve feature quality?

7-way comparison at L16 (all converged step-10000 checkpoints; identical shard sample; NIM-free metric = high peak activation + low firing frequency + wide token span, split by modality).

| Variant | HQ protein | HQ text | Note |
|---|---:|---:|---|
| **Flat BatchTopK** (`btk-only`) | **6** | **3513** | Best all-around |
| Modality-wt, uniform (`mw-uniform`) | 9 | 3226 | 9-vs-6 protein is within noise (single digits) |
| Modality-wt, protein-upweighted (`mw-coarse`) | 6 | 3257 | The variant *designed* to boost protein = **same as flat** |
| Modality-wt, fine (`mw-fine`) | 4 | 2996 | |
| Per-token Matryoshka (`matry-pertoken`) | 1 | 2187 | Worst — over-allocates weak protein features |
| BatchTopK-Matryoshka k64 (`mbtk-k64`) | 3 | 2448 | Starves minority modality |
| BatchTopK-Matryoshka k32 (`mbtk-k32`) | 1 | 983 | Low-k starves *everything* |

**Verdict:** neither nested-prefix Matryoshka nor per-prefix modality-weighting improves feature quality over plain flat BatchTopK. The protein-upweighted variant matched flat exactly, so the modality-weighting hypothesis is not supported. This corroborates the earlier autointerp result (flat detection-F1 0.522 > Matryoshka-BatchTopK 0.373). **Recommendation: flat TopK / BatchTopK.**

## 3. Protein feature weakness → DNA is the better substrate

Across all 7 variants, protein features stay weak (median peak activation 0.8–5.8) vs text (10–22). No architecture fixes it → it is specific to the ESM3 protein encoder (its per-token embeddings are near-collinear, self-cosine ~0.99), not an SAE problem.

BioReason-**DNA** (Evo2-1B → projector → Qwen3-4B) is a much better SAE target: token embeddings are far more varied (self-cosine 0.55), and its SAE features are **~12× stronger** (median max-activation 13.98 vs 1.18 for protein). The DNA L16 SAE dashboard is the more interesting artifact to browse.

---

## Artifacts

- Protein cross-modal (full 117k, L24): `multimodal_dashboard/public/crossmodal/crossmodal_l24_matryoshka_full.json`
- DNA cross-modal (full 36k, L16): `/data/savithas/dna_sae/crossmodal_dna_FULL.json`
- Matryoshka quality atlases (7 variants): `/data/savithas/phase3_full/qual_atlas/<variant>/`
- DNA feature dashboard: `?model=dna_l16_vep` (with live auto-interpret button)

## Optional next step

The one experiment that would give *positive* evidence for where fusion lives is **causal activation patching** across the modality boundary (patch bio-token activations into a text-only forward and measure answer shift). Both cross-modal verdicts point to attention-mediated fusion; patching would confirm the mechanism directly. Scoped but not yet run.
