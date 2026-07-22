# Sparse Autoencoders on a Multimodal Protein-Reasoning Model: Interpretable, Localizable — but Not More Decodable

*Savitha S. — July 2026. Companion mini-paper to the BioReason-Pro SAE recipe. Structured after Wilber & Cao's CodonFM SAE paper — and, in one important way, its mirror image.*

## Abstract

Codon and protein foundation models encode biology in superposition, and sparse autoencoders (SAEs) have been shown to pull that biology apart into cleaner, more decodable features than the raw neurons — most recently for **EnCodon/CodonFM**, where SAE features beat raw hidden dimensions on 96.9% of gene-level GO concepts. We apply the same lens to **BioReason-Pro**, a *multimodal* model that fuses ESM3 protein embeddings, a GO-graph encoder, and a Qwen3-4B reasoning LLM to predict protein function through a written reasoning trace, and we report a deliberately honest result: **on this model, the SAE does *not* provide better decodability than the raw residual stream.** On dense per-protein probes the SAE never beats raw (InterPro domains: 0.99 ≈ 0.99, saturated); a *sparse* SAE probe beats raw only for a handful of reasoning-band concepts that clear a strict leakage floor (4 of 12). The value of the SAE here is not decodability but **structure and interpretability**: sparse features that *localize* to protein domains (a kinesin-motor feature with domain-F1 = 0.98), reasoning-band features that carry genuine antimicrobial-defense mechanisms above the leak floor, and a modality-balancing recipe that is *required* to make protein features exist at all (20–50× more protein-selective features). We trace the decodability ceiling to the ESM3 encoder — protein token embeddings are nearly collinear (self-cosine ≈ 0.99), so per-token protein signal is real but swamped — and we ship an auto-interp pipeline and an interactive multimodal feature atlas. The negative result is the contribution: **SAEs are not a universal decodability win; whether they help depends on the encoder and the modality.**

---

## 1. Introduction

BioReason-Pro is unusual: it doesn't just embed a protein, it *reasons* about it. ESM3 residue embeddings and a GO-graph encoder are projected into the token stream of a Qwen3-4B LLM, which then writes a free-text reasoning trace ("…the N-terminal P-loop NTPase fold positions duplex DNA…") and predicts GO function. That makes a single residual stream carry three modalities at once — **protein-residue tokens, reasoning-text tokens, and `<go>` graph slots** — and makes it a natural, hard test for SAE interpretability: can we pull apart what the model knows, and does the sparse dictionary decode biology better than the dense stream?

We trained TopK SAEs on the layer-30 residual stream and interpreted the features. Our contributions:

1. **A balanced multimodal SAE recipe + a leakage-aware evaluation discipline.** Naïvely, one modality (text) monopolizes the dictionary; balancing is not optional. And because the reasoning text *names* the function, ordinary decodability is inflated by leakage — we measure everything against a random-SAE **leak floor**.
2. **The honest decodability result:** unlike CodonFM, the SAE here never beats raw on dense probes, and beats it only marginally (sparse) for a few reasoning concepts. The encoder, not the SAE, is the ceiling.
3. **Interpretable structural features** with a *localization* metric (domain-F1), separating genuinely localized detectors from high-AUROC features that fire outside their domain.
4. **A characterization of cross-modal features** showing that protein↔reasoning links are **prompt-mediated correlations, not causal feature-to-feature flow.**
5. **An auto-interp pipeline, a grounded synthesis-quality scorer, and an interactive feature atlas** for browsing per-band firing, domain-F1, enrichment, and steering.

---

## 2. SAE setup: turning a multimodal residual stream into sparse features

For each layer-30 hidden state `x ∈ ℝ^2560`, the SAE encodes `x` into an overcomplete sparse code `z ∈ ℝ^40960` (expansion factor 16) and decodes back to `x̂`, trained with reconstruction loss and TopK sparsity (**k = 128**, ~0.31% active) plus the OpenAI auxiliary-k revival loss. Activations come from ~8k proteins run through BioReason-Pro; each row is one token, tagged `protein`/`text`/`go` in a row-aligned sidecar so any downstream analysis can select a band.

**Modality balancing is required, not a nicety.** In the natural mix, protein tokens are ~2% of the stream and text tokens monopolize the dictionary. Our reported SAE (`sae-l30-exp16-balanced`) trains with `--balance-modality`: it **drops the `<go>` slots** (an ablation shows they are model-unused — they are a fixed ontology reduction, not per-protein terms) and **downsamples text to ~50/50 with protein** (text-keep ≈ 0.19). We use per-token input normalization and a **batch-aggregated loss** (`--aggregate-loss` + `--normalize-loss`): in a multimodal stream, tokens sit at very different distances from the shared center and are reconstructed to very different degrees, so a per-token loss starves the minority modality's features and mis-aims dead-latent revival — batch aggregation fixes both. (Details in §9 and the recipe.)

**The evaluation discipline — the leak floor.** The reasoning trace often *states* the function, so a random-initialized SAE already "decodes" it. We therefore train a probe on a random-SAE as a **leak floor** and trust only the *margin* `sae_sparse − random`. We also distinguish **scoring** (label-grounded per-feature AUROC — trustworthy) from **labeling** (an LLM auto-interp read of top windows — a hypothesis; §8). Every number below is reported against these controls.

---

## 3. Does the SAE recover biology better than raw? — Mostly **no** (the mirror image of the codon result)

This is the central, honest finding, and it inverts the CodonFM story.

**On dense per-protein decodability, the SAE never beats raw.** Ranking proteins by mean-pooled representation and scoring against InterPro domains, the SAE, raw hidden state, and even a random SAE all saturate together:

| detector (per-protein, InterPro domain) | AUROC |
|---|---|
| SAE (sparse) | 0.990 |
| raw hidden | 0.989 |
| random SAE | 0.981 |

The SAE is a **lossy re-expression** of the residual stream; it cannot manufacture decodability the encoder didn't already have, and biology "doesn't build up" with depth here the way it does in the codon model.

![layer decodability](charts/layer_decodability.png)
> **Decodability across layers.** SAE (sparse), raw, and random-SAE probes track each other at every depth — the SAE does not open a gap over raw. Contrast Jared & Fan's codon result, where SAE features beat neurons on 96.9% of GO-slim concepts.

**Where the SAE *does* add something is narrow and sparse.** On the reasoning band, a *sparse* SAE probe beats raw for a few concepts — but only those that clear the leak floor. Of 12 designed GO concepts, **only 4 clear a margin ≥ 0.05**: defense→fungus (+0.111), defense→bacterium (+0.108), structural-molecule (+0.063), plasma-membrane (+0.055). The rest sit *at* the leak floor (the text just names them). And for the echo-vs-synthesis distinction, the SAE adds nothing at all (raw 0.916 ≈ SAE 0.914).

**Why:** the ceiling is the **ESM3 encoder**, not the SAE. Protein token embeddings are nearly collinear (self-cosine ≈ 0.99), so every protein token's code is dominated by the same high-magnitude component — the signal separating one protein from another is real but swamped per-token (§9). This reframes the whole project: the interesting SAE contributions are not decodability, but **localization, sparse interpretable concepts, and the multimodal representational picture.**

---

## 4. Structural features: Fisher enrichment + a localization metric (domain-F1)

To label features we compute, per feature, the most over-represented **GO or InterPro term** among the proteins where it fires (hypergeometric test + per-feature AUROC, on the proteins where the feature actually fires so sparse features aren't diluted). This surfaces clean structural detectors — e.g. a GPCR-rhodopsin feature (IPR000276, FDR 3e-84) and a kinesin-motor feature.

![enrichment](charts/enrichment.png)
> **Per-feature GO+InterPro enrichment.** Each structural feature's best-over-represented term with its hypergeometric FDR and rank-sum AUROC.

But **AUROC is not localization.** A feature can score AUROC 0.98 for a domain yet fire *outside* it. So we add **domain-F1** = (per-position precision: of residues where the feature fires, the fraction inside the domain) × (per-region recall: of domain regions, the fraction with ≥1 firing residue):

| feature | concept | AUROC | **domain-F1** | verdict |
|---|---|---|---|---|
| **F18393** | kinesin motor | 0.98 | **0.98** | 📍 genuinely localized (fires inside the motor domain) |
| F7369 | GPCR (IPR000276) | high | 0.51 | ⚠️ half in-domain |
| **F4647** | (AUROC oversell) | **0.98** | **0.00** | ❌ high AUROC, fires *outside* the domain |

Applying domain-F1 across all AUROC-passing structural features, **only ~17% (59 of 872) genuinely localize.** The lesson generalizes: for structural interpretability, trust the localization metric, not the ranking metric.

![per-feature structural](charts/per_feature_structural.png)
> **Structural features by localization.** The localized minority (domain-F1 high) vs the AUROC-oversell majority.

---

## 5. Reasoning-band features: a distributed antimicrobial-defense circuit (the flagship reasoning result)

The strongest *reasoning* result is antimicrobial defense — and it is a distributed circuit, not a single feature. The defense→fungus probe recruits 12 features; ranked by **label-grounded AUROC** (not auto-interp labels), they split cleanly:

| tier (per-feature fungal AUROC) | features | meaning |
|---|---|---|
| **genuine fungal** ≥ 0.94 | F2808 (0.99), F35336 (0.99), F32785 (0.99), F23726 (0.98), F36488 (0.94) | individually predict fungal defense |
| weak / correlate 0.6–0.72 | F7665 cell-wall, F5047 motility, F22156 | marginal |
| co-occurring ~0.5–0.59 | F2082 chromatin, F15775 autophagy, F28215, F29332 | NOT fungal — the L1 probe picked them as co-predictors |

The genuine set has a sensible structure: **pathogen-specific detectors** (F23726 "filamentous fungi"; F22077 "bacterial LPS") sitting on a **shared innate-immune core** (F35336/F2808/F32785 = fungus ∩ bacterium). And these are *genuine reasoning*, not label echo: on the reasoning band F36488 fires on "redox gating of pattern-recognition receptors… potentiate defense circuits against fungal invasion… AP2/ERF control over pathogenesis-related promoters"; flip the *same* feature to the prompt band and it's just echoed `GO:` accessions.

Two honest caveats, both visible in the table: a sparse probe selects *predictive* features, not features that *mean* the concept (≈⅓ are co-occurring correlates), and **the protein band has no clean defense feature** — a combined protein+reasoning probe recruits 16 protein features but none enrich for any defense term. Defense is a reasoning-band phenomenon; function is weak in the residues, structure is strong.

---

## 6. Cross-modal features: prompt-mediated, not causal

Do protein features and reasoning features *link*? We measure it two ways, and they disagree in a way that is itself the answer.

- **Pairing (unsupervised):** correlate one bio feature's per-protein activation against every reasoning feature's → its single best partner (kinesin F18393 ↔ F15673, r = 0.62; GPCR F7369 ↔ F3184, r = 0.89).
- **Combined probe (supervised):** L1-logistic on `[protein ; reasoning]` features for a concept → a distributed set.

They **disagree** — the kinesin pairing partners are *not* recruited by the kinesin probe — because one measures co-firing and the other prediction. More importantly, a 2×2 causal test shows the link is **prompt-mediated, not a causal feature-to-feature flow**: clamping a reasoning feature writes its concept into the trace, but the paired protein feature is causally inert.

![cross-modal](../analysis/figures/fig2_crossmodal_combined.png)
> **Cross-modal structure.** Pairing correlations and the combined-probe recruitment; the causal 2×2 (fusion control) shows cross-modal is correlational/prompt-mediated, not fused.

---

## 7. Synthesis features and a grounded synthesis-quality scorer

Beyond named concepts, does the model do genuine *mechanistic synthesis* in its reasoning? We built a grounded scorer (`synth_span_scorer.py`, no LLM) that ranks reasoning features by: (0) a **frequency filter** dropping broadband "reasoning-mode" features (act-freq > 0.2); (1) firing on **long contiguous phrases** (not single tokens); (2) phrase quality = **mechanism verbs + inference/conclusion language + beyond-prompt named entities − echo/accession density**, scored per-window.

It *discovers* genuine mechanistic features automatically: **F12706** (DNA-TF motif recognition), **F15088** (mRNA translation control — poly(A)/eIF4E/CCR4-NOT), **F30993** (transporter alternating-access), **F5221** (RAS signaling — SOS1/GAP/RAF), **F36488** (plant-immunity defense synthesis).

![synthesis novelty](charts/synthesis_novelty.png)
> **Synthesis vs echo.** Mechanism/entity density separates genuine synthesis features from accession-echo features.

But the honest ceiling reappears: at the *representation* level, echo-vs-synthesis is a diffuse property of the raw residual stream — a probe reads it at 0.916 from raw vs 0.914 from the SAE. **The SAE does not localize "synthesis" into a clean feature;** it surfaces *individual* mechanistic features you can read, not a synthesis direction.

---

## 8. The auto-interp pipeline (and why we don't trust its labels alone)

Naming 40,960 features by hand is impossible, so we built an auto-interp pipeline: for a feature, take its top-firing windows in a chosen band, **strip cited `GO:`/`IPR` accessions** (so the LLM interprets the *reasoning*, not the model reading a label back), mark the **full active phrase** (span-aware, not a fixed window around the peak — critical for phrase-firing features like an RRM-fold feature that spans "RNA recognition motif … β-α-β topology"), and ask an LLM for the single concept.

**We treat these labels as hypotheses, not measurements** — and the pipeline's own failures show why. Auto-interp *missed* F23726 (label "amino acid transport") even though its fungal AUROC is 0.975 and 6/8 windows fire on *Aspergillus/Candida/S. pombe*; and it *oversold* F15775 (label "autophagy", sounds antimicrobial) whose fungal AUROC is only 0.572. The discipline: **rank by AUROC (score) → read the marked windows → use the auto-interp label only as a naming hint.**

![auto-interp walkthrough — F23726 fungal feature: top reasoning windows fire on named fungi; label-grounded AUROC 0.975 confirms fungal, while the auto-interp label was a miss](charts/echo_synthesis.png)
> *Illustrative auto-interp walkthrough (to be rendered as a dashboard snapshot): a feature's top reasoning windows with the peak phrase marked, its per-band firing, its GO+IPR enrichment, and its label-grounded AUROC — the four pieces we read together before trusting a feature.*

---

## 9. Modality balancing: the representational difference, feature quality, and layer trends

Balancing is where the multimodal story becomes concrete. Training a matched pair — unbalanced (~2% protein) vs balanced (~50/50) — at nine layers and counting **protein-selective** features (fires on >1% of protein tokens, <0.1% of text):

| Layer | protein-selective (unbal → bal) |
|------:|:-------------------------------:|
| L14 | 27 → **572** |
| L16 | 12 → **650** |
| L24 | 34 → **928** (peak) |
| L30 | 12 → **663** |
| L32 |  9 → **655** |

**A 20–50× increase at every layer** — unbalanced, protein gets a couple dozen dedicated features; balanced, it gets ~570–930. Balancing isn't starving text; it's carving out capacity the natural mix never gave protein (peak richness L22–L24).

![balancing selectivity grid](../analysis/figures/fig_layer_selectivity_grid.png)
> **Per-feature modality selectivity across layers.** Left = unbalanced, right = balanced. The dense stripe of protein-selective features up the left edge of every balanced panel is nearly absent when unbalanced.

**But balancing changes *which* features carry protein, not *where the tokens sit*.** In the raw residual stream — and in the SAE code, balanced or not — protein and text occupy separate regions at every layer:

![layer UMAP grid](../analysis/figures/fig_layer_umap_grid.png)
> **Token geometry across layers.** Rows = layers, columns = raw / unbalanced-SAE / balanced-SAE. Protein (blue) and text (purple) separate in every panel — the split is a property of the *encoder* that the SAE preserves, which is also why the raw cross-modal metric reads ~0.

**And here is the representational limit.** Even with 5.5× more protein features, protein *code spread* stays near zero (mean pairwise cosine ≈ 0.006 vs text ≈ 0.024) — the ESM3 collapse. Protein tokens are nearly collinear, so per-token protein codes are dominated by one shared component; the discriminative signal is real (enrichment still finds a protein-kinase feature at p = 3.6e-14) but swamped. **This is a limitation of the encoder, not the SAE**, and it's why our Matryoshka + modality-weighting variants never beat flat TopK (they starved protein further) — the fix is encoder alignment (Prot2Text-style), not a fancier SAE.

**How much to balance?** A sweep at layer 16 puts the sweet spot at **~50–60% protein**: protein-selective features plateau there, and past ~40% dead latents climb (≈46% dead at 70% — the low-diversity protein embeddings can't fill a protein-dominated dictionary).

![balance sweep](../analysis/figures/fig_balance_sweep.png)
> **Balance-split sweep (L16).** Protein-selective features peak at ~50–60% protein; dead latents rise monotonically past 40%. 50/50 is the sweet spot — the config behind our reported SAE.

---

## 10. An interactive multimodal feature atlas

Interpretability is exploratory, so we ship a browser-based atlas over the SAE features: per-band firing windows (protein / reasoning / prompt / answer), activation-frequency and metric histograms, sortable feature cards, a **domain-F1 localization badge**, GO+IPR enrichment, cross-modal pair links, and a **🔍 live auto-interp** button (span-aware, accession-stripped). It lets a researcher move from the global UMAP of decoder vectors, to a filtered feature subset, to the residue- or token-level evidence for a single feature — the same loop that produced every result above.

---

## 11. What we found, and what we didn't

**Found:** BioReason-Pro's SAE yields (i) genuinely *localized* structural features (kinesin domain-F1 0.98), with a localization metric that exposes the ~83% of high-AUROC features that don't localize; (ii) a distributed, reasoning-band **antimicrobial-defense circuit** — pathogen detectors on a shared innate-immune core — cleanly above a strict leak floor; (iii) individual **mechanistic-synthesis features** surfaced by a grounded scorer; and (iv) a balancing recipe that is *required* to give protein any features at all (20–50×).

**Did not find (and this is the point):** the SAE does **not** provide better decodability than raw — the mirror image of the codon result — because the ESM3 protein representation is collapsed and the reasoning-band signal is largely leakage. Cross-modal links are **prompt-mediated, not causal**. And "synthesis" is a diffuse residual property the SAE doesn't isolate.

The honest conclusion is a useful one for the field: **SAEs are not a guaranteed decodability win — whether they help is set by the encoder and the modality.** For a multimodal reasoning model, the SAE's contribution is *structure and interpretability* — localized concepts you can read and steer, and a clear diagnosis of where the ceiling is — with the next unlock being **encoder alignment**, not a bigger dictionary.

---

*Recipe, training config, auto-interp pipeline, synthesis scorer, and the feature atlas are open-sourced with this work. Sources: `RESULTS_WRITEUP.md`, `MICROBIAL_DEFENSE_RESULTS.md`, `DASHBOARD_REVIEW_CHECKLIST.md`, `PHASE3-LAYER-BALANCING.md`, `CROSSMODAL_PAIRS.md`.*
