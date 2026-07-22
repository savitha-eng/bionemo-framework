# Sparse Autoencoders on a Multimodal Protein-Reasoning Model

*Savitha S. — July 2026. Companion mini-paper to the BioReason-Pro SAE recipe.*

## Abstract

BioReason-Pro is a multimodal model that fuses ESM3 protein embeddings, a GO-graph encoder, and a Qwen3-4B reasoning LLM to predict protein function through a written reasoning trace. We train sparse autoencoders (SAEs) on its residual stream and pull the fused representation apart into interpretable features. We find (i) **reasoning-band features that decode genuine biological concepts** above a strict leakage floor — most cleanly a distributed *antimicrobial-defense circuit* with pathogen-specific detectors on a shared innate-immune core; (ii) **protein-domain features that localize**, measured with a domain-F1 metric that separates genuinely localized detectors (a kinesin-motor feature, domain-F1 = 0.98) from high-AUROC features that fire *outside* their domain; and (iii) a **cross-modal structure** linking protein features to reasoning features that is correlational and prompt-mediated, not a causal fusion. Getting there required solving a multimodal-training problem — a single dictionary is monopolized by the majority (text) modality, so **loss balancing is mandatory** (it yields 20–50× more protein-selective features). We ship an auto-interp pipeline, a grounded synthesis-quality scorer, and an interactive feature atlas. A side observation: the SAE does not out-decode the raw residual stream on dense probes — the ESM3 encoder is the ceiling — which is why the SAE's value here is *interpretable structure*, not raw decodability.

---

## 1. Introduction

Most SAE interpretability work targets a single modality — a protein language model, a codon model, a genome model. BioReason-Pro is different: it *reasons* about a protein in text. ESM3 residue embeddings and a GO-graph encoder are projected into the token stream of a Qwen3-4B LLM, which writes a free-text reasoning trace ("…the N-terminal P-loop NTPase fold positions duplex DNA…") and predicts GO function. A single residual stream therefore carries **three modalities at once** — protein-residue tokens, reasoning-text tokens, and `<go>` graph slots — and the interpretability question becomes: *what does each modality's features encode, and do they connect?*

Our contributions:

1. **Reasoning-band concept probes** with a leakage-aware evaluation, surfacing a distributed antimicrobial-defense circuit (and showing which "concepts" are real vs leakage).
2. **Protein-domain features + a localization metric (domain-F1)** that separates genuine domain detectors from AUROC artifacts.
3. **A cross-modal exploration** aligning protein features with reasoning features, and a causal test showing the link is prompt-mediated, not fused.
4. **A multimodal-SAE training recipe** whose central lesson is loss balancing — without it, the minority (protein) modality gets almost no features.
5. **An auto-interp pipeline, a synthesis-quality scorer, and an interactive feature atlas.**

---

## 2. Background & methods

### 2.1 How BioReason-Pro works

BioReason-Pro predicts protein function not as a classifier head but as a **reasoning task**. For a protein it (a) encodes the sequence with **ESM3** into per-residue embeddings, (b) encodes candidate GO structure with a **GO-graph encoder** into a fixed set of `<go>` memory slots, (c) projects both into the token embedding space of a **Qwen3-4B** LLM, and (d) has the LLM generate a natural-language reasoning trace that dissects the protein's domains and mechanisms before emitting GO terms. The consequence for us: the layer-30 residual stream we read is a **fused, multimodal** representation — one token is a protein residue, the next is a `<go>` slot, the next is a reasoning word — and any feature we learn lives in one (or across) those bands. We tag every activation row with its band (`protein` / `text` / `go`) in a row-aligned sidecar so downstream analyses can select a modality.

### 2.2 Training a multimodal SAE: loss balancing is the whole game

The first thing we learned is that **you cannot train a multimodal SAE naïvely.** In the natural token mix, protein residues are ~2% of the stream and reasoning text dominates. Left alone, text tokens monopolize the dictionary: they activate essentially the whole live dictionary (~15,800 features at layer 16) while protein collapses into ~230. The SAE "works" — good reconstruction, low dead-latent % — but it has almost no protein vocabulary.

The fix is **modality balancing at load time** (`--balance-modality`): we drop the `<go>` slots (an ablation shows they are model-unused — a fixed ontology reduction, not per-protein terms), and **downsample text to ~50/50 with protein** (text-keep ≈ 0.19). This one change is dramatic — measured as *protein-selective* features (fires on >1% of protein tokens, <0.1% of text):

| Layer | protein-selective (unbalanced → balanced) |
|------:|:-----------------------------------------:|
| L16 | 12 → **650** |
| L24 | 34 → **928** (peak) |
| L30 | 12 → **663** |
| L32 |  9 → **655** |

A **20–50× increase at every layer** — balancing carves out capacity the natural mix never gave protein (richest around L22–L24), without starving text.

![balancing selectivity grid](../analysis/figures/fig_layer_selectivity_grid.png)
> **Per-feature modality selectivity across layers.** Left = unbalanced, right = balanced. The dense stripe of protein-selective features up the left edge of every balanced panel is nearly absent when unbalanced.

Two more multimodal-training lessons:

- **Batch-aggregated loss, not per-token** (`--aggregate-loss` + `--normalize-loss`). In a multimodal stream tokens sit at very different distances from the shared center and are reconstructed to very different degrees; a per-token loss ratio starves the minority modality's features (FVU term) and mis-aims dead-latent revival toward already-well-reconstructed tokens (AuxK term). Batch aggregation fixes both. (Single-modality recipes like evo2 can skip it; multimodal cannot.)
- **How much to balance?** A sweep puts the sweet spot at **~50–60% protein** — past ~40% dead latents climb (≈46% dead at 70%), because the low-diversity ESM3 protein embeddings can't fill a protein-dominated dictionary.

![balance sweep](../analysis/figures/fig_balance_sweep.png)
> **Balance-split sweep (L16).** Protein-selective features peak at ~50–60% protein; dead latents rise past 40%. 50/50 is the reported config.

### 2.3 SAE architecture and evaluation discipline

For each `x ∈ ℝ^2560` we train a TopK SAE to a sparse code `z ∈ ℝ^40960` (expansion 16, **k = 128**) with reconstruction + auxiliary-k revival loss. Our reported model is `sae-l30-exp16-balanced`.

Two disciplines govern every number below:

- **The leak floor.** The reasoning text often *states* the function, so a random-initialized SAE already "decodes" it. We train a probe on a random SAE as a **leak floor** and trust only the margin `sae_sparse − random`.
- **Scoring vs labeling.** We separate **scoring** (label-grounded per-feature AUROC — trustworthy) from **labeling** (an LLM auto-interp read of a feature's top windows — a hypothesis). Our auto-interp pipeline takes a feature's top-firing windows in a chosen band, **strips cited `GO:`/`IPR` accessions** (so the LLM interprets reasoning, not label echo), marks the **full active phrase** (span-aware), and asks for one concept. We treat the label as a hint and always confirm by AUROC + reading the windows — because the pipeline both *misses* (F23726, fungal AUROC 0.975, mislabeled "amino acid transport") and *oversells* (F15775 "autophagy", fungal AUROC only 0.572).

---

## 3. Results

### 3.1 Reasoning-band probes: a distributed antimicrobial-defense circuit

The cleanest reasoning result is antimicrobial defense. Probing 12 designed GO concepts against the leak floor, **only 4 clear a margin ≥ 0.05** — defense→fungus (+0.111), defense→bacterium (+0.108), structural-molecule (+0.063), plasma-membrane (+0.055); the other 8 sit at the floor (the text just names them). So most "decodable" concepts are leakage — and the defense concepts are the real signal.

Defense is a *distributed circuit*, not one feature. Ranking the defense→fungus features by **label-grounded AUROC**:

| tier (per-feature fungal AUROC) | features | meaning |
|---|---|---|
| **genuine fungal** ≥ 0.94 | F2808 (0.99), F35336 (0.99), F32785 (0.99), F23726 (0.98), F36488 (0.94) | individually predict fungal defense |
| weak / correlate 0.6–0.72 | F7665 cell-wall, F5047 motility, F22156 | marginal |
| co-occurring ~0.5–0.59 | F2082 chromatin, F15775 autophagy, F28215, F29332 | NOT fungal — the probe picked them as co-predictors |

The genuine set has a biological shape: **pathogen-specific detectors** (F23726 "filamentous fungi"; F22077 "bacterial LPS") on a **shared innate-immune core** (F35336/F2808/F32785 = fungus ∩ bacterium). And it is *genuine reasoning*, not echo: on the reasoning band F36488 fires on "redox gating of pattern-recognition receptors… potentiate defense circuits against fungal invasion… AP2/ERF control over pathogenesis-related promoters"; flip the same feature to the prompt band and it's just echoed `GO:` accessions.

![enrichment](charts/enrichment.png)
> **Per-feature GO+InterPro enrichment.** Best over-represented term per feature with hypergeometric FDR and rank-sum AUROC — the label-grounded scoring behind the tiers above.

Honest caveat: a sparse probe selects *predictive* features, not features that *mean* the concept (≈⅓ are co-occurring correlates), and the **protein band has no clean defense feature** — defense is a reasoning-band phenomenon.

### 3.2 Protein-domain features: Fisher enrichment + a localization metric

For protein-band features we compute the most over-represented **GO or InterPro** term among the proteins where each feature fires (hypergeometric + per-feature AUROC), surfacing clean structural detectors — a GPCR-rhodopsin feature (IPR000276, FDR 3e-84), a kinesin-motor feature, and more.

But **AUROC is not localization** — a feature can score 0.98 for a domain and fire *outside* it. So we add **domain-F1** = (per-position precision: of firing residues, fraction inside the domain) × (per-region recall: of domain regions, fraction with ≥1 firing residue):

| feature | concept | AUROC | **domain-F1** | verdict |
|---|---|---|---|---|
| **F18393** | kinesin motor | 0.98 | **0.98** | 📍 genuinely localized |
| F7369 | GPCR (IPR000276) | high | 0.51 | ⚠️ half in-domain |
| **F4647** | (AUROC oversell) | **0.98** | **0.00** | ❌ fires *outside* the domain |

Across all AUROC-passing structural features, **only ~17% (59 of 872) genuinely localize.** For structural interpretability, trust the localization metric, not the ranking metric.

![per-feature structural](charts/per_feature_structural.png)
> **Structural features by localization.** The localized minority vs the AUROC-oversell majority.

### 3.3 Cross-modal exploration: aligning protein and reasoning features

Do a protein's domain features connect to what the model *says* about it? We aligned the two bands two ways:

- **Pairing (unsupervised):** correlate one protein feature's per-protein activation vector against every reasoning feature's → its single best partner. Kinesin F18393 ↔ F15673 (r = 0.62); GPCR F7369 ↔ F3184 (r = 0.89).
- **Combined probe (supervised):** an L1-logistic on `[protein ; reasoning]` features for a concept → a distributed cross-band set.

The two methods **disagree** — the pairing's kinesin partners are *not* recruited by the kinesin probe — because one measures co-firing and the other prediction. And a **2×2 causal test settles the interpretation**: clamping a reasoning feature writes its concept into the trace, but the paired protein feature is causally inert. So the cross-modal link is **correlational and prompt-mediated — the model "talks about motors" when the motor detector fires — not a causal protein→reasoning feature flow.** (Consistently, protein and text separate in the raw residual stream at every layer, so there's no fused sub-space for the SAE to find.)

![cross-modal](../analysis/figures/fig2_crossmodal_combined.png)
> **Cross-modal structure.** Pairing correlations and combined-probe recruitment; the fusion-control 2×2 shows the link is prompt-mediated, not fused.

### 3.4 Synthesis features: reading genuine mechanism

Beyond named concepts, a grounded scorer (`synth_span_scorer.py`, no LLM) ranks reasoning features by good-synthesis quality — frequency-filtered (drop broadband "reasoning-mode" features), firing on long contiguous phrases, scored for mechanism verbs + inference language + beyond-prompt named entities − echo. It *discovers* mechanistic features automatically: **F12706** (DNA-TF motif recognition), **F15088** (mRNA translation control — poly(A)/eIF4E/CCR4-NOT), **F30993** (transporter alternating-access), **F5221** (RAS signaling — SOS1/RAF).

![synthesis novelty](charts/synthesis_novelty.png)
> **Synthesis vs echo.** Mechanism/entity density separates genuine synthesis features from accession-echo.

---

## 4. A note on decodability (why the SAE's value is structure, not raw signal)

A natural question is whether the SAE *out-decodes* the raw residual stream, as it does in codon models. Here, mostly it does not. On dense per-protein probes, SAE, raw, and even a random SAE saturate together (InterPro domains: 0.990 / 0.989 / 0.981), and the sparse SAE beats raw only for the few reasoning concepts that clear the leak floor. The reason is the **encoder, not the SAE**: ESM3 protein token embeddings are nearly collinear (self-cosine ≈ 0.99), so per-token protein signal is real but swamped — even balanced, protein code spread stays ≈ 0.006 vs text ≈ 0.024. This is why our Matryoshka + modality-weighting variants never beat flat TopK (they starved protein further), and why the next unlock is **encoder alignment** (Prot2Text-style), not a bigger dictionary. The SAE's contribution here is interpretable, localizable, steerable *structure* — not decodability.

![layer decodability](charts/layer_decodability.png)
> **Decodability across layers.** SAE, raw, and random-SAE probes track each other — the SAE doesn't open a gap over raw (the mirror image of the codon-model result).

---

## 5. An interactive multimodal feature atlas

Interpretability is exploratory, so we ship a browser-based atlas: per-band firing windows (protein / reasoning / prompt / answer), activation-frequency and metric histograms, sortable feature cards, a **domain-F1 localization badge**, GO+IPR enrichment, cross-modal pair links, and a **🔍 live auto-interp** button (span-aware, accession-stripped). It closes the loop that produced every result above — from the global UMAP of decoder vectors, to a filtered feature subset, to the residue- or token-level evidence for a single feature.

---

## 6. Conclusion

Applying SAEs to a multimodal protein-reasoning model, we recover **interpretable structure across all three of its faces**: reasoning-band concepts that pass a strict leakage test (a distributed antimicrobial-defense circuit), protein-domain features that genuinely *localize* (domain-F1), and a cross-modal alignment between them that we show is prompt-mediated rather than causal. The enabling method is loss balancing — the multimodal-training lesson that a single dictionary must be forced to give the minority modality room. The honest limitation is that the SAE does not out-decode the raw stream, because the protein encoder is collapsed; the payoff is not decodability but a set of *readable, localizable, steerable* features and a clear diagnosis of where the ceiling is. All training artifacts, the auto-interp pipeline, the synthesis scorer, and the feature atlas are open-sourced.

---

*Sources: `RESULTS_WRITEUP.md`, `MICROBIAL_DEFENSE_RESULTS.md`, `DASHBOARD_REVIEW_CHECKLIST.md`, `PHASE3-LAYER-BALANCING.md`, `CROSSMODAL_PAIRS.md`.*
