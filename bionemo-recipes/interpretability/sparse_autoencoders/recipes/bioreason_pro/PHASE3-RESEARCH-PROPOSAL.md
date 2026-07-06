# Sparse Autoencoders for Biology–Language Multimodal Models

**A proposal — why this is novel and publishable (blog post → workshop paper → full publication)**

_Draft, 2026-07-06. Savitha S._

## Thesis

Sparse autoencoders (SAEs) are the standard tool for interpreting transformer internals, and in the last ~12 months they were extended to **vision–language** multimodal models (SAE-V; Zhang et al.; Pach et al.). **No published work applies SAEs to a biology–language fusion model** — where a protein/DNA foundation-model encoder is fused into an LLM. We are the first to do so, and along the way we are (1) establishing the methodology multimodal-bio SAEs require, (2) producing a concrete scientific finding about *how* (and whether) these models fuse modalities at the feature level, and (3) proposing novel SAE architectures for the multimodal setting.

## 1. The gap

SAEs decompose transformer activations into sparse, monosemantic features and are now the default interpretability substrate for LLMs. The multimodal extension is recent and entirely **vision–language**:

- **SAE-V** — Lou et al., 2025 (ICML), [arXiv:2502.17514](https://arxiv.org/pdf/2502.17514): SAEs over LLaVA/Chameleon activations + a cross-modal cosine score for data filtering.
- Pach et al., 2025, [arXiv:2504.02821](https://arxiv.org/pdf/2504.02821); Zhang et al., 2024, [arXiv:2411.14982](https://arxiv.org/pdf/2411.14982).
- Crosscoders for cross-layer/model feature comparison — Lindsey et al., 2024.
- **Orlov et al., 2026** (bio-SAE field review) names *"scaling SAE analysis to multimodal architectures"* as a **top-4 field priority** — but frames it as vision, and **does not consider biology–language fusion at all.**

**No prior work has applied SAEs to a biology-LLM fusion model.** That is the gap we fill.

## 2. Why now: biology–LLM fusion is the obvious next domain

A wave of models now fuse a biological foundation model into an LLM, and *how* they integrate the two modalities is unknown:

- **BioReason** — Evo 2 (DNA) → projector → Qwen3-4B.
- **BioReason-Pro** — ESM3 (protein) + GO-graph encoder → Qwen3-4B.
- **Peter's vocab-integration model** — a DNA tokenizer + MoE on Nemotron-3-Super.

The SAE methodology transfers directly from vision-language MLLMs (hook the LLM residual stream, train an SAE, interpret features). The open questions are biology-specific and important:

- **Which DNA/protein features carry the task signal?**
- **How does the LLM align biological content with text concepts — or does it?**
- **How do different fusion architectures (projector vs. token-integration vs. MoE) compare on common interpretability ground?**

## 3. What we've established so far (BioReason-Pro, ESM3 + Qwen, layer 16)

**(a) Multimodal SAEs need modality-aware training.** Two findings that don't arise in single-modality SAEs:

- **Normalization is essential** because the modalities occupy wildly different magnitude regimes — protein-token residuals have ~**50× the norm of text tokens** (median ‖·‖ ≈ 2287 vs ≈ 45). Without per-token input normalization, the reconstruction loss (and the learned dictionary) is dominated by the loud bio tokens.
- **Modality loss-balancing dramatically improves feature quality for both modalities.** The protein tokens are a small fraction of the stream, so an unbalanced SAE starves them: at layer 16 we go from **98 protein-heavy features (unbalanced) → 1,840 (balanced)** — a **~19× increase** — while text-heavy features drop from 20,476 → 13,739, i.e. **far cleaner modality separation** and higher-quality reasoning features rather than a text-dominated dictionary. (See §5 for the architecture experiments this motivated.)

**(b) An interpretation pipeline for multimodal-bio SAE features.** We built a live auto-interpretation system that goes well beyond generic LLM labeling:

- **Per-band labeling** — each feature is characterized separately on its protein, DNA, reasoning, question, and answer token bands, with a cross-band "shared-concept" synthesis to flag genuinely cross-band features.
- **Reliability-first, enrichment-grounded protein labels.** Raw amino-acid windows are uninterpretable by an LLM ("fires on leucine"). Instead, for the proteins a feature fires on we run a **Fisher over-representation test** against GO annotations and UniProt keywords/domains — a *statistical* label with a p-value, no hallucination. Examples: **F9422 = mitochondrial transit-peptide / targeting-sequence detector** (UniProt *Transit peptide* p=4.9e-9; fires on N-terminal Arg/Leu-rich mitochondrial presequences — verified by inspecting the proteins and token windows, which the top-3-GO view alone had mislabeled as "ribosomal"); **F33529 = protein-kinase detector** (UniProt *Protein kinase* domain, p=3.6e-14).
- **Sample-level specialization for text features** — for a reasoning feature we enrich the GO terms of the *proteins of the samples it fires on*, revealing whether the reasoning specializes by protein function.

**(c) Promising reasoning features are already emerging** in the layer-16 analysis — clean, specific concepts in the model's *reasoning*:

- **F7856 — "embryonic development"** (shared across reasoning/answer; its samples' proteins are enriched for developmental GO terms, p=2.2e-22).
- **F6217 — "establishment of localization / transport"** (its samples' proteins enriched for *establishment of localization*, p=2.4e-17).

**(d) A concrete scientific finding: fusion is attention-mediated, not feature-level — and cross-modal SAE metrics require an aligned representation.** We looked hard for features that bind a biological concept to its text: **we found none.** At full scale (117k proteins; and a properly-trained balanced-DNA control), *zero* features selectively co-fire on both modalities of the same sample beyond structural boundary markers. Investigating *why* yielded the key insight:

- **SAE-V's cross-modal cosine operates on raw hidden states** (per its Algorithm 1, the cosine compares the *raw* activations of a feature's top text vs. top vision tokens; the SAE only selects/ranks tokens). It therefore inherits the raw-subspace geometry.
- In BioReason-Pro, **protein and text residuals are near-orthogonal** (mean cross-modal cosine ≈ 0; DNA is 0.37), so the SAE-V metric reads ≈0 *by construction* — it cannot see fusion even if it exists.
- **This is a property of the encoder, not the method.** SAE-V worked on LLaVA/Chameleon because those vision encoders are *text-aligned* (CLIP is image–text contrastive; Chameleon shares a token space). ESM3/Evo2 are **not** text-aligned — we project them into the LLM's space, but *same dimensionality ≠ same directions*.

This reframes the "null" into a **precondition finding**: *cross-modal SAE features require a directionally-aligned representation.* It is directly testable (see §6), and it is a genuinely new statement about multimodal-bio interpretability.

## 4. Contributions

1. **First SAE analysis of a biology–LLM fusion model** (methodology + open-source recipe: extract → train → interpret).
2. **A modality-aware training recipe** (normalization + loss-balancing) and an **enrichment-grounded interpretation pipeline** (GO/UniProt Fisher tests + per-band + sample-level specialization) that make bio-multimodal SAE features actually interpretable.
3. **A scientific result:** fusion in current projector-based bio-LLMs is attention-mediated, not feature-level, and cosine-based cross-modal metrics require encoder–text alignment — with a concrete experiment to test it.
4. **Novel SAE architectures for the multimodal setting** (§5).

## 5. Novel architectures — modality-aware SAEs

Our balanced-loss and Matryoshka experiments are the **tip of the iceberg** of modality-aware SAE design:

- **Loss-level modality awareness (done).** We tried per-prefix Matryoshka SAEs and per-modality-weighted losses. Finding: balancing helps utilization and separation, but **Matryoshka/modality-weighting did not improve feature quality over a plain balanced BatchTopK** — reducing dead latents is not the same as producing interpretable features. Useful negative result; motivates architectural (not just loss) modality-awareness.
- **Architecture-level modality awareness (proposed): an MoE-style SAE with a dedicated head per modality.** A single shared dictionary is a poor fit when modalities have different magnitude regimes and feature vocabularies (it invites magnitude domination and minority-modality starvation — both of which we measured). Instead, **route each token to a modality-specific expert** (its own encoder, dictionary, and decoder), by known band (protein/DNA vs. text) or a learned router, with **per-expert input normalization**. This structurally removes starvation and magnitude domination. Crucially, add a **shared "fusion" expert** trained with an explicit cross-modal objective (or on modality-boundary tokens) so that *cross-modal features become a first-class, measurable component* rather than something we hunt for in an entangled shared basis. This turns "do these models fuse at the feature level?" into a directly quantifiable architectural question — and is a novel SAE contribution in its own right.

## 6. Next steps

- **Test the alignment hypothesis (highest-value).** Run our pipeline on **Prot2Text-V2** ([`xiao-fei/Prot2Text-V2-11B-Instruct-hf`](https://huggingface.co/xiao-fei/Prot2Text-V2-11B-Instruct-hf); ESM2-3B → adapter → LLaMA-3.1-8B with **H-SCALE contrastive alignment** — arch-matched to BioReason but *aligned*). If cross-modal features appear here but not in BioReason, alignment is the prerequisite. Recipe ≈ a thin extractor + our existing pipeline.
- **Baselines / validation.** Replicate **SAE-V on LLaVA-Next** (confirm our cross-modal metric reproduces their positive result — rules out a pipeline bug) and analyze **BioReason-DNA** (does DNA fuse differently than protein?).
- **Causal confirmation.** Activation patching across the modality boundary to directly test attention-mediated fusion.
- **Architecture.** Prototype the **MoE-style modality-head SAE** (§5).
- **Scale interpretation.** Full protein-feature enrichment catalog; structural grounding via AlphaFold-DB/OpenFold (map a feature's firing residues onto the fold — e.g. do the kinase feature's residues line the catalytic pocket?).

## 7. Publication path

- **Blog post (now-ish):** "SAEs for biology–language models" — the recipe, the reasoning features, and the alignment finding. Immediate, establishes priority.
- **Workshop paper (near-term):** the methodology + the alignment-is-a-precondition result + the aligned-vs-unaligned comparison (Prot2Text-V2). A tight, novel contribution.
- **Full publication (if the aligned model shows fusion features and/or the MoE-SAE works):** the first systematic interpretability study of biology–language fusion, with a new modality-aware SAE architecture.

Applying SAEs to multimodal biology models is itself novel; combined with a concrete mechanistic finding and a new architecture, the work supports at least a workshop paper, plausibly a full publication.
