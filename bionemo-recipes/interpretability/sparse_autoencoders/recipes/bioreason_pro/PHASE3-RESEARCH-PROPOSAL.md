# Sparse Autoencoders for Biology–Language Models

*Savitha S. — July 2026*

## Goal

Bring SAE interpretability to **multimodal biology–language models** — LLMs with a protein or DNA foundation model fused in — starting with BioReason. The concrete deliverable is a short workshop paper introducing interpretability methods for these models, plus a working recipe and a set of methods I can hand to our own multimodal-model efforts. The pipeline (extract activations → train SAE → interpret features) already runs end to end on BioReason-Pro, so the starting point is real; the end point is the paper and the transfer to our models.

## Why it's novel, and why we care

Sparse autoencoders are the standard way to pull interpretable features out of a transformer. We haven't found any prior work applying them to **biology–language fusion models**. The relevant multimodal-SAE literature is thin: the Orlov et al. bio-SAE systematic review (bioRxiv, March 2026) names "scaling SAE analysis to multimodal architectures" as a top-4 field priority and doesn't even consider biology + natural-language fusion, and the closest published methodological precedent is SAE-V (Lou et al., ICML 2025) on vision–language MLLMs — we lift their cross-modal feature-weighting metric directly.

There's also a reason the business should care. When we fuse a biology model into an LLM (BioReason = Evo 2 + Qwen; BioReason-Pro = ESM3 + Qwen; Peter's DNA-tokenizer MoE on Nemotron), is the model really *combining* biology and language into something new, or doing a dressed-up version of tool-calling — reading the encoder's output like a lookup and largely ignoring the rest? That bears directly on whether multimodal bio models are worth the investment over a cheaper pipeline that just calls a bio tool, and SAEs give us a direct way to check.

## What I've found so far

I built the full pipeline on BioReason-Pro (ESM3 + Qwen) — activation extraction, SAE training, and feature interpretation. Three findings stand out.

**Balancing the modalities is the key training lever.** Protein tokens are ~50× larger in norm than text and are a small slice of the data, so a naively trained SAE mostly ignores them. Rebalancing the mix to ~50/50 protein:text raises the number of protein features from ~100 to ~1,800 (Figure 1); at the individual-feature level, the protein-selective region of the dictionary goes from nearly empty to populated (Figure 2). The SAE gains a dedicated protein vocabulary instead of forcing protein onto text-owned features.

![Figure 1](analysis/figures/fig1_balancing.png)
> **Figure 1.** Modality loss-balancing at layer 16: protein-selective features rise ~100 → ~1,800, and the modalities separate cleanly rather than the dictionary being dominated by text.

![Figure 2](analysis/figures/fig5_feature_selectivity.png)
> **Figure 2.** The same effect per feature: x = fraction of text tokens a feature fires on, y = fraction of protein tokens. Protein-selective features live in the top-left. The unbalanced loss leaves that corner nearly empty (~15 features); balancing fills it (~660).

**The model does learn real, interpretable reasoning features.** My interpretation pipeline labels a feature by the proteins it fires on and what they have in common, and it surfaces clean concepts from the model's reasoning — a "cell junction" feature, an "embryonic development" feature, a "transport / localization" feature — that line up with the proteins the concept applies to. The protein-side labels are still preliminary (many features are weak or fire too often to trust), so I treat them as candidates to verify.

**The headline result: fusion doesn't appear to happen at the feature level.** Looking for features that tie a biological concept to its text, I find only structural markers. The standard cross-modal metric (from SAE-V) compares the model's raw activations, and in BioReason-Pro the protein and text activations point in almost entirely separate directions — the SAE inherits that separation regardless of balancing (Figure 3), so the metric reads near zero by construction (Figure 4; the DNA model overlaps somewhat more). This looks like a property of the encoder: the vision–language models where fusion showed up use encoders trained against text (CLIP), whereas ESM3 and Evo 2 have never seen text. My working hypothesis is that cross-modal features only emerge once the two modalities are actually aligned — which is directly testable.

![Figure 3](analysis/figures/fig6_raw_vs_sae_umap.png)
> **Figure 3.** Same tokens, raw vs SAE (UMAP). The raw residual (a) already separates the modalities; the SAE codes inherit it whether the loss is unbalanced (b) or balanced (c). Balancing changes *which* features carry protein, not the token-level geometry.

![Figure 4](analysis/figures/fig3_geometry.png)
> **Figure 4.** Layer-16 activations. Protein (ESM3) and text form near-separate clusters (left); DNA (Evo 2) and text overlap more (right) — which is why a raw-activation cross-modal metric reads near zero for an encoder never aligned to text.

## The methods I want to develop

This is where the novelty is — interpretability methods built for the multimodal case, not borrowed wholesale from the single-modality setting.

- **Test the alignment hypothesis directly.** Run the same pipeline on **Prot2Text-V2**, which has BioReason's shape (protein encoder → adapter → LLM) but was explicitly trained to align protein and text. Cross-modal features appearing there but not in BioReason would be strong support. Baselines alongside it: reproduce SAE-V on LLaVA-Next to confirm the metric behaves, and run the DNA model.
- **A modality-aware SAE architecture.** My balancing and Matryoshka experiments were a first pass, and they point to a structural fix rather than another loss tweak. I want to try a **mixture-of-experts SAE** — an expert per modality plus a shared "fusion" expert — so each modality gets its own capacity and cross-modal fusion becomes something we *measure* (does the fusion expert fire?) instead of hunt for.
- **Complementary analyses:** how reasoning features evolve across layers, how SAE features compare to what attention is doing, and grounding protein features in structure via AlphaFold / OpenFold.

## Applying it to our models

The recipe and the methods are model-agnostic, so they carry straight to our own multimodal models — Peter's DNA-tokenizer MoE on Nemotron and whatever fusion models the team builds next. Whatever I learn about *whether and where* fusion happens, and about how to give each modality real capacity in the SAE, transfers directly. This is work I'm driving end to end across teams.

## Risks

- The alignment hypothesis may not hold — cross-modal features might fail to appear even in Prot2Text-V2. That would still be a clean, publishable result about *when* feature-level fusion is possible.
- The MoE-SAE may not beat flat balancing (Matryoshka didn't). Then the contribution is the recipe, the balancing method, and the negative architectural result.
- Protein-feature interpretation is currently weak and needs the labeler and verification hardened.

The downside is protected: even if the fusion story turns out negative, the working recipe, the "bio encoders aren't text-aligned" finding, and the transfer to our own models all stand on their own.

## Deliverable and venues

A short **workshop paper** first — a natural fit for mechanistic-interpretability workshops, bio-ML workshops, and broader ML venues — growing into a full paper if the alignment result holds and the architecture pans out.
