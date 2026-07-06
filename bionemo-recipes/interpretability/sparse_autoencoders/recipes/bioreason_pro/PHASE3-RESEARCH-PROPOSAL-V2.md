# Sparse Autoencoders for Biology–Language Models

*Savitha S. — July 2026*

## Motivation

Extending SAEs to multimodal models is a nascent field: most notable work in this space has been done on vision–language models. To our knowledge, there is no prior work on applying SAEs to biology–language fusion models, i.e., BioReason-style models where a protein or DNA foundation model that has never seen text is fused into an LLM. Indeed, the Orlov et al. bio-SAE systematic review (bioRxiv, March 2026) names "scaling SAE analysis to multimodal architectures" as a top-4 field priority within digital biology, signaling that this is a high-impact area to research and publish in. The two nearest precedents are both vision-language: SAE-V (Lou et al., ICML 2025) on general vision-language MLLMs, whose cross-modal feature-weighting metric we lift directly; and a Matryoshka-SAE study of the radiology model MAIRA-2 (Bouzid, Bannur et al., ICML 2025), which pulled clinically meaningful concepts — pleural effusion, cardiomegaly, medical devices — out of a domain-specialised MLLM.

This gap is worth closing for a few reasons. First, these models raise a question no one has answered: when we fuse a biology encoder into an LLM (BioReason = Evo 2 + Qwen; BioReason-Pro = ESM3 + Qwen; Peter's DNA-tokenizer MoE on Nemotron), how deeply do the two modalities actually integrate — does the model build features that genuinely mix biology and language, or does it keep them in separate representations and combine them only through attention? SAEs are the most direct way to determine this. Second, this could also offer our team insight into how multimodal fusion models reason as compared to traditional tool-calling methods. Finally, the recipe and learnings from applying an SAE to BioReason-Pro will be directly applicable to the multimodal models being developed on our team by Peter and John.

## Initial learnings and results

I built the full pipeline on BioReason-Pro (ESM3 + Qwen) — activation extraction, SAE training, and feature interpretation. Some key learnings from my work so far:

- **Balancing the modalities is what makes a bio-LLM SAE work.** Protein tokens are ~50× larger in norm than text and are a small slice of the data, so a naively trained SAE mostly ignores them. Rebalancing the training mix to ~50/50 protein:text raises the number of protein features from ~100 to ~1,800 (Figure 1a); at the individual-feature level, the protein-selective region of the dictionary goes from nearly empty to populated (Figure 1b). This holds at every one of the nine layers I tested — a 20–50× increase in protein-selective features. I'm currently testing other balancing ratios.

![Figure 1](analysis/figures/fig1_balancing_combined.png)
> **Figure 1.** The balancing effect, two views. **(a)** Layer-16 feature counts by modality: protein-selective features rise ~100 → ~1,800 with modality balancing. **(b)** The same effect per feature — x = fraction of text tokens a feature fires on, y = fraction of protein tokens; protein-selective features live in the top-left, and balancing fills that corner (~15 → ~660).

- **The SAE surfaces interpretable reasoning features.** I built an interpretation pipeline that first splits each example into bands — prompt, question, reasoning, and answer text (and, for the DNA model, reference vs variant sequence) — so a feature is labeled by *where* it fires and reasoning features are separated from prompt/answer boilerplate. Labeling is domain-aware (distinct protein vs DNA concept taxonomies) and flags purely syntactic or polysemantic features. This surfaces clean concepts on the text side — a "cell junction" feature, an "embryonic development" feature, a "transport / localization" feature. For **protein** features I ground the labels in biology rather than tokens: a Fisher exact over-representation test on GO terms and UniProt keywords/domains, run over the proteins a feature fires on against a background set, which recovers real signal (e.g. a protein-kinase feature at p = 3.6e-14). The protein side is still the weak point — many features fire too often to trust, and the labeler keys on a feature's single strongest token — so I treat protein labels as candidates to verify, and hardening this is the main remaining work.

- **So far, I haven't found any cross-modal (fusion) features — the most interesting result.** Searching for features that tie a biological concept to its text turns up only structural markers, in protein or DNA, balanced or not. As for *why*: the cross-modal metric (from SAE-V) compares the model's raw activations, and in BioReason-Pro the protein and text activations point in almost entirely separate directions — the SAE inherits that separation regardless of balancing (Figure 2), so the metric reads near zero by construction (Figure 3; the DNA model overlaps somewhat more). One hypothesis — to test, not a conclusion — is that this is a property of the encoder: the vision–language models where fusion shows up use encoders trained against text (CLIP), whereas ESM3 and Evo 2 never did. Whether alignment is really the deciding factor is exactly what I want to test next.

![Figure 2](analysis/figures/fig6_raw_vs_sae_umap.png)
> **Figure 2.** Same tokens, raw vs SAE (UMAP). The raw residual (a) already separates the modalities; the SAE codes inherit it whether the loss is unbalanced (b) or balanced (c). Balancing changes *which* features carry protein, not the token-level geometry.

![Figure 3](analysis/figures/fig3_geometry.png)
> **Figure 3.** Layer-16 activations. Protein (ESM3) and text form near-separate clusters (left); DNA (Evo 2) and text overlap more (right) — which is why a raw-activation cross-modal metric reads near zero for an encoder never aligned to text.

## Next steps

- **Test the alignment hypothesis directly:** run the same pipeline on **Prot2Text-V2**, which has BioReason's shape (protein encoder → adapter → LLM) but was explicitly trained to align protein and text. Cross-modal features appearing there but not in BioReason would support it. Baselines alongside it: reproduce SAE-V on LLaVA-Next to confirm the metric behaves, and run the DNA model.
- **(Bonus) a modality-aware SAE architecture** — a mixture-of-experts SAE with an expert per modality plus a shared "fusion" expert, so each modality gets its own capacity and cross-modal fusion becomes something we *measure* (does the fusion expert fire?) rather than hunt for.

## Deliverable

The first two results already make a blog post or short workshop paper — the first SAE interpretability study of a biology–language fusion model (precedents for the format and community interest: Goodfire's [*Under the Hood of a Reasoning Model*](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model) and the MAIRA-2 radiology study). If the alignment test or the MoE-SAE architecture pans out, it grows into a full paper. Either way the recipe and methods are model-agnostic, so they transfer to our team's own multimodal models regardless of how the science lands. Natural venues are mechanistic-interpretability and bio-ML workshops.
