# Sparse Autoencoders for Biology–Language Models

*Savitha S. — July 2026*

## Goal

Bring SAE interpretability to **multimodal biology–language models** — an LLM with a protein or DNA foundation model fused in — starting with BioReason-Pro. The deliverable is a short workshop paper (the first SAE study of a biology–language fusion model), plus a recipe and set of methods that transfer directly to the multimodal models our team is building.

## Why it matters

There is no prior work applying SAEs to biology–language fusion models; the nearest precedents are vision-language (SAE-V, Lou et al. 2025; the MAIRA-2 radiology Matryoshka-SAE, Bouzid, Bannur et al. 2025), and the Orlov et al. bio-SAE review (bioRxiv, March 2026) names this a top-4 field priority. Beyond novelty, SAEs let us ask how deeply these models actually integrate the two modalities — shared features vs. attention-only — which bears on whether a fused model earns its cost over a cheaper tool-calling pipeline, and the methods carry straight to Peter's and John's models.

## What I've found (BioReason-Pro: ESM3 + Qwen)

- **Balancing the modalities is what makes a bio-LLM SAE work.** Protein tokens are ~50× larger in norm and a small slice of the data, so a naive SAE ignores them; a ~50/50 protein:text training mix lifts protein features from ~100 to ~1,800, and this holds across all nine layers tested (sweet spot ~50–60%). *(Appendix, Figure A1.)*
- **The SAE surfaces interpretable reasoning features** via a banded autointerp pipeline (prompt/question/reasoning/answer) labeled by a 70B-instruct NIM — e.g. "cell junction," "transport/localization." Protein-side labels are still weak, so I treat them as candidates.
- **No cross-modal (fusion) features so far.** Under the SAE-V metric I find only structural markers. We hypothesize this is because the protein and text feature spaces are near-orthogonal — the bio encoder was never trained against text — and the metric reads ~zero by construction. This is a hypothesis I can test directly. *(Appendix, Figure A2.)*

## Next steps

- **Refine the protein features and autointerp pipeline** — harden the NIM labeler and ground protein features in structure via AlphaFold / OpenFold.
- **Test the alignment hypothesis** — run the same pipeline on **Prot2Text-V2** (a protein→adapter→LLM model trained to align protein and text); cross-modal features there but not in BioReason would confirm alignment is the deciding factor. LLaVA-Next as a known-positive control; DNA to check dataset-specificity.
- **Develop a modality-aware architecture** — a mixture-of-experts SAE (an expert per modality + a shared "fusion" expert) built for the modality imbalance we see.

## Deliverable

Completing the first two next steps makes a substantial blog post or short workshop paper (cf. Goodfire's [*Under the Hood of a Reasoning Model*](https://www.goodfire.ai/blog/under-the-hood-of-a-reasoning-model)). An interesting result from the alignment test or the new architecture would grow it into a full-scale paper; comparing SAE features against pure attention is a further avenue.

---

## Appendix — details and figures

**Balancing (Figure A1).** Left alone, the SAE spends almost its whole dictionary on text. At the individual-feature level, protein-selective features (fire on protein, ~never on text) go from ~15 to ~660 when the loss is balanced; text-selective features actually *sharpen* too (protein stops piggybacking on them). Sweeping the mix, protein features plateau at ~50–60% protein and dead latents blow up past ~40% (the low-diversity ESM3 embeddings can't fill a protein-dominated dictionary).

![Figure A1](analysis/figures/fig5_feature_selectivity.png)
> **Figure A1.** Per-feature modality selectivity at layer 16 — protein-selective features (blue, top-left) rise from ~15 (unbalanced) to ~660 (balanced).

**Why the cross-modal metric reads zero (Figure A2).** The SAE-V metric compares the model's raw activations. In BioReason-Pro the protein and text activations point in almost entirely separate directions, and the SAE inherits that separation regardless of balancing — so the metric is near zero by construction. The vision-language models where fusion shows up use encoders trained against text (CLIP); ESM3 and Evo 2 never saw text, which is why the alignment test is the key experiment.

![Figure A2](analysis/figures/fig2_crossmodal_combined.png)
> **Figure A2.** *(a)* Same tokens, raw vs SAE (UMAP): the modality separation is in the raw residual and survives encoding. *(b)* Layer-16 activations: protein/text near-orthogonal (DNA/text overlap more).

**Interpretation caveat.** The dashboard's per-feature GO-AUC labels use an all-token metric that is prone to prompt-leakage (BioReason-Pro prompts contain the protein's annotations). Recomputed leakage-free (protein-band only, held-out, quality-filtered), the protein signal is weak at every layer — ~1 modest, generic feature per layer — which is why protein labels are candidates and the alignment experiment is the real unlock.
