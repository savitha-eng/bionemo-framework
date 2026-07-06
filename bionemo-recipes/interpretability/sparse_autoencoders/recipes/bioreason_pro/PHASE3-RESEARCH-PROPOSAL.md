# Sparse Autoencoders for Biology–Language Models

*Savitha S. — July 2026*

Sparse autoencoders (SAEs) are the standard way to pull interpretable features out of a transformer, and over the past year people have started applying them to vision–language models (SAE-V, Lou et al. 2025; Zhang et al. 2024; Pach et al. 2025; crosscoders, Lindsey et al. 2024). No one has done this for **biology–language models** — the kind where a protein or DNA foundation model is fused into an LLM. The recent bio-SAE review (Orlov et al. 2026) even lists "scaling SAE analysis to multimodal architectures" as an open priority, but only in the vision sense. I've built this analysis out for biology, and I think it's both a genuine research contribution and a way to answer a question the business actually cares about.

That question: when we fuse a biology model into an LLM (BioReason = Evo 2 + Qwen; BioReason-Pro = ESM3 + Qwen; Peter's DNA-tokenizer MoE on Nemotron), is the model really *combining* biology and language into something new, or is it doing a dressed-up version of tool-calling — reading the encoder's output like a lookup and largely ignoring the rest? The answer bears directly on whether multimodal bio models are worth the investment over a cheaper pipeline that just calls a bio tool. SAEs give us a direct way to check, and the method carries over cleanly from the vision-language work.

## What I've done so far

I've focused on BioReason-Pro (ESM3 + Qwen) and gotten the whole pipeline working end to end — extracting activations, training SAEs, and interpreting the features. A few things stand out.

**Training an SAE on a multimodal bio model is not the same as on a plain LLM.** The two modalities live at very different scales: protein tokens are roughly 50× larger in norm than text tokens, so without careful normalization the protein tokens swamp the objective. The bigger lever is balancing the modalities during training. Protein tokens are a small slice of the data, and left alone the SAE mostly ignores them; when I rebalance the mix to about 50/50 protein:text, the number of protein features jumps from ~100 to ~1,800 (Figure 1), and the two modalities separate cleanly instead of the dictionary being swallowed by text.

![Figure 1](analysis/figures/fig1_balancing.png)
> **Figure 1.** Modality loss-balancing at layer 16: protein-selective features rise ~100 → ~1,800, and the modalities separate cleanly rather than the dictionary being dominated by text.

![Figure 2](analysis/figures/fig5_feature_selectivity.png)
> **Figure 2.** The same effect at the level of individual SAE features. Each dot is one feature: x = fraction of text tokens it fires on, y = fraction of protein tokens. Protein-selective features live in the top-left. Under the unbalanced loss that corner is nearly empty (~15 features); balancing fills it (~660) — the SAE gains a dedicated protein vocabulary instead of forcing protein onto text-owned features.

**The model does learn real, interpretable reasoning features.** I built an interpretation pipeline that labels a feature by looking at which proteins it fires on and what those proteins have in common. It's surfacing clean, specific concepts from the model's reasoning — a "cell junction" feature, an "embryonic development" feature, an "establishment of localization / transport" feature — and these aren't trivial keyword matches. They recur across the model's reasoning and line up with exactly the proteins the concept applies to. Doing this on a *biology* reasoning model — and tying the features back to protein function — is new; the SAE method itself is the same standard approach recent reasoning-interpretability work uses (e.g. Goodfire's DeepSeek-R1 study).

**The most interesting result is about how the model fuses the two modalities — and so far, it doesn't, at least not the way you'd expect.** I went looking for features that tie a biological concept to its text, and beyond structural markers I haven't found any. Working out why is, I think, the real result. The standard cross-modal metric (from SAE-V) compares the model's raw activations, and in BioReason-Pro the protein and text activations point in almost entirely separate directions (the DNA model overlaps somewhat more), so that metric reads near zero almost automatically. This looks like a property of the encoder rather than a flaw in the method: the vision-language models where fusion showed up use encoders trained against text (CLIP), whereas ESM3 and Evo 2 have never seen text — we project them into the LLM's space, but that doesn't make them point the same way. My working hypothesis is that cross-modal features only emerge once the two modalities are actually aligned, and that's something I can test directly (below).

![Figure 3](analysis/figures/fig3_geometry.png)
> **Figure 3.** Layer-16 activations (UMAP). Protein (ESM3) and text form near-separate clusters (left); DNA (Evo 2) and text overlap more (right) — which is why a raw-activation cross-modal metric reads near zero for an encoder that was never aligned to text.

I'd flag two honest caveats. The protein-feature labels are still preliminary — many protein features are weak or fire too often to trust — so I treat them as candidates to verify, not conclusions. And the labeler currently keys on a feature's single strongest token, which sometimes misses recurring phrases; that's a fix I already have in mind.

## Why it's worth publishing

No one has applied SAEs to a biology–language fusion model, and the field's own review calls it out as open. On top of that first, I have a concrete (if early) mechanistic finding — fusion doesn't appear to happen at the feature level, plausibly because the bio encoder isn't aligned to text — a clean experiment to test it, and a proposal for a new SAE architecture for the multimodal case. That's enough for a blog post now, a workshop paper as the story firms up, and a full paper if the alignment result holds and the new architecture pans out. It's also a contribution the interpretability community would find useful: a working recipe and set of pitfalls for anyone bringing SAEs to a fused multimodal model.

## Where I'm taking it

The key next experiment is to test the alignment hypothesis head-on: run the same pipeline on **Prot2Text-V2**, which has the same shape as BioReason (protein encoder → adapter → LLM) but was explicitly trained to align protein and text. If cross-modal features show up there but not in BioReason, that's strong support. Alongside it I'm standing up two baselines — reproducing SAE-V on LLaVA-Next to confirm my metric behaves, and running the DNA model — plus three directions I'm keen on: studying how reasoning features change across the model's layers (a general question for multimodal models, not just this one), comparing what the SAE finds against what attention is doing, and grounding protein features in structure by mapping them onto AlphaFold / OpenFold predictions.

On architecture: my balancing and Matryoshka experiments were a first pass, and they told me the fix is probably structural rather than another loss tweak. I want to try a mixture-of-experts SAE — a separate expert per modality plus a shared "fusion" expert — as a cleaner way to give each modality its own capacity and to make cross-modal fusion something we can measure directly instead of hunt for. It's still a proposal, and a natural fit for the MoE-based fusion models we're already building.
