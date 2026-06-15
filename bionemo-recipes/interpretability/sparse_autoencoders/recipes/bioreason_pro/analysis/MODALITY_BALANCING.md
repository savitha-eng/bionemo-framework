# Modality balancing for the BioReason-Pro SAE

**Status:** open research direction. This doc has (1) the per-modality feature-coverage analysis that
motivates it, (2) a prioritized experiment plan, (3) literature notes. For team review.

BioReason-Pro is multimodal — the residual stream carries **protein** (ESM3), **GO** (graph memory),
and **text** (reasoning) tokens. A single SAE is trained on all of them. The question: *is the SAE
learning the biology, or mostly the text?*

---

## 1. The problem — per-modality feature coverage (measured)

`scripts/per_band_coverage.py` counts, per band, **how many SAE features ever fire on that band's
tokens** (held-out val300, 8× / top_k 128 / normalize_loss). The global "dead %" hides the story:

| layer | protein feats (of 20,480) | GO feats | text feats | **bio total** | dead-everywhere |
|---|---|---|---|---|---|
| **L28** | **1,222 (6.0%)** | **670 (3.3%)** | 19,321 (94.3%) | **1,658** | 5.1% |
| L30 | 210 (1.0%) | 258 (1.3%) | 19,654 (96.0%) | 299 | 3.8% |
| L32 | 309 (1.5%) | 284 (1.4%) | 19,296 (94.2%) | 382 | 5.3% |

**Token mix of the corpus: protein 15.4%, GO 5.6%, text 79.0%.**

Findings:
- **The SAE massively under-serves the bio modalities.** Bio tokens are **21%** of the corpus but get
  ~**8% (L28)** / ~**1.5% (L30)** / ~**1.9% (L32)** of the features. ~98% of features at L32 never fire
  on a protein token.
- **L28 fits ~4× more bio features than L32** (1,658 vs 382). The earlier layer keeps more bio-specific
  features before the model folds protein/GO info into its text representation — another reason L28 is
  the better layer for biology/steering.
- The skew (94% text features) is **worse than the 80/20 token split**, because (a) the reconstruction
  loss rewards covering the dominant modality and (b) the bio bands are intrinsically **low-rank**
  (`LAYER_ANALYSIS.md`).

**This is the core motivation:** if we want the SAE to learn *protein/GO* concepts (not just text), we
must counteract this allocation. (Protein-band feature AUC is currently ~0.59 — near the random
baseline — consistent with there being only ~300 protein features at L32 to find concepts in.)

---

## 2. Experiment plan (prioritized)

Success metrics throughout: **per-band feature coverage** (this analysis), **per-band feature AUC**
(esp. protein, vs the ~0.59 floor), **# bio-labeled features**; guardrails: loss-recovered ≥ 0.97,
text-band AUC not regressing. Each is ~8-min train + eval on the subset.

### Tier 1 — cheap, high-value
1. **Normalized AuxK loss.** The dead-latent revival (AuxK) is still computed in *raw* space while the
   primary FVU is normalized → it revives toward high-magnitude directions. Route it through
   `_loss_targets` (normalized space) for consistency. *Hypothesis:* lower dead%, better bio revival.
2. **Modality-balanced loss reweighting** *(most promising)*. Weight each token's reconstruction loss by
   **inverse band frequency** (text ×~1, protein/GO up-weighted) so the SAE doesn't optimize ~only text.
   *Hypothesis:* directly raises bio feature coverage + AUC. This is the class-imbalance fix applied to
   the token stream.
3. **Per-modality `pre_bias`.** Replace the single global geometric-median center (text-dominated) with
   one center per band. Protein-normalized and text-normalized are ~orthogonal, so a per-modality center
   fits each better. *Hypothesis:* fewer dead bio features.
4. **Per-band dead tracking + targeted AuxK.** Track dead-ness per band; revive features dead on the
   *rare* modalities. *Hypothesis:* forces capacity to protein/GO.

### Tier 2 — capacity & normalization
5. **Reserved capacity / per-modality top_k** — reserve a feature subset for bio tokens, or raise `top_k`
   on protein/GO tokens.
6. **Sink-token filtering** — the 4 stable outlier channels (0,4,19,21, ~7× norm) were found but never
   filtered; they may distort the normalized *direction* even though `normalize_input` divides out scale.
7. **RMSNorm-match normalization** — `normalize_input` does `(x−mean)/std`; the model's RMSNorm does
   `x/RMS(x)·γ` (no mean-subtract, learned per-dim γ). Match it so the SAE reads the residual exactly as
   the model's sublayers do. (Likely second-order.)

### Tier 3 — architecture (exploratory)
8. **Modality-conditioned SAE** — give the encoder the token's band (a learned band-embedding / per-band
   bias) so it can behave per-modality. *Risk:* trivial "is-a-protein-token" detector features.
9. **Per-modality SAEs** (baseline) — separate SAEs per band; cleaner bio features but no cross-modal.
10. **Hook post-RMSNorm activations** instead of the raw residual.

### Tier 4 — causal / eval
11. **Input ablation** — zero the protein/GO embeddings, measure GO-prediction degradation → does the
    bio data actually get used, or is text carrying it?
12. **Cross-modal fusion, done right** — exclude prompt boilerplate + control for top-token selection
    (the naive SAE-V "+0.59 fusion" was a prompt-template artifact — see `SAE_RESULTS.md` §6).

---

## 3. Literature notes

Multimodal-residual-stream SAE balancing specifically is **under-explored** (part of why this is a real
direction). The relevant established ideas to borrow from:

- **Class-imbalance ML** — inverse-frequency **loss reweighting**, **over/under-sampling**, and **focal
  loss** (Lin et al., 2017). Directly applicable: protein/GO tokens are the minority "class."
- **Data-mixing for LM pretraining** — DoReMi (Xie et al., 2023) and domain-reweighting: the closest
  large-scale analogue ("one source dominates → reweight to balance").
- **SAE capacity / dead latents** — Gao et al. "Scaling and evaluating SAEs"; Anthropic monosemanticity
  (feature splitting/absorption). Covers dead latents + feature allocation, but **not** modality balance.

**Honest caveat:** the protein band is intrinsically **low-rank**, so reweighting/oversampling can
re-allocate *attention* to protein tokens but **cannot manufacture structure that isn't there** — there
may be a ceiling on protein-feature quality. So treat "does protein-band AUC beat 0.59?" as the empirical
test, not an assumption. *(TODO: a proper literature search to ground citations — the above are
analogues, not a multimodal-SAE-balancing paper.)*

---

## 4. Reproduce

```bash
# per-modality coverage for a layer
python scripts/per_band_coverage.py --sae <ckpt> --store val300_L<L> --layer L --out-json out.json
# results above are in /scratch/savithas/phase3_subset/per_band_coverage_l{28,30,32}.json
```
