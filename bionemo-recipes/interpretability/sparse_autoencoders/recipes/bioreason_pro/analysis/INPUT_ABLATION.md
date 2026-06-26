# Causal input ablation — does BioReason-Pro *fuse* the protein modality, or just tool-call it?

**TL;DR.** Zeroing the **protein-structure embedding** makes the model significantly worse at predicting
its *own* reasoning (ΔCE = +0.161 ± 0.090, t=31.1, **93% of proteins affected**) and answer
(ΔCE = +0.071 ± 0.058, t=21.3, **90% of proteins**) on 300 held-out proteins. Because the protein's GO
annotations are *also* present as text in the prompt, a model that merely "tool-called" the protein would
be unaffected — so this is **causal evidence of genuine multimodal fusion**: the protein embedding shapes
the model's chain-of-thought. Separately, zeroing the **GO-graph embedding has zero effect**
(ΔCE = 0.000) — that channel is unused; GO information reaches the LLM via text. **Protein is the live
multimodal channel.**

This result is notable because the **SAE-feature analysis says the opposite** (no cross-modal features;
see `SAE_RESULTS.md` / cross-modal section): a text-dominated SAE decomposes the fused direction into
separate unimodal features, so feature-level cosine misses the fusion that the causal test reveals.

---

## 1. Question

BioReason-Pro is multimodal: a protein **ESM3 embedding** and a **GO-graph embedding** are injected into
placeholder token slots, then a Qwen3 LLM reasons and predicts GO terms. We want to know whether the
model **integrates** the protein representation into its generation (*fusion*, like a true multimodal
model) or whether the non-text channels are inert and all the work is done by text in the prompt
(*tool-calling*, where bio info would only ever arrive as text).

The discriminating test: **the prompt already contains the GO terms as text** (the `go_pred` list). So if
removing the protein *embedding* — while leaving all text intact — still degrades the model's output,
the embedding must be carrying information the text does not, i.e. the model fuses it.

## 2. Method

`scripts/ablation_eval.py` (Env A; loads the SFT model via `load_bioreason_pro_sft`):

1. **Teacher-force** each held-out (validation-split) protein's full sequence
   `prompt + <think> reasoning </think> + answer` through the model.
2. **Measure next-token cross-entropy** `CE = F.cross_entropy(logits[:-1], input_ids[1:])` and average it
   over two token spans, identified by the `<think>` (id 151667) and `</think>` (id 151668) markers:
   - **reasoning** tokens (between the markers),
   - **answer** tokens (after `</think>`).
3. **Ablate** by wrapping the model's embedding producers to return zeros (shape-preserved so the
   injection's count-check still passes):
   - `no_protein` → `process_protein_embeddings` returns zeros,
   - `no_go` → `process_go_aspects` returns zeros,
   - `no_both` → both.
4. **ΔCE = CE(ablated) − CE(intact)**, per token-span, averaged over proteins.

**Interpretation.** ΔCE > 0 ⇒ the ablated channel was *causally lowering* the model's loss on its own
text ⇒ the model uses that channel to generate. ΔCE ≈ 0 ⇒ the channel is inert for generation.

**Why this beats the SAE cosine.** It is *causal* and *model-level* — it does not depend on how an SAE
chooses to decompose the residual stream, so it is immune to the feature-splitting confound that makes
the SAE under-report cross-modal structure.

## 3. Results (n = 300 held-out validation proteins, full model)

| condition | reasoning CE | ΔCE | answer CE | ΔCE |
|---|---|---|---|---|
| **intact** | 1.1061 | — | 0.3512 | — |
| **no_protein** | 1.2671 | **+0.161** (≈15%) | 0.4226 | **+0.071** (≈20%) |
| no_go | 1.1061 | +0.000 | 0.3512 | +0.000 |
| no_both | 1.2671 | +0.161 | 0.4226 | +0.071 |

**Per-protein significance** (paired ΔCE = no_protein − intact, n=300):

| span | mean ΔCE | sd | paired t | % proteins ΔCE>0 |
|---|---|---|---|---|
| **reasoning** | **+0.161** | 0.090 | **31.1** | **93.0%** |
| **answer** | **+0.071** | 0.058 | **21.3** | **89.7%** |

- **Protein embedding is causally used** for both reasoning and answer generation, and the effect is
  **systematic, not a few outliers**: ~9 in 10 individual proteins are hurt by removing it (t ≫ 5).
  Absolute effect is larger on reasoning; *relative* effect is larger on the answer (20% vs 15%).
- **GO-graph embedding is unused** (exactly 0.000; `no_both == no_protein`). The model obtains GO
  information from the text `go_pred`, not the graph channel.
- Replicated at n=200 (ΔCE +0.165 / +0.073) — stable.

## 4. Interpretation & the SAE tension

- **The model fuses** the protein modality into its reasoning — stronger evidence of "deeper than
  tool-calling" than any feature-level metric, because it is causal.
- **Standard SAEs hide this.** The SAE trained on the (≈80% text) token stream allocates ≤8% of features
  to bio and tends to split a cross-modal direction into a protein-only + a text-only feature, so the
  per-sample SAE-V cosine (`crossmodal_cooccur.py`) reads ≈0 for protein↔text/reasoning/answer. The
  fusion is in the model; the vanilla SAE just doesn't surface it. → motivates **modality-balanced SAE
  training** (`MODALITY_BALANCING.md`) to recover the cross-modal features.
- **The GO-graph channel is vestigial** for generation — a useful negative result (the multimodal value
  is in the protein/ESM3 channel, not the GO graph encoder, which only re-expresses a fixed ontology
  summary; see `go-graph-slots-not-resolvable`).

## 5. Caveats / next steps

- Teacher-forced CE on the model's *own* (GPT-5-distilled SFT) targets; a generation-mode + GO-prediction-F1
  version would test task accuracy directly.
- Add per-protein CIs / paired t (in progress) and a within-protein permutation test.
- Run per-layer (does the protein contribution enter at a specific depth?) and on the **balanced-loss SAE**
  to check whether it now surfaces the cross-modal features the ablation proves exist.

## 6. Reproduce

```bash
ENVA=/data/savithas/bioreason-pro/.venv
CUDA_VISIBLE_DEVICES=<gpu> $ENVA/bin/python scripts/ablation_eval.py \
  --num-proteins 300 --split validation
# prints per-condition mean CE on reasoning/answer + paired per-protein dCE significance.
```
