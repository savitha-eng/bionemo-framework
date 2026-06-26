# BioReason-Pro SAE — interpretation findings (focused pass, small/8k model)

First focused interpretation pass on the **8k-subset** SAEs, per Jared's go-ahead ("training charts
look healthy, best next step is to start interpreting … begin finding anything interesting"). Layer
**L28** (our pick: most faithful + earliest, good for steering), TopK SAE, expansion 8 (n=20,480),
top_k=128, `normalize_input` + `normalize_loss`. Held-out 300-protein eval (1.07M tokens). The full
117k-protein extraction is running in parallel; this is the small-model read while that completes.

Companions: `SAE_RESULTS.md`, `DATASET_AND_MODALITY_STATUS.md`, `MODALITY_BALANCING.md`.
Artifacts: `feature_tables/features_L28_normloss.csv` (all features), `feature_tables/feature_contexts_L28_normloss.md`
(decoded snippets), `multimodal_dashboard/` (React app).

---

## 1. What the dictionary contains (L28)

- **19,441 features**; **4,635 carry a confident GO concept** (held-out de-biased per-protein AUC > 0.65), max **AUC 0.978**.
- **Band composition (where features fire):**

| band_class | count | note |
|---|---|---|
| text-heavy | 19,171 | the overwhelming majority |
| protein-heavy | 121 | scarce bio capacity (ESM3 slots) |
| go-heavy | 124 | scarce bio capacity (GO memory) |
| protein-go-shared | 16 | fire on both bio bands |
| mixed | 7 | |
| cross-modal | 2 | |

- **Top GO concepts** (by best-feature AUC): *cytosol* (0.978), *positive regulation of biological
  process* (0.959), *nucleic acid binding* (0.944), *protein-containing complex* (0.944), *plasma membrane*.
- **Feature-splitting** is healthy: e.g. ~6+ distinct "cytosol" features (15891, 9222, 8489, 11519, 13468…),
  each firing on <1% of tokens — monosemantic, sparse.

## 2. The key caveat — high AUC is partly prompt label-leakage, not pure reasoning

Reading the decoded contexts (`feature_contexts_L28_normloss.md`), the **strongest "biology" features
fire on the GO-annotation tokens that are listed verbatim in the input prompt**, not (only) on novel
reasoning:

- **F15891 "cytosol" (0.978):** fires on `ol` in `GO:0005829 cytos⟦ol⟧` — the term as printed in the
  prompt's annotation block.
- **F10418 "positive regulation…" (0.959):** fires on `of` in `positive regulation⟦of⟧ <X>` — the GO
  phrase template.

A subset *are* genuine reasoning features (fire mid-sentence in the model's analysis):
- **F13219:** "…it directly implements GO:0045944 positive⟦regulation⟧ of transcription by RNA pol II…"
- **F6644:** "…a direct route for GO:1903078 positive⟦regulation⟧ of protein localization to plasma membrane…"

**Implication:** a feature's high per-protein GO-AUC can come from the GO term *appearing in the prompt*
(the label co-occurs with the input), not from the model independently representing that function. This
is the same lesson as the retracted SAE-V "fusion" artifact (`SAE_RESULTS.md` §6): a summary metric
looked strong until we read the actual tokens. **To separate "detects the GO string" from "reasons about
the function," score features on response/reasoning-only text** (exclude the prompt's annotation block),
and/or run the input-ablation causal test.

## 3. Bio vs. text (consistent with the rank/coverage story)

Every top GO predictor is **text-heavy** — the interpretable biology is carried by the reasoning text,
with only **121 protein-heavy + 124 go-heavy** bio-band features (vs 19,171 text). Matches the
under-served-biology finding (bio tokens ~21% of corpus, ~1–8% of features; protein band low-rank ~3).
**Whether this is intrinsic or a small-N (8k/300-protein) artifact is the question the running full-train
extraction will answer.**

## 4. Dashboard status

`multimodal_dashboard/` (Jared's ESM2 app + a Multimodal view) is built with `feature_metadata.parquet`
(GO label/AUC, band/fusion class, UMAP atlas) and `feature_examples.parquet` (top examples per feature).
**Gap:** the in-app activation highlights (`sequence_window`) are still placeholders (`[text] <pid> @tok<n>`);
the decoded snippets exist in markdown but aren't yet wired into the parquet the React app reads.
**Next:** populate `feature_examples.sequence_window` with the decoded windows (CPU tokenizer pass,
no model forward) so the dashboard renders real highlighted context, then explore.

## 5. Open / next

1. **Wire decoded highlights into the dashboard parquet** → make it explorable.
2. **Response-only re-scoring** of the top GO features (exclude prompt annotation block) → how many survive?
3. Re-run this whole pass on the **full-train** SAE once trained → does bio coverage/rank rise with 14.6× data?
