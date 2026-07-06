# Modality balancing across layers (BioReason-Pro)

*Savitha S. — July 2026 · companion to the main proposal*

A quick cross-layer check on the balancing result from the proposal: does modality loss-balancing create protein-selective SAE features at *every* layer, or is layer 16 special? I trained a matched pair of SAEs — unbalanced (~2% protein, the natural mix) vs balanced (~50/50 protein:text) — at nine layers, and measured per-feature modality selectivity on the same protein+text token sample.

A feature is **protein-selective** if it fires on >1% of protein tokens and <0.1% of text tokens (top-left corner of the scatter), and **text-selective** the other way around.

## Result: balancing creates a protein vocabulary at every layer

| Layer | protein-selective (unbal → bal) | text-selective (unbal → bal) |
|------:|:-------------------------------:|:----------------------------:|
| L14   | 27 → **572**  | 1787 → 1395 |
| L16   | 12 → **650**  | 1972 → 2248 |
| L18   | 15 → **714**  | 2564 → 2469 |
| L20   | 21 → **646**  | 2735 → 2403 |
| L22   | 13 → **764**  | 2360 → 2223 |
| L24   | 34 → **928**  | 2009 → 1893 |
| L28   | 31 → **709**  | 3107 → 2773 |
| L30   | 12 → **663**  | 1741 → 2888 |
| L32   |  9 → **655**  | 1393 → 3171 |

The effect is robust and layer-independent: unbalanced, protein gets a couple dozen dedicated features at best; balanced, it gets **~570–930**, a 20–50× increase, at every layer. The peak is around **L24** (34 → 928), and the mid-stack layers (L22–L24) are the richest. Text-selective counts stay in the same ballpark — balancing isn't starving text, it's carving out capacity the natural mix never gave protein.

*(Measured on a 2,000-protein-token + 2,000-text-token sample per layer; counts scale with the sample but the unbal→bal ratio is stable. "Selective" here means near-exclusive firing, so these numbers are stricter — and smaller — than the "protein-heavy feature" count in Figure 1 of the proposal.)*

![selectivity grid](analysis/figures/fig_layer_selectivity_grid.png)
> **Per-feature modality selectivity across layers.** Left column = unbalanced loss, right column = balanced. Each dot is a feature; x = fraction of text tokens it fires on, y = fraction of protein tokens. The dense stripe up the left edge of every balanced panel is the protein-selective population — nearly absent in every unbalanced panel.

## Token geometry across layers (raw vs SAE)

Balancing changes *which* features carry protein (above), but it doesn't change where the tokens sit. At every layer, protein and text separate in the raw residual stream, and the SAE codes inherit that separation whether the loss is unbalanced or balanced. The modality split is a property of the encoder that the SAE preserves at all depths — which is the same reason the raw-activation cross-modal metric reads near zero throughout (see the main proposal).

![UMAP grid](analysis/figures/fig_layer_umap_grid.png)
> **Token geometry across layers.** Rows = layers (L14–L32); columns = raw residual, unbalanced SAE code, balanced SAE code (same protein+text tokens, UMAP). Blue = protein, purple = text. Protein and text form separate regions in every panel — the separation is there in the raw residuals and survives encoding regardless of balancing.
