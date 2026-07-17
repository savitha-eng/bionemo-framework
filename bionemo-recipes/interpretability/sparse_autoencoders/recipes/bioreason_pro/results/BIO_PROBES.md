# Biological / Structural Probes — Results

_Do the SAE (and the underlying representation) encode protein structure, and does the SAE isolate
interpretable structural features? Self-contained results + experimental details._

## Experimental setup

| item | value |
|---|---|
| Model | BioReason-Pro SFT (ESM3 protein embeddings + Qwen3-4B reasoning LLM) |
| SAE | `sae-l30-exp16-balanced`, TopK, layer 30 residual, **input 2560 → 40,960 features (expansion 16×), top-k 128** |
| Data | 8,000 held-out-distribution proteins (`phase3_subset_8k/L30_subset8k`); AlphaFold v6 structures for 7,613 (95%) |
| Band probed | **protein / residue band** (ESM3 per-residue embeddings), not the reasoning band |
| Layer | **L30 residual** for the per-feature + structural probes. Decodability was **swept L16/28/30/32** (below) → structure is **flat-high at every depth**, so L30 is not special. *Caveat:* the paper reads ESM3 pre-projection / earlier residues; our probes are on the LLM residual stream. |
| Representations compared | **SAE-svd256** (SAE 40,960→256 via SVD) vs **raw** (2560-d, or PCA-256) vs **random-SAE** |
| Probe | logistic regression, held-out AUROC; per-feature analysis uses per-feature AUROC (rank-sum) |

**Two questions, two methods (this distinction matters):**
1. **Decodability** — "is structure in the representation?" → dense probe on SVD-256 / raw. (Compresses+mixes
   features; measures decodability + SAE-vs-raw, but *cannot name individual features*.)
2. **Which features** — "does the SAE isolate an interpretable structural feature?" → **per-feature AUROC**
   (each of the 40,960 features scored individually). *This is the interpretability question.*

## Labels probed

| probe | label (Y) | source | granularity | circular? |
|---|---|---|---|---|
| InterPro domain (presence) | protein has domain X? | `interpro_ids` | per-protein | non-circular |
| InterPro domain (boundary) | residue in domain X? | `interpro_location` spans | per-residue | non-circular |
| 3D contact / burial | residue buried (core) vs surface? | AlphaFold contact number (Cβ<8Å, top/bottom tercile) | per-residue | non-circular |

## Result 1 — Decodability (dense probe, SAE-svd256 vs raw vs random)

| probe | SAE-svd256 | raw | random | verdict |
|---|---|---|---|---|
| InterPro domain, per-protein (60 domains) | **0.990** | 0.989 | 0.981 | SAE ≈ raw (near-saturated) |
| InterPro domain, per-residue (boundaries) | **0.982** | 0.982 | 0.947 | SAE = raw |
| 3D contact (buried vs surface) | 0.873 | 0.886 (pca256) / 0.890 (full) | 0.811 | **raw > SAE** |

![InterPro domain probe](charts/interpro_probe.png)
![contact dimension sweep](charts/contact_dimsweep.png)

**Dimension-sweep control (is 256 too small for the 40,960-d SAE?):** SAE-svd vs raw-pca at matched K —

| K | SAE-svd | raw-pca |
|---|---|---|
| 256 | 0.869 | 0.882 |
| 512 | 0.871 | 0.891 |
| 1024 | 0.866 | 0.890 |
| 2048 | 0.854 | 0.867 |

**raw beats SAE at every matched dimension** → "SAE loses on 3D contact" is real, not a compression artifact.

**Layer sweep (does structure peak at a depth, or is it flat?):** InterPro-domain decodability (mean over 60 domains) across the LLM residual stream —

| layer | SAE-svd256 | raw | random |
|---|---|---|---|
| L16 | 0.991 | 0.990 | 0.979 |
| L28 | 0.983 | 0.985 | 0.968 |
| L30 | 0.990 | 0.989 | 0.981 |
| L32 | 0.983 | 0.984 | 0.966 |

![layer decodability](charts/layer_decodability.png)

**Flat-high (~0.98–0.99) at every depth, and SAE ≈ raw at every depth.** Domain identity is carried from the
ESM3 input essentially unchanged through the LLM stack — it doesn't "build up" with depth, so **the L30 result
is not a layer artifact**. (The tiny L28/L32 dip is within noise.) Consistent with "the encoder is the ceiling."

![per-residue burial example](charts/contact_example.png)
*What "buried vs surface" means: one protein's per-residue 3D contact number (red = buried core).*

**Takeaway (decodability):** structure is **near-perfectly decodable from the ESM3 residues** (domains ~0.99),
and the **SAE never beats raw** — it re-expresses the residue representation (the encoder is the ceiling),
losing slightly on the harder 3D-contact probe. Expected: an SAE is a lossy re-expression; its value is
interpretability, not decodability.

## Result 2 — Which features? YES, the SAE isolates clean structural-residue features

Per-feature AUROC of each of the 40,960 SAE features against InterPro-domain-membership labels (residue band):
the best single feature per domain. **This corrects an earlier expectation** — the residue band is NOT feature-
barren for structure; it just isn't for GO *function*.

![per-feature structural](charts/per_feature_structural.png)

| InterPro domain | best feature | AUROC |
|---|---|---|
| Protein kinase domain (IPR000719) | **F16026** | **0.98** |
| Protein kinase-like (IPR011009) | F16026 | 0.97 |
| Armadillo (ARM) fold (IPR016024) | F679 | 0.95 |
| WD40 repeat (IPR015943) | F11009 | 0.92 |
| Homeodomain-like (IPR009057) | F5540 | 0.90 |
| Ig-like (IPR036179 / IPR007110 / IPR013783) | F17703 / F25050 / F1386 | 0.87 |
| Zinc-finger (IPR013083) | F8306 | 0.80 |
| P-loop NTPase (IPR027417) | F7079 | 0.75 |

**So the SAE HAS individual, nameable structural-residue features** — e.g. F16026 fires on protein-kinase-domain
residues at 0.98 AUROC; different features cleanly pick out different folds (kinase, ARM, WD40, Ig-like, Zn-
finger). 8 of 10 abundant domains have a clean (>0.80) single feature.

**Reconciliation with "SAE ≈ raw":** the SAE doesn't *decode* domains better than raw (both ~0.99), but it
*isolates* each fold into a **single nameable feature** — an interpretable handle raw's entangled dimensions
don't give. So on the residue band the SAE adds **interpretability** (nameable structural features) even though
it adds no **decodability**. (Buried-vs-surface per-feature number pending — the first pass had an AUROC bug,
re-running.)

## Result 3 — Does the residue feature "reach into" the reasoning? (text↔structure) — NO (it's prompt-mediated)

F16026 (kinase-catalytic residues, 0.98) fires **only on protein residues, never on reasoning tokens**. Its 50
proteins' reasoning traces *are* consistently about kinase catalysis — but this is **not feature-level fusion**:

| what the reasoning says | echo or synthesis? | evidence |
|---|---|---|
| **identity** — "it's a protein kinase domain" | **echo** of the prompt | 29/50 proteins have the InterPro kinase annotation *in the prompt*; reasoning cites the IPR IDs verbatim |
| **mechanism** — "bilobal fold, phosphotransfer, activation loop, catalytic scaffold" | **novel** (not copied) | ~83–90% of reasoning words absent from prompt; mechanism terms not in the given annotation |

**Interpretation:** the prompt hands the model the kinase *label*; the model elaborates generic textbook kinase
*mechanism* from that label (**label-conditioned recall, not residue-conditioned inference**). F16026 (residue
channel) and the reasoning (text channel) are **two independent descendants of "this is a kinase"** — a fact the
model got from the **prompt**, not from residues feeding reasoning. Consistent with **no feature-level cross-modal
fusion** (fusion, where it exists, is attention-mediated). So the residue-band structural features are real and
nameable, but they do **not** demonstrate structure→reasoning information flow at the feature level.

**Note on the paper:** the paper did **not** run an InterPro-domain *probe*. It used InterPro domains only to
*group per-residue attention scores by domain* (§4.5.5–6, eEFSec/CFAP61 case studies). The decodability probe and
per-feature structural analysis here are **our additions**, not a reproduction of a paper result.

## Bottom line
- **Structure lives in the ESM3 residues at ceiling** (domains ~0.99, 3D burial ~0.89); the **SAE re-expresses
  it, never beats raw.**
- Contrast: **function (GO) is weakly but *non-circularly* decodable from residues (~0.83, above the 0.79
  random baseline).** ⚠️ **The reasoning-band GO "~0.95" is LEAKAGE, not a representation result** — a *random*
  projection of the reasoning band also scores ~0.957 (trained probe, L16), because the model's reasoning text
  literally states the function it is predicting. So GO is trivially decodable there by *anything*; it is NOT
  evidence the reasoning representation encodes function. Non-circular functional signal lives *only* in the
  protein residues (weak). [corrected per Jared's random-baseline discipline; prior "function emerges in
  reasoning ~0.95" claim retracted.]
- The **SAE's interpretability value is in the reasoning band**, not the residues (see the reasoning-feature +
  steering + synthesis results in `RESULTS.md`).

_Scripts: `interpro_probe.py`, `residue_domain_probe.py`, `contact_residue_probe.py` (+ `download_alphafold.py`),
`per_feature_structural.py`, `make_probe_charts.py`._
