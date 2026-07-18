# BioReason-Pro SAE — Complete Results Overview

_Single source-of-truth synthesis of all interpretability experiments. **Every result below is layer 30**
(SAE `sae-l30-exp16-balanced`, TopK k128, 2560→40,960 features, `normalize_input=True`), unless a layer
sweep is explicitly named. Dashboard: `phase3_subset_l30_dash`, model key `l30`._

## The framework — 4 tools, 4 questions (they are NOT interchangeable)

| tool | question it answers | trained model? | script |
|---|---|---|---|
| **Single-feature AUROC** | is there ONE feature that *is* this concept? (monosemanticity) | no | `feature_biology_table.py` |
| **Linear-regression probe** | is the concept *decodable*; does SAE beat raw; (sparse →) which features? | **yes** | `probe_v2.py`, `probe_trained.py` |
| **Enrichment (Fisher / GSEA)** | what biology do a feature's top proteins share? (unsupervised) | no | `enrichment_probe.py`, `gsea_enrichment.py` |
| **Domain-F1 (Polina)** | does a structural feature fire *inside* the domain and *not outside*? | no | `domain_f1.py` |

Plus two experimental layers: **steering** (causal) and **cross-modal alignment** (representational).

---

## 1. Single-feature AUROC — the feature→biology map (on BOTH InterPro AND GO)

`feature_biology_table.py`: scored all 40,960 features against **ground-truth dataset annotations** (UniProt/
InterPro — NOT model output). Term selected per feature by Fisher enrichment on its top-activating proteins,
then per-feature **per-protein** rank-sum AUROC for that term.

- **1,187 features** pass FDR < 0.01, split into:
  - **872 InterPro-domain features** (structural) — AUROC **0.97–1.00**
  - **315 GO-term features** (functional) — AUROC **0.68–0.89**
- Clean examples: F2062 kinase, F11009 WD40, F16964 LRR, **F7369 GPCR (IPR000276)**, **F3623 P450 (IPR001128)**, F11025 RNA-binding.

**Significance:** the SAE isolates clean, nameable **structural** features (~0.99) — its core interpretability
win. **Functional (GO)** features are real but weaker (0.68–0.89): function is less cleanly monosemantic than
structure. **Caveat:** the AUROC is *selection-biased* (best term AND best feature chosen on the same data) —
treat 0.99 as best-case; a train/test correction (`best_single_train_test`) drops detectors ~0.01.

## 2. Domain-F1 (Polina) — the honest correction to the AUROC story ⭐ new

Per-**position** precision (of firing residues, fraction inside the domain) × per-**region** recall (of domain
regions, fraction with ≥1 firing residue), over all 872 structural features (`domain_f1.py`, 3000 proteins).

- **corr(AUROC, domain-F1) = 0.23** — the two metrics rank features very differently → domain-F1 is non-redundant.
- **Only 148/872 (~17%)** of AUROC-selected structural features are *genuinely* localized (F1 ≥ 0.5).
- AUROC oversell exposed: **F4647** kinesin conserved-site AUROC 0.982 but **F1 = 0.0** — it fires, but *never
  on the domain's residues* (high per-protein AUROC just means "these proteins activate it more overall").
- Cleanly localized: **F18393** kinesin (0.98), **F13950** RNA-binding (0.96), **F30032** kinase-like (0.95),
  **F18647 / F37746** protein-kinase (0.93), **F18162** HLH (0.93).

**Significance:** the honest count of real, domain-localized structural features is **~1/6** of the AUROC list.
This is the most important methodological correction we added — AUROC per-protein overstates localization.

## 3. Linear-regression probes — decodability + SAE-vs-raw

### 3a. Structure/function decodability (protein/residue band — `probe_v2`, `BIO_PROBES.md`)
| probe | SAE-svd256 | raw | random | verdict |
|---|---|---|---|---|
| InterPro domain (per-protein) | 0.990 | 0.989 | 0.981 | **saturated, SAE ≈ raw** |
| InterPro domain (per-residue boundary) | 0.982 | 0.982 | 0.947 | SAE = raw |
| 3D contact (buried vs surface) | 0.873 | **0.886** | 0.811 | **raw > SAE (SAE loses)** |
| GO function (protein band) | 0.76–0.81 | — | ~0.79 | **weak, non-circular** |

Layer sweep L16/28/30/32: **flat-high (~0.98) everywhere** → the ESM3 encoder is the ceiling; structure
doesn't "build up" with depth. **The SAE never beats raw** — expected (it's a lossy re-expression).

### 3b. Reasoning-concept probes (reasoning band — `probe_trained`, L30, sparse L1 + dense)
12 designed concepts. **MEAN: sparse SAE 0.986 > raw 0.954 > random/leak 0.946 > SAE-svd 0.938.**

| concept | sparse AUROC | n-features | raw | random (leak) |
|---|---|---|---|---|
| **defense → fungus** | **0.993** | **12** | 0.971 | **0.882** |
| defense → bacterium | 0.993 | 36 | 0.917 | 0.885 |
| kinase activity | 0.992 | 80 | 0.985 | 0.979 |
| reproduction | 0.974 | 257 | 0.941 | 0.958 |
| mitochondrion | 0.988 | 159 | 0.964 | 0.975 |

Antifungal L30 feature IDs: `36488, 35336, 2808, 23726, 29332, 32785, 7665, 15775, 22156, 2082, 28215, 5047`.

**Significance:** the SAE never wins on *dense* decodability, **but the sparse SAE probe beats raw** — its
nonlinear TopK basis makes concepts more linearly separable with a handful of features. **Honest caveat:**
reasoning-band GO is *leakage-prone* (random projection ≈ 0.95 because the text names the function), so only
the *margin above the leak floor* is real. **Microbial-defense concepts have it (leak floor only ~0.88);
reproduction/mitochondrion sit AT the leak floor** and are not trustworthy reasoning-representation signals.

## 4. Echo-vs-synthesis probe — non-circular reasoning (`echo_synthesis_probe`)

**Goal (corrected framing):** find the *clean reasoning features that fire MORE on synthesis (novel elaboration)
than on echo (restatement)* — i.e. which concepts the model **adds beyond the given annotations**. Per-feature
AUROC(activation | synthesis-token vs echo-token); token is **synthesis** if its word is novel, **echo** if
copied from the prompt. Non-circular (label = reasoning-process property, not the GO answer). Shuffle-null 0.506.
AUROC here is the **rank** ("more synthesis-leaning than others"), NOT a claim any feature "is synthesis."

⚠️ **RETRACTED (selection flaw):** the original ranking had **no frequency filter**, so ~100%-firing near-dense
features topped it (F39979, F39744, F5147, F14759, F7099 all fire on ~100% of tokens — always-on, not clean
features; their 0.71 AUROC is a weak magnitude bias). **Those are not features, and the auto-interp labels I
gave them are invalid.** `echo_synthesis_probe.py` now applies a `freq ≤ 2%` filter (`synthesis_formalize.py`
regenerates the clean list from cached activations).

**CLEAN synthesis-leaning features** (freq ≤ 2%, each carries a specific biology it *elaborates*):

| feature | AUROC | freq | biology (top firing phrase) |
|---|---|---|---|
| **F11654** | 0.601 | 0.09% | calcium/calmodulin — "sterically gated by calcium/calmodulin" |
| **F16620** | 0.592 | 0.11% | DNA strand — "nucleates DNA strand separation" |
| **F29986** | 0.591 | 0.04% | lipid/cholesterol — "reduced surface levels of LDLR" |
| **F7351** | 0.584 | 0.06% | membrane topology — "on the cytosolic face of" |
| **F29616** | 0.577 | 0.06% | protein fold — "jellyroll fold characteristic of ConA-like" |
| **F35498** | 0.567 | 0.05% | axonal transport — "powers retrograde transport along axon" |
| **F11877 / F8875** | 0.56 | 0.00% | mitochondrial matrix / inner-membrane translocase |

**Echo features** (restate given IDs): fire on IPR/GO accession strings. **F35387 is excluded — it fires on the
`<think>` assistant-start token (positional, not biology)**; the freq filter alone doesn't catch positional features.

**Trained probe (echo vs synthesis): CV AUROC 0.932** (dense-SVD256 0.914, best-single 0.71, shuffle-null 0.505).
**Controls now run (`echo_synthesis_probe.py`):**
- **raw-residual baseline = 0.916 ≈ SAE 0.914** → **the SAE adds nothing over raw.** Echo-vs-synthesis is NOT an
  SAE-feature story — the distinction lives in the residual stream itself. (So "distributed *SAE* representation"
  is **withdrawn** — SAE ≈ raw.)
- **same-token-type = 0.916 (no collapse)** → it is **NOT** the accession-ID surface confound; even restricting to
  ordinary-word tokens on both sides, echo-vs-synthesis stays decodable at 0.916. So the distinction is real, but
  it likely reflects **copy-vs-generate token mechanics** (a residual-stream property), not a special SAE concept.

**Net:** the *clean individual features* (F11654 calcium, F16620 helicase…) remain useful interpretable handles,
but "synthesis" as a global probe is a **residual-stream property (SAE ≈ raw), not an SAE-feature phenomenon.**

**Auto-interp validation** (`autointerp_validate.py` — do a labeled feature's top proteins match its label?):
**cross-fire partners 19/19 match** (anchored to bio features: F15673 "Motor"→P-loop NTPase/kinesin, F13384→
oxidoreductase/P450, F4783→LRR); **free-floating synthesis labels are shakier** (~⅓ clean-agree, ~½ partial, ~⅕
mismatch — e.g. F423 "Response regulator" but top proteins are GPCRs). **Auto-interp labels are hypotheses, not
ground truth** — anchored ones validate, free-floating ones need the enrichment cross-check.

## 5. Cross-modal alignment — two senses (this is where "co-fire" lives)

**Sense A — cross-*feature* pairing** (`crossmodal_pairing`): a bio residue-feature's per-protein activation
vector Pearson-correlated (across 8k proteins) against every reasoning feature's per-protein vector.
- **20/20 bio features have a concept-matched reasoning partner at r ≥ 0.4**; permutation-null max-r 0.07–0.25.
- GPCR F7369↔F3184 (r=0.89), P450 F3623↔F13384 (0.87), RNA-binding F11025↔F10235 (0.86), LRR F16964↔F4783 (0.81).
- Multiple independent bio features for the same fold **converge on one reasoning feature** (4 GPCR features → F3184).

**Sense B — same-*feature* co-firing** (SAE-V φ / Eq.7): does ONE feature fire in both bands? **NULL.**

**Cross-modal LOCALIZATION (SAE-V spirit — the tightened version, `crossmodal_localization`):** per-protein
pooled correlation can be an artifact (same critique as domain-F1). So require *localization on both sides*:
bio feature domain-F1 ≥ 0.5 (fires on the domain residues) AND reasoning partner labeled on-concept (auto-interp).
**13/20 pairs pass both** — e.g. F18393 kinesin (F1 0.98) → F15673 "Motor Function", F4888 collagen (0.92) →
F29088 "Collagen Binding", F16964 LRR (0.80) → F4783 "Leucine-rich repeat", F11836 histone (0.63) → F21642
"H2A-H2B Dimer". Caveat: **GPCR bio features have LOW domain-F1 (0.22–0.51)** — that alignment is weak on the
bio side despite high r.

**Significance:** the SAE basis is **concept-localized across modalities** (13/20 pairs, both sides) — a real,
tightened representational result. **But** it is *correlational, prompt-mediated*, **not causal and not feature-
level fusion** (§6 proves this).

## 6. Steering — causal tests

- **L30 synapse cluster: 67% genuine concept injection**, feature-specific (matched-load control 0%), LLM-judged
  → reasoning features causally **write** their concept.
- **Answer-change: NEGATIVE** — steering shifts the reasoning *language*, not the final GO prediction.
- **L32 dose-controlled: 0%** (p95 per-feature dosing) → **L30 is the steering sweet spot**, earned not assumed.
- **Cross-modal 2×2 (v2, DECISIVE — cluster writer + concept-absent proteins):** the reasoning cluster now
  works as a writer, cleanly isolating the residue effect. **P450: reason main-effect +49** (29 coherent
  cytochrome/heme words vs 3 baseline), **residue main-effect −3 (≈0)**; **GPCR: reason +4, residue +0**. →
  **the reasoning feature causally writes the concept; the structure/residue feature is causally inert.** The
  cross-modal alignment (§5) is prompt-mediated, NOT a causal structure→reasoning flow. (D12/D13_*_2x2_cluster)
- **S2 — does a domain-F1-LOCALIZED structure feature rescue causality? NO.** Injecting F18393 (kinesin,
  domain-F1 **0.98**) on residues: residue main-effect **+2** (noise, 12 proteins); F13950 (RNA-binding, 0.96):
  **+0**. So it's not that F7369 was poorly localized — **even a perfectly localized structure feature does not
  feed reasoning.** The read/write dissociation is fundamental. (D14/D15_*_2x2_localized)

---

## The narrative in one paragraph

**The SAE's value is interpretability, not decodability** — it never beats raw on decoding, but isolates clean
nameable features. **Structure** is strongly monosemantic yet only ~17% of features are genuinely domain-localized
(domain-F1). **Function (GO)** is weakly encoded in residues; its features are weaker. **Reasoning** has separable
elaboration-vs-restatement features, microbial-defense concepts are cleanly decodable above leakage, and reasoning
features causally write concepts but don't change the answer. **Cross-modally**, structure and reasoning share
concept-aligned features (correlational, prompt-mediated) with no feature-level fusion; whether there is any causal
structure→reasoning flow is what the 2×2 decides.

## Honest negatives / retractions (kept visible on purpose)
- "Function emerges in reasoning (~0.95)" — **RETRACTED**: leakage (random projection scores the same).
- Per-protein AUROC overstates structural localization — corrected by domain-F1 (only ~17% localized).
- Cross-modal alignment is **not** causal/fusion — population correlation, prompt-mediated.
- Steering does **not** flip the GO answer (concept-injection, not goal-redirection).
- Feature-biology AUROC is selection-biased (best-of-terms-and-features).

_Scripts: `feature_biology_table.py`, `domain_f1.py`, `probe_v2.py`, `probe_trained.py`, `echo_synthesis_probe.py`,
`crossmodal_pairing.py`, `steer_generation.py`, `steer_crossmodal.py`, `enrichment_probe.py`, `gsea_enrichment.py`.
Companion docs: `BIO_PROBES.md`, `RESULTS_REVIEW.md`, `OVERNIGHT_RESULTS.md`._
