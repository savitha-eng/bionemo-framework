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

Per-feature AUROC(activation | synthesis-token vs echo-token), where a reasoning token is **synthesis** if its
word is novel (not in the given InterPro/GO annotations) and **echo** if copied. Label is a property of the
reasoning *process*, not the GO answer → **non-circular**. Shuffle-null max 0.506.
- **Synthesis features** (reason beyond the prompt): F39979 (0.71, transmembrane/clathrin), F39744 (0.71,
  sec61/translocation), F5147 (HSP70), F14759 (kinase cascade), F7099 (MEK/ERK), F11654 (Ca²⁺).
- **Echo features** (restate given IDs): F34302/F37366 (fire on IPR IDs), F20205 (GO IDs).
- Now also has a **trained L1 probe** for restatement-vs-elaboration decodability + signed feature selection.

## 5. Cross-modal alignment — two senses (this is where "co-fire" lives)

**Sense A — cross-*feature* pairing** (`crossmodal_pairing`): a bio residue-feature's per-protein activation
vector Pearson-correlated (across 8k proteins) against every reasoning feature's per-protein vector.
- **20/20 bio features have a concept-matched reasoning partner at r ≥ 0.4**; permutation-null max-r 0.07–0.25.
- GPCR F7369↔F3184 (r=0.89), P450 F3623↔F13384 (0.87), RNA-binding F11025↔F10235 (0.86), LRR F16964↔F4783 (0.81).
- Multiple independent bio features for the same fold **converge on one reasoning feature** (4 GPCR features → F3184).

**Sense B — same-*feature* co-firing** (SAE-V φ / Eq.7): does ONE feature fire in both bands? **NULL.**

**Significance:** the SAE basis is **concept-organized across modalities** (parallel handles for "GPCR-ness" in
residues and in text) — a real representational result. **But** it is a *population correlation, prompt-mediated*
(both features are children of the annotation the prompt hands the model), **not causal and not feature-level
fusion.** The 2×2 steering (§6) is the decisive causal test of this.

## 6. Steering — causal tests

- **L30 synapse cluster: 67% genuine concept injection**, feature-specific (matched-load control 0%), LLM-judged
  → reasoning features causally **write** their concept.
- **Answer-change: NEGATIVE** — steering shifts the reasoning *language*, not the final GO prediction.
- **L32 dose-controlled: 0%** (p95 per-feature dosing) → **L30 is the steering sweet spot**, earned not assumed.
- **Cross-modal 2×2 (v1, single features):** residue-injection main-effect ≈ 0 → structure causally inert — BUT
  a *null-on-null* (the single co-firing reasoning feature was also a weak writer). **v2 running:** reasoning arm
  = co-firing **cluster** (Q1) on concept-**absent** proteins (Q2) — the decisive version. _[verdict pending]_

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
