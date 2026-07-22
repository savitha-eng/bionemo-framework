# Master validation checklist (L30) — validate everything, fast

_The ONE page to keep open. Covers every result type in validation order: (A) probe results → (B) features
within the probes → (C) combined "cross-modal" probe → (D) structural bio → (E) cross-modal pairs →
(F) synthesis-vs-echo. Deeper detail is in the linked docs, but the IDs + how-to-check live here._

**Setup.** Dashboard `?model=l30_balanced` at `localhost:5177`; NIM auto-interp on `5199` (the 🔍 button).
Forward both: `ssh -L 5177:localhost:5177 -L 5199:localhost:5199 <pod>`.
**Golden rule: trust rank / AUROC / domain-F1 / leak-margin — NOT magnitude.** Protein-band features are all
low-magnitude (~0.3) by design; magnitude ≠ quality.

**The one discipline that governs everything: the LEAK FLOOR.** The reasoning text often *names* the function,
so a random projection already decodes it. We train a probe on a random-init SAE = the leak floor. **Only the
margin `sae_sparse − leak` is trustworthy.** A high raw AUROC means nothing on its own.

---

## A. Probe results — which concepts are REAL (start here)
The 12-concept reasoning probe (`probe_reasoning_l30_fair.json`). Margin ≥ 0.05 = clears the leak floor.
**Only 4 of 12 are real:**

| concept | trust | #feat | margin | validate by |
|---|---|---|---|---|
| **defense → fungus** | ✅ | 12 | **+0.111** | auto-interp the feats → defense/fungal (§B below) |
| **defense → bacterium** | ✅ | 36 | **+0.108** | auto-interp → immune/LPS |
| structural molecule | ✅ | 73 | +0.063 | localization on collagen/keratin feats |
| plasma membrane | ✅ | 362 | +0.055 | ⚠️ real but diffuse (362 feats = weak) |
| nucleus / catalytic / sexual-repro / reproduction / oxidoreductase / mitochondrion / **kinase** / transporter | ❌ leak | 80–386 | +0.006…+0.035 | **don't trust** — match a random projection |

**What to check:** the 8 ❌ concepts are leakage — if you auto-interp their features you'll see GO-term echo,
not mechanism. The 4 ✅ are the only ones worth your time. Full feature-ID lists: `REASONING_CONCEPT_FEATURES.md`.

---

## B. Features WITHIN the real probes — rank by SCORE, not by auto-interp label · **reasoning band**
The probe's nonzero-coef features. **Scoring (trustworthy) ≠ labeling (a hypothesis)** — see the box below.
Rank each feature by its **label-grounded per-feature AUROC** (predicts the GO concept); use auto-interp only
as a naming hint and confirm by reading the windows.

**defense → fungus (12 feats)** — AUROC-ranked (random feature ≈ 0.50):

| tier (per-feature AUROC) | features | meaning |
|---|---|---|
| **GENUINE fungal** ≥0.94 | F2808 (0.99), F35336 (0.99), F32785 (0.99), F23726 (0.98), F36488 (0.94) | individually predict fungal defense = the real core + detectors |
| weak / correlate 0.6–0.72 | F7665 cell-wall, F22156, F5047 motility | marginal |
| co-occurring ~0.5–0.59 | F2082 chromatin, F15775 autophagy, F29332, F28215 | NOT fungal — L1 picked them as co-predictors |

**defense → bacterium (36 feats, top):** `35336, 38712, 36769, 8824, 2724, 32785, 20730, 22077, 13479, …`
**fungus ∩ bacterium = {F35336, F2808, F32785}** = shared innate-immune core (biologically sensible).

> **SCORING vs LABELING — the discipline that governs §B and §C.**
> - **Score (trust it):** supervised, GO-label-grounded. (i) L1 probe coef = helps *predict* the concept in
>   combination; (ii) per-feature AUROC = *individually* means the concept. F23726 = 0.975 vs random ≈ 0.50.
> - **TWO label systems, both weak — neither is the score:**
>   - **Static dashboard `label` = coarse GO-SLIM enrichment** (NOT auto-interp). Only ~59 distinct terms over
>     10,579 features → collapses onto giants: "catalytic activity" ×875, "regulation of gene expression" ×323,
>     **"none" ×6122 (58%)**. The slim vocab lacks specific terms, so F23726 (fungal 0.975) → "regulation of gene
>     expression (0.69)", and F36488 (fungal 0.943, gorgeous synthesis) → **"none"**. Treat the static label as
>     nearly useless for specific concepts.
>   - **Auto-interp label (🔍 button, live) = one unsupervised LLM read of top windows.** A hypothesis — it
>     **missed** F23726 ("Amino acid transport", but 6/8 windows are *Aspergillus/filamentous fungi/H. sativum*)
>     and **oversells** plausible correlates (F15775 "Autophagy" *sounds* antimicrobial, AUROC 0.572 = co-occurring).
> - **Rule:** rank by AUROC (score) → read the windows → use auto-interp as a name hint. Ignore the static slim label.
> - **Fix in flight:** replace the go-slim label with `feature_biology_table.py`'s specific GO+IPR enrichment
>   (AUROC-scored), gate at AUROC≥0.75 + term-size filter, fall back to auto-interp when "none".

**What to check:** the top-5 GENUINE features on the reasoning band show mechanism ("antifungal activity vs
*H. sativum*", "redox gating"); flip to the **prompt band** → just GO-ID echo. The bottom-4 co-occurring ones
(AUROC ~0.5) are what "a sparse probe selects *predictive* features, not features that *mean* the concept" means.

---

## C. Combined ("cross-modal") probe — does it recruit REAL bio features? · NEW, just validated
Concatenate `[protein-band ; reasoning-band]` features → one L1 probe on a concept. It recruits from BOTH
bands. **The question: are the protein-side recruits genuine, or spurious?** Validated for defense→fungus
(`crossmodal_probe_defense_ids.json`), AUROC 0.978, **16 protein + 20 reasoning** features:

- **PROTEIN side = SPURIOUS.** Only **1/16 localizes** (F25284 ion-transport, domF1 0.80). The rest enrich for
  *generic* domains that merely co-occur in defense proteins — F28287 kinase (domF1 0.33), F33248 trypsin
  (0.18), F15988 DNA-binding (0.00) — and **none enrich for any defense term.** → the protein band has **no
  clean defense detector.** (Consistent: function is weak in residues, structure is strong.)
- **REASONING side = GENUINE.** Auto-interp confirms defense: F36488/F2808 "Defense response to fungus",
  F35336 "Defense Response", F32785 "Immune Response" (`autointerp_defense_probe_reasoning.json`).

**What to check:** run domain-F1 on any combined-probe's protein recruits → if they don't localize AND don't
enrich for the concept, they're co-occurrence artifacts. **The combined probe finds the concept in the
REASONING band; its protein-band recruits are noise.** (This is why the pairing and probe disagree — §E.)

---

## D. Structural bio features — the most trustworthy · **protein band**
Real residue-level localization. Filter the atlas by `localized = true` (148 features).

| Feature | Concept | domain-F1 | check |
|---|---|---|---|
| **F18393** | kinesin motor | **0.98 📍** | gold standard — fires *inside* the motor domain |
| **F7369** | GPCR (IPR000276) | 0.51 ⚠️ | moderate — ~half the firing in-domain |
| **F4647** | (oversell demo) | **0.00 ❌** | AUROC 0.98 but fires *outside* domain — why domain-F1 exists |

**Check:** F18393 green 📍 badge + residue span aligns with domain; F4647 red badge despite high AUROC.

---

## E. Cross-modal PAIRS — correlational, not causal · **both bands side-by-side**
`CROSSMODAL_PAIRS.md` (20 pairs). One bio feature ↔ its single best-correlated reasoning partner.

| bio (protein) | ↔ | reasoning | r |
|---|---|---|---|
| **F18393** kinesin | ↔ | **F15673** | 0.62 |
| **F7369** GPCR | ↔ | **F3184** | 0.89 |

**Read as** co-firing across proteins ("model talks about motors when the motor detector fires"), **not** a
causal path (2×2 confirmed prompt-mediated). ⚠️ **Pairs ≠ combined-probe lists** — different questions:
- **Pairing** = *unsupervised* Pearson corr of per-protein activation vectors → one best partner. "Which
  reasoning feature co-fires with THIS bio detector."
- **Combined probe** (§C) = *supervised* L1 on a concept label → distributed predictive set. "Which features
  decode THIS concept." Sparsity knob = `C` (lower = fewer). Empirically the two **disagree** (kinesin pairing
  features were NOT recruited by the kinesin probe) — expected, they answer different things.

---

## F. Synthesis vs echo — weakest; SAE ≈ raw, NO clean feature · **reasoning band**
Echo/synthesis is real (~0.93 separable) but **diffuse in the residual stream — the SAE doesn't localize it**
(raw 0.916 ≈ SAE 0.914; best single feature only 0.713). Review as "mechanism-talk vs ID-echo," not "the
synthesis feature." (`echo_synthesis_l30.json`)

- **Echo / restatement** (GO/IPR accession dumps): F15558, F23456, F32788.
- **Synthesis / elaboration** (mechanism + relational language): **F23543** ("modulates, catalyze, perturbs,
  dampens, chaperone"), **F38531** ("protein–protein, RNA–protein, membrane–cytoskeleton").
- **Best defense-synthesis exemplar: F36488** (fungal AUROC 0.943) — genuine plant-immunity mechanism:
  "redox gating of pattern-recognition… potentiate defense circuits against fungal invasion", "GCC-box occupancy
  on pathogenesis-related promoters integrates ethylene with salicylic-acid and jasmonic-acid pathways", "AP2/ERF
  control over defense-associated promoters → defense response to fungus". Score + windows both say genuine
  fungal-defense synthesis — yet its static label is "none/text-heavy" (the labeling failure, not the feature).
- Individually-interpretable synthesis-leaning **biology** (real as biology, weak as synthesis, AUROC ~0.56–0.60):
  F11654 Calcium, F16620 Helicase, F35498 Dynein, F29616 Ig-fold, F423 Response-regulator, F36032 Axon, …

**Note (auto-interp fix):** span-firing features (e.g. **F10235** RRM: "RNA recognition motif … β-α-β
topology … α-β plait superfamily") were mislabeled by the old ±window auto-interp. The server now uses
**span-aware windows** (marks the full active phrase) — re-click 🔍 to get the phrase-level label.

---

## Where each result lives (if you need depth beyond this page)
| result | doc / file |
|---|---|
| 12-concept probe + all feature IDs | `REASONING_CONCEPT_FEATURES.md` |
| microbial-defense flagship (full write-up) | `MICROBIAL_DEFENSE_RESULTS.md` |
| combined-probe validation (IDs) | `crossmodal_probe_defense_ids.json`, `autointerp_defense_probe_reasoning.json` |
| cross-modal pairs (all 20 + domain-F1) | `CROSSMODAL_PAIRS.md` |
| structural domain-F1 (872 feats) | `domain_f1_l30.json` |
| synthesis vs echo probe | `echo_synthesis_l30.json` |
| bio enrichment (GO+IPR, per-feature AUROC) | `feature_biology_table.json` |

**Validation order if time-crunched:** A (which concepts real) → B (auto-interp the 4 real ones) → D (click
F18393 vs F4647 to see domain-F1 work) → C (protein-side spurious) → E/F (skim, know the caveats).
