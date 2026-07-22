# Dashboard review checklist (L30) — what to click, in order

_One page to keep open beside the dashboard. Model: `?model=l30_balanced` (the 8k build's
`feature_examples` is still rendering — stay on `l30_balanced`). Servers: dashboard `localhost:5177`,
NIM auto-interp `5199` (needed for the 🔍 Auto-interpret button)._

**Golden rule: trust rank/AUROC/domain-F1, NOT magnitude.** Protein-band features are all low-magnitude
(~0.3) by design — magnitude ≠ quality.

---

## Tier 1 — Structural bio features (strongest; review first) · **protein band**
Real residue-level localization. Trust the **📍 domain-F1 badge**, not magnitude. Filter the atlas by
`localized = true` (148 features) to see the whole set.

| Feature | Concept | domain-F1 | what to check |
|---|---|---|---|
| **F18393** | kinesin motor | **0.98 📍** | gold standard — firing residues sit *inside* the motor domain |
| **F7369** | GPCR rhodopsin-like (IPR000276) | 0.51 ⚠️ | moderate — real domain, but only ~half the firing is in-domain |
| **F4647** | (AUROC-oversell demo) | **0.00 ❌** | AUROC 0.98 but fires *outside* the domain — *this is why domain-F1 exists* |

**Do:** open F18393 → confirm 📍 green badge + residue span aligns with domain. Then open F4647 → see the
red "not-localized" badge despite high AUROC. That contrast justifies the whole metric.

---

## Tier 2 — Microbial-defense reasoning (flagship; reasoning-band only) · **reasoning band**
Switch to the **reasoning band**, then hit 🔍 Auto-interpret (accessions are auto-stripped so it labels the
*reasoning*, not the ID-echo).

**Pathogen-specific detectors:**
- **F23726** — "Filamentous fungi" (antifungal)
- **F22077** — "Bacterial lipopolysaccharide" (LPS)

**Shared innate-immune core (fungus ∩ bacterium):**
- **F35336 / F2808 / F32785** — "Defense / innate immune response"

**Do:** on the reasoning band, confirm the windows show genuine mechanism ("antifungal activity vs *H.
sativum*", "redox gating") — **not** just `GO:xxxx` lists. Then flip the *same* feature to the **prompt
band** → you'll see it's just GO-ID echo. That per-band contrast is the finding. (Answer band = restatement.)

---

## Tier 3 — Cross-modal pairs (correlational; not causal) · **both bands, side by side**
Open a bio feature and its reasoning partner together. Full list in `CROSSMODAL_PAIRS.md`.

| bio (protein band) | ↔ | reasoning partner | r |
|---|---|---|---|
| **F18393** kinesin | ↔ | **F15673** | 0.62 |
| **F7369** GPCR | ↔ | **F3184** | 0.89 |

**Read as:** "the model talks about motors when the motor detector fires" — co-firing across proteins, **not**
a causal path (2×2 confirmed prompt-mediated). ⚠️ Note: these pairs do **not** match the combined-probe
feature lists — different question (co-firing vs prediction).

---

## Tier 4 — Synthesis vs echo (weakest; SAE ≈ raw, NO clean feature) · **reasoning band**
Echo/synthesis is real (~0.93 separable) but **diffuse in the residual stream — the SAE doesn't localize it**
(raw 0.916 ≈ SAE 0.914; best single feature only 0.713). So review as "mechanism-talk vs ID-echo," not "the
synthesis feature."

**Echo / restatement direction** (top words = GO/IPR accession dumps — confirm they're just IDs):
- **F15558**, **F23456**, **F32788**

**Synthesis / elaboration direction** (top words = mechanism/relational language):
- **F23543** — "modulates, catalyze, perturbs, dampens, chaperone, precedes" (mechanism verbs)
- **F38531** — "protein–protein, RNA–protein, membrane–cytoskeleton" (relational reasoning)

**Individually-interpretable synthesis-leaning biology features** (real as *biology*, weak as *synthesis*,
per-token AUROC ~0.56–0.60):
- F11654 Calcium · F16620 Helicase · F35498 Dynein motor · F29616 Ig-like fold · F29986 LDL-receptor
  · F7351 Peripheral membrane · F423 Response regulator · F36032 Axon extension · F37913 Embryo dev
  · F8875 Chloroplast envelope · F23525 P-body · F28070 Scaffold · F7096 Membrane loc · F11877 Oxoglutarate DH

**Do:** open F23543 (reasoning band) → mechanism verbs. Open F15558 → GO-ID dump. See the difference — but
remember the SAE isn't what separates them.

---

## Two methods, two questions (so you don't expect them to agree)
- **Cross-modal pairing** (Tier 3): *unsupervised* Pearson correlation of one bio feature's per-protein vector
  vs each reasoning feature's → one best partner. "Which reasoning feature co-fires with THIS bio detector."
- **Combined probe**: *supervised* L1 logistic on a concept label over `[protein ; reasoning]` features → a
  distributed predictive set. "Which features (either band) decode THIS concept." Sparsity knob = `C` (lower =
  fewer). Empirically the two **disagree** (kinesin pairing features were *not* recruited by the kinesin probe)
  — they answer different questions.

---
_Sources: `domain_f1_l30.json` (Tier 1) · `MICROBIAL_DEFENSE_RESULTS.md` (Tier 2) · `CROSSMODAL_PAIRS.md`
(Tier 3) · `echo_synthesis_l30.json` + `autointerp_synthesis_clean_l30.json` (Tier 4)._
