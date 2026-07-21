# Microbial-defense reasoning features — results page (L30)

_The flagship non-leaked reasoning result. The model reasons about antimicrobial defense in text, using a
distributed set of features with pathogen-specific detectors on a shared innate-immune core. Everything here is
reasoning-band (protein band has no clean defense feature — see §5). All L30, train split._

---

## 1. The probe: only 4 of 12 GO concepts are real (the rest leak)
Designed-label linear probe (`probe_trained.py`): per GO concept, mean-pool **reasoning-band** SAE activations
per protein → logistic regression on 4 representations. The **random-SAE = leak floor** (the text names the
function, so any projection decodes it). **Only the margin `sae_sparse − random` is trustworthy.**

| concept | sparse | leak floor | **margin** | verdict |
|---|---|---|---|---|
| **defense → fungus** | 0.993 | 0.882 | **+0.111** | ✅ real (cleanest) |
| **defense → bacterium** | 0.993 | 0.885 | **+0.108** | ✅ real |
| structural molecule | 0.985 | 0.922 | +0.063 | ✅ real |
| plasma membrane | 0.978 | 0.923 | +0.055 | ⚠️ real but diffuse (362 feats) |
| reproduction / catalytic / mitochondrion / nucleus / kinase / transporter / oxidoreductase / sexual-repro | ~0.98 | ~0.95–0.98 | +0.01–0.03 | ❌ **leakage** |

## 2. Defense is DISTRIBUTED (not one feature) with a shared core
All real concepts: decoder-cosine ~0 (distinct directions), low co-firing → distributed. Feature count scales
with breadth (antifungal 12 → plasma-membrane 362). Cross-concept overlap is biologically sensible:
- **fungus ∩ bacterium = {F2808, F32785, F35336}** = shared innate-immune core.
- The two probe *designs* (reasoning-only vs combined) agree on {F2808, F5047, F7665} — **consistent**.

## 3. Auto-interp of the defense features (reasoning-band, accession-stripped)
Labels = LLM reading each feature's top-firing **reasoning** windows (ignoring cited GO IDs = echo):

**Pathogen-specific detectors:**
- **F23726 — "Filamentous fungi"** (antifungal)
- **F22077 — "Bacterial lipopolysaccharide"** (LPS = the canonical bacterial signature)

**Shared innate-immune core (fungus ∩ bacterium):**
- **F35336 / F2808 / F32785 — "Defense Response / Innate Immune Response"**

**Mechanism features:** F7665 "Cell wall adaptation" (fungal target), F15775 "Autophagy" (antimicrobial),
F38712 "Leukocyte activation".
**Generic / auto-interp misses:** F2082 chromatin, F28215 protein-localization, F5047 cell-motility (~⅓ of labels).

## 4. These are GENUINE reasoning, not echo (the band matters)
Split by band, F36488/F35336/F32785 show:
- **Prompt band = ECHO** — fires on given `GO:0009620 (response to fungus)` accessions.
- **Reasoning band = SYNTHESIS** — *"Plant immunity hinges on redox gating of pattern-recognition receptors"*,
  *"growth inhibition of H. sativum, Verticillium albo-atrum"* (antifungal activity vs named fungi), *"type I
  interferon, the most specific immune pathway"* — mechanisms **beyond** the given annotations.
- **Answer band = RESTATEMENT** — back to GO-ID lists (less genuine reasoning).
→ Interpret the **reasoning band only**; the earlier "just echoes GO IDs" read was a band-pooling artifact.

## 5. The protein (bio) band has NO clean defense feature
A **combined** protein+reasoning probe (`concatenate` both bands) decodes defense at 0.978 recruiting 15
protein-band + 22 reasoning-band features — but the 15 protein-band ones **do not enrich for any defense GO
term** (best is circadian rhythm, FDR ~0.6, not significant). So the protein band has only weak spurious
correlates, **no residue-level defense detector.** Consistent with the thesis: **function (defense) is weak in
the residues; structure is strong.** Domain-F1 is N/A here (defense is a function, not a domain with residue
spans — unlike F18393 kinesin, F1 0.98).

## 6. Honest bottom line
- ✅ The model **genuinely reasons about antimicrobial defense in text** — a distributed feature set with
  pathogen-specific detectors (fungi / LPS) on a shared innate-immune core, cleanly above the leak floor,
  consistent across probe designs.
- ⚠️ It is a **reasoning-band** phenomenon — **no residue-level defense feature** (function is weak in residues).
- ⚠️ ~⅓ of the distributed features are generic/co-occurring correlates (a sparse probe selects *predictive*
  features, not features that *mean* the concept); the defense-labeled subset is the genuine signal.
- This is the **Phase-2 flagship example notebook**: leak-floor discipline (4/12 real), distributed representation,
  pathogen-specific + shared core, per-band interpretation.

_Features: `REASONING_CONCEPT_FEATURES.md` · probe: `probe_trained.py` · labels: `autointerp_microbial_defense.json`._
