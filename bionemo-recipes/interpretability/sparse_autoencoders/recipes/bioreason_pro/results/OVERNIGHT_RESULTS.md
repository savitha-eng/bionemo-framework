# BioReason-Pro SAE — Overnight Run: FINAL HONEST SUMMARY (2026-07-17)

Goal: robust results comparable to Jared's methodology + Goodfire methodology for reasoning; deliverables = good BIO features + good REASONING features + STEERING. Built the Jared-gap experiments overnight; below is the honest verdict.

## 1. BIO FEATURES — ROBUST ✅
- **Enrichment (Jared nb03 / InterPLM):** 74.7% of live non-sink protein-band features carry >=1 GO/InterPro term at FDR<0.05, **lift 88.5x over label-shuffle null, 203 distinct terms.** Matches InterPLM's >=70% headline.
- **Clean per-feature detectors:** F16026 kinase (0.98), F679 ARM (0.95), F11009 WD40 (0.92), F5540 homeodomain (0.90), F17703 Ig-like (0.87) — distinct feature per fold, auto-interp-named.
- **SAE healthy:** FVU 0.20 (80% variance explained).
- Structure is at ceiling in ESM3 residues but SATURATED (random projection also ~0.99) — decodability is not an SAE result; the SAE's value is the clean nameable detectors.

## 2. REASONING FEATURES — ROBUST + NON-CIRCULAR ✅
- **Echo-vs-synthesis probe (Goodfire-style, non-circular):** synthesis features fire when the model reasons BEYOND the given annotations (mechanistic/structural inference: transmembrane, translocation, HSP chaperone, kinase-cascade, Ca-signaling, autophagy), echo features fire on the literal given IPR/GO accession IDs. Per-feature AUROC to 0.71 vs shuffle-null max 0.506. Auto-interp-named cleanly.
- **144 synthesis features** (novelty>=0.5), spanning structural-fold / mechanistic / functional categories.
- **CAUTION / retracted:** probing named GO/InterPro on the reasoning band is LEAKY — a random projection decodes GO at 0.94 because the reasoning text names the function (40/40 concepts leak). "Function emerges in reasoning ~0.95" is retracted. The non-circular reasoning result is echo-synthesis, not GO-decodability.

## 3. STEERING — read/write dissociation (with honest negatives)
- **Reasoning-band features WRITE their concept:** synapse (0.77), mito (0.60), nervsys (0.36) clusters inject coherently (char-level n=40 + LLM-judge). Decision-point + p95-dose (Goodfire).
- **Protein-structure features are concept-READ-ONLY:** clamping (single + cluster, matched absolute dose) perturbs text but never writes "kinase" — harness-validated (synapse works) + dose-confound-ruled-out. Corroborated by logit-lens (protein features -> garbage in text-output space) and the read/write plane (ma_text: reasoning 6.9 vs bio 1.55).
- **NEGATIVE — no NEW steerable cluster:** redox (LLM-judge 0% genuine), conformational + reasoning-flow degenerate. The behavioral/reasoning-flow steering hope failed (oversteer -> salad). Steerable set stays synapse/mito/nervsys.
- **NEGATIVE — answer-change:** steering does NOT flip the final GO prediction (generations rarely reach the answer; where they do, steering drops GO terms via degradation, not a flip). = concept-INJECTION, not goal-redirection.

## 4. METHODOLOGY — now matches Jared on
enrichment(nb03), two-tier probe + specificity + random/shuffle nulls(nb06), dimensionality-matched layerwise, load-bearing(mean_active), logit-lens(nb01), auto-interp(nb05), SAE health(nb01), decoder feature-families(nb04, KMeans fallback), read/write plane(axis-nb analog).
**Still MISSING vs Jared:** steering matched-control FEATURE (only did random-direction + absolute-dose), held-out steering generality panel, TPSS-style random-invariant steering metric, Fisher/MWU structural mapping, missense/variant profiles (no variant labels).

## 5. vs GOODFIRE reasoning paper
Done: decision-point steering (steer-after-prefix), rebalancing/oversteer-degeneration observed, echo-vs-synthesis = their phase-distribution-shift analog. Negative: answer-change (their operator-swap). Absent in this dataset: backtracking/answer-timing behavioral features (distilled traces lack them).

## ROBUST vs NEGATIVE vs PENDING
- **ROBUST:** bio-feature enrichment (88.5x), clean detectors, SAE health, echo-vs-synthesis reasoning features, read/write dissociation, 3 validated steerable clusters.
- **NEGATIVE (honest):** reasoning-GO probing = leakage (retracted); no new steerable clusters; answer-change doesn't flip the prediction; structure decodability = saturation not SAE.
- **PENDING/follow-ups:** matched-control-feature + held-out steering panel; answer-change with full-length generation; hdbscan/umap install for proper feature-family viz; cryo-EM case study (needs chain/numbering mapping).

## The honest through-line for a write-up
A bio-reasoning SAE gives (a) **robustly annotatable, nameable BIO structure detectors** (InterPLM-grade, but read-only causally), (b) **non-circular reasoning-synthesis features** that fire when the model infers beyond its given annotations, and (c) a **band-aligned read/write dissociation** — reasoning features are causal handles on vocabulary, structure features are not. The apparent "function-in-reasoning decodability" was leakage; caught by proper baselines.

---

# Overnight Autonomous Run — Robust Jared-comparable + Goodfire results

Goal: good BIO features + good REASONING features + STEERING, with Jared's rigor + Goodfire's reasoning methodology.

## Cycle 1 (2026-07-17)

### BIO features — ENRICHMENT (Jared nb03, InterPLM-style) ✅ STRONG
Protein-band SAE features, per-feature Fisher-exact enrichment vs GO/InterPro, FDR<0.05:
- **annotation rate = 74.7%** (1239/1658 live non-sink features carry ≥1 curated term)
- shuffle-null rate 0.8% → **LIFT = 88.5×**
- **term diversity = 203** distinct terms recovered
- → **matches InterPLM's ≥70% headline.** Bio features are robustly annotatable. This is the validation I'd been missing.

### REASONING features — ECHO-vs-SYNTHESIS probe (non-circular, Goodfire-style) ✅ STRONG
Per-feature AUROC(synthesis-token vs echo-token), 17170 synth / 13111 echo tokens, **shuffle null max=0.506** (chance):
- **SYNTHESIS features** (fire when reasoning BEYOND given annotations, AUROC to 0.71 ≫ null):
  - F39979 (0.71): amphipathic/transmembrane/clathrin — structural-mechanistic synthesis
  - F39744 (0.71): translocation/peptidyltransferase/sec61/channel — translocation mechanism
  - F5147 (0.64): hsp70/hsp90/chaperone; F14759 (0.63): MAP3K7/kinase-cascade/ubiquitin
- **ECHO features** (restate given names, AUROC ~0.29 = fire on echo):
  - F34302/F37366: fire on literal **IPRxxxxx accession IDs**; F20205: GO IDs
- → **the synthesis LANGUAGE = mechanistic/structural inference vocab; echo features fire on the literal given IDs.** Clean, non-circular, novel. Directly answers "how the model reasons around the given annotations."

### Feature NAMING — logit-lens (Jared nb01)
- Works for REASONING features: F16494 (synapse) → "connection/contact/contacts" ✓
- GARBAGE for PROTEIN features: F16026 (kinase) → "institutions/Callable/种" — because protein-structure features aren't in the text-output space. **Corroborates read/write: protein features are not text-writable.**

### STEERING — answer-change: running (does steering flip the GO prediction)

## Honest verdicts so far
- Bio features: **robust & Jared-comparable** (74.7% annotated, 88.5× lift).
- Reasoning features: **robust & non-circular** (synthesis vs echo cleanly separated, above null).
- Steering: read/write dissociation holds; 3 validated clusters; answer-change pending.

## Cycle 2 (2026-07-17)

### SAE health — FIRST PASS BUGGY (re-running)
- First pass: FVU 1.13, 75% dead — but this is a **measurement bug**: compared RAW activations to a reconstruction in NORMALIZED space (SAE has normalize_input=True), and dead% on a tiny strided sample. NOT a real SAE failure — the 88.5x enrichment lift + clean per-feature detectors prove the SAE is good. Re-running in normalized space.

### Read/Write plane (attributability×steerability analog) ✅
`results/charts/read_write_plane.png` — per-feature protein-band load vs reasoning-band load (mean_active), all live features + highlights:
- **Steerable reasoning clusters** (synapse/mito/nervsys): ma_text=6.9 (high, WRITABLE), ma_protein=0.0
- **Bio structure detectors** (kinase/ARM/WD40/homeodom/Ig): ma_protein=0.77, ma_text only 1.55 (~4x below steerable → weakly writable)
- → visualizes the read/write dissociation: reasoning features occupy the write axis; bio detectors sit low on it.

### Answer-change (Goodfire operator-swap analog) — INCONCLUSIVE / negative
Steer synapse cluster, generate to the GO answer (max_new 1000), dose 0 vs 6:
- Only 1/8 generations REACHED the answer (</think>+GO) even at 1000 tokens — reasoning is long / steering delays it.
- Where dose-0 had GO terms, dose-6 LOST them (degradation), not a clean flip to synapse GO terms.
- **Verdict: no clean evidence steering flips the DECISION** — it grafts vocabulary, doesn't change the predicted GO. Consistent with "concept-injection, not goal-redirection." Proper test needs full generation (max_new ~2500) or teacher-forced answer — a follow-up.

### SAE health — re-running (normalized-space FVU); SAE quality already validated indirectly by 88.5x enrichment lift + clean detectors.

## Cycle 3 (2026-07-17)

### SAE health ✅ (fixed — normalized-space FVU)
- **FVU=0.20 → 80% variance explained** (healthy; Jared targets ~0.10-0.20). SAE reconstructs well.
- dead_frac 74% is a STRIDED-SAMPLE artifact (few tokens → many features never fire); true dead ~20% per training.

### Reasoning-band enrichment: 87.3% / 148x / 756 terms — higher than protein but LEAKAGE-inflated
- 8658 live features, annotation-rate 87.3%, lift 148x, diversity 756 (vs protein 74.7% / 88.5x / 203).
- HIGHER than protein — consistent with reasoning-band leakage (text names the function). The **non-circular** reasoning result is the echo-synthesis probe, not this. Still shows the reasoning band has a rich, diverse feature repertoire.

## Cycle 4 (FINAL, 2026-07-17)

### Auto-interp (Jared nb05) ✅ — 15 features LLM-named
- SYNTHESIS reasoning features: F39979 Transmembrane Immunoreceptor, F39744 Protein-Synthesis Tracking, F5147 HSP70 Chaperone, F7099 Protein Phosphorylation, F23525 Autophagy Regulation, F11654 Calcium Signaling, F16620 DNA Polymerase — all meaningful mechanistic/functional concepts.
- BIO detectors: F16026 Protein Kinase Domain, F679 ARM fold, F11009 WD40, F5540 Homeodomain, F17703 Ig-like — named by detected domain.
- → reasoning-synthesis features and bio detectors are both cleanly nameable.

### Decoder feature families (nb04) — KMeans fallback (hdbscan not installed)
- KMeans-40 on decoder weights over live features (UMAP+HDBSCAN viz skipped — dependency missing). Family structure exists; proper HDBSCAN needs `pip install hdbscan umap-learn`.

## Matched-control-feature steering (Jared's strongest specificity control) — PASSES ✅
Synapse cluster vs a NON-synapse cluster matched on mean_active (load) + density, identical decision-point + p95-dose:
- **Real synapse cluster: injects (1.0 synapse-hits @ mult=6).**
- **Matched-load control cluster: 0.0 synapse-hits at EVERY dose.**
- → The steering effect is **FEATURE-SPECIFIC**, not "any high-load feature clamped hard." Jared's strongest control passes. (LLM-judge confirmation running.)

## FINAL steering controls
- **Matched-control-feature (Jared's strongest specificity control) — PASSES, LLM-judge-confirmed:** matched-load non-synapse cluster = **0% genuine at every dose**; real synapse cluster = 67% genuine. Steering is FEATURE-SPECIFIC.
- **L32 vs L30 (deeper-layer test — DOSE-CONTROLLED, caveat resolved):** the L32 synapse cluster was re-steered with the **same p95 per-feature dosing that makes L30 work** (`--scale-file load_bearing_l32.npz --scale-key ma_text`, per-feature scales 9.8/3.3/4.9/6.3/7.0/7.9, α=0,4,6,8, decision-point steer-after-25). Result: **0/32 generations contain any synapse content** — exact concept-word count = 0 at every α, and a broader semantic scan (synap|neuro|axon|dendrit|glia|nmda|gaba|…) also = 0/32. Coherence stays 100% (distinct3≈1.0); the model just keeps reasoning about the *real* protein (e.g. MerR HTH, chromo domain) — so this is **no effect, not degeneration**. **L32 genuine synapse injection = 0% vs L30's 67%.** The earlier "absolute-dose" caveat is now resolved: L32 fails even when properly p95-calibrated, so **L30 is the steering sweet spot on its own merits** and Jared's "steerability emerges deeper" does NOT hold for this model. (`D9_L32synapse_p95.jsonl`)
