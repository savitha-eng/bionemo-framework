# BioReason-Pro SAE — technical results writeup (L30)

Every claim below carries its number and the source doc it came from. Numbers are **not** invented;
where a source gives only a qualitative statement it is reproduced as such.

**Model.** BioReason-Pro = ESM3 residue embeddings + a GO-graph encoder + a Qwen3-4B reasoning LLM
that predicts protein GO function via a text reasoning trace.

**SAE under study.** `sae-l30-exp16-balanced`: TopK SAE on **layer-30** Qwen3 residual activations,
expansion 16, top-k = 128, 2,560 → **40,960** features, `normalize_input=True`, multimodal (protein
residue + reasoning text tokens in one stream, per-band tagged). _(RESULTS_OVERVIEW.md; README.md)_

**Recipe sanity.** First decoder-LLM SAE recipe in `bionemo-framework`; hooks the real fused
forward. Smoke test drove **FVU 0.90 → 0.04 (≈96% variance explained) in 350 steps**. Honest
normalized-space variance-explained is **0.76–0.81** (raw-space ~0.97 is inflated by the dominant
magnitude direction; `normalize_loss` on by default: L32 dead latents 12.8%→5.3%, var-exp
0.76→0.81, loss-recovered 0.94→0.98; rescued L28 54.9%→5.1% dead). _(README.md)_

---

## 0. The two disciplines that govern everything

**Leak floor.** The reasoning text often *names* the function, so a random-init SAE already decodes
it. Train the identical probe on a random SAE = leak floor. **Only the margin `sae_sparse − leak` is
trustworthy.** _(DASHBOARD_REVIEW_CHECKLIST.md §intro)_

**Scoring vs labeling.** Trust the **label-grounded per-feature AUROC** (does one feature predict the
ground-truth GO term; random ≈ 0.50) and the L1 probe coefficient. Do **not** trust either label
system: (a) static dashboard label = coarse GO-slim enrichment; (b) auto-interp = one LLM read of top
windows = a hypothesis. **Golden rule: trust rank / AUROC / domain-F1 / leak-margin — NOT magnitude**
(protein-band features are all ~0.3). _(DASHBOARD_REVIEW_CHECKLIST.md §B)_

---

## 1. Leak-floor probe — only 4 of 12 GO concepts are real

Designed-label L1 linear probe on mean-pooled **reasoning-band** SAE activations per protein
(`probe_trained.py`). Margin ≥ 0.05 clears the leak floor. Mean across 12 concepts: **sparse SAE
0.986 > raw 0.954 > random/leak 0.946 > SAE-svd 0.938.** _(RESULTS_OVERVIEW.md §3b)_

| concept | trust | #feat | sparse | leak floor | margin |
|---|---|---|---|---|---|
| defense → fungus | REAL | 12 | 0.993 | 0.882 | **+0.111** |
| defense → bacterium | REAL | 36 | 0.993 | 0.885 | **+0.108** |
| structural molecule | REAL | 73 | 0.985 | 0.922 | +0.063 |
| plasma membrane | REAL (diffuse) | 362 | 0.978 | 0.923 | +0.055 |
| nucleus | leak | 362 | — | — | +0.035 |
| catalytic (enzyme) | leak | 386 | — | — | +0.024 |
| sexual reproduction | leak | 106 | — | — | +0.021 |
| reproduction | leak | 257 | — | — | +0.016 |
| oxidoreductase | leak | 104 | — | — | +0.016 |
| mitochondrion | leak | 159 | — | — | +0.013 |
| kinase activity | leak | 80 | 0.992 | 0.979 | +0.013 |
| transporter activity | leak | 96 | — | — | +0.006 |

_Sources: DASHBOARD_REVIEW_CHECKLIST.md §A; REASONING_CONCEPT_FEATURES.md; MICROBIAL_DEFENSE_RESULTS.md §1; RESULTS_OVERVIEW.md §3b._
Kinase at 0.992 is **leakage** (leak floor 0.979). "Function emerges in reasoning ~0.95" is
**RETRACTED** — leakage. _(RESULTS_OVERVIEW.md §Honest negatives)_

---

## 2. Scoring vs labeling — the receipts

Per-feature AUROC is GO-label-grounded; auto-interp is a hypothesis. Documented failures:

| feature | per-feature AUROC | auto-interp label | reality |
|---|---|---|---|
| **F23726** | **0.975** | "amino acid transport" (**MISS**) | 6/8 top windows = *Aspergillus* / filamentous fungi / *H. sativum* |
| **F15775** | **0.572** | "Autophagy" (**OVERSELL**) | co-occurring correlate, not a defense feature |

Static GO-slim label vocabulary: only ~59 distinct terms over 10,579 features → **"none" ×6122
(58%)**, "catalytic activity" ×875, "regulation of gene expression" ×323. Consequently F23726
(fungal 0.975) → "regulation of gene expression (0.69)"; F36488 (fungal 0.943, strong synthesis) →
"none". **Fix (in recipe):** `eval_enrichment.py` / `feature_biology_table.py` — specific GO+IPR
enrichment, AUROC-gated (≥0.75) + term-size filter, auto-interp fallback on "none".
_(DASHBOARD_REVIEW_CHECKLIST.md §B box)_

**Rule:** rank by AUROC → read windows → auto-interp is a name hint only.

---

## 3. Tier 1 — Structural features (protein band, most trustworthy)

**Feature→biology map** (`feature_biology_table.py`, scored vs UniProt/InterPro ground truth, not
model output): 1,187 features pass FDR < 0.01 → **872 InterPro-domain (structural) AUROC 0.97–1.00**;
**315 GO-term (functional) AUROC 0.68–0.89**. Function is less monosemantic than structure. AUROC is
selection-biased (best term × best feature); train/test correction drops detectors ~0.01.
_(RESULTS_OVERVIEW.md §1)_

**Domain-F1** = per-**position** precision (fraction of firing residues *inside* the domain)
× per-**region** recall (fraction of domain regions with ≥1 firing residue), over 872 structural
features, 3000 proteins (`domain_f1.py`). _(RESULTS_OVERVIEW.md §2)_

- **corr(AUROC, domain-F1) = 0.23** → non-redundant metrics.
- **Only 148 / 872 (~17%)** genuinely localize (F1 ≥ 0.5).

| feature | concept | AUROC | domain-F1 | note |
|---|---|---|---|---|
| **F18393** | kinesin motor | — | **0.98** | gold standard, fires inside motor domain |
| F33072 | kinesin motor | — | 0.95 | localized |
| F13950 | RNA-binding | 1.00 | 0.96 | localized |
| F30032 | protein kinase | 0.98 | 0.95 | localized |
| F18647 / F37746 | protein kinase | 0.98 | 0.93 | localized |
| F18162 | HLH DNA-binding | 1.00 | 0.93 | localized |
| F8277 | zinc-finger C2H2 | 0.98 | 0.93 | localized |
| F7369 | GPCR (IPR000276) | — | 0.51 | moderate |
| **F4647** | kinesin conserved-site | **0.982** | **0.00** | **AUROC oversell** — fires *outside* the domain |

_(DASHBOARD_REVIEW_CHECKLIST.md §D; RESULTS_OVERVIEW.md §2, §7A)_

---

## 4. Tier 2 — Microbial-defense reasoning circuit (flagship, reasoning band)

Distributed feature set: pathogen-specific detectors on a shared innate-immune core. All real
concepts are distributed (decoder-cosine ~0, low co-firing); feature count scales with breadth
(antifungal 12 → plasma-membrane 362). _(MICROBIAL_DEFENSE_RESULTS.md §2)_

**defense → fungus, per-feature AUROC tiers** (random ≈ 0.50):

| tier | features (AUROC) | meaning |
|---|---|---|
| genuine fungal ≥0.94 | F2808 (0.99), F35336 (0.99), F32785 (0.99), F23726 (0.98), F36488 (0.94) | real core + detectors (5 features) |
| weak correlate 0.6–0.72 | F7665 cell-wall, F22156, F5047 motility | marginal |
| co-occurring ~0.5–0.59 | F2082 chromatin, F15775 autophagy (0.572), F29332, F28215 | L1 co-predictors, not "meaning" |

- **Shared core, fungus ∩ bacterium = {F2808, F32785, F35336}** = "Defense / Innate Immune Response."
- **Pathogen-specific:** F23726 filamentous fungi; **F22077 bacterial LPS**.
- **Band matters:** prompt band = echo (fires on given `GO:0009620` accessions); reasoning band =
  synthesis (*"redox gating of pattern-recognition receptors," "growth inhibition of H. sativum,
  Verticillium albo-atrum," "type I interferon"*); answer band = restatement.

_(MICROBIAL_DEFENSE_RESULTS.md §§2–4; DASHBOARD_REVIEW_CHECKLIST.md §B; REASONING_CONCEPT_FEATURES.md)_

**Protein band has NO clean defense feature.** Combined `[protein ; reasoning]` L1 probe decodes
defense→fungus at **AUROC 0.978**, recruiting **16 protein-band + 20 reasoning-band** features.
Protein side is spurious: only **1/16 localizes** (F25284 ion-transport, domain-F1 0.80); the rest
enrich for generic co-occurring domains (F28287 kinase 0.33, F33248 trypsin 0.18, F15988 DNA-binding
0.00) and **none enrich for any defense term** (best circadian rhythm, FDR ~0.6). Reasoning side is
genuine (F36488/F2808 "Defense response to fungus," F35336 "Defense Response," F32785 "Immune
Response"). Domain-F1 N/A (defense is a function, not a residue-span domain).
_(DASHBOARD_REVIEW_CHECKLIST.md §C; MICROBIAL_DEFENSE_RESULTS.md §5)_

_(A second count exists — MICROBIAL_DEFENSE_RESULTS.md §5 reports 15 protein + 22 reasoning at 0.978;
the validated §C numbers 16 + 20 are used here.)_

---

## 5. Tier 3 — Cross-modal (correlational, prompt-mediated)

Two methods that **disagree** — by design, they answer different questions.

**(a) Unsupervised pairing** (`crossmodal_pairing.py`): a bio feature's per-protein activation vector
Pearson-correlated against every reasoning feature's; keep the single best partner. 20/20 bio
features find a concept-matched partner at r ≥ 0.4; permutation null max-r 0.07–0.25.
_(CROSSMODAL_PAIRS.md; RESULTS_OVERVIEW.md §5)_

| bio feat | domain-F1 | reasoning partner | r | concept |
|---|---|---|---|---|
| **F18393** kinesin | 0.98 | **F15673** "Motor" | 0.62 | kinesin motor |
| F33072 kinesin | 0.95 | F15673 "Motor" | 0.63 | kinesin motor |
| F7369 GPCR | 0.51 | F3184 "Transmembrane" | 0.89 | GPCR |
| F3623 P450 | 0.49 | F13384 "Residue Range" | 0.87 | cytochrome P450 |
| F13950 RNA-binding | 0.95 | F10235 | 0.86 | RNA-binding |
| F16964 LRR | 0.80 | F4783 "LRR" | 0.81 | leucine-rich repeat |
| F4888 collagen | 0.92 | F29088 "Collagen Binding" | 0.71 | collagen triple helix |
| F11836 histone | 0.63 | F21642 "H2A-H2B Dimer" | 0.65 | histone-fold |

Convergence: 4 GPCR→F3184, 4 kinesin→F15673, 3 RNA-binding→F10235. **13/20 pairs** pass localization
on both sides (bio domain-F1 ≥ 0.5 AND reasoning partner on-concept). GPCR pairs have high r but weak
bio-side localization (0.22–0.51). _(CROSSMODAL_PAIRS.md; RESULTS_OVERVIEW.md §5, §7B)_

**(b) Supervised combined probe** (§4): L1 on `[protein ; reasoning]` for a concept. The kinesin
*pairing* features are **not** recruited by the kinesin *probe* — the two methods answer different
questions (co-firing vs concept-decoding). _(DASHBOARD_REVIEW_CHECKLIST.md §E)_

**Causal 2×2** (`steer_crossmodal.py`) — cross-modal is **prompt-mediated, not causal**:
- P450: reasoning main-effect **+49** (29 coherent cytochrome/heme words vs 3 baseline); residue
  main-effect **−3 (≈0)**. GPCR: reason **+4**, residue **+0**.
- Localized-structure rescue fails: inject F18393 kinesin (domain-F1 **0.98**) on residues →
  residue main-effect **+2** (noise); F13950 (0.96) → **+0**.
- Sense B (same feature firing in both bands, SAE-V φ / Eq.7) = **NULL**.

→ reasoning features causally **write** their concept; structure/residue features are causally
**inert**. _(RESULTS_OVERVIEW.md §6; CROSSMODAL_PAIRS.md)_

---

## 6. Synthesis-quality scorer (method contribution, reasoning band)

`synth_span_scorer.py` ranks all reasoning features by good-synthesis quality, **grounded, no LLM**:

- **(0) Frequency filter** — drop broadband features (`act_freq > 0.20`; a feature firing on ~all
  tokens is a reasoning-mode signal, not a concept — e.g. **F2124 at 94%**, F11654 at 32%); prefer
  `log_freq < −3`.
- **(1) Phrase filter** — keep only features firing on long **contiguous** phrases; single-token =
  lexical/echo → dropped. **1,307 of 8,842 qualify.**
- **(2) Per-window score** = mechanism-verbs + inference/conclusion language + beyond-prompt named
  entities − echo/accession density. Per-window (not averaged) so mixed features are flagged
  (`frac_synth < 1.0`). Output: `synth_span_ranked.json`.

**Discovered mechanistic-reasoning features:**

| feature | concept | exemplar phrase |
|---|---|---|
| **F12706** | DNA-TF motif recognition | "T-rich major-groove signature… TAAT-centered" |
| **F15088** | mRNA translation control | "cytoplasmic poly(A)-binding protein… eIF4E/eIF4G… CCR4–NOT" |
| **F30993** | transporter alternating-access | "alternating-access antiport… outward-open cavity… coordinates Na⁺" |
| **F5221** | RAS signaling | "GEF such as SOS1, GAPs RASA1/NF1, effectors RAF1/BRAF/PIK3CA" |
| **F36488** | plant-immunity defense synthesis | "redox gating of pattern-recognition… defense response to fungus" |
| F32573 | CCR4–NOT deadenylase | via CNOT1 |
| F25291 | GPCR activation | **MIXED** (frac_synth 0.86; some windows IPR echo) |

Two valued kinds: **mechanism-elaboration** (F12706, F30993, F5221) and **inference/conclusion**
(F22404: *"absence of catalytic motifs… argue for a carrier rather than a channel; consequently…"*).
_(DASHBOARD_REVIEW_CHECKLIST.md §G)_

**Prior echo-vs-synthesis probe** (`echo_synthesis_probe.py`): trained CV AUROC 0.932, but **SAE ≈
raw** (SAE 0.914 vs raw 0.916; best single feature 0.713) → residual-stream property, not an
SAE-feature phenomenon. "Distributed SAE synthesis representation" is **withdrawn**; individual clean
features (freq ≤ 2%: F11654 calcium, F16620 helicase, F35498 dynein, F29616 Ig-fold) remain useful
handles. _(DASHBOARD_REVIEW_CHECKLIST.md §F; RESULTS_OVERVIEW.md §4)_

---

## 7. Honest negatives / retractions (kept visible)

- **No causal cross-modal flow** — structure/residue features causally inert; alignment prompt-mediated (§5).
- **No residue-level function detector** — protein band has no clean defense/GO feature (§4).
- **SAE never beats raw on decodability** — ties on saturated domain probes (0.99 ≈ 0.989), loses on
  3D burial (SAE-svd256 0.873 vs raw 0.886). Value is interpretability, not accuracy. _(RESULTS.md §4)_
- **Auto-interp is a hypothesis** — misses genuine (F23726) and oversells correlates (F15775) (§2).
- **"Function emerges in reasoning ~0.95" RETRACTED** — leakage (§1).
- **Per-protein AUROC overstates localization** — corrected by domain-F1 (~17% localize) (§3).
- **Global "synthesis" is SAE ≈ raw** — residual-stream property (§6).

**Root cause of the negatives (prompt-anchoring):** the prompt's `go_pred` already contains **68% of
ground-truth GO terms** (≥80% of truth for 47% of proteins); the final answer echoes it **38%**. The
model **refines a given speculation**, not derives function from residues — so the answer can't be
steered, structure need not feed reasoning, and reasoning is echo-heavy. What it adds is
mechanism/elaboration (the microbial-defense circuit + synthesis features). _(RESULTS_OVERVIEW.md §8)_

---

## Where each result lives

| result | doc / file |
|---|---|
| leak-floor probe + all feature IDs | `REASONING_CONCEPT_FEATURES.md`, `probe_trained.py` |
| microbial-defense flagship | `MICROBIAL_DEFENSE_RESULTS.md` |
| master methodology (A–G) | `DASHBOARD_REVIEW_CHECKLIST.md` |
| combined-probe validation | `crossmodal_probe_defense_ids.json` |
| cross-modal pairs (20 + domain-F1) | `CROSSMODAL_PAIRS.md`, `crossmodal_pairing_l30.json` |
| structural domain-F1 (872 feats) | `domain_f1_l30.json`, `domain_f1.py` |
| good-synthesis ranked features | `synth_span_ranked.json`, `synth_span_scorer.py` |
| bio enrichment (GO+IPR, AUROC) | `feature_biology_table.json` |
| full synthesis | `RESULTS_OVERVIEW.md`, `RESULTS.md` |
