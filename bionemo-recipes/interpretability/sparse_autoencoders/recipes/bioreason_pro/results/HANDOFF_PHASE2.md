# Handoff Prompt — Phase 2: dashboard + interpretability PRs (+ Phase-1 eps fix)

_Copy the block below into the publish session (the one that did Phase 1). Phase 1 (train pipeline)
is reviewed and merge-ready; this covers the dashboard + probe/steer/autointerp scripts._

---

```
Continue the BioReason-Pro SAE recipe publishing in NVIDIA-BioNeMo/bionemo-interpretability
(already cloned at /data/savithas/bionemo-interpretability). Phase 1 (5 PRs: scaffold, model-integration,
extraction, training, eval) is REVIEWED and merge-ready. Two tasks:

TASK A — apply the ONE Phase-1 review fix (quick), on the bioreason-pro/eval branch:
  Read /data/savithas/bioreason_pro_review_feedback.md. The only should-fix: eval.py's reconstruction
  target hand-rolls normalization with `std + 1e-5`, but TopKSAE._normalize uses `+1e-8`. Replace the
  hand-rolled `mu/std/xn` in accumulate_recon with `xn, _ = sae._normalize(x)` (keep
  `recon = sae.decoder(codes) + sae.pre_bias`). Optionally address the nits (pyarrow>=23.0.0 pin looks
  wrong — verify; add an assert that kept tokens are contiguous at extract.py token_index). Amend the
  eval branch.

TASK B — draft Phase-2 PRs (dashboard + interpretability). Same discipline as Phase 1: small,
same-function-grouped, DRAFT PRs stacked in dependency order, mirroring recipes/codonfm/ layout.
Source (working code) is at
/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro/
(scripts/ + multimodal_dashboard/). Distill ~110 research scripts down to the canonical ones below.

PR STACK (base each on the prior; base the first on the eval branch):
  P2-1 Dashboard build: scripts/dashboard.py + launch_dashboard.py (from our dashboard.py) — builds the
       feature_examples.parquet + serves it. GO-BAND: make --drop-go DEFAULT-ON (a reviewer browsing
       go-centered features just sees the fixed unused ontology). Document "interpret GO via text
       accessions, not the <go> band."
       PROTEIN-BAND: keep the --protein-n-examples lever (higher example cap for the low-magnitude
       protein band; protein features fire ~10-40x weaker so fewer clear the TopK cutoff).
       SURFACE LOCALIZATION (important): the dashboard currently shows go_auc but NOT domain-F1, so a
       reviewer can't tell a real localized structural feature (F18393, F1 0.98) from an AUROC-oversold
       one (F4647, AUROC 0.98 but F1 0.0). (1) DATA: dashboard.py must write domain_f1/dom_precision/
       dom_recall/ipr_domain/ipr_auroc/localized into feature_metadata.parquet (merge from the domain_f1
       output). (2) DISPLAY: App.jsx must render domain_f1 + a "localized" badge next to each feature.
       CRITICAL for reviewing bio features: their examples are low-magnitude/single-residue and can't
       show localization by eye -- the metric must be visible.
  P2-2 Dashboard React app: <model>_dashboard/ (from multimodal_dashboard/) — mirror codon_dashboard/
       (src/, package.json, vite.config.js, index.html). MUST .gitignore node_modules/, dist/, *.log
       (do NOT commit them). One documented build (npm/vite).
  P2-3 Enrichment + feature-biology: scripts/eval_enrichment.py (from feature_biology_table.py +
       gsea_enrichment.py) — Fisher/GSEA per feature vs GO/InterPro, + per-feature AUROC.
  P2-4 Domain-F1 (Polina's metric): scripts/eval_domain_f1.py (from domain_f1.py) — per-position
       precision x per-region recall for structural features.
  P2-5 SAE-vs-raw probe + FLAGSHIP EXAMPLE NOTEBOOK: scripts/probe.py (from probe_v2.py / probe_trained.py)
       — SAE-svd vs raw vs random(leak-floor), sparse L1, per-band.
       *** SHIP A DEMO NOTEBOOK built on this — the reasoning-concept probe, centered on the MICROBIAL-
       DEFENSE finding (the cleanest reasoning result). The notebook should tell the honest story:
       (1) design GO-concept labels, mean-pool reasoning-band SAE activations per protein;
       (2) train logistic probes on 4 representations: sae-sparse(L1), sae-svd256, raw, random(=LEAK FLOOR);
       (3) show that of 12 designed concepts, ONLY 4 clear the leak floor (defense→fungus +0.111,
           defense→bacterium +0.108, structural-molecule +0.063, plasma-membrane +0.055) — the other 8 are
           leakage (random projection matches them);
       (4) for defense→fungus/bacterium, show the SPARSE-SELECTED feature IDs and that they are DISTRIBUTED
           (decoder-cosine ~0, distinct+complementary) with a SHARED microbial-defense core (fungus∩bacterium
           = {2808,32785,35336});
       (5) the takeaway: the model represents microbial defense as a distributed, orthogonal feature set with a
           shared antifungal/antibacterial core — a real, non-leaked reasoning-representation finding.
       fungus features: 36488,35336,2808,23726,29332,32785,7665,15775,22156,2082,28215,5047 ;
       bacterium: 35336,38712,36769,8824,2724,32785,20730,22077,13479,25134.
       Full lists + all-12-concepts in reasoning_concept_features_real.json / REASONING_CONCEPT_FEATURES.md.
       NON-NEGOTIABLE: the random-projection LEAK-FLOOR baseline is the whole point — it's what proves defense
       is real and the other 8 are fake. Never drop it; never report the absolute AUROC without the margin.
  P2-6 Steering: scripts/steer.py (from steer_generation.py + steer_crossmodal.py) — p95-dose,
       decision-point, set-mode clamp, concept-absent, coherence + LLM-judge.
  P2-7 Auto-interp: scripts/autointerp.py + autointerp_validate.py — LLM feature labels + the
       enrichment cross-check.

FIX-LIST TO PRESERVE (Phase-2 code was hard-won — a naive cleanup WILL regress these):
  - Enrichment/biology: FREQUENCY FILTER (drop near-dense/always-on features, per-token freq<=~0.02/0.10)
    before ranking — without it, ~100%-firing junk features top the list. Also the winner's-curse
    TRAIN/TEST split (select feature on train, report test AUROC).
  - domain-F1: per-POSITION precision + per-REGION recall (not per-protein AUROC — that OVERSELLS
    localization; only ~17% of AUROC-high features are truly localized).
  - probe: the RANDOM-projection LEAK-FLOOR baseline is mandatory on the reasoning band (random ~0.95
    because the text names the function) — report the margin over the leak floor, never the absolute.
  - steering: p95 per-feature dose (scale by mean_active), decision-point (steer-after-N), set-mode
    clamp with denormalization, concept-ABSENT proteins, coherence gate + LLM-judge. Use a co-firing
    CLUSTER as the writer, not a single feature (single features are weak writers -> null-on-null).
  - autointerp labels are HYPOTHESES: keep the enrichment cross-check (autointerp_validate) — free-
    floating labels validated only ~1/3 of the time; anchored (cross-fire) ones 19/19.
  - GO band: --drop-go default-on (see P2-1).

CAVEAT TO DOCUMENT (don't hide it): all interpretability analyses run on the TRAIN split (the only
activation stores that exist). Protein/structure band is ESM3-sourced (frozen -> memorization minor);
reasoning band carries a memorization caveat on top of leakage. Note this in the interp README section.

NORMALIZATION: any script that reconstructs (probe/steer/dashboard example-scoring) must use the SAE's
own _normalize/decode (normalized space for scoring; forward()/decode(codes, info) for a raw-space
substitution) — do NOT hand-roll normalization (same class of bug as the Phase-1 eps nit).

ENV/GIT: recipe venv, no pip (use uv). Never commit dashboard node_modules/dist/*.log. DRAFT PRs; the
user reviews before ready. End commits with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

START: apply Task A (eps fix) on the eval branch, then draft P2-1..P2-7 stacked. Post each branch name
so the user (and reviewer) can check the diffs against this fix-list.
```

---

_Phase-1 review verdict: correct + faithful to sae-l30-exp16-balanced; one eps should-fix (Task A above);
matryoshka/modality-weighting correctly absent; loss-balancing (aggregate_loss) correct. Full detail in
`/data/savithas/bioreason_pro_review_feedback.md`._
