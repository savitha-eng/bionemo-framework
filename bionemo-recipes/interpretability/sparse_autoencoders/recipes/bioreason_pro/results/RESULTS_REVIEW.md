# BioReason-Pro SAE — Results Review (START HERE)

Single illustrated page to review the analysis. **Layer L30** for all feature/steering/reasoning work (decodability also swept L16/28/32 + ESM3). Full narrative in `OVERNIGHT_RESULTS.md`; steering example traces in `traces_*_normfixed.md`.

## How to review, in order
1. This page (charts + examples below)
2. **Dashboard** (interactive features): `cd multimodal_dashboard && npm run dev` → browse feature cards, examples, atlas UMAP (l30_balanced set)
3. `OVERNIGHT_RESULTS.md` (final summary at top) → `results/charts/` → `traces_synapse_normfixed.md` (before/after steering)
4. Probing code: `scripts/` (see table at bottom)

---

## 1. BIO FEATURES — robust, Jared/InterPLM-comparable
![enrichment](charts/enrichment.png)
- **74.7% of protein-band features annotated (GO/InterPro, FDR<0.05), 88.5x lift over shuffle-null, 203 terms** — matches InterPLM's >=70%.
- Clean per-feature detectors (each fold = its own feature):

![per-feature](charts/per_feature_structural.png)

| feature | detects | AUROC |
|---|---|---|
| F16026 | Protein kinase domain | 0.98 |
| F679 | ARM fold | 0.95 |
| F11009 | WD40 | 0.92 |
| F5540 | Homeodomain | 0.90 |
| F17703 | Ig-like | 0.87 |

- SAE healthy: FVU 0.20 (80% variance explained). Structure decodability is flat-high across depth (below) — but that's saturation (random ties it), so decodability is not the SAE's value; the nameable detectors are.

![layers](charts/layer_decodability.png)

## 2. REASONING FEATURES — robust + NON-CIRCULAR (echo vs synthesis)
![echo-synthesis](charts/echo_synthesis.png)
- **Synthesis features** fire when the model reasons BEYOND its given annotations (mechanistic inference); **echo features** fire on the literal given IPR/GO IDs. Per-feature AUROC to 0.71 vs shuffle-null 0.506.
- Auto-interp labels (LLM):
| F39979 | reasoning-synthesis | Transmembrane Immunoreceptor Protein |
| F39744 | reasoning-synthesis | Protein Synthesis Tracking |
| F13295 | reasoning-synthesis | Cell Signaling Pathway |
| F5147 | reasoning-synthesis | HSP70 Chaperone Function |
| F14759 | reasoning-synthesis | Cellular Signaling Pathway |
| F7099 | reasoning-synthesis | Protein Phosphorylation Event |
| F23525 | reasoning-synthesis | Autophagy Regulation |
| F11654 | reasoning-synthesis | Calcium Signaling Pathway |
| F35387 | reasoning-synthesis | Protein Structure Prediction |
| F16620 | reasoning-synthesis | DNA Polymerase Activity |
| F16026 | protein-detector | Protein Kinase Domain Detected |
| F679 | protein-detector | ARM fold protein domain |
| F11009 | protein-detector | WD40 Domain Detection |
| F5540 | protein-detector | Homeodomain Detection |
| F17703 | protein-detector | Ig-like domain detected |

- ⚠️ Probing named GO on the reasoning band is LEAKY (retracted); the non-circular result is echo-synthesis above.

## 3. STEERING — read/write dissociation
![read-write](charts/read_write_plane.png)
- **Reasoning-band clusters WRITE their concept** (synapse 0.77 / mito 0.60 / nervsys 0.36, LLM-judged). See `traces_synapse_normfixed.md` for before/after.
- **Protein-structure features are concept-READ-ONLY** (perturb but can't write "kinase"; harness + dose controlled).
- **Negatives (honest):** no NEW steerable cluster (redox/conformational/reasoning-flow judged 0%/degenerate); answer-change does NOT flip the GO prediction.
- Matched-control-feature steering: RUNNING (synapse vs load-matched non-synapse cluster).

## Probing code (scripts/)
| script | purpose |
|---|---|
| probe_v2.py | two-tier probe (overlap+CV+specificity+nulls), both bands |
| enrichment_probe.py | nb03 enrichment (annotation-rate + lift) |
| echo_synthesis_probe.py | non-circular reasoning probe (synthesis vs echo) |
| per_feature_structural.py | per-feature bio detectors |
| interpro/contact/residue_domain_probe.py | structural decodability |
| load_bearing.py, per_token_sink_freq.py | load-bearing / sinks |
| logit_lens.py, auto_interp.py, sae_health.py | naming / health |
| steer_generation.py, llm_judge_steering.py | steering + LLM-judge |

## Matched-control steering — feature-SPECIFICITY confirmed
Synapse cluster injects synapse (1.0 @ mult=6); a load-matched NON-synapse cluster injects 0.0 at every dose → the steering is feature-specific, not generic (Jared's strongest control passes).
