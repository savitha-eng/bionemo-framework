# Phase 3 — BioReason-Pro SAE — STATUS (2026-06-29)

## LATEST (2026-06-29 PM)

**Balanced dashboards rebuilt clean.** Root-caused a chain of failures: the `build_dashboard` omega step
loads the full residual matrix to GPU (276 GiB → OOM); my `--max-shards 10` workaround then crippled the
metadata (555-protein sample → wrong feature set, dropped 6 of the autointerp top-10, e.g. F30664). Fix:
added `build_dashboard.py --skip-omega` (omega is artifact-prone + unused; skipping removes the OOM and the
297 GB Z) → metadata now builds on the FULL 8k store with reliable stats and all features. Also fixed a slow
`dashboard.py` loader (`to_pylist`+`np.isin` on 29M rows → `to_numpy` + contiguity cap) and a near-miss node
OOM (7×297 GB examples → capped to 3000 proteins). Rebuild running on GPU 0-6.

**Cross-modal viewers.** `crossmodal_viewer.html` (full-sequence, works now, L28 — promising: F27582 PHD-finger,
F38758 proteasome/protease, F9861 estrogen-receptor) vs `crossmodal_explorer.html` (ranked table, empty until
the full-dataset sweep runs). Sweep + balanced full-seq viewer queued behind the rebuild.

**Full-117k cross-modal (L24): 455 co-fire, 0 aligned (cosine>0.3)** — the model fuses (ablation +15% CE) but
the SAE decomposes. Definitive.

## NEXT EXPERIMENT — Matryoshka SAE (scoped + smoke-launched 2026-06-29)

**Why:** our flat TopK shows feature absorption/splitting (always-on 0.33/0.33/0.33 artifacts, 530 noisy
band-mass "cross-modal" features, spurious F34377). Matryoshka SAEs (arXiv:2503.17547; used in microsoft/
maira-2-sae on a radiology multimodal LLM) directly target absorption → cleaner, more monosemantic, hierarchical
features. NOT inherently multimodal — it improves feature QUALITY, which gives us cleaner features to then judge
for cross-modality.

**Does it replace balancing? NO — complementary.** Matryoshka fixes feature quality; balancing fixes which
features get learned (modality coverage). If text is 80% of tokens, even Matryoshka fills early prefixes with
text features. So: **train Matryoshka ON balanced data.** (MAIRA-2 sidestepped this by training only on text
tokens — excluding image tokens — which carry fused image info via attention; we instead train on protein+text
and balance, because we want features on the protein residues themselves.)

**Can we reuse the HF SAE? NO** — maira-2-sae weights are for MAIRA-2's layer-15 (4096-dim radiology) activations;
our model is Qwen3-4B (2560-dim, protein+bio-text). Different activation space → weights useless. We reuse the
METHOD only.

**Implementation (done):** opt-in `matryoshka_groups` flag on our existing `TopKSAE` (sae/src/sae/architectures/
topk.py) — NOT a new dependency. Group fractions (MAIRA-2 scheme `0.5,0.25,0.125,0.0625,0.0625`) → cumulative
prefix bounds; recon loss = mean FVU over nested prefixes (each prefix must reconstruct x alone, pressuring
low-index latents to learn general features → coarse→fine hierarchy). Inference (encode/decode) is identical to
TopK, so Matryoshka checkpoints load in plain TopKSAE — all dashboards/eval/DDP/Triton tooling unchanged.
`train.py --matryoshka-groups "0.5,0.25,0.125,0.0625,0.0625"`. Unit smoke passed (loss runs, grads flow, config
round-trips). Real smoke: `matryoshka_smoke.sh` (L24, 8k store, in-loader 50/50, 1 GPU) — confirming FVU drops.

**Plan:** smoke → full L24 Matryoshka run on balanced data → compare cross-modal cleanliness vs flat TopK
(do the artifacts shrink? are shared concepts cleaner?). Steering deferred until features are higher quality.

---
# (earlier) Phase 3 — BioReason-Pro SAE — STATUS (2026-06-29)

Consolidated status snapshot. For deep conventions/methods see `HANDOFF.md`. This file is the
quick "where are we" review.

## ✅ Concretely accomplished

1. **First decoder-LLM SAE recipe in bionemo.** Full extract→train→eval pipeline for BioReason-Pro-SFT
   (Qwen3-4B multimodal: ESM3 protein embeds + GO graph encoder + Qwen3 LLM, 36 layers, hidden 2560).
   SAEs (TopK, exp16, k128) trained across the full depth (L12–L35) on the full 117k-protein dataset.

2. **Causal fusion proven (the thesis).** Zeroing protein/GO input embeddings raises the model's *own*
   reasoning-token CE by **+15%** (reasoning ΔCE +0.161, t=31.1, 93% of proteins; answer +0.071, t=21.3;
   n=300, p≪0.001). The model genuinely fuses modalities. Write-up: `analysis/INPUT_ABLATION.md` (pushed
   to the savitha-eng fork). Script: `scripts/ablation_eval.py`.

3. **Definitive cross-modal result (L24, full 117k proteins).** Per-sample SAE-V Eq.7 metric:
   **455 / 40,960 features co-fire on both bio + English, but 0 are aligned (cosine > 0.3).**
   → The model fuses, but the SAE *decomposes* that fusion into separate per-modality features rather
   than localized cross-modal ones. Strong honest negative; pairs with the ablation positive.
   The tension IS the result: fusion is a distributed computation, not a single feature.

4. **Modality balancing.** Fast in-loader 50/50 balancing (drop `<go>`, downsample text to ≈protein at
   load time — no store rewrite; `train.py --balance-modality --balance-protein-frac`). **7 balanced
   SAEs trained:** L16, L18, L22, L24, L28, L30, L32. Produces visibly distinct bio-vs-English feature
   clusters in the UMAP (vs the unbalanced model where protein features were starved/smeared).

5. **Multimodal dashboard** (Vite/DuckDB, on Jared's ESM2 template): modality filter, UMAP, per-band
   examples, reasoning/answer split, full-sequence cross-modal viewer.

6. **Autointerp** (label + detection-F1 + GO-AUC) on top-10 L24-balanced features. Clean highlights:
   **F30664** (regulation of cellular processes, det-F1 0.78 / GO-AUC 0.78), **F32128** (nucleic-acid
   binding, protein, 0.67/0.69), **F12369** (nuclear localization, 0.59/0.75). Independently confirmed
   the cross-modal candidate **F34377 spurious** (det-F1 0.14, GO-AUC 0.00).
   - **det-F1** = does the LLM's label predict which held-out windows actually fire (detection task,
     positives vs random negatives). High = the explanation is real; low = vague/wrong/uninterpretable.

7. **Fixed go-token eval leakage.** Balanced SAEs never trained on `<go>`, so running them on `<go>` at
   eval produced OOD-artifact "go features" (135 of them). `build_dashboard.py --drop-go` /
   `dashboard.py --drop-go` zero `<go>` activations at encode → go vanishes from all stats. Verified:
   L24/L28 balanced now `go_frac>0.5 = 0`. (Text mentions like "GO:0005622 (organelle)" are real prompt
   text and correctly kept.)

8. **Fixed build_dashboard OOM.** It loaded the full residual matrix to GPU for the omega step
   (29M tokens → 276 GiB → OOM). Cap metadata build to `--max-shards 10` (~2M tokens, like the original
   good build); `dashboard.py` still does the full 8k for 50 examples/feature.

## 🔬 What you can view NOW (`http://localhost:5174/`)

- **All 7 balanced dashboards** (metadata + UMAP clean): `?model=l16_balanced` … `l32_balanced`.
  Examples refreshing to 50/feature (CPU decode phase).
- **Band-mass cross-modal features**: any balanced model → modality filter → "cross-modal"
  (e.g. l24_balanced has 530). Subset-level; fires on both bands.
- **Autointerp viewer**: `/autointerp_viewer.html` — top-10 with LLM label, det-F1, GO-AUC, inspect links.
- **Full-sequence cross-modal viewer**: `/crossmodal_viewer.html`.
- **Cross-modal explorer**: `/crossmodal_explorer.html` — built, populates as the sweep lands (below).

## ⏳ Still waiting on

- **Balanced dashboards — 50-example refresh**: metadata/UMAP done & clean for all 7; per-feature
  examples refreshing (`dashboard.py`, CPU decode, ~30–60 min) → 50 examples/feature.
- **Cross-modal full-dataset sweep** (`crossmodal_sweep.sh`): full-117k per-sample protein↔text metric
  on all 7 balanced layers, with `--dump-json` → `multimodal_dashboard/public/crossmodal/crossmodal_l{L}.json`.
  Running on GPU 0-6, ~hours each. The explorer's strict per-feature alignment list is empty until these land.
- **30/70 L24** (in-loader) — secondary.

## Two cross-modal metrics (don't conflate)

| | Band-mass (in dashboard now) | Per-sample cosine (sweep → explorer) |
|---|---|---|
| Question | fires on both bands somewhere in 8k subset? | within ONE protein, are bio & text firings the SAME concept (aligned)? |
| Rigor | loose (includes always-on artifacts) | strict (the real fusion test) |
| Data | 8k subset | full 117k |

The completed L24 run only logged the 455/0 summary (predates `--dump-json`), so the inspectable
full-dataset list requires re-running = the sweep. That is why the explorer is empty before the sweep.

## Key scripts

- `scripts/ablation_eval.py` — causal input ablation (the thesis).
- `scripts/crossmodal_cooccur.py` — per-sample SAE-V cross-modal, now with `--dump-json`.
- `scripts/build_dashboard.py` / `dashboard.py` — dashboard, both with `--drop-go`.
- `scripts/autointerp_top10.py` — label + detection-score; `autointerp_biology.py` — GO-AUC for protein feats.
- `scripts/make_dash_subset.py` — file-level subset of a full store for dashboard builds (no re-extraction).
- `/data/savithas/phase3_full/parallel_rebuild_balanced.sh`, `crossmodal_sweep.sh` — orchestration.
