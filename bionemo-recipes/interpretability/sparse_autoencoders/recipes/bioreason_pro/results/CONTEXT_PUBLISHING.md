# Context: Publishing the BioReason-Pro SAE Recipe (codonfm-style)

_Scoping doc for the next task — turn the research sprawl into a publishable recipe in the style of Polina's
evo2 / Jared's codonfm recipes, in the new repo. Written by the research agent; hand-off ready._

---

## 1. Goal

Take the working code for the multimodal model (**BioReason-Pro** = ESM3 protein embeddings + GO-graph encoder
+ Qwen3-4B reasoning LLM) SAE and publish it as a clean recipe. **MVP = (a) training pipeline, (b) dashboard,
(c) a few steering/probe/auto-interp scripts-or-notebooks** modeled on Jared's codonfm.

---

## 2. The target pattern — follow `codonfm`, NOT `evo2`

There are 3 reference recipes. **`codonfm` (Jared's) is the richest and the right model** — it's the only one
with dashboard + eval + enrichment + residue-F1. `evo2` (Polina's) is minimal (extract/train only) but is the
canonical **streaming-extract + train.py** pattern (the skill says copy evo2's `train.py` — it's the only one
wiring all 4 opt-in flags). **Note: reference recipes ship SCRIPTS, not notebooks** — Jared's/Polina's steering
& probe *notebooks* live in their separate research repos (the "gold standard" notebooks reviewed earlier), not
in bionemo-recipes.

**codonfm layout (the manifest to match):**
```
recipes/codonfm/
├── README.md                 # numbered pipeline: extract → train → eval → analyze → dashboard
├── 1b.sh / run.py            # orchestrator
├── pyproject.toml, .gitignore
├── run_configs/config.yaml
├── codon_dashboard/          # clean React app (package.json, index.html, vite.config.js)
└── scripts/
    ├── download_*.py         # data acquisition
    ├── extract.py            # streaming activations → ActivationStore
    ├── train.py              # universal SAE train (copy evo2's)
    ├── eval.py               # loss-recovered / reconstruction / dead-latents
    ├── analyze.py            (1277 lines — logit-lens + computed annotations + auto-interp)
    ├── eval_gene_enrichment.py (618 — GSEA vs GO/InterPro/Pfam per feature)
    ├── eval_swissprot_f1.py  (860 — residue-level annotation F1)
    ├── dashboard.py          # build dashboard data
    └── launch_dashboard.py
```

---

## 3. What we have (the sprawl to distill)

`recipes/bioreason_pro/`:
- **`src/bioreason_pro_sae/`** (the reusable core, ~330 lines) — `model_loader.py` (187), `data.py` (147),
  `viz/`. **This is the crux and it's already clean-ish.**
- **`scripts/` — 111 python files.** ~90% are research one-offs; **~12–15 are the publishable core.**
- **`multimodal_dashboard/`** — React app (WORKS, live) + build logs + node_modules + dist committed (needs cleanup).
- **`results/`, `analysis/`** — our findings (not shipped, but source the README's "what it finds").
- Existing `extract.py` (538 lines) and `train.py` (20 KB) — **the training pipeline already works.**

---

## 4. MVP file manifest — what to include, mapped from our scripts

| target file | maps from (ours) | state | effort |
|---|---|---|---|
| `README.md` | new (model on codonfm) | write fresh | S |
| `run.sh` orchestrator | new | write fresh | S |
| `pyproject.toml`, `.gitignore`, `run_configs/config.yaml` | ours (exist) | prune | S |
| **`src/…/model_loader.py`** | ours (187 ln) | **works; external-dep risk (see §5)** | **L** |
| **`src/…/data.py`** | ours (147 ln) | works | M |
| `scripts/extract.py` | ours (538 ln) | clean + license + doc | M |
| `scripts/train.py` | **copy evo2's** (per skill) + our wandb defaults | swap | S |
| `scripts/eval.py` | ours `eval.py` / `eval_loss_recovered.py` | consolidate | M |
| `scripts/dashboard.py` + `launch_dashboard.py` | ours `dashboard.py` | clean | M |
| `<model>_dashboard/` React app | ours `multimodal_dashboard/` | strip logs/node_modules; clean build | M |
| **PROBE scripts (pick 3–5):** | | | |
| `eval_enrichment.py` | `feature_biology_table.py` + `gsea_enrichment.py` + `enrichment_probe.py` | consolidate | M |
| `eval_domain_f1.py` | `domain_f1.py` | clean (Polina's metric ✓) | S |
| `probe.py` (SAE vs raw vs random) | `probe_v2.py` / `probe_trained.py` | consolidate | M |
| `crossmodal.py` | `crossmodal_pairing.py` (+ `layer_robustness.py`) | clean | M |
| `echo_synthesis.py` | `echo_synthesis_probe.py` | clean (freq-filter ✓) | S |
| **STEERING:** `steer.py` | `steer_generation.py` + `steer_crossmodal.py` | clean | M |
| **AUTO-INTERP:** `autointerp.py` + `autointerp_validate.py` | ours (exist) | clean | M |
| Notebooks (optional MVP+) | wrap steer/probe/autointerp as demo `.ipynb` | new | M–L |

_Effort: S≈½ day, M≈1–2 days, L≈3–5 days._

---

## 5. The crux / hard parts (where the real work + risk is)

1. **`model_loader.py` external dependencies (the #1 risk).** It imports the **authors' bioreason-pro repo**
   (`bioreason2/models/protein_llm.py`) unchanged + **ESM3** (`esm3_sm_open_v1`) + an **unsloth `sys.modules`
   stub** (protein_llm does a top-level `from unsloth import …` only used on the training path). For a
   publishable recipe this needs: pinned install instructions for the external repo + ESM3 weights + the HF SFT
   checkpoint, OR vendoring the minimal model code. **This is what makes/breaks reproducibility** and is most of
   the L-effort. Unlike esm2 (HF-native `AutoModel`), this model is NOT a clean `from_pretrained`.
2. **Distilling 111 scripts → ~12.** Lots of near-duplicates (7 `crossmodal_*`, 8 `autointerp_*`, many
   `add_*_metric` sidecar patches). Decide the canonical one per function; the rest are provenance, not shipped.
3. **Dashboard cleanup.** The React app works but has `node_modules/`, `dist/`, `*.log` committed. Needs a clean
   `.gitignore`, a documented build, and the band-tagging (protein/go/reasoning/answer) is model-specific.
4. **The token↔position contract** (data.py) — protein-token placeholder, GO band, reasoning=response, accession
   handling — is the model-specific glue reviewers will scrutinize; must be documented.
5. **Fixes to preserve** (do NOT regress — these were hard-won): train.py's 4 opt-in flags; freq-filter in
   echo-synthesis; domain-F1 (Polina's per-position/per-region); winner's-curse train/test; the leak-floor
   (random-SAE) baseline in probes; p95-dose + decision-point + concept-absent in steering.

---

## 6. Work estimate

- **Training-only MVP** (README + extract + train + eval + src, reproducible): **~1 week**, dominated by the
  model_loader external-dependency reproducibility (§5.1).
- **+ Dashboard**: **+3–4 days** (React cleanup + dashboard.py + band-tagging docs).
- **+ Probe/steer/autointerp scripts** (5–7 consolidated): **+1 week**.
- **+ Demo notebooks**: **+3–5 days**.
- **Total for the full MVP the user described**: **~3 weeks** focused, with the external-dep repro as the main
  risk that could balloon if ESM3/authors-repo licensing or packaging is thorny.

---

## 7. Recommendation on who does it

- **Fresh agent for the mechanical distillation** (copy/clean/license/README scaffolding) — well-defined, high
  volume, low ambiguity. Good hand-off with this doc + the manifest.
- **Keep me (this agent) for the two judgment-heavy pieces:** (a) the model_loader/data external-dependency
  reproducibility (I know why the unsloth stub + ESM3 + band contract are shaped as they are), and (b)
  **reviewing** that the distilled scripts preserve the fixes in §5.5 (a fresh agent will re-introduce the exact
  bugs we spent this project fixing — e.g. dropping the freq filter, the leak-floor baseline, or train.py's flags).
- **Concrete split:** fresh agent scaffolds → I review each probe/steer script against the fix-list → I own
  model_loader repro. That plays to strengths and avoids regressions.

---

## 8. Open decisions for the user
1. **New repo target** — standalone, or a `recipes/bioreason_pro/` in the new repo's tree? (affects import paths)
2. **External model deps** — vendor the minimal BioReason-Pro model code, or ship install instructions + pinned refs?
3. **Notebooks vs scripts** — codonfm ships scripts; do we want notebooks too (more demo-friendly, more maintenance)?
4. **Scope of probes** — which of {enrichment, domain-F1, SAE-vs-raw probe, cross-modal, echo-synthesis} are MVP
   vs later? (I'd MVP: enrichment + domain-F1 + one probe; defer cross-modal/echo-synthesis to a v2.)
5. **Layer** — ship L30 (the validated sweet spot) as the default; mention the layer-sweep finding in the README.

_Sources: `recipes/codonfm/` (target), `recipes/evo2/` (train.py), `sae/` (universal train/eval), our
`bioreason_pro/{src,scripts,multimodal_dashboard}`. Fix-list: `results/RESULTS_OVERVIEW.md` + `LAYER_SWEEP.md`._
