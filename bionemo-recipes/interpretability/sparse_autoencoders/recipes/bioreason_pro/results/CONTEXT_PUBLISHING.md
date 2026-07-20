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
| **`src/…/model_loader.py`** | ours (187 ln) | **works as-is; just needs the authors' repo documented as a prereq (§5)** | **M** |
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

1. **Model loading — ONE checkpoint + ONE documented prerequisite (not a research problem; already solved).**
   The weights are a single self-contained HF checkpoint — **`wanglab/bioreason-pro-sft`** — with *everything
   bundled*: Qwen3-4B LLM (`model-*.safetensors`), **ESM3 (`protein_model/pytorch_model.bin`)**, the GO stack
   (`go_embedding.pt`, `go_encoder.pt`, `go_projection.pt`, `go-basic.obo`), `protein_projection.pt`, tokenizer.
   One `hf download`. **BUT** its config is `"architectures": ["Qwen3ForCausalLM"]` with **no `auto_map` / no
   modeling code**, so `AutoModelForCausalLM.from_pretrained` alone yields a *plain text LLM* that ignores the
   protein/GO tensors. The multimodal-assembly code (splice ESM3 + GO embeddings into the token stream at
   `protein_token_id`/`go_token_id`) lives in the authors' GitHub repo **`bowang-lab/BioReason-Pro`** — which the
   HF README itself directs users to for the inference guide.
   **Our `model_loader.load_bioreason_pro_sft` already handles this** with two inputs: `ckpt_dir` (the HF
   checkpoint) + `bioreason_pro_root` (a clone of the authors' repo, put on `sys.path` for `import bioreason2`).
   The `unsloth` stub and the `process_go_aspects` patch are minor consequences of using the authors' code.
   → **DECISION (per user): document the authors' repo as a PREREQUISITE** — not vendor. So the recipe's setup is:
   `hf download wanglab/bioreason-pro-sft` + `git clone bowang-lab/BioReason-Pro@<pin>` + pass `--bioreason-root`.
   This is a **codonfm-style prerequisite step**, not an L-effort — model loading is *solved*, it just needs
   documenting + a pinned commit + an env note (the checkpoint's `config.json` carries `unsloth_version`).
2. **Distilling 111 scripts → ~12.** Lots of near-duplicates (7 `crossmodal_*`, 8 `autointerp_*`, many
   `add_*_metric` sidecar patches). Decide the canonical one per function; the rest are provenance, not shipped.
   **This is now the largest single chunk of work** (was overshadowed by the model-loading concern).
3. **Dashboard cleanup.** The React app works but has `node_modules/`, `dist/`, `*.log` committed. Needs a clean
   `.gitignore`, a documented build, and the band-tagging (protein/go/reasoning/answer) is model-specific.
4. **The token↔position contract** (data.py) — protein-token placeholder, GO band, reasoning=response, accession
   handling — is the model-specific glue reviewers will scrutinize; must be documented.
5. **Fixes to preserve** (do NOT regress — these were hard-won): train.py's 4 opt-in flags; freq-filter in
   echo-synthesis; domain-F1 (Polina's per-position/per-region); winner's-curse train/test; the leak-floor
   (random-SAE) baseline in probes; p95-dose + decision-point + concept-absent in steering.

---

## 6. Work estimate

_(Revised down — model loading is solved, not a risk. The work is now mostly mechanical distillation + cleanup.)_
- **Training-only MVP** (README w/ prereq steps + extract + train + eval + src): **~3–4 days.** `model_loader`/
  `data` work as-is; the model dependency is a documented prerequisite (`hf download` + `git clone @pin`), not code.
- **+ Dashboard**: **+3–4 days** (React cleanup: strip `node_modules`/`dist`/logs, documented build, band-tagging docs).
- **+ Probe/steer/autointerp scripts** (5–7 consolidated from the 111): **+1 week** — now the *largest* chunk.
- **+ Demo notebooks**: **+3–5 days**.
- **Total for the full MVP**: **~2–2.5 weeks** focused. No single ballooning risk anymore — it's steady distillation
  + cleanup, and the main quality risk is regressing a fix (§5.5), which is why review matters more than raw effort.

---

## 7. Recommendation on who does it

- **Fresh agent for the mechanical distillation** (copy/clean/license/README scaffolding, prereq-step docs) —
  well-defined, high volume, low ambiguity. This is now most of the work. Good hand-off with this doc + manifest.
- **Keep me (this agent) mainly for REVIEW** — that the distilled scripts preserve the fixes in §5.5 (a fresh
  agent will re-introduce the exact bugs we spent this project fixing — dropping the freq filter, the leak-floor
  baseline, domain-F1's per-region metric, train.py's 4 flags). Plus documenting the token↔position/band contract
  (§5.4) and sanity-checking the prereq setup (the `hf download` + pinned `bowang-lab/BioReason-Pro` clone actually
  loads). Model loading itself is *solved* — it just needs the prereq written down, not re-engineered.
- **Concrete split:** fresh agent scaffolds + distills → I review each probe/steer/train script against the
  fix-list and verify the prereq load works end-to-end. Review-heavy, not build-heavy for me.

---

## 8. Open decisions for the user
1. **New repo target** — standalone, or a `recipes/bioreason_pro/` in the new repo's tree? (affects import paths)
2. ~~External model deps — vendor or document?~~ **RESOLVED (user): document as a prerequisite.** README setup =
   `hf download wanglab/bioreason-pro-sft` + `git clone https://github.com/bowang-lab/BioReason-Pro` (**pin the
   validated commit `86d3e516e6fbbb7bc646d229520be206bf6293f9`**) + pass `--bioreason-root`. Optional: a tiny
   `check_env.py` that verifies the clone imports (`import bioreason2`) + the checkpoint loads before a run.
3. **Notebooks vs scripts** — codonfm ships scripts; do we want notebooks too (more demo-friendly, more maintenance)?
4. **Scope of probes** — which of {enrichment, domain-F1, SAE-vs-raw probe, cross-modal, echo-synthesis} are MVP
   vs later? (I'd MVP: enrichment + domain-F1 + one probe; defer cross-modal/echo-synthesis to a v2.)
5. **Layer** — ship L30 (the validated sweet spot) as the default; mention the layer-sweep finding in the README.

_Sources: `recipes/codonfm/` (target), `recipes/evo2/` (train.py), `sae/` (universal train/eval), our
`bioreason_pro/{src,scripts,multimodal_dashboard}`. Fix-list: `results/RESULTS_OVERVIEW.md` + `LAYER_SWEEP.md`._
