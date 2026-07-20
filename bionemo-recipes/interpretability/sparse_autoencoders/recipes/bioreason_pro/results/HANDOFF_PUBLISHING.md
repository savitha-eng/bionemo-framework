# Handoff Prompt — Publish BioReason-Pro SAE Recipe (Day 1: training/core pipeline)

_Copy the block below into a fresh session. This conversation stays as the research + decision record.
Full scoping detail lives in `CONTEXT_PUBLISHING.md` (same dir), which the prompt reads first._

---

```
You are cleaning up working BioReason-Pro SAE code into a publishable recipe (codonfm/evo2 style) and opening it
as small, reviewable DRAFT PRs. The code already works — this is packaging, not building.

FIRST, read these on disk (full context + the fix-list):
- bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro/results/CONTEXT_PUBLISHING.md
  (scoping doc: file manifest, PR plan, §5.5 fix-list, the prerequisite decision)
- .../recipes/bioreason_pro/results/RESULTS_OVERVIEW.md and LAYER_SWEEP.md (what the code produces)
Invoke the `sae-recipe-v1` skill — it has the extract→train→eval pattern and train.py gotchas.

TARGET REPO (the new repo — clone via SSH, the key has access; confirmed working):
  git@github.com:NVIDIA-BioNeMo/bionemo-interpretability.git  (default branch: main)
CONFIRMED LAYOUT (already inspected) — the recipe goes at `sparse_autoencoders/recipes/bioreason_pro/`
(does NOT exist yet — create it). Match `recipes/codonfm/` EXACTLY:
  sparse_autoencoders/
  ├── README.md, pyproject.toml, .ci_build.sh, uv.lock, .gitignore
  ├── sae/                         # shared universal train/eval + tests (already there — reuse, don't fork)
  └── recipes/
      ├── esm2/, codonfm/, evo2/   # references
      └── bioreason_pro/           # <-- CREATE THIS, mirroring codonfm:
          ├── README.md, pyproject.toml, .gitignore
          ├── run.py  and/or  <size>.sh    (orchestrator)
          ├── run_configs/config.yaml
          ├── src/bioreason_pro_sae/       # OUR src already matches this convention (model_loader.py, data.py)
          ├── scripts/
          │   ├── _paths.py                # <-- codonfm CENTRALIZES paths here. USE THIS to fix our hardcoded
          │   │                            #     absolute paths cleanly instead of scattering args.
          │   ├── extract.py, train.py, eval.py, dashboard.py, launch_dashboard.py, download_*.py ...
          └── <model>_dashboard/           # React app (like codon_dashboard/: src, package.json, vite.config.js)
- Copy `evo2`'s scripts/train.py (only one wiring all 4 opt-in flags; change only docstring + wandb default).
  Use codonfm as the structural template for everything else (README numbered pipeline, eval.py, _paths.py, dashboard).
- Create a WORKING BRANCH, push via SSH (works from this env). Open DRAFT PRs (needs `gh` or the web UI — the
  user's env has gh; if a PR can't be opened here, push the branch and tell the user to open the draft).
- PRs small, same-function grouped, dependency order (below). User reviews + marks ready before publishing.

SCOPE — DAY 1 ONLY: the training / core pipeline. Do NOT touch the analysis scripts (day 2 — user still reviewing
them). Do NOT distill all 111 scripts. Source code lives at
bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro/ (src/ + scripts/).

MODEL LOADING (SOLVED — just document it as a prerequisite, do NOT vendor):
- Weights = ONE self-contained HF checkpoint `wanglab/bioreason-pro-sft` (ESM3, GO, projections all bundled).
- Its config is Qwen3ForCausalLM with no modeling code, so from_pretrained alone gives a plain LLM. The multimodal
  assembly lives in the authors' repo `bowang-lab/BioReason-Pro`.
- Our `src/bioreason_pro_sae/model_loader.load_bioreason_pro_sft(ckpt_dir, bioreason_pro_root)` handles it:
  ckpt_dir = HF snapshot, bioreason_pro_root = a clone of the authors' repo (for `import bioreason2`).
- README PREREQUISITE: `hf download wanglab/bioreason-pro-sft` + `git clone
  https://github.com/bowang-lab/BioReason-Pro` pinned at commit 86d3e516e6fbbb7bc646d229520be206bf6293f9 + pass
  --bioreason-root. Our validated local paths: checkpoint at
  /data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b ;
  authors' clone at /data/savithas/bioreason-pro.
- MUST verify the prereq load actually works before finalizing PR2.

PR STACK (draft PRs, dependency order):
  PR1 Scaffold: README (with the prereq above), pyproject.toml, .gitignore, run.sh, run_configs/config.yaml
  PR2 Model integration: src/bioreason_pro_sae/{model_loader.py, data.py} — load + token/band contract.
      Only judgment-heavy PR; document why the unsloth stub + GO patch + placeholder-token contract exist.
  PR3 Extraction: scripts/extract.py (streaming → ActivationStore)
  PR4 Training: scripts/train.py (copy evo2's latest)
  PR5 Eval: scripts/eval.py (loss-recovered / reconstruction / dead-latents)

FIXES TO PRESERVE (do not regress — CONTEXT_PUBLISHING.md §5.5): train.py's 4 opt-in flags
(--aggregate-loss, --dead-count-global, --mix-shards, --presample-shards); never copy an older train.py.

ENV/GIT: recipe venv at recipes/bioreason_pro/.venv (no pip — use `VIRTUAL_ENV=... /root/.local/bin/uv pip`).
Never commit dashboard node_modules/dist/*.log. Use `gh` for the draft PRs. End commit messages with
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

START: read the docs + invoke sae-recipe-v1, clone the target repo and inspect its structure, create a branch,
verify the prereq load works, then open DRAFT PR1 + PR2. Keep the user posted as each PR lands.
```

---

_The commit pin `86d3e51` is what our working clone is on (validated). PRs are DRAFTs against
`NVIDIA-BioNeMo/bionemo-interpretability`; the user reviews before marking ready. Day 2 (analyses) is a
separate handoff once the user has reviewed that code + results._
