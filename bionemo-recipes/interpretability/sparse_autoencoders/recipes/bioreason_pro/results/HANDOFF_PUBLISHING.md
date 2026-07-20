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

TARGET REPO + AUTHORITATIVE ARCHITECTURE (the new repo — this is where the work goes):
https://github.com/NVIDIA-BioNeMo/bionemo-interpretability.git
- Its `sparse_autoencoders/` directory SPECIFIES THE ARCHITECTURE to conform to:
  https://github.com/NVIDIA-BioNeMo/bionemo-interpretability/tree/main/sparse_autoencoders
  Clone it FIRST and inspect that layout — it is AUTHORITATIVE for where files go (recipe dir naming, sae/
  package location, scripts/ layout, README/pyproject conventions). Match it exactly.
- Create a WORKING BRANCH. Open DRAFT PRs against this repo (the user reviews + marks ready before publishing).
- Keep PRs small, same-function grouped, opened in dependency order (below).

REFERENCE (for CONTENT/patterns only — the target repo above wins on LAYOUT): the latest upstream SAE recipes
https://github.com/NVIDIA-BioNeMo/bionemo-recipes/tree/main/interpretability/sparse_autoencoders — follow
`codonfm` (README/dashboard/eval pattern) and COPY `evo2`'s scripts/train.py (only one wiring all 4 opt-in flags;
change only docstring + wandb default). If the target repo's `sparse_autoencoders/` layout differs from this
upstream, the TARGET REPO layout wins.

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
