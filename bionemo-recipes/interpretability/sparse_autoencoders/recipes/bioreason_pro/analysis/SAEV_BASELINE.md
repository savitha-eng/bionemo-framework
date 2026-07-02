# SAE-V Reference Replication — a control for our cross-modal findings

**Status:** proposed (2026-06-29). Owner: a dedicated agent, run in parallel with the BioReason-Pro work.
**One-line goal:** establish whether our null cross-modal result on BioReason-Pro reflects *our framework*
or *the model itself*, by reproducing a published multimodal-SAE result with a reference implementation.

---

## ⭐ AGENT KICKOFF (read this first)

You are a dedicated agent on a separate dev pod. All paths below are on shared NFS `/data` (visible from any
pod). Prefix: `/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/`.

**Reading order:**
1. **This doc** (the brief — goal, plan, resources, decision table, guardrails).
2. **Familiarize with the phase-3 subtree** — our extract→train→eval pattern you'll reuse:
   - `recipes/bioreason_pro/scripts/crossmodal_cooccur.py` — the per-sample Eq.7 cross-modal metric you must validate.
   - `recipes/bioreason_pro/scripts/train.py` + `sae/src/sae/architectures/topk.py` — our SAE + training entrypoint (you'll train an SAE on the VLM with these).
   - `sae/src/sae/activation_store.py` — the parquet ActivationStore contract (extractor → store → train).
   - `recipes/bioreason_pro/scripts/extract.py` — how we stream model activations into a store (the pattern to mirror for the VLM).
   - `recipes/bioreason_pro/STATUS.md` + `analysis/INPUT_ABLATION.md` — *why* this matters (the 455/0 null + the +15% ablation).
3. External: `github.com/PKU-Alignment/{SAE-V, SAELens-V, TransformerLens-V}` and `huggingface.co/PKU-Alignment/SAE-V` (pretrained ckpts).

**Work in a NEW dir on shared NFS: `/data/savithas/saev-baseline/`** (already created, empty). Clone the PKU
forks into it; do NOT modify the phase-3 subtree (read it for patterns only). Use `/data` (not `~/` or
`/scratch`) so the work survives a pod change. The node's GPUs are currently free.

**Make a FRESH venv** — do NOT reuse the bioreason-pro env (`/data/savithas/bioreason-pro/.venv`); it's pinned to
our Qwen3/ESM3 stack and the PKU forks + VLMs will want different (often older) torch/transformers versions.
Create a clean venv for this work (`python -m venv /data/savithas/saev-baseline/.venv`), install the forks' pinned
requirements there, and `pip install -e` our `sae` package into it for Phase 2 (so you can reuse our
TopKSAE/train.py/ActivationStore against the VLM activations). Record the exact versions that worked.

**Opening task:** see "Plan" below — **prioritize Phase 2** (train our SAE on the VLM); run Phase 1 in parallel
only if it's quick. **Hard-stop at ~1 day of env setup per fork** — if a fork won't run, report the blocker and
pivot, don't rabbit-hole. Append findings to a `## Results` section at the bottom of this doc.

### 🧭 PLAN FIRST, then hand off (execution may be done by Codex — no chat context)
**Before cloning anything or writing code:** (1) read this brief in full, (2) skim the phase-3 subtree files
listed above (especially `crossmodal_cooccur.py`, `train.py`, `activation_store.py`, `extract.py`), then
(3) **write a concrete step-by-step plan to `/data/savithas/saev-baseline/PLAN.md`** and **wait for human approval
before executing.**

⚠️ **Execution will likely be handed to a *different* agent (Codex) that has NONE of this conversation's
context.** Therefore *everything must live in files*, not in chat memory:
- `PLAN.md` must be self-contained and executable by a fresh agent: exact model ID to start with, the exact
  residual-stream hook (module path + layer), how to map the forward into our `ActivationStore` (cite the
  `extract.py` pattern), the **exact `train.py` command** (all flags), and the **exact `crossmodal_cooccur.py`
  command** to run the metric. Use **absolute paths** and **literal commands** — assume the executor can read
  files and run bash but knows nothing else.
- Record env setup as runnable commands (venv creation, `pip install` lines, fork commits/versions) in PLAN.md
  so Codex can reproduce the environment deterministically.
- Keep a running `## Results` log (in this file or a sibling `RESULTS.md`): what worked, configs, metrics,
  gotchas — for both the handoff and the future multimodal-SAE skill.
- Do NOT rely on Claude-Code-specific features (skills, etc.); write the plan in plain "read file / run command /
  edit file" terms that any agent can follow.

---

---

## Why this matters (the scientific question)

Our headline BioReason-Pro finding is a **tension**:
- **The model fuses** — zeroing protein/GO input embeddings raises the model's own reasoning-token CE by
  **+15%** (n=300, t=31, p≪0.001). Fusion is real and causal.
- **The SAE does NOT show fused features** — on the full 117k-protein dataset, 455/40,960 latents co-fire on
  both protein and text bands, but **0 are aligned** (within-sample SAE-V Eq.7 cosine > 0.3). The co-firing is
  on *unrelated* content. The SAE decomposes the fusion into separate per-modality features.

This is interesting **only if we trust our pipeline.** Right now we can't fully distinguish three explanations:
1. **Real finding** — protein+text fusion is genuinely distributed, not localized to single features.
2. **Framework bug** — our extraction / SAE / cross-modal metric (omega, band-mass, per-sample cosine) is wrong.
3. **Model-specific difficulty** — BioReason-Pro is just harder to interpret than a "nice" multimodal model.

A reference replication separates these. **SAE-V** (PKU-Alignment, ICML 2025) is the paper whose Eq.7 cross-modal
cosine metric we use; it interpreted vision-language models (LLaVA-NeXT-7B, Chameleon-7B) and **reported clean
cross-modal features.** If we can reproduce *their* result, our metric/pipeline is sound — and our BioReason-Pro
null is a real finding (explanation 1/3, not 2). If we *can't*, we've found our bug (explanation 2).

**Caveat (important):** SAE-V supports *vision*-language models only. TransformerLens-V will not support
BioReason-Pro's architecture (ESM3 embeds + GO-graph encoder + Qwen3). So this is an **indirect** control: we
validate the *method* on their model, not interpret BioReason-Pro directly. That's still decisive for ruling
out a framework bug and for calibrating "what a clean cross-modal result looks like."

---

## SAE-V exact configs (from their HF checkpoint card)

| Base model | HF ID (inferred) | Gated? | Hook layer | act dim | expansion | # features |
|---|---|---|---|---|---|---|
| LLaVA-NeXT-7B (Mistral) | `llava-hf/llava-v1.6-mistral-7b-hf` | **open** | **16** | 4096 | **16** | 65,536 |
| Chameleon-7B / Anole | `facebook/chameleon-7b` (gated) / `leloy/Anole-7b-v0.1-hf` (open) | mixed | 8 | 4096 | 32 | 131,072 |

~30k of the features actually activate during training (high dead rate — matches our experience).

**Recommended target: LLaVA-NeXT-Mistral-7B, layer 16, expansion 16.** It's open (no gating) AND uses
expansion 16 — *the same as our `l24_balanced` / Matryoshka runs* — so Phase 2 becomes a near-exact
apples-to-apples comparison. (For the downscale-first pass, still bring the pipeline up on a small open VLM
like Qwen2-VL-2B; then scale to LLaVA-NeXT-Mistral-7B layer 16 for the headline comparison.)

## Resources

- Paper repo (poster/docs): https://github.com/PKU-Alignment/SAE-V
- Source (forks — real code): https://github.com/PKU-Alignment/SAELens-V , https://github.com/PKU-Alignment/TransformerLens-V
- **Pretrained SAE checkpoints:** https://huggingface.co/PKU-Alignment/SAE-V  ← use these; do NOT retrain in Phase 1
- Models they interpret: LLaVA-NeXT-7B, Chameleon-7B (HF)
- Our metric to cross-check: `scripts/crossmodal_cooccur.py` (per-sample Eq.7 cosine), `scripts/build_dashboard.py` (omega)

---

## Reference commands (adapt for the VLM; full templates in `/data/savithas/phase3_full/run_*.sh`)

These are the literal phase-3 commands to adapt — copy the flags, change the store/layer/checkpoint. Put the
final adapted versions in `PLAN.md`.

```bash
# 0) env (fresh venv; install forks' pins there, then our sae package editable)
python -m venv /data/savithas/saev-baseline/.venv && source /data/savithas/saev-baseline/.venv/bin/activate
pip install -e /data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/sae   # our SAE/ActivationStore/train

# 1) EXTRACT — mirror recipes/bioreason_pro/scripts/extract.py: run the VLM forward, register a
#    forward_hook on the chosen residual-stream layer, append hidden[mask] to an ActivationStore
#    (a dir of shard_*.parquet + metadata.json). Tag each token's modality (image vs text) in a
#    token_labels.parquet sidecar (mirror how we tag protein/text). Cast bf16->fp16/fp32 (Arrow can't store bf16).

# 2) TRAIN our SAE on the VLM activations (TopK; flags mirror our runs):
torchrun --nproc_per_node <N> scripts/train.py \
  --cache-dir <store>/layer<L> --layer <L> --dp-size <N> \
  --model-type topk --expansion-factor 16 --top-k 128 --normalize-input --normalize-loss \
  --auxk 2048 --auxk-coef 0.03125 --dead-tokens-threshold 20000000 \
  --init-pre-bias --aggregate-loss --dead-count-global --mix-shards 10 --presample-shards 8 \
  --lr 1e-4 --lr-schedule cosine --warmup-steps 1000 --batch-size 4096 --n-epochs 1 \
  --wandb --wandb-project saev-baseline --checkpoint-dir <ckpt_dir>
# NOTE: dead_latents spikes to ~95% around step ~1k then RECOVERS to ~20-25% by ~step 5-6k (AuxK). Normal — don't panic.

# 3) CROSS-MODAL METRIC (the thing to validate): per-sample Eq.7 cosine + dump the feature list
python scripts/crossmodal_cooccur.py --sae <ckpt>/checkpoint_final.pt --store <store> --layer <L> \
  --pair image-text --shards 9999 --dump-json <out>.json
# "image-text" requires the token_labels band column to use those band names; adapt --pair to your band tags.
```

## Plan (Phase 2 PRIORITIZED, time-boxed)

**Shared prerequisite (both phases need this):** get a VLM running and **hook its residual stream at one layer**
to dump activations. TransformerLens-V gives the hook; a plain HF `forward_hook` may also work and avoids the
fork entirely — **try that first if the fork fights you.**

### ⬇️ DOWNSCALE FIRST — bring the pipeline up on a SMALL VLM before the 7B
Do **not** start on a 7B model. Get the whole loop (load VLM → hook → extract → ActivationStore → train our SAE
→ run our metric) working end-to-end on a **small HF-native VLM** first — it makes the iteration loop minutes
instead of hours and de-risks everything before you spend compute on the big model. Good candidates (pick
whichever loads cleanest with a hookable text-stream residual):
- **Qwen2-VL-2B-Instruct** — small, strong, HF-native, easy hooks (recommended first target).
- **SmolVLM-Instruct (2.2B)** or **SmolVLM-256M/500M** — tiny, fastest possible bring-up.
- **Moondream2 (~1.9B)** — very small.
- **TinyLLaVA / LLaVA-Phi (~3B)** — small but keeps the *LLaVA architecture*, so it's closer to SAE-V's setup.

**Gating (HF access):** the recommended small models are **open / not gated** — `Qwen2-VL-2B-Instruct`
(Apache-2.0), `HuggingFaceTB/SmolVLM*` (Apache-2.0), `vikhyatk/moondream2` (Apache-2.0), TinyLLaVA, and
`llava-hf/*` LLaVA-NeXT (open). So **downscaling on Qwen2-VL-2B or SmolVLM needs no permission request.** The one
to watch is **`facebook/chameleon-7b` — GATED** (Meta license, must request access on HF + `hf auth login`). So:
if SAE-V's pretrained checkpoint (Phase 1) is the *Chameleon* one, request Chameleon access early; if it's the
*LLaVA-NeXT* one, no gating. Either way, Phase 2 can proceed immediately on an open small model.

Downscale the rest too for the first pass: fewer image-text samples, smaller expansion (e.g. 8), one layer.
Once the pipeline produces sane SAE features + a cross-modal readout on the small model, **then** scale up:
- **Phase 2** can stay on a small/mid VLM (the control is "does our pipeline find cross-modal features on a
  multimodal model" — any real VLM works); use a bigger one only if results are ambiguous.
- **Phase 1 cannot downscale** — it loads SAE-V's *pretrained checkpoint*, which is tied to their exact model
  (LLaVA-NeXT-7B / Chameleon-7B). So Phase 1 requires the 7B; that's another reason to lead with Phase 2.

### Phase 2 — Replicate the SAE with OUR library  ★ PRIORITY
The *stronger* control, and it leans on our **already-validated** training code, so it's also the more robust
path. Steps:
1. Extract the VLM's residual-stream activations into our `ActivationStore` (mirror `extract.py`; cast fp16/fp32).
2. Train an SAE with **our `train.py` / TopKSAE** (same flags we use: normalize-input/-loss, auxk, aggregate-loss).
3. Run **our** `crossmodal_cooccur.py` (per-sample Eq.7) + band-mass on it; compare to SAE-V's reported cross-modal
   features (qualitatively — same kinds of image+text features?).
- **Comparable cross-modal features** → our *entire* pipeline (extract→train→interpret) is sound on a multimodal
  model → any BioReason-Pro null is **intrinsic to protein+text**, not our code. **Definitive control.** ✅
- **Worse** → our training/extraction has a gap → fix before trusting BioReason-Pro SAEs. 🐛

**Why prioritize this over Phase 1:** Phase 1 depends on getting their *entire* inference stack + checkpoint
format working (SAELens-V); Phase 2 only needs the VLM forward + a hook, then our own (known-good) training. So
Phase 2 is both the stronger result and the lower fork-dependency risk. **If Phase 1 setup isn't working
quickly, drop it and put everything on Phase 2.** This phase is also the artifact that makes a multimodal-SAE
skill credible (see below).

### Phase 1 — Validate OUR metric against THEIR pretrained SAE  (parallel / opportunistic)
Run **in parallel** if their forks come up easily, else defer. **Use their released checkpoint — no training.**
1. Load a pretrained SAE-V checkpoint on its VLM; reproduce **one** cross-modal result from the paper.
2. Run **our** `crossmodal_cooccur.py` on their SAE's activations.
3. **Compare:** does our metric flag the same features SAE-V reports?
   - **Match** → our metric is correct → the BioReason-Pro 455/0 null is trustworthy. ✅
   - **Mismatch** → bug in our cross-modal metric → fix it. 🐛

Both phases share the VLM+hook prerequisite, so they can run in **one repo, in parallel**: extract once, then
(2) train our SAE *and* (1) load theirs off the same activations.

---

## Success criteria / decision table

| Phase 2 (our SAE on VLM) | Phase 1 (our metric on their SAE, if run) | Conclusion |
|---|---|---|
| comparable cross-modal features | matches / not run | ✅ Our full pipeline works on a multimodal model → BioReason-Pro null is a real finding about protein+text fusion |
| worse than SAE-V | matches | ⚠️ Metric OK but training/extraction gap → fix it, retrain BioReason-Pro |
| worse than SAE-V | mismatches | 🐛 Metric bug too → fix `crossmodal_cooccur.py`/omega first, then revisit training |

Phase 2 alone is decisive; Phase 1 is a useful corroboration of the metric specifically but not required for the
main conclusion.

---

## Guardrails for the agent
- **Use the pretrained checkpoint in Phase 1.** Do not retrain until Phase 1 validates.
- **Time-box setup:** if the forks aren't running after ~1 day, STOP and report the blocker — don't rabbit-hole.
- Report which exact models/datasets/commits were used (their repos under-document this).
- Keep it isolated from the BioReason-Pro work (separate env/dirs; the node's GPUs are currently free).
- Log findings here as you go (append a "## Results" section).

---

## ✍️ Document findings for the multimodal-SAE skill (do this AS YOU GO)

This experiment is also the raw material for a future **"train + interpret an SAE on a multimodal foundation
model" skill.** Keep a running, structured log (append to `## Results` here, or a sibling `SKILL_NOTES.md`) of
the *reusable* lessons — not just the final result. Specifically capture:
- **Per-model hook recipe:** which model, exact residual-stream hook site (module path / layer), how you got
  `[B, S, H]` out, image-token vs text-token handling, dtype casting.
- **Extraction → store:** how you mapped the VLM forward into our `ActivationStore` (the pattern that worked).
- **Training config that trained well:** expansion, top_k, normalize-input/-loss, auxk, LR/schedule, and the
  FVU / dead-% it reached (so the skill has known-good defaults for a *multimodal* residual stream).
- **Cross-modal readout:** how the metric applied, what a clean cross-modal feature looks like on a VLM.
- **Gotchas / dead-ends:** fork dependency pins, gating, hook surprises, anything that cost time — these are the
  highest-value lines in a skill (they save the next person the same debugging).
- **What generalizes vs what was model-specific.**
Two validated data points — BioReason-Pro (protein+text) + this VLM (image+text) — make the skill credible.

## Connection to a "multimodal-SAE" skill
We've accumulated reusable multimodal-SAE lessons already: normalized-FVU honesty (raw var-exp is misleading),
two distinct dead-latent metrics, modality balancing (token rebalancing vs in-loader), go-token eval leakage,
omega-as-artifact vs band-mass vs per-sample cosine, the build_dashboard OOM (don't load residuals to GPU),
Matryoshka for feature absorption. With **two** validated data points — BioReason-Pro (protein+text) *and* a
reproduced SAE-V (vision+text) — those lessons generalize into a credible skill: *"train + interpret an SAE on a
multimodal foundation model."* The SAE-V replication is what turns our anecdotes into a method.

## SAE-V cross-modal algorithm hyperparameters (Table 5 — use these EXACTLY for faithful reproduction)
| param | Cosine sim | Cooccurrence | L0 |
|---|---|---|---|
| top-K | 5 | 5 | 5 |
| text token vocab size | 32000 | 32000 | 32000 |
| vision token vocab size | 64 | 64 | 64 |
| activation bound | 1 | 1 | 1 |
| sample data size | 1000 | 1000 | 1000 |

NOTE: our BioReason-Pro cross-modal used DIFFERENT settings (top-K=8, activation/tau=2.0, full 117k samples).
To compare to SAE-V, match top-K=5, activation-bound=1, sample=1000. (vocab sizes are model properties.)
