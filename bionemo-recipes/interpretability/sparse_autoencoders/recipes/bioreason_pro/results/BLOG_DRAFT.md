# Reading a protein-reasoning model's mind — and keeping ourselves honest about it

_A sparse-autoencoder interpretability study of NVIDIA's BioReason-Pro. What its internal
"concept dictionary" really contains, what it doesn't, and the leak-floor discipline that
separates the two._

---

BioReason-Pro answers a deceptively simple question — "what does this protein do?" — in a
deceptively complex way. It reads a protein as ESM3 residue embeddings, folds in a Gene-Ontology
(GO) graph encoder, and then a Qwen3-4B language model **reasons in text** toward a GO-function
prediction. We wanted to know *how* it reasons: which concepts live inside the model, whether it
genuinely synthesizes biology or just parrots its prompt, and whether the protein "sequence" view
and the text "reasoning" view ever talk to each other.

Our tool is a sparse autoencoder (SAE) trained on the model's own residual stream. The short
version of what we found: the SAE isolates beautiful, nameable **structural** features; a genuine,
**distributed microbial-defense reasoning circuit**; and almost everything a first pass would call
"cross-modal fusion" that dissolves under scrutiny into prompt-mediated correlation. The longer
version — and the reason we trust the first version at all — is a set of disciplines we imposed on
ourselves to avoid fooling ourselves. That discipline is the real subject of this post.

## What we built

We trained a TopK SAE (`sae-l30-exp16-balanced`) on **layer-30** residual activations of the Qwen3
decoder: expansion factor 16, top-k = 128, taking the 2,560-dim residual to a **40,960-feature**
dictionary, with `normalize_input=True`. Crucially it is **multimodal in one stream** — protein
residue tokens and reasoning-text tokens flow through together, each tagged by band, so we can ask
per-band questions of the same feature.

The SAE lives in a self-contained recipe (the first **decoder-LLM** SAE recipe in
`bionemo-framework`) that hooks the real fused forward — ESM3 embeddings and the GO band spliced
into their placeholder slots exactly as in production inference. The training loop is boring in the
best way: a smoke test drove the fraction-of-variance-unexplained from **0.90 → 0.04 (≈96% variance
explained) in 350 steps**. (One honesty note we surface loudly in the recipe: measured in
*normalized* space, where every token is weighted equally, the honest variance-explained is ~0.76–0.81,
not the ~0.97 that raw-space accounting inflates it to — the residual stream is dominated by a single
magnitude direction, and a raw-space metric rewards the SAE for re-inserting it. We report the
normalized number.)

## How we kept ourselves honest

Interpretability is unusually easy to fool yourself in, because the model's own text tells you the
answer. Two disciplines governed everything downstream.

### The leak floor

BioReason-Pro reasons in words, and those words frequently **name the function** the protein has.
So if you train a linear probe on the SAE's reasoning-band features to predict a GO concept, a high
accuracy proves nothing — a *random-init* SAE decodes the same concept nearly as well, because the
concept is written on the page.

Our fix: for every concept, train the identical probe on a **random SAE** and call that the **leak
floor**. Only the **margin `sae_sparse − leak`** is trustworthy. When we did this across 12 designed
GO concepts, the mean sparse-SAE AUROC was 0.986 — impressive until you see the mean **leak floor is
0.946**. Once you subtract it, **only 4 of 12 concepts clear a +0.05 margin**:

| concept | sparse | leak floor | margin | verdict |
|---|---|---|---|---|
| defense → fungus | 0.993 | 0.882 | **+0.111** | real (cleanest) |
| defense → bacterium | 0.993 | 0.885 | **+0.108** | real |
| structural molecule | 0.985 | 0.922 | +0.063 | real |
| plasma membrane | 0.978 | 0.923 | +0.055 | real but diffuse (362 feats) |
| nucleus, catalytic, kinase, mitochondrion, transporter, reproduction, oxidoreductase, sexual-repro | ~0.98 | ~0.95–0.98 | +0.006…+0.035 | **leakage** |

Kinase activity scores 0.992 — and is **leakage**, because a random projection scores 0.979. The
retraction "function emerges in reasoning (~0.95)" is exactly this mistake caught in our own earlier
work. The leak floor is the single most important thing in this study.

### Scoring is not labeling (and magnitude is a trap)

When we do trust a feature, we rank it by a **label-grounded per-feature AUROC** — does this one
feature, on its own, predict the ground-truth GO annotation? That's a *score*. What an SAE dashboard
usually shows you instead is a *label*: an LLM's one-shot read of the feature's top-firing windows.
A label is a **hypothesis**, and we have the receipts to prove it:

- Auto-interp **missed** F23726 — a genuine filamentous-fungi detector at AUROC **0.975** — and
  called it "amino acid transport," even though 6 of its 8 top windows are *Aspergillus*,
  *H. sativum*, "filamentous fungi."
- Auto-interp **oversold** F15775 "autophagy": it *sounds* antimicrobial, but its per-feature AUROC
  is **0.572** — a co-occurring correlate, not a defense feature.

So the rule is: **rank by AUROC, read the windows to confirm, treat the auto-interp label as a
naming hint.** And never rank by magnitude — every protein-band feature is low-magnitude (~0.3) by
design; magnitude tells you nothing about quality.

## Three tiers of findings, strongest first

### Tier 1 — Structural features (the SAE's cleanest win)

The SAE isolates crisp structural detectors: of 1,187 features passing FDR < 0.01, 872 are
InterPro-domain features scoring AUROC 0.97–1.00. But per-protein AUROC **overstates
localization** — it only says "these proteins activate the feature more," not "the feature fires
*on the domain*." So we adopted a residue-level metric, **domain-F1** = per-position precision
(fraction of firing residues inside the domain) × per-region recall (fraction of domain regions
with ≥1 firing residue).

The two metrics barely agree (correlation 0.23), and the gap is the story: **only 148 of 872
(~17%)** AUROC-selected structural features genuinely localize (F1 ≥ 0.5).

- **F18393 (kinesin motor) — domain-F1 = 0.98.** The gold standard: it fires *inside* the motor
  domain and nowhere else.
- **F4647 (kinesin conserved-site) — AUROC 0.982 but domain-F1 = 0.00.** It fires, reliably, on the
  right proteins — but *never on the domain's residues.* This is the AUROC oversell in one feature,
  and the reason domain-F1 exists.

### Tier 2 — The microbial-defense reasoning circuit (the flagship)

The cleanest reasoning result is not one "defense feature" but a **distributed circuit**:
pathogen-specific detectors sitting on a **shared innate-immune core**. Ranked by per-feature AUROC,
the antifungal probe's 12 features split cleanly:

| tier (per-feature AUROC) | features | reading |
|---|---|---|
| genuine fungal ≥0.94 | F2808 (0.99), F35336 (0.99), F32785 (0.99), F23726 (0.98), F36488 (0.94) | the real core + detectors |
| weak correlate 0.6–0.72 | F7665 cell-wall, F22156, F5047 motility | marginal |
| co-occurring ~0.5–0.59 | F2082 chromatin, F15775 autophagy, F29332, F28215 | L1 picked them as co-predictors, not "meaning" |

The biology is sensible: **fungus ∩ bacterium = {F2808, F32785, F35336}** — a shared innate-immune
core — while **F23726 detects filamentous fungi** and **F22077 detects bacterial lipopolysaccharide
(LPS)**, the canonical pathogen signatures. And it is genuine reasoning, not echo: split by band,
these features fire on given `GO:0009620` accessions in the *prompt* band (echo) but on
mechanism — *"redox gating of pattern-recognition receptors," "growth inhibition of H. sativum"* —
in the *reasoning* band (synthesis).

The honest caveat is the punchline of the whole thesis: this is a **reasoning-band phenomenon
only**. A combined protein+reasoning probe decodes defense at AUROC 0.978 but recruits 16
protein-band features of which exactly **1 localizes** and **none enrich for any defense term** (the
best is "circadian rhythm," FDR ~0.6). **The protein band has no residue-level defense feature.**
Function is weak in the residues; structure is strong. Domain-F1 doesn't even apply — defense is a
function, not a domain with residue spans.

### Tier 3 — Cross-modal alignment (the weakest, and the most seductive)

Do the protein view and the reasoning view share concepts? We asked two ways, and they **disagree** —
which is itself the finding.

- **Unsupervised pairing:** take a bio feature's per-protein activation vector, Pearson-correlate it
  against every reasoning feature's, keep the single best partner. Every one of 20 bio features
  finds a concept-matched partner (permutation null 0.07–0.25): F18393 kinesin ↔ **F15673 "Motor"**
  (r = 0.62), F7369 GPCR ↔ F3184 "Transmembrane" (r = 0.89). Multiple bio features for one fold even
  converge on one reasoning feature (4 kinesin → F15673).
- **Supervised combined probe:** train one L1 probe on `[protein ; reasoning]` for a concept, see
  which features it recruits.

The kinesin *pairing* features are **not** the ones the kinesin *probe* recruits. That's not a bug —
they answer different questions ("what co-fires with this detector?" vs "what decodes this concept?").
But it should make you suspicious of any "fusion" story, and the causal test confirms the suspicion:
a 2×2 intervention shows the reasoning feature causally **writes** its concept (injecting a P450
reasoning cluster: +49 coherent cytochrome/heme words) while the structure/residue feature is
causally **inert** (residue main-effect ≈ 0). Even a perfectly localized structure feature — F18393
kinesin at domain-F1 0.98 — moves reasoning by +2 (noise) when injected. **Cross-modal alignment is
prompt-mediated correlation, not a causal feature-to-feature flow.**

## A method contribution: scoring good reasoning automatically

Reading windows by hand doesn't scale, so we built a grounded synthesis-quality scorer
(`synth_span_scorer.py`) that ranks reasoning features **with no LLM in the loop**. It (0) drops
**broadband** features that fire on everything (act_freq > 0.20 — e.g. F2124 at 94% activation is a
"reasoning-mode" signal, not a concept), (1) keeps only features that fire on long **contiguous
phrases** (1,307 of 8,842 qualify; single-token firing is lexical echo), and (2) scores each phrase
for **mechanism-verbs + inference/conclusion language + beyond-prompt named entities − echo/accession
density**, per-window so mixed features are flagged rather than averaged out.

It **discovered** genuine mechanistic-reasoning features on its own: **F12706** (DNA-TF motif
recognition — *"T-rich major-groove signature… TAAT-centered"*), **F15088** (mRNA translation control),
**F30993** (transporter alternating-access), **F5221** (RAS signaling — *"GEF such as SOS1… effectors
RAF1/BRAF"*), and **F36488** (plant-immunity defense synthesis, the same feature the fungal probe
ranks at AUROC 0.94). This is how you *find* good reasoning instead of eyeballing it.

## A labeling finding worth stating plainly

The deployed dashboard's static labels come from a coarse ~59-term GO-slim vocabulary that is
**58% "none"** (6,122 of 10,579 features) and collapses onto giants ("catalytic activity" ×875).
So our genuine filamentous-fungi feature F23726 reads "regulation of gene expression," and the
gorgeous plant-immunity synthesizer F36488 reads "none." The feature is fine; **the label vocabulary
is the failure.** The fix already lives in the recipe's `eval_enrichment.py`: specific GO+IPR
enrichment, gated by per-feature AUROC, with auto-interp as fallback.

## What we did NOT find (kept visible on purpose)

- **No causal cross-modal flow.** Structure/residue features are causally inert; alignment is
  prompt-mediated. Even a domain-F1-0.98 feature doesn't feed reasoning.
- **No residue-level function detector.** The protein band encodes structure at ceiling but has no
  clean defense (or GO-function) feature.
- **The SAE never beats raw on decodability.** It ties raw on saturated domain probes and *loses* on
  harder ones (3D burial 0.873 vs 0.886). That's expected — an SAE is a lossy re-expression; its
  value is interpretability, not accuracy.
- **Auto-interp is a hypothesis, not ground truth** — it misses genuine features and oversells
  correlates.
- **"Synthesis" is not an SAE-feature phenomenon.** Echo-vs-synthesis is decodable (~0.93) but
  SAE ≈ raw (0.914 vs 0.916) — it's a residual-stream property. The individual synthesis features
  are still useful handles; the global claim is withdrawn.

Why so many negatives? A single fact explains them: the prompt hands the model a `go_pred`
speculation that already contains **68% of the ground-truth GO terms**, and the final answer echoes
it 38% of the time. The model largely **refines a given answer** rather than deriving it from
residues — so you can't steer a prediction that's two-thirds handed over, and structure doesn't need
to feed reasoning. What the model *adds* is mechanism and elaboration on top — which is exactly what
the microbial-defense circuit and the synthesis features capture.

That is the honest shape of the result: a real, distributed, interpretable reasoning circuit sitting
on top of a largely pre-specified answer, surrounded by a lot of correlational structure that only
looks like fusion until you subtract the leak floor.
