# BioReason-Pro SAE Interpretability — Experiments & Probes, Explained

_A plain-language guide to what each experiment is, how it works, its context in the BioReason-Pro paper,
and what we found. (Paper details are from our reading of BioReason-Pro §2.7, §4.5 — verify specifics.)_

## The model
**BioReason-Pro** (Wang Lab) is a **multimodal biological reasoning model**:
- **ESM3** encodes each protein (sequence → structure-aware per-residue embeddings).
- A **GO-graph encoder** adds ontology context.
- A **Qwen3-4B reasoning LLM** takes the protein embeddings + a text prompt (InterPro domains, a predicted-GO
  hint, UniProt summary, protein–protein interactions) and **reasons in text** to predict the protein's
  function (Gene Ontology terms) — 73.6% Fmax on CAFA.

So the model literally writes a reasoning trace ("This protein contains a chromo domain … which binds
methyl-lysine on histone tails …") before answering. That reasoning trace is our main object of study.

**The paper's own interpretability (§4.5) is correlational, no SAEs, no causal intervention:** it shows the
model's *attention* concentrates on structurally-critical cryo-EM **contact residues** (§4.5.5–6, e.g. eEFSec/
7ZJW, CFAP61/8J07), and that the LLM *reorganizes* the protein embeddings by function across layers (§4.5.4).
Our work adds the **mechanistic / causal** layer using SAEs.

## The tool: Sparse Autoencoders (SAEs)
We train an SAE on the LLM's residual stream at **layer 30**. It decomposes each 2560-dim activation into
**40,960 sparse features**, each ideally one human-nameable concept. Two token "bands" we analyze separately:
- **protein / residue band** — the ESM3 embeddings (one per amino acid): the model's *sequence/structure* view.
- **reasoning band** — the text the model generates: its *functional reasoning*.

---

## Experiment 1 — Feature discovery & validation (GO-AUROC "overlap")
**Biological question:** do individual SAE features correspond to specific biological concepts?
**Method:** pool a feature's activation over each protein's reasoning tokens → a per-protein score; compute
AUROC against each GO annotation. High AUROC ⇒ the feature selectively fires on proteins with that function.
This is **not a trained classifier** — it uses the feature's own activation (Jared calls it "GO-label overlap"
= monosemanticity, distinct from "probing").
**Result:** clean monosemantic **reasoning features** — e.g. **F16494** fires on synapse proteins (AUROC 0.99),
on words like *synapse / postsynaptic / cleft / boutons / connectivity*; similarly for mitochondrion, hormone
response (F39407 → glucocorticoids/cortisol/progesterone), oxidoreductase, transporter, etc.
`scripts/validate_features.py`

## Experiment 2 — Steering (the causal test)
**Question:** are these features *causally used*, or merely correlated?
**Method:** take a cluster of features for one concept (e.g. 6 synapse features) and, while the model
generates, **clamp them ON** in the residual stream; read whether the concept appears and whether the text
stays coherent. **Controls:** a random matched-norm direction (should do nothing); selectivity (synapse
cluster should inject synapse, not mito). The SAE only supplies the direction — the **original model** generates.
**Result:** **feature-specific causal injection** — synapse/mito/nervous-system clusters inject their concept;
random does nothing; ER/transporter don't steer; kinase/oxidoreductase are baseline-confounded. But it's
mostly **lexical injection** (concept words grafted onto intact reasoning, ~73%) rather than genuine
**reasoning-redirection** (~31%, by an independent LLM-judge). _[A normalization-calibration fix was just
applied; numbers being refreshed.]_ `scripts/steer_generation.py`, validation `scripts/analyze_traces.py` +
`scripts/llm_judge_steering.py`

## Experiment 3 — Synthesis vs echo (circularity check)
**Question:** when a feature fires on "hormone," is the model *synthesizing* new reasoning, or just *echoing*
the prompt (which already contains GO/InterPro hints)? Echo would be circular and uninteresting.
**Method:** reconstruct the *words* a feature fires on; novelty = fraction NOT present in that protein's prompt.
**Result:** most reasoning features are **genuine synthesis** — F39407 fires on glucocorticoids/cortisol/
progesterone/estradiol (novelty ~0.64), words the model *generated*, not copied. So the features reflect the
model's own reasoning. `scripts/synthesis_probe_word.py` (word-level; the earlier subword version had a
junk-feature artifact now fixed).

## Experiment 4 — Structural probes (non-circular, sequence-grounded)
_Parallels the paper's §4.5.5–6 (attention→structure), but we probe the **representation** directly with a
trained linear classifier ("probing" in Jared's sense), non-circular because the target is structure, not a
functional description._
- **InterPro-domain probe** — predict which structural domain a residue belongs to, from its activation.
  Result: **~0.99 AUROC, SAE ≈ raw** → structural domains are at *ceiling* in the ESM3 residues.
  `scripts/interpro_probe.py`, `scripts/residue_domain_probe.py`
- **Contact-residue probe** — predict buried (3D core) vs surface, using **AlphaFold** structures.
  Result: **raw 0.89 > SAE 0.87** → 3D burial is encoded in residues; the SAE doesn't beat raw.
  `scripts/contact_residue_probe.py` (+ `download_alphafold.py`)

**Takeaway:** **structure lives in the residues at ceiling; the SAE re-expresses it, doesn't improve it.**
Contrast: GO *function* is weak in residues (~0.83) but ~0.95 in the reasoning band → **function emerges in the
reasoning**. And the SAE never beats raw on decodability (expected — its value is interpretability + steering).

## Experiment 5 — Cross-modal fusion (the open question, most relevant to Jared)
_Context: SAE-V (vision+text) found features that fuse modalities. We ask the analogous question._
**Question:** do **protein-band** features and **reasoning-band** features *fuse* (co-fire on the same
biology), or does the model keep modalities separate?
**Status/result:** no clear cross-modal fusion features — consistent with BioReason-Pro having **no explicit
protein–text alignment objective**. This motivates testing a model trained *with* alignment, to see if that
unlocks clean cross-modal features. _(This is the highest-value new probe to build out.)_

---

## The honest one-paragraph arc (for the conversation)
We applied SAEs to a multimodal protein+reasoning model (a first), and found **interpretable reasoning-band
features that track biology** (synapse, hormone, mitochondrion …), validated by overlap. They're **causally
real** — clamping them injects the concept into the model's reasoning (feature-specific, random-null control) —
though mostly as lexical injection (~31% genuine redirection, honestly measured). Representationally,
**structure is at ceiling in the ESM3 residues while function emerges in the reasoning**, and the SAE adds
*interpretability*, not decodability (it never beats raw). We find **no cross-modal fusion features**, which
fits the lack of an alignment objective and motivates trying an aligned model. Open/in-flight fixes: steering
magnitude calibration (found + fixed) and word-level autointerp quality.
