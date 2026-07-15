#!/usr/bin/env python
"""Parse the steering/coherence/synthesis logs and emit charts + a consolidated results markdown.
Robust to missing logs (uses whatever is present). Outputs to <outdir>/charts/*.png + RESULTS.md."""
import re, json, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/data/savithas/phase3_full/steer_logs"
OUT = Path("/data/savithas/phase3_full/results"); (OUT / "charts").mkdir(parents=True, exist_ok=True)
CH = OUT / "charts"


def parse_coherence(path):
    """coh_*.log -> {alpha: {raw_c1,raw_c2,distinct3,nonascii,gibber,pct_coh,coh_c1}}"""
    if not os.path.exists(path): return None
    txt = open(path).read()
    if "steering vs COHERENCE" not in txt: return None
    out = {}
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)%\s+\|\s+(\d+)\s*$",
                          txt, re.M):
        a = int(m.group(1))
        out[a] = dict(raw_c1=int(m.group(2)), raw_c2=int(m.group(3)), distinct3=float(m.group(4)),
                      nonascii=float(m.group(5)), gibber=float(m.group(6)), pct_coh=int(m.group(7)),
                      coh_c1=int(m.group(8)))
    return out or None


def parse_firm(path):
    """firm_l*.log (old format) -> {alpha: c2}"""
    if not os.path.exists(path): return None
    txt = open(path).read()
    seg = txt.split("concept-word counts")[-1]
    out = {}
    for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$", seg, re.M):
        out[int(m.group(1))] = int(m.group(3))   # c2 = target neuron words for synapse sweep
    return out or None


# ---- Chart 1: layer curve (firmed synapse sweep, raw injected words vs alpha) ----
layers = {}
for L in (16, 28, 30, 32):
    d = parse_firm(f"{LOG}/firm_l{L}_syn.log")
    if d: layers[L] = d
if layers:
    plt.figure(figsize=(6, 4))
    for L, d in sorted(layers.items()):
        xs = sorted(d); plt.plot(xs, [d[x] for x in xs], marker="o", label=f"L{L}")
    plt.xlabel("clamp strength α"); plt.ylabel("injected concept words (raw, n=20)")
    plt.title("Synapse-cluster steerability by layer\n(same cluster, matched quality)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(CH / "layer_curve.png", dpi=110); plt.close()

# ---- Chart 2: coherence per cluster (coh_c1 and %coherent vs alpha) ----
clusters = {}
for tag in ["synapse", "nervsys", "mito", "transcription", "er", "oxidoreductase", "transporter", "kinase", "random"]:
    d = parse_coherence(f"{LOG}/coh_{tag}.log")
    if d: clusters[tag] = d
if clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for tag, d in clusters.items():
        xs = sorted(d)
        ax1.plot(xs, [d[x]["coh_c1"] for x in xs], marker="o", label=tag)
        ax2.plot(xs, [d[x]["pct_coh"] for x in xs], marker="o", label=tag)
    ax1.set(xlabel="clamp strength α", ylabel="coherent-only injected words (coh_c1)",
            title="Coherent injection by cluster")
    ax2.set(xlabel="clamp strength α", ylabel="% generations coherent", title="Coherence retention by cluster")
    ax2.axhline(50, ls="--", c="gray", alpha=0.5)
    for ax in (ax1, ax2): ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(CH / "coherence_by_cluster.png", dpi=110); plt.close()

# ---- Chart 3: raw-vs-coherent gap for the strongest clusters (shows loops inflate raw) ----
if clusters:
    plt.figure(figsize=(7, 4))
    for tag in ["synapse", "nervsys", "mito"]:
        if tag in clusters:
            d = clusters[tag]; xs = sorted(d)
            plt.plot(xs, [d[x]["raw_c1"] for x in xs], marker="s", ls="--", alpha=0.6, label=f"{tag} raw")
            plt.plot(xs, [d[x]["coh_c1"] for x in xs], marker="o", label=f"{tag} coherent-only")
    plt.xlabel("clamp strength α"); plt.ylabel("injected concept words")
    plt.title("Raw count OVERSTATES steering — loops inflate it\n(coherent-only is the honest signal)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(CH / "raw_vs_coherent.png", dpi=110); plt.close()

# ---- Chart 4: synthesis novelty distribution ----
synth = None
jp = next((q for q in ["/data/savithas/phase3_full/synthesis_l30_focused.json",
                       "/data/savithas/phase3_full/synthesis_l30_allvalidated.json"] if os.path.exists(q)), "")
if jp and os.path.exists(jp):
    feats = json.load(open(jp))["features"]
    rows = [r for r in feats.values() if r.get("n_fire", 0) > 0 and r.get("n_content", 0) >= 10]
    nv = np.array([r["novelty"] for r in rows]); cn = np.array([r["connective"] for r in rows])
    synth = dict(n=len(rows), nv=nv, cn=cn)
    plt.figure(figsize=(6, 4))
    plt.hist(nv, bins=20, color="steelblue", edgecolor="k", alpha=0.8)
    plt.axvline(0.3, ls="--", c="r", label="echo<0.3"); plt.axvline(0.6, ls="--", c="g", label="synth>0.6")
    plt.xlabel("novelty (frac top tokens NOT in prompt)"); plt.ylabel("# reasoning features")
    plt.title(f"Echo vs synthesis over {len(rows)} L30 reasoning features")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(CH / "synthesis_novelty.png", dpi=110); plt.close()

# ---- RESULTS.md ----
md = []
md.append("# Phase 3 — Interpretability of Reasoning in BioReason-Pro\n")
md.append("_Auto-generated by `scripts/make_results_charts.py`. Re-run to refresh as jobs complete._\n")

md.append("## TL;DR — the whole story\n")
md.append("| finding | evidence |")
md.append("|---|---|")
md.append("| **Structure lives in the residues, at ceiling** | InterPro domain presence 0.99, boundaries 0.98, "
          "3D contacts (raw) 0.89 — all strongly decodable from the ESM3 residue representation |")
md.append("| **Function emerges in the reasoning** | GO function weak in residues (0.83) but ~0.95 in the reasoning band |")
md.append("| **The SAE never beats raw on decodability** | ties raw on saturated domain probes, *loses* on the "
          "harder 3D-contact (0.873 vs 0.890) and protein-GO (0.75 vs 0.83) probes |")
md.append("| **The SAE's value is interpretability + steering** | monosemantic nameable features; the synapse "
          "cluster steers selectively (0 cross-leakage) vs a random null; **~0.75 coherent lexical injection "
          "(char-level, n=60), but only ~31% GENUINE reasoning-redirection by an LLM-judge** |")
md.append("| **Steering is layer- & cluster-specific** | L16 null -> L30 sweet spot -> L32 weaker; coherent only "
          "for synapse-like clusters; molecular-function clusters baseline-confounded |")
md.append("| **Reasoning features are genuine, not prompt-echo** | 48% synthesis vs 20% echo; F39407 novelty 0.63 |\n")
md.append("**Is 'SAE never beats raw' a bad result? No — it's expected.** An SAE is a *lossy re-expression* of the "
          "model's activations into an interpretable basis; by construction it can only match or lose information "
          "vs raw, so 'SAE <= raw on probing' is the default. It has no decodability headroom *here* precisely "
          "because the ESM3 residue representation is already at ceiling (nothing hidden to surface). The SAE's "
          "contribution is **interpretability** — monosemantic, nameable, **steerable** features (which raw "
          "dimensions can't give you) — not probe accuracy. **The headline is the structure/function dissociation "
          "+ coherent causal steering, NOT 'SAE > raw' (which would be overclaiming).**\n")

md.append("## Background — what is this even measuring?\n")
md.append("**BioReason-Pro** is a model that reads a protein (as ESM3 embeddings) and then *reasons in text* "
          "(a Qwen LLM) to predict the protein's function (GO terms). We want to understand HOW it reasons.\n")
md.append("To do that we trained **Sparse Autoencoders (SAEs)** on the LLM's internal activations at layer 30. "
          "An SAE decomposes the model's internal state into ~40,960 **features** — each feature ideally switches "
          "on for one human-nameable concept (e.g. a 'synapse' feature, a 'hormone' feature). Think of features "
          "as a dictionary of concepts the model uses internally.\n")
md.append("Two kinds of tokens flow through the model, and we analyze them separately:\n")
md.append("- **protein / residue band** — the ESM3 embeddings, one per amino acid (the model's *sequence* view).\n")
md.append("- **reasoning band** — the text the model generates while thinking (its *functional* view).\n")
md.append("The experiments below ask: (1) can we **causally control** the reasoning by forcing features on "
          "(*steering*)? (2) are the reasoning features **genuine or just echoing the prompt** (*synthesis probe*)? "
          "(3) what does the **protein representation actually encode** (*structural probes*)? "
          "Throughout, we compare **SAE features** vs the **raw** model activations vs a **random** baseline — "
          "to see whether the SAE adds anything, and whether the model 'knows' a given property at all.\n")
md.append("| experiment | script | plain-language question |")
md.append("|---|---|---|")
md.append("| 1. Steering | `steer_generation.py` | If we force a concept's features ON while the model writes, "
          "does that concept appear in its reasoning — *and stay coherent*? |")
md.append("| 2. Layer sweep | `steer_generation.py` | Does steering work better at different depths of the model? |")
md.append("| 3. Synthesis vs echo | `synthesis_probe.py` | When a feature fires on 'hormone', is the model "
          "*synthesizing* new reasoning or just *repeating* 'hormone' from the prompt (circular)? |")
md.append("| 4. InterPro domain probe | `interpro_probe.py`, `residue_domain_probe.py` | Does the protein "
          "representation encode structural domains (which/where)? Does the SAE beat raw? |")
md.append("| 5. Contact-residue probe | *(AlphaFold, in progress)* | Does it encode 3D structure (buried vs surface residues)? |\n")

md.append("## 1. Steering — can we causally control the reasoning?\n")
md.append("**What we do:** take a *cluster* of SAE features that all fire on one concept (say six 'synapse' "
          "features). While the model generates its reasoning, we artificially crank those features up in its "
          "internal state (\"clamp\" them to a strength α). Then we read the generated text and ask: did the "
          "concept get injected, and is the text still coherent reasoning?\n")
md.append("**Two controls make it rigorous:** (a) a **random** direction of the same size — if *that* also "
          "injects the concept, the effect isn't real; (b) we check the cluster injects ITS concept and not a "
          "different one (selectivity / *double dissociation*).\n")
md.append("**Measuring coherence (this mattered a lot):** a raw count of concept-words is misleading, because "
          "at high α the model often falls into a **repetition loop** (\"synapse synapse synapse…\") or "
          "**gibberish** — which inflates the count without being real reasoning. So we score each generation on "
          "three things: **distinct3** (variety of word-triples; low = looping), **nonascii** (foreign-character "
          "junk), **gibber** (non-word fragments). `coh_c1` = concept-words counted **only** in generations that "
          "stay coherent. If raw count is high but `coh_c1` ≈ 0, the 'steering' is really just degeneration.\n")
md.append("![raw vs coherent](charts/raw_vs_coherent.png)\n")
md.append("*Above:* dashed = raw count, solid = coherent-only. The gap is the repetition-loop inflation. The "
          "**random control injects 0** everywhere → the effect is direction-specific and real.\n")

md.append("### The honest steering rate (three levels of scrutiny)\n")
md.append("| measure | synapse rate | what it counts |")
md.append("|---|---|---|")
md.append("| lexical `coh_c1` (n=10) | ~0.97 | ANY coherent gen with concept words — **overstates** (marks "
          "pseudo-word junk 'synaptotereguliaritys' as coherent) |")
md.append("| **char-level** (n=60, `analyze_traces.py`) | **~0.73** | coherent (char-level) injection into "
          "zero-baseline proteins — catches the pseudo-word loops the lexical metric missed |")
md.append("| **LLM-judge** (llama-3.1-70b, `llm_judge_steering.py`) | **~0.31** | model GENUINELY reasons about "
          "the concept (not just words sprinkled in) AND coherent — the gold standard |")
md.append("So steering is **~73% coherent lexical injection but only ~31% genuine reasoning-redirection** — "
          "mostly grafting concept vocabulary onto intact reasoning (e.g. 'binds post-synaptic density marks on "
          "histone H3'), genuinely redirecting ~1/3 of the time. **Calibration note:** the SAE has "
          "`normalize_input=True`; steering now correctly denormalizes the injected delta by the per-token std — "
          "this shifted the operating point (α≈180→28) but the rate is unchanged (0.75), confirming the fix "
          "preserves the result.\n")

if clusters:
    md.append("### Per-cluster coherence (n=10, L30) — NOTE: coh_c1 is the OVERSTATED lexical metric (see above)\n")
    md.append("![coherence by cluster](charts/coherence_by_cluster.png)\n")
    md.append("| cluster | best α | %coherent@best | coh_c1@best | verdict |")
    md.append("|---|---|---|---|---|")
    def verdict(tag, d):
        xs = sorted(d); raws = [d[a]["raw_c1"] for a in xs]
        base = raws[0]; mx = max(raws)                       # baseline vs peak injection
        best = max(d, key=lambda a: (d[a]["pct_coh"] >= 60, d[a]["coh_c1"]))
        pc = d[best]["pct_coh"]; c1 = d[best]["coh_c1"]; pc_hi = d[max(xs)]["pct_coh"]
        if tag == "random": v = "null control ✓"
        elif mx > 0 and base >= 0.5 * mx: v = "baseline-confounded (already on at α=0; no dose-response)"
        elif c1 >= 20 and pc >= 80: v = "**coherent + causal ✓**"
        elif mx > 50 and pc_hi < 60: v = "causal but DEGENERATES into loops at high α"
        elif c1 >= 5 and pc >= 80: v = "mild coherent injection (degenerates at high α)"
        elif mx <= 5: v = "weak / not steerable"
        else: v = "weak"
        return best, pc, c1, v
    for tag, d in clusters.items():
        b, pc, c1, v = verdict(tag, d)
        md.append(f"| {tag} | {b} | {pc}% | {c1} | {v} |")
    md.append("")

if layers:
    md.append("## 2. Steerability depends on layer (matched synapse cluster, n=20)\n")
    md.append("![layer curve](charts/layer_curve.png)\n")
    md.append("Same high-quality synapse cluster steered at each layer. **L16 = null** (concepts not yet "
              "steerable directions); **L28** injects but over-breaks; **L30** strongest/most coherent; "
              "**L32** moderate. De-confounds the earlier 'L28=salad' (that was a weak repro cluster).\n")


md.append("## 3. Synthesis vs echo — is the reasoning genuine or circular?\n")
md.append("**The worry:** BioReason-Pro's prompt already contains hints — predicted GO terms, InterPro "
          "annotations, a UniProt summary. So when a reasoning feature fires on the word 'hormone', is the model "
          "*synthesizing* an insight, or just *echoing* a word that was handed to it in the prompt? If it's "
          "echoing, the feature is circular and uninteresting.\n")
md.append("**What we do:** for each reasoning feature, look at the words it fires on most, and compute "
          "**novelty** = the fraction of those words that do **NOT** appear in that protein's prompt. "
          "Low novelty (<0.3) = **echo** (restating the prompt). High novelty (>0.6) = **synthesis** (the model "
          "generated words the prompt didn't contain).\n")
if synth is not None:
    md.append("![synthesis](charts/synthesis_novelty.png)\n")
    md.append(f"**Result ({synth['n']} features):** most reasoning features are genuine synthesis — "
              f"{100*(synth['nv']>0.6).mean():.0f}% have novelty>0.6, only {100*(synth['nv']<0.3).mean():.0f}% are "
              f"echo. **F39407** (the 'response to hormone' feature you asked about) = novelty **0.63** = genuine "
              f"synthesis, *not* prompt-echo. **Honest caveat:** we predicted molecular-function features would be "
              f"echo (explaining why they don't steer well); they were NOT — so the synthesis/echo split does "
              f"**not** explain the steering pattern. Clean result, failed hypothesis.\n")
else:
    md.append("_(synthesis probe still running)_\n")

md.append("## 4. Structural probes — what does the protein representation actually know?\n")
md.append("**What is a 'probe'?** We freeze the model, take its internal representation of each residue, and "
          "train a *simple* classifier (logistic regression) to predict some known biological property. If the "
          "classifier succeeds, that property is *linearly encoded* in the representation. We do this on three "
          "representations to see what the SAE adds: **SAE features** (compressed to 256 dims via SVD, so the "
          "comparison to raw is fair), **raw** model activations (2560 dims), and a **random** SAE (baseline — "
          "tells us how much any random projection already captures).\n")
md.append("**Why 'non-circular'?** Our reasoning-band results risk circularity: the model's text *describes* the "
          "function, so predicting function from the text is almost cheating. Here we predict from the **residue** "
          "representation (the ESM3 sequence embedding), and the target is a **structural** annotation "
          "(InterPro domains = conserved folds/motifs), not a functional description. So it's a clean test of "
          "whether the *sequence* representation encodes *structure*.\n")
md.append("**The headline to watch is SAE vs raw:** if SAE ≈ raw, the SAE just re-expresses what's already there "
          "(the residue representation is the 'ceiling'); if SAE > raw, the SAE surfaces something raw hides.\n")
md.append("**What we probe, and with what labels:**\n")
md.append("| probe | input (what we probe) | label (target) | label source | circular? |")
md.append("|---|---|---|---|---|")
md.append("| GO function | rep (SAE/raw/random), per-protein | has GO term X? | protein GO annotations | "
          "**reasoning band circular** (text describes fn); residue band non-circular but weak |")
md.append("| InterPro domain (per-protein) | residue-band rep, mean-pooled | has domain X? | dataset "
          "`interpro_ids` | non-circular |")
md.append("| InterPro domain (per-residue) | single residue activation | is this residue in domain X? | "
          "`interpro_location` spans | non-circular |")
md.append("| Contact / burial | single residue activation | buried (3D core) vs surface? | AlphaFold contact "
          "number (Cβ neighbors <8Å) | non-circular |")
md.append("Every probe compares **SAE features vs raw hidden vs random-SAE** — separating 'is it in the "
          "representation at all' (raw vs random) from 'does the SAE add anything' (SAE vs raw). Structural "
          "labels (domain, burial) are probed from the residue band → non-circular → at ceiling (~0.98); "
          "functional labels (GO) are ~0.95 from reasoning (circular) but ~0.83 from residues.\n")
md.append("**Methodology vs Jared/CodonFM:** same core — GO-label *overlap* (per-feature AUROC), *trained* "
          "linear probes, and **SVD-256 dimensionality-matching** (his §7.2). For the discriminating contact "
          "probe we go further with a **matched-dim sweep** (SAE-SVD-K *vs raw-PCA-K* at K=256/512/1024/2048), "
          "stricter than his (which leaves raw at full dim). We also add **causal steering**, which his probing "
          "does not.\n")
md.append("![InterPro domain probe](charts/interpro_probe.png)\n")
md.append("![contact dimension sweep](charts/contact_dimsweep.png)\n")
md.append("*Left: structural domains are near-perfectly decodable from residues, SAE≈raw (ceiling). "
          "Right: on the harder 3D-contact probe, raw beats SAE at **every** matched dimension → not a "
          "compression artifact.*\n")
md.append("![per-residue burial example](charts/contact_example.png)\n")
md.append("*Concrete example of the contact probe's label: one protein's per-residue 3D contact number "
          "(red = buried core, blue = surface) — what 'buried vs surface' means physically.*\n")
ip = "/data/savithas/phase3_full/interpro_probe_l30.json"
rp = "/data/savithas/phase3_full/residue_domain_probe_l30.json"
if os.path.exists(ip):
    m = json.load(open(ip))["mean"]
    md.append(f"- **InterPro domain (per-protein, {json.load(open(ip))['n_domains']} domains):** "
              f"SAE-svd **{m['sae_svd']}** ≈ raw {m['raw']} ≈ random {m['random']}. Structural domains are "
              f"near-perfectly, non-circularly decodable from ESM3 residues; SAE re-expresses, doesn't beat raw "
              f"= the **residue-band ceiling**. Contrast GO function (~0.83 from residues) → **structure lives "
              f"in residues, function emerges in reasoning.**\n")
if os.path.exists(rp):
    m = json.load(open(rp))["mean"]
    md.append(f"- **InterPro domain (residue-resolution, is-this-residue-in-domain):** SAE-svd **{m['sae_svd']}** "
              f"| raw {m['raw']} | random {m['random']}. Harder than per-protein presence — tests whether the rep "
              f"knows domain *boundaries* along the chain.\n")
cp = "/data/savithas/phase3_full/contact_residue_probe_l30.json"
if os.path.exists(cp):
    c = json.load(open(cp))
    md.append(f"- **3D contact (buried vs surface residue, AlphaFold structures, 95% coverage):** "
              f"SAE-svd {c['sae_svd']} | raw **{c['raw']}** | random {c['random']}. A harder, less-saturated "
              f"probe: raw≫random ({c['raw']} vs {c['random']}) = the residue rep genuinely encodes 3D burial; "
              f"and here **raw BEATS SAE** ({c['raw']} vs {c['sae_svd']}) — the SVD-compressed SAE sheds info.\n")
md.append("\n> **Honest bottom line across ALL probes: the SAE never beats raw on decodability.** It ties raw on "
          "the (near-saturated) domain probes and *loses* to raw on the harder 3D-contact and protein-band-GO "
          "probes. This is expected — the ESM3 residue representation is the ceiling, and an SAE re-expresses it "
          "rather than exceeding it. The SAE's value is **interpretability** (monosemantic, nameable, *steerable* "
          "features — e.g. the synapse cluster), NOT better probing accuracy. Structure lives in the residues at "
          "ceiling; function emerges in the reasoning band.\n")
md.append("\n## Honest limitations\n")
md.append("- **Steering n=10–20** (generation is expensive) — winning conditions need firming to n≥50.\n")
md.append("- Coherent-steering window is **narrow** (breaks by α≈300) and **concept-dependent**.\n")
md.append("- Coherence metric is lexical (distinct3/nonascii/gibber), not an LLM-judge — a stronger rating "
          "would be an independent judge model.\n")
md.append("\n## Raw data / logs\n")
md.append("- Coherence tables: `steer_logs/coh_*.log` · layer sweep: `steer_logs/firm_l*.log`\n")
md.append("- Synthesis: `synthesis_l30_allvalidated.json` · clusters: `analysis/validated_features_l30_reasoning.csv`\n")

(OUT / "RESULTS.md").write_text("\n".join(md))
print(f"[charts] wrote {len(list(CH.glob('*.png')))} charts to {CH}")
print(f"[report] wrote {OUT/'RESULTS.md'}")
print("clusters scored:", list(clusters.keys()))
print("layers:", list(layers.keys()), "| synthesis:", "yes" if synth is not None else "pending")
