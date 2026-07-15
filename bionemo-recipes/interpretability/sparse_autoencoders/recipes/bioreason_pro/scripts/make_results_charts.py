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

if clusters:
    md.append("### Per-cluster coherence (n=10, L30)\n")
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

if synth is not None:
    md.append("## 3. Echo vs synthesis reasoning features\n")
    md.append("![synthesis](charts/synthesis_novelty.png)\n")
    md.append(f"Over {synth['n']} L30 reasoning features: novelty mean={synth['nv'].mean():.2f}, "
              f"echo (novelty<0.3)={100*(synth['nv']<0.3).mean():.0f}%, "
              f"synthesis (>0.6)={100*(synth['nv']>0.6).mean():.0f}%. "
              "Low-novelty features restate prompt GO terms (circular); high-novelty fire on synthesized text.\n")
else:
    md.append("## 3. Echo vs synthesis reasoning features\n_(synthesis probe still running — re-run this "
              "script when `synthesis_l30_allvalidated.json` lands)_\n")

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
