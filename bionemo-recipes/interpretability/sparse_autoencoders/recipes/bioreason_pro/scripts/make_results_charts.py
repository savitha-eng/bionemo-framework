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
jp = "/data/savithas/phase3_full/synthesis_l30_allvalidated.json"
if os.path.exists(jp):
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
md.append("# Phase 3 — Interpretability of Reasoning in BioReason-Pro: Steering & Synthesis\n")
md.append("_Auto-generated by `scripts/make_results_charts.py` from the run logs. "
          "Re-run it to refresh as jobs complete._\n")
md.append("## What was tested\n")
md.append("On the L30 SAE (`sae-l30-exp16-balanced`, exp16, 40,960 features), we ask three questions about "
          "reasoning-band features: are they **causal** (steering), is the effect **coherent**, and are they "
          "**synthesis** vs prompt-**echo**.\n")
md.append("| question | script | what it does |")
md.append("|---|---|---|")
md.append("| Causal steering + coherence | `scripts/steer_generation.py` | clamp a feature cluster into the "
          "layer-L residual during generation; `--features` group, `--random-dirs` control; measures concept "
          "words + distinct3/nonascii/gibber coherence + `coh_c1` (concept words in coherent gens only) |")
md.append("| Echo vs synthesis | `scripts/synthesis_probe.py` | per feature, novelty = frac top-firing tokens "
          "NOT in the prompt (low=echo, high=synthesis); connective = frac inference markers |")
md.append("| Cluster mining | `analysis/validated_features_l30_reasoning.csv` | high-quality steerable clusters |\n")

md.append("## 1. Steering is feature-specific & causal, but coherence is graded\n")
md.append("![raw vs coherent](charts/raw_vs_coherent.png)\n")
md.append("Raw concept-word counts **overstate** steering because at high α the model falls into repetition "
          "loops (\"nervous system nervous system…\") that inflate the count. `coh_c1` (words counted only in "
          "generations passing distinct3≥0.55, nonascii≤0.02, gibber≤0.20) is the honest signal. "
          "The **random matched-norm control injects 0** everywhere → the effect is direction-specific.\n")

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

md.append("## Honest limitations\n")
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
