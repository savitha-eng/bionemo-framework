#!/usr/bin/env python
"""Layer curve: InterPro-domain decodability (SAE-svd / raw / random) across layers -> charts/layer_decodability.png.

Reads interpro_probe_l{L}.json for the layers present and plots mean AUROC vs layer. Answers: is structure
flat-high across depth (ESM3 carries it from input) or does it peak at a particular layer?
"""
import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "/data/savithas/phase3_full"
rows = []
for f in sorted(glob.glob(f"{base}/interpro_probe_l*.json")):
    L = int(re.search(r"_l(\d+)\.json", f).group(1))
    d = json.load(open(f))
    m = d.get("mean", {})
    rows.append((L, m.get("sae_svd"), m.get("raw"), m.get("random"), d.get("n_domains")))
rows.sort()
if not rows:
    raise SystemExit("no interpro_probe_l*.json found")

L = [r[0] for r in rows]
print(f"{'layer':>6} {'SAE-svd':>8} {'raw':>6} {'random':>7} {'ndom':>5}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>8} {r[2]:>6} {r[3]:>7} {r[4]:>5}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(L, [r[1] for r in rows], "o-", label="SAE (SVD-256)", lw=2)
ax.plot(L, [r[2] for r in rows], "s-", label="raw residue (2560-d / PCA-256)", lw=2)
ax.plot(L, [r[3] for r in rows], "^--", label="random-SAE (floor)", color="gray", lw=1.5)
ax.set_xlabel("LLM layer"); ax.set_ylabel("mean held-out AUROC (60 InterPro domains)")
ax.set_title("InterPro-domain decodability across layers")
ax.set_ylim(0.9, 1.0); ax.grid(alpha=0.3); ax.legend(loc="lower right")
fig.tight_layout()
out = "results/charts/layer_decodability.png"
Path("results/charts").mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"[wrote] {out}")
