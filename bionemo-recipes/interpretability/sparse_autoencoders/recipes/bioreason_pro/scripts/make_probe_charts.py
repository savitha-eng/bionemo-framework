#!/usr/bin/env python
"""Charts for the sequence-grounded structural probes (InterPro domain + AlphaFold contact residue)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

D = "/data/savithas/phase3_full"
OUT = Path(D) / "results" / "charts"; OUT.mkdir(parents=True, exist_ok=True)

# --- 1. InterPro domain probe (per-protein + residue): SAE vs raw vs random ---
ip = json.load(open(f"{D}/interpro_probe_l30.json")); rp = json.load(open(f"{D}/residue_domain_probe_l30.json"))
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["InterPro domain\n(per-protein)", "InterPro domain\n(per-residue)"]
sae = [ip["mean"]["sae_svd"], rp["mean"]["sae_svd"]]
raw = [ip["mean"]["raw"], rp["mean"]["raw"]]
rnd = [ip["mean"]["random"], rp["mean"]["random"]]
x = np.arange(2); w = 0.25
ax.bar(x - w, sae, w, label="SAE-svd256", color="steelblue")
ax.bar(x, raw, w, label="raw", color="darkorange")
ax.bar(x + w, rnd, w, label="random", color="gray")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("held-out AUROC"); ax.set_ylim(0.9, 1.0)
ax.set_title("Structural domains: near-perfectly decodable from residues\nSAE ≈ raw (residue is the ceiling)")
ax.legend(); ax.grid(alpha=0.3, axis="y")
for i, (s, r, n) in enumerate(zip(sae, raw, rnd)):
    for xoff, v in [(-w, s), (0, r), (w, n)]:
        ax.text(i + xoff, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "interpro_probe.png", dpi=120); plt.close()

# --- 2. Contact-residue dimension sweep: SAE-svd vs raw-pca across K ---
cs = json.load(open(f"{D}/contact_dimsweep_l30.json"))
Ks = [256, 512, 1024, 2048]
saeK = [cs.get(f"sae_svd{k}") for k in Ks]; rawK = [cs.get(f"raw_pca{k}") for k in Ks]
plt.figure(figsize=(6, 4))
plt.plot(Ks, saeK, "o-", label="SAE-svd", color="steelblue")
plt.plot(Ks, rawK, "s-", label="raw-pca", color="darkorange")
plt.axhline(cs["raw_full2560"], ls="--", color="darkorange", alpha=0.5, label=f"raw-full 2560 ({cs['raw_full2560']})")
plt.axhline(cs["random"], ls=":", color="gray", label=f"random ({cs['random']})")
plt.xscale("log", base=2); plt.xticks(Ks, Ks)
plt.xlabel("compression dimension K"); plt.ylabel("buried-vs-surface AUROC")
plt.title("3D contact (buried vs surface residue)\nraw beats SAE at every matched K — not a dim artifact")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUT / "contact_dimsweep.png", dpi=120); plt.close()

# --- 3. EXAMPLE: one AlphaFold protein's per-residue contact number (buried vs surface) ---
import glob
import biotite.structure.io.pdbx as pdbx, biotite.structure as struc
cif = None
for f in sorted(glob.glob(f"{D}/af_structures/*.cif")):
    if 200 < os.path.getsize(f) and os.path.getsize(f) < 400000:  # a modest-size protein
        cif = f; break
if cif:
    acc = Path(cif).stem
    arr = pdbx.get_structure(pdbx.CIFFile.read(cif), model=1); arr = arr[struc.filter_amino_acids(arr)]
    cb = arr[(arr.atom_name == "CB") | ((arr.res_name == "GLY") & (arr.atom_name == "CA"))]
    xyz = cb.coord; d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    cn = ((d < 8) & (d > 0.1)).sum(1)
    plt.figure(figsize=(9, 3.2))
    plt.plot(cn, color="teal", lw=0.8)
    hi, lo = np.quantile(cn, 0.67), np.quantile(cn, 0.33)
    plt.axhline(hi, ls="--", color="firebrick", alpha=0.6, label=f"buried (>{hi:.0f} contacts, core)")
    plt.axhline(lo, ls="--", color="royalblue", alpha=0.6, label=f"surface (<{lo:.0f} contacts)")
    plt.fill_between(range(len(cn)), hi, cn, where=(cn >= hi), color="firebrick", alpha=0.25)
    plt.xlabel("residue position"); plt.ylabel("contact number (3D neighbors <8Å)")
    plt.title(f"Example: {acc} — per-residue 3D burial (the probe's label; red=buried core)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "contact_example.png", dpi=120); plt.close()
    print(f"example protein: {acc} ({len(cn)} residues, {(cn>=hi).sum()} buried, {(cn<=lo).sum()} surface)")
print(f"[wrote] {OUT}/interpro_probe.png, contact_dimsweep.png, contact_example.png")
