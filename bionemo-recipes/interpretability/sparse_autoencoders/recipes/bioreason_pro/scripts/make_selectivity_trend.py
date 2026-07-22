"""Concise 1-panel trend: protein-selective feature count per layer, balanced vs unbalanced.
Reads the cached layer_balance_counts.json (no SAE re-encoding needed)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC = "/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro"
d = json.load(open(f"{REC}/analysis/figures/layer_balance_counts.json"))
L = sorted(int(k) for k in d)
bal = [d[str(x)]["bal_ps"] for x in L]
unbal = [d[str(x)]["unbal_ps"] for x in L]
BLUE, GREY, INK, MUTED = "#2563eb", "#9aa3ad", "#1a1a1a", "#6b6b6b"

fig, ax = plt.subplots(figsize=(7.2, 3.7)); fig.patch.set_facecolor("white")
ax.plot(L, bal, "-o", color=BLUE, lw=2.2, ms=6, zorder=3)
ax.plot(L, unbal, "-o", color=GREY, lw=2.0, ms=5, zorder=2)
# direct labels (no legend box)
ax.text(L[-1] + 0.4, bal[-1], "balanced", color=BLUE, va="center", fontsize=11, fontweight="bold")
ax.text(L[-1] + 0.4, unbal[-1] + 34, "unbalanced\n(natural mix)", color="#79818b", va="center", fontsize=9.5, linespacing=1.1)
# peak marker
pk = L[bal.index(max(bal))]
ax.annotate(f"peak L{pk}\n{max(bal)} features", (pk, max(bal)), xytext=(pk, max(bal) + 90),
            ha="center", fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color="#c9c3b6", lw=1))
ax.set_xlabel("layer", fontsize=10, color=MUTED)
ax.set_ylabel("protein-selective features", fontsize=10, color=MUTED)
ax.set_title("Modality balancing gives protein a vocabulary at every layer",
             fontsize=12, color=INK, fontweight="bold", loc="left")
ax.set_xticks(L); ax.set_xlim(L[0] - 0.5, L[-1] + 3.5); ax.set_ylim(0, max(bal) + 150)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cfcbc2")
ax.tick_params(labelsize=8.5, colors=MUTED)
ax.grid(axis="y", color="#eeece7", lw=0.8, zorder=0)
fig.tight_layout()
fig.savefig(f"{REC}/results/charts/fig_selectivity_trend.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_selectivity_trend.png")
