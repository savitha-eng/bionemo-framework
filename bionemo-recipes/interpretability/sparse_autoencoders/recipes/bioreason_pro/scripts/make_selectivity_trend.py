"""Protein- and text-selective feature counts per layer, balanced vs unbalanced.
Log y-axis so the ~20-50x protein jump and the roughly flat, high text count both read.
Reads the cached layer_balance_counts.json (no SAE re-encoding needed)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC = "/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro"
d = json.load(open(f"{REC}/analysis/figures/layer_balance_counts.json"))
L = sorted(int(k) for k in d)
p_bal = [d[str(x)]["bal_ps"] for x in L]
p_unb = [d[str(x)]["unbal_ps"] for x in L]
t_bal = [d[str(x)]["bal_ts"] for x in L]
t_unb = [d[str(x)]["unbal_ts"] for x in L]
BLUE, ORANGE, INK, MUTED = "#2563eb", "#e0932f", "#1a1a1a", "#6b6b6b"

fig, ax = plt.subplots(figsize=(7.6, 4.3)); fig.patch.set_facecolor("white")
ax.plot(L, p_bal, "-o", color=BLUE, lw=2.4, ms=6, zorder=4)
ax.plot(L, p_unb, "--o", color=BLUE, lw=1.6, ms=4.5, alpha=0.55, zorder=3)
ax.plot(L, t_bal, "-o", color=ORANGE, lw=2.4, ms=6, zorder=4)
ax.plot(L, t_unb, "--o", color=ORANGE, lw=1.6, ms=4.5, alpha=0.55, zorder=3)
ax.set_yscale("log")

# direct labels at the right edge
ax.text(L[-1] + 0.4, t_bal[-1], "text, balanced", color=ORANGE, va="center", fontsize=10, fontweight="bold")
ax.text(L[-1] + 0.4, t_unb[-1] * 0.86, "text, unbalanced", color="#c8934c", va="center", fontsize=9)
ax.text(L[-1] + 0.4, p_bal[-1], "protein, balanced", color=BLUE, va="center", fontsize=10, fontweight="bold")
ax.text(L[-1] + 0.4, p_unb[-1] * 0.82, "protein, unbalanced", color="#7f9ede", va="center", fontsize=9)

# protein peak + the gain (label placed in the empty band below the protein-balanced line)
pk = L[p_bal.index(max(p_bal))]
ax.annotate(f"peak L{pk}: {max(p_bal)} protein features", (pk, max(p_bal)), xytext=(pk, 300),
            ha="center", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color="#c9c3b6", lw=1))
ax.text(L[0], 7.0, "balancing: 20–50× more protein features, text stays high",
        fontsize=9, color=MUTED, style="italic")

ax.set_xlabel("layer", fontsize=10, color=MUTED)
ax.set_ylabel("modality-selective features (log scale)", fontsize=10, color=MUTED)
ax.set_title("Balancing gives protein a vocabulary without starving text",
             fontsize=12.5, color=INK, fontweight="bold", loc="left")
ax.set_xticks(L); ax.set_xlim(L[0] - 0.5, L[-1] + 5.2); ax.set_ylim(6, 4600)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cfcbc2")
ax.tick_params(labelsize=8.5, colors=MUTED)
ax.grid(axis="y", which="both", color="#eeece7", lw=0.7, zorder=0)
fig.tight_layout()
fig.savefig(f"{REC}/results/charts/fig_selectivity_trend.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_selectivity_trend.png")
