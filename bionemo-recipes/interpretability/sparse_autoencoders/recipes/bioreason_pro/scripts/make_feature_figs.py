"""Anthropic-style SAE feature-activation figures for the BioReason-Pro blog.

Sequential orange ramp = activation magnitude (one hue, light->dark; dataviz rule).
Real top-activating windows, per-token highlighting, monospace so layout is
deterministic. Flowing top-down cursor layout (no fixed-panel whitespace); char/line
metrics computed from the figure size so token advance matches the glyph advance.
"""
import numpy as np, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/"
       "bioreason_pro/multimodal_dashboard/public/l30_balanced/feature_examples.parquet")
OUT = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/"
       "bioreason_pro/results/charts/")
df = pq.read_table(PUB).to_pandas()
INK, MUTED = "#1a1a1a", "#6b6b6b"
CMAP = plt.cm.Oranges
MONO = "DejaVu Sans Mono"
FS = 9.0
FIG_W = 9.4


def windows(fid, band, n=3, ctx=5, maxlen=30):
    sub = df[(df.feature_id == fid) & (df.band == band)].nlargest(n, "max_activation")
    out = []
    for _, x in sub.iterrows():
        toks = str(x["sequence"]).split(); act = np.asarray(x["activations"], float)
        m = min(len(toks), len(act))
        if not m:
            continue
        toks, act = toks[:m], act[:m]
        pk = int(act.argmax()); thr = 0.30 * act[pk]
        fire = np.where(act >= thr)[0]
        s, e = (int(fire.min()), int(fire.max())) if len(fire) else (pk, pk)
        lo, hi = max(0, s - ctx), min(m, e + ctx + 1)
        if hi - lo > maxlen:
            lo, hi = max(0, pk - maxlen // 2), min(m, pk + maxlen // 2 + 1)
        a = act[lo:hi]; a = a / (a.max() + 1e-9)
        out.append(list(zip(toks[lo:hi], a)))
    return out


class Cursor:
    """Top-down flowing layout on one axis; metrics derived from figure size."""
    def __init__(self, fig, ax, fig_h):
        self.fig, self.ax = fig, ax
        self.cw = 0.60 * FS / (72 * FIG_W)          # glyph advance in axes-fraction x
        self.lh = 1.5 * FS / (72 * fig_h)           # line height in axes-fraction y
        self.max_cols = int(0.98 / self.cw)
        self.y = 0.995

    def gap(self, k=1.0):
        self.y -= self.lh * k

    def header(self, text, big=False):
        self.ax.text(0.008, self.y, text, transform=self.ax.transAxes, family=MONO,
                     fontsize=FS + (3.5 if big else 1.8), fontweight="bold", color=INK, va="top")
        self.gap(1.9 if big else 1.5)

    def sub(self, text):
        self.ax.text(0.008, self.y, text, transform=self.ax.transAxes, family=MONO,
                     fontsize=FS - 1.2, color=MUTED, va="top")
        self.gap(1.35)

    def wins(self, ws):
        for w in ws:
            col = 0
            for tok, a in w:
                t = tok if tok else "·"
                if col + len(t) + 1 > self.max_cols:
                    self.gap(); col = 0
                if a > 0.06:
                    self.ax.text(0.008 + col * self.cw, self.y, t, transform=self.ax.transAxes,
                                 family=MONO, fontsize=FS, va="top",
                                 color=(INK if a < 0.55 else "white"),
                                 bbox=dict(boxstyle="square,pad=0.10", fc=CMAP(0.18 + 0.82 * a), ec="none"))
                else:
                    self.ax.text(0.008 + col * self.cw, self.y, t, transform=self.ax.transAxes,
                                 family=MONO, fontsize=FS, va="top", color=MUTED)
                col += len(t) + 1
            self.gap(1.25)
        self.gap(0.6)

    def rule(self):
        self.ax.axhline(self.y + self.lh * 0.3, 0.008, 0.99, color="#e0ddd6", lw=1.0)
        self.gap(0.7)


def new_fig(h):
    fig = plt.figure(figsize=(FIG_W, h)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    return fig, Cursor(fig, ax, h)


# ===== FIGURE A: auto-interp walkthrough (F36488, per-band contrast) =====
fig, c = new_fig(3.7)
c.header("Feature 36488  —  antimicrobial-defense reasoning", big=True)
c.sub("fungal-defense AUROC 0.943   ·   the SAME feature read on two bands   ·   orange = activation strength")
c.gap(0.6)
c.header("Reasoning band  →  genuine mechanism")
c.wins(windows(36488, "reasoning", 3))
c.rule()
c.header("Prompt band  →  echoes the given GO accessions")
c.wins(windows(36488, "prompt", 3))
fig.savefig(OUT + "fig_autointerp_walkthrough.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_autointerp_walkthrough.png")

# ===== FIGURE B: feature gallery =====
GAL = [
    (15088, "reasoning", "Feature 15088  —  mRNA translation control", "poly(A)-binding protein · eIF4E/eIF4G · CCR4-NOT deadenylase   ·   reasoning band"),
    (30993, "reasoning", "Feature 30993  —  transporter (alternating-access antiport)", "outward-open cavity · cystine binding · Na+ coordination   ·   reasoning band"),
    (5221,  "reasoning", "Feature 5221  —  RAS signaling cascade", "GEF SOS1 · GAPs RASA1/NF1 · effectors RAF1/BRAF/PIK3CA   ·   reasoning band"),
]
fig, c = new_fig(5.2)
c.header("SAE feature gallery", big=True)
c.sub("reasoning-concept, localized-domain, and mechanistic-synthesis features   ·   orange = activation strength")
c.gap(0.6)
for i, (fid, band, title, sub) in enumerate(GAL):
    c.header(title)
    c.sub(sub)
    c.wins(windows(fid, band, 2))
    if i < len(GAL) - 1:
        c.rule()
fig.savefig(OUT + "fig_feature_gallery.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_feature_gallery.png")

# ===== FIGURE C: cross-modal alignment (pairing scatter + paired windows) =====
P = np.load("/data/savithas/phase3_full/pooled_protein_l30_00e1d8.npz")
R = np.load("/data/savithas/phase3_full/pooled_reasoning_l30_00e1d8.npz")
keep = P["keep"] & R["keep"]
SP, SR = P["SAE"][keep], R["SAE"][keep]
DOT = "#3b6ea5"


def scatter(ax, pf, rf, label):
    x, y = SP[:, pf], SR[:, rf]
    r = np.corrcoef(x, y)[0, 1]
    m = (x > 0) | (y > 0)
    ax.scatter(x[m], y[m], s=7, c=DOT, alpha=0.30, edgecolors="none")
    ax.set_xlabel(f"protein-band F{pf} activation", fontsize=8.5, color=MUTED)
    ax.set_ylabel(f"reasoning-band F{rf}", fontsize=8.5, color=MUTED)
    ax.set_title(label, fontsize=10.5, color=INK, fontweight="bold", loc="left", family=MONO)
    ax.text(0.95, 0.08, f"r = {r:.2f}", transform=ax.transAxes, ha="right", fontsize=12,
            color=INK, fontweight="bold", family=MONO)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcbc2")
    ax.tick_params(labelsize=7, colors=MUTED)


FIG_C_H = 5.4
fig = plt.figure(figsize=(FIG_W, FIG_C_H)); fig.patch.set_facecolor("white")
fig.text(0.02, 0.965, "Cross-modal alignment: protein detectors <-> reasoning features",
         family=MONO, fontsize=FS + 3.5, fontweight="bold", color=INK)
fig.text(0.02, 0.925, "each dot = one protein; a protein feature's per-protein activation vs its best-correlated reasoning feature (unsupervised pairing)",
         family=MONO, fontsize=FS - 1.2, color=MUTED)
ax1 = fig.add_axes([0.09, 0.60, 0.37, 0.27]); scatter(ax1, 7369, 3184, "GPCR pair")
ax2 = fig.add_axes([0.60, 0.60, 0.37, 0.27]); scatter(ax2, 18393, 15673, "kinesin pair")
axt = fig.add_axes([0, 0, 1, 1]); axt.axis("off")
ct = Cursor(fig, axt, FIG_C_H); ct.y = 0.50
ct.header("What the aligned GPCR pair fires on   (F7369  <->  F3184)")
ct.sub("protein detector (residues) — the structural signature")
ct.wins(windows(7369, "protein", 1))
ct.gap(0.4)
ct.sub("reasoning partner (text) — the mechanism the model writes for those proteins")
ct.wins(windows(3184, "reasoning", 1))
fig.savefig(OUT + "fig_crossmodal_alignment.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_crossmodal_alignment.png")
