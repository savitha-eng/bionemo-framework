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
from PIL import Image, ImageChops


def autocrop(path, pad=14):
    """Hard-trim white borders so there is zero trailing whitespace."""
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad), min(im.width, r + pad), min(im.height, b + pad)))
        im.save(path)

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
fig, c = new_fig(6.4)
c.header("Feature 36488  —  antimicrobial-defense reasoning", big=True)
c.sub("fungal-defense AUROC 0.943   ·   the SAME feature read on all three text bands   ·   orange = activation strength")
c.gap(0.6)
c.header("Reasoning band  →  genuine mechanism (synthesis, across many proteins)")
c.wins(windows(36488, "reasoning", 6, ctx=5, maxlen=34))
c.rule()
c.header("Prompt band  →  echoes the given GO accessions (label-reading)")
c.wins(windows(36488, "prompt", 2))
c.rule()
c.header("Answer band  →  restates named effectors + GO terms")
c.wins(windows(36488, "answer", 2))
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

# ===== FIGURE D: auto-interp pipeline diagram =====
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, cx, cy, w, h, text, fc="#f5f2ec", ec="#c9c3b6", tc=INK, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.4, transform=ax.transAxes, mutation_aspect=0.45))
    ax.text(cx, cy, text, transform=ax.transAxes, ha="center", va="center", fontsize=fs, family=MONO,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>",
                                 mutation_scale=13, color="#9a948a", lw=1.6))


fig = plt.figure(figsize=(9.8, 7.2)); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.text(0.5, 0.975, "The auto-interp pipeline", ha="center", family=MONO, fontsize=15, fontweight="bold", color=INK)
ax.text(0.5, 0.945, "read a feature per band, name it, then never trust the name alone",
        ha="center", family=MONO, fontsize=9.5, color=MUTED)
# --- band separation strip ---
BW = 0.30
box(ax, 0.19, 0.875, BW, 0.072, "prompt band\nGO/IPR echo", fc="#f2f0ea", ec="#cfc9bc", fs=8.8)
box(ax, 0.50, 0.875, BW, 0.072, "reasoning band\nmechanism (synthesis)", fc="#fdecd9", ec="#e0932f", fs=8.8, bold=True)
box(ax, 0.81, 0.875, BW, 0.072, "answer band\nrestatement", fc="#f2f0ea", ec="#cfc9bc", fs=8.8)
ax.text(0.5, 0.818, "read the feature PER BAND — we interpret the reasoning band (accessions stripped)",
        ha="center", family=MONO, fontsize=8.5, color=MUTED, style="italic")
# --- two tracks ---
LX, RX, W, H = 0.26, 0.74, 0.42, 0.082
ax.text(LX, 0.762, "LABELING  (a hypothesis)", ha="center", family=MONO, fontsize=10, fontweight="bold", color="#c26a1a")
ax.text(RX, 0.762, "SCORING  (trustworthy)", ha="center", family=MONO, fontsize=10, fontweight="bold", color="#5f8f4c")
LY = [0.685, 0.560, 0.435, 0.310]
box(ax, LX, LY[0], W, H, "top-firing reasoning windows")
box(ax, LX, LY[1], W, H, "strip GO: / IPR: accessions")
box(ax, LX, LY[2], W, H, "mark the full active phrase\n(span-aware → long phrases)")
box(ax, LX, LY[3], W, H, "LLM names the concept", fc="#fdecd9", ec="#e0932f")
for i in range(3):
    arrow(ax, LX, LY[i] - H / 2, LX, LY[i + 1] + H / 2)
box(ax, RX, LY[0], W, H, "GO / InterPro labels")
box(ax, RX, LY[1], W, H, "per-feature AUROC\n(does it predict the concept?)")
box(ax, RX, LY[2], W, H, "AUROC score", fc="#e6f0e0", ec="#7fae6b")
box(ax, RX, LY[3], W, H, "grounded synthesis score\n(long phrases · mechanism · − echo)", fc="#e6f0e0", ec="#7fae6b")
for i in range(2):
    arrow(ax, RX, LY[i] - H / 2, RX, LY[i + 1] + H / 2)
box(ax, 0.5, 0.16, 0.88, 0.088,
    "discipline:  rank by score  →  read the marked windows  →  the LLM label is only a hint",
    fc="#f5f2ec", ec="#b9b3a6", fs=10.5, bold=True)
arrow(ax, LX, LY[3] - H / 2, 0.40, 0.16 + 0.044)
arrow(ax, RX, LY[2] - H / 2, 0.60, 0.16 + 0.044)
ax.text(0.5, 0.052, 'e.g. F23726 labeled "amino acid transport" (miss) — but AUROC 0.975 → genuinely fungal',
        ha="center", family=MONO, fontsize=8.5, color=MUTED, style="italic")
fig.savefig(OUT + "fig_autointerp_pipeline.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_autointerp_pipeline.png")

# ===== FIGURE E: rich fungal feature card (F23726, multi-example, 3 bands, recurring vocab) =====
fig, c = new_fig(6.0)
c.header("Feature 23726  —  filamentous fungi (pathogen detector)", big=True)
c.sub("fungal-defense AUROC 0.975   ·   recurring vocabulary across examples: filamentous · fungi · Aspergillus · Helminthosporium · Peronospora · antifungal")
c.gap(0.5)
c.header("Reasoning band  →  names specific fungi + mechanism (multiple proteins)")
c.wins(windows(23726, "reasoning", 4, ctx=6, maxlen=32))
c.rule()
c.header("Prompt band  →  GO-accession echo    ·    Answer band  →  restatement")
c.sub("prompt (given annotations)")
c.wins(windows(23726, "prompt", 1))
c.gap(0.3)
c.sub("answer (final GO output)")
c.wins(windows(23726, "answer", 1))
fig.savefig(OUT + "fig_fungal_feature_card.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_fungal_feature_card.png")

# ===== FIGURE F: domain-F1 localization panel (nucleic-acid-binding, robust; not kinesin-only) =====
DOM = [
    (13950, "RNA-binding domain (RRM)", "domain-F1 0.95 · 60 regions · fires on the RNP β-sheet residues"),
    (8277, "Zinc finger, C2H2-type", "domain-F1 0.93 · 48 regions"),
    (18162, "Helix-loop-helix DNA-binding", "domain-F1 0.93 · 27 regions"),
    (4647, "the AUROC-oversell contrast", "AUROC 0.98  BUT  domain-F1 0.00 — fires OUTSIDE any domain (why AUROC alone misleads)"),
]
fig, c = new_fig(5.4)
c.header("Protein-domain features localize  —  domain-F1", big=True)
c.sub("residue-band firing (amino acids; orange = activation). domain-F1 rewards firing INSIDE the annotated domain, not merely correlating with it.")
c.gap(0.5)
for i, (fid, dom, sub) in enumerate(DOM):
    c.header(f"Feature {fid}  —  {dom}")
    c.sub(sub)
    c.wins(windows(fid, "protein", 1, ctx=(12 if fid == 4647 else 2), maxlen=48))
    if i < len(DOM) - 1:
        c.rule()
c.gap(0.2)
c.sub("(kinesin F18393 scores the highest domain-F1, 0.98, but on only 15 regions — small-n; this RNA / zinc / HLH nucleic-acid-binding panel is the robust result.)")
fig.savefig(OUT + "fig_domain_f1_panel.png", dpi=170, facecolor="white", bbox_inches="tight")
print("wrote fig_domain_f1_panel.png")

# hard-crop all generated figures
for f in ["fig_autointerp_walkthrough", "fig_feature_gallery", "fig_crossmodal_alignment",
          "fig_autointerp_pipeline", "fig_fungal_feature_card", "fig_domain_f1_panel"]:
    autocrop(OUT + f + ".png")
print("autocropped all")
