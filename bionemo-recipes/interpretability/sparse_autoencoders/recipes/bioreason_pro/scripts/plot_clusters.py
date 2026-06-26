# Quick standalone UMAP scatter of SAE features colored by band_class (from modality_clusters output).
# Lets you SEE whether protein/go/prompt/reasoning features form distinct clusters, no dashboard needed.
#   python scripts/plot_clusters.py <atlas.parquet> <out.png>
import sys
import numpy as np, pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

atlas, out = sys.argv[1], sys.argv[2]
t = pq.read_table(atlas)
x = np.asarray(t.column("x").to_pylist()); y = np.asarray(t.column("y").to_pylist())
bc = np.asarray(t.column("band_class").to_pylist(), dtype=object)
COL = {"protein": "#2a9d8f", "go": "#e9c46a", "prompt": "#8d99ae", "reasoning": "#e76f51",
       "text": "#8d99ae", "dead": "#dddddd"}
order = ["dead", "text", "prompt", "reasoning", "go", "protein"]  # bio on top
fig, ax = plt.subplots(figsize=(11, 9))
for b in order:
    m = bc == b
    if not m.any():
        continue
    ax.scatter(x[m], y[m], s=(3 if b in ("dead", "text", "prompt", "reasoning") else 18),
               c=COL.get(b, "#000"), label=f"{b} ({int(m.sum())})",
               alpha=(0.25 if b in ("dead", "text", "prompt", "reasoning") else 0.9),
               edgecolors="none")
ax.set_title(f"SAE feature UMAP by band — {atlas.split('/')[-1]}")
ax.legend(markerscale=2, loc="best", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(out, dpi=130)
print(f"wrote {out}  ({len(x)} features)")
