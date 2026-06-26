# Per-feature firing across 4 groups: protein, go, prompt-text, response(reasoning)-text.
# Uses token_labels_with_role.parquet (band + role). Surfaces bio vs prompt vs reasoning features.
import argparse, glob
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

p = argparse.ArgumentParser()
p.add_argument("--sae", required=True); p.add_argument("--store", required=True)
p.add_argument("--layer", type=int, required=True); p.add_argument("--device", default="cuda")
p.add_argument("--encode-batch", type=int, default=8192)
a = p.parse_args()
dev = a.device if torch.cuda.is_available() else "cpu"

tl = pq.read_table(f"{a.store}/token_labels_with_role.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
role = np.array(tl.column("role").to_pylist(), dtype=object)
# 4 groups
grp = np.full(len(band), "?", dtype=object)
grp[band == "protein"] = "protein"
grp[band == "go"] = "go"
grp[(band == "text") & (role == "prompt")] = "prompt_text"
grp[(band == "text") & (role == "response")] = "reasoning_text"
GROUPS = ["protein", "go", "prompt_text", "reasoning_text"]
gtok = {g: int((grp == g).sum()) for g in GROUPS}
gcode = np.select([grp == g for g in GROUPS], list(range(len(GROUPS))), default=-1)
gt = torch.from_numpy(gcode).to(dev)

ck = torch.load(a.sae, map_location="cpu")
sae = TopKSAE(**ck["model_config"]).to(dev).eval(); sae.load_state_dict(ck["model_state_dict"]); H = sae.hidden_dim
fired = torch.zeros(len(GROUPS), H, dtype=torch.bool, device=dev)
shards = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
row0 = 0
with torch.no_grad():
    for sp in shards:
        X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
        for s in range(0, n, a.encode_batch):
            e = min(n, s + a.encode_batch)
            c = sae.encode(torch.from_numpy(X[s:e]).to(dev)) > 0
            gg = gt[row0 + s:row0 + e]
            for gi in range(len(GROUPS)):
                m = gg == gi
                if m.any(): fired[gi] |= c[m].any(0)
        row0 += n
f = fired.cpu().numpy()
print(f"[8k L28] {H} features ; tokens/group: " + "  ".join(f"{g}={gtok[g]:,}" for g in GROUPS))
print(f"{'group':16}{'#feats fire':>12}{'% of dict':>10}")
for gi, g in enumerate(GROUPS):
    print(f"{g:16}{int(f[gi].sum()):>12,}{100*f[gi].sum()/H:>9.1f}%")
prot, go, pt, rt = f
print("\n--- feature taxonomy ---")
print(f"  bio (protein|go)            : {int((prot|go).sum()):,}")
print(f"  fire on prompt-text         : {int(pt.sum()):,}")
print(f"  fire on reasoning-text      : {int(rt.sum()):,}")
print(f"  reasoning-ONLY (rt & ~pt & ~prot & ~go): {int((rt&~pt&~prot&~go).sum()):,}  <- candidate 'pure reasoning' features")
print(f"  prompt-ONLY  (pt & ~rt & ~prot & ~go)  : {int((pt&~rt&~prot&~go).sum()):,}")
print(f"  shared prompt+reasoning (pt & rt)      : {int((pt&rt).sum()):,}")
print(f"  bio that also fire on reasoning        : {int((prot|go).astype(bool) & rt).sum() if True else 0:,}")
