"""RESIDUE-CONTEXT VIEW: for a protein-band SAE feature, show WHICH residues it fires on with the InterPro
domain span marked -- the noise-proof way to verify localization (the dashboard can't, because protein-band
features are low-magnitude). For each of a few domain-containing proteins, prints a compact ASCII track:
the domain region and the feature's firing residues along the sequence, so you SEE whether they overlap.

Usage: residue_view.py <sae.pt> <store> <layer> --feature 18393 --domain IPR001752 --n-proteins 30"""
import argparse, sys, glob, json
from pathlib import Path
import numpy as np, torch, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "src")
from bioreason_pro_sae.model_loader import _install_unsloth_stub; _install_unsloth_stub()
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
import bioreason_pro_sae.data as brp

p = argparse.ArgumentParser()
p.add_argument("sae"); p.add_argument("store"); p.add_argument("layer", type=int)
p.add_argument("--feature", type=int, required=True)
p.add_argument("--domain", required=True, help="InterPro id whose span to mark, e.g. IPR001752")
p.add_argument("--n-proteins", type=int, default=30, help="top proteins (by feature activation) to show")
p.add_argument("--tau", type=float, default=0.0, help="fires if activation > tau")
p.add_argument("--width", type=int, default=90, help="ASCII track width")
a = p.parse_args(); dev = "cuda"

# residue rows + within-protein residue index
tl = pq.read_table(f"{a.store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64)
N = len(band); prot = band == "protein"
df = pd.DataFrame({"pid": rpid[prot].astype(str), "tidx": rtidx[prot], "row": np.arange(N)[prot]})
df = df.sort_values(["pid", "tidx"]).reset_index(drop=True)
df["res"] = df.groupby("pid").cumcount()  # 0-indexed residue position

# interpro spans for the target domain
tr, _, _ = brp.load_reasoning_splits(max_length_protein=2000)
loc_of = {}
for pid, loc in zip(tr["protein_id"], tr["interpro_location"]):
    if not loc: continue
    try: loc_of[str(pid)] = json.loads(loc) if isinstance(loc, str) else loc
    except Exception: pass
def spans(pid):
    s = loc_of.get(pid, {}).get(a.domain)
    if not s: return []
    return [tuple(x) for x in s] if isinstance(s[0], list) else [tuple(s)]
has_domain = set(pid for pid in df["pid"].unique() if spans(pid))
df = df[df["pid"].isin(has_domain)].reset_index(drop=True)
print(f"[residue-view] F{a.feature} vs {a.domain}: {len(has_domain)} proteins carry the domain", flush=True)

# gather the feature's activation on those proteins' residues (targeted store read)
ck = torch.load(a.sae, map_location="cpu"); sae = TopKSAE(**ck["model_config"]).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
keep_arr = np.sort(df["row"].to_numpy()); act = {}  # global row -> feature activation
order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
row0 = 0
with torch.no_grad():
    for sp in order:
        n = pq.read_metadata(sp).num_rows
        lo = int(np.searchsorted(keep_arr, row0)); hi = int(np.searchsorted(keep_arr, row0 + n))
        rows = keep_arr[lo:hi]  # global keep-rows in this shard (O(log), not O(n))
        if len(rows):
            X = shard_table_to_array(pq.read_table(sp))
            enc = sae.encode(torch.from_numpy(np.ascontiguousarray(X[rows - row0])).to(dev))[:, a.feature].cpu().numpy()
            for r, v in zip(rows, enc): act[int(r)] = float(v)
        row0 += n

# per protein: firing residues + in-domain fraction; rank proteins by peak activation
recs = []
for pid, g in df.groupby("pid", sort=False):
    res = g["res"].to_numpy(); rows = g["row"].to_numpy()
    acts = np.array([act.get(int(r), 0.0) for r in rows])
    L = len(res)
    fire = res[acts > a.tau]
    sp = spans(pid)
    in_dom = sum(1 for r in fire if any(s - 1 <= r <= e - 1 for s, e in sp))
    recs.append({"pid": pid, "L": L, "peak": acts.max(), "fire": set(int(x) for x in fire),
                 "n_fire": len(fire), "in_dom": in_dom, "spans": sp})
recs.sort(key=lambda r: -r["peak"])

print(f"\nF{a.feature} firing vs {a.domain} domain span   ('#'=fires in-domain  '.'=fires outside  '='=domain, no fire)")
print(f"{'protein':16} {'len':>4} {'fires':>5} {'in-dom':>6}  track (seq scaled to {a.width} cols)")
for r in recs[:a.n_proteins]:
    L = max(1, r["L"]); track = [" "] * a.width
    def col(pos): return min(a.width - 1, int(pos / L * a.width))
    for s, e in r["spans"]:
        for c in range(col(s - 1), col(e - 1) + 1): track[c] = "="
    for f in r["fire"]:
        indom = any(s - 1 <= f <= e - 1 for s, e in r["spans"])
        track[col(f)] = "#" if indom else "."
    prec = r["in_dom"] / max(1, r["n_fire"])
    print(f"  {r['pid'][:15]:15} {r['L']:>4} {r['n_fire']:>5} {r['in_dom']:>4}({prec:.0%}) |{''.join(track)}|")
tot_fire = sum(r["n_fire"] for r in recs[:a.n_proteins]); tot_in = sum(r["in_dom"] for r in recs[:a.n_proteins])
print(f"\n[summary] over shown proteins: {tot_in}/{tot_fire} firing residues in-domain = {tot_in/max(1,tot_fire):.0%} precision")
print("  (# clustered inside the ==== domain region => LOCALIZED; . scattered outside => not)")
