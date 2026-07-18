# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
"""DOMAIN-SPECIFIC F1 (Polina's metric, from her evo2-SAE PR) for the structural residue-band features.

For a region label (an InterPro domain that spans residues [start,end]), a good feature fires on SOME
positions inside the region and NONE outside. So the two halves are measured at different granularities:
  - PRECISION is PER-POSITION : of all residues where the feature fires, what fraction are inside a
                                domain-D region?  (penalizes firing outside domains)
  - RECALL    is PER-REGION   : of all domain-D regions, what fraction have >=1 firing residue inside?
                                (a feature firing on part of a region still gets full credit for it)
  F1 = 2PR/(P+R). Contrast with per-feature AUROC (rank-sum, per-position both sides), also reported here.

Candidate features per domain = the AUROC-best feature from feature_biology_table.json (so this ENRICHES
that table with an F1 column rather than re-searching all 40,960). "Fires" = SAE activation > 0 (TopK-natural).

Usage: domain_f1.py <sae.pt> <store> <layer> [--max-prot 3000] [--out ...] [--bio-table ...]
"""
import argparse, glob, json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, torch, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "src")
from bioreason_pro_sae.model_loader import _install_unsloth_stub; _install_unsloth_stub()
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
import bioreason_pro_sae.data as brp_data

DFULL = "/data/savithas/phase3_full"
p = argparse.ArgumentParser()
p.add_argument("sae"); p.add_argument("store"); p.add_argument("layer", type=int)
p.add_argument("--max-prot", type=int, default=3000, help="subsample whole proteins (keeps regions intact)")
p.add_argument("--tau", type=float, default=0.0, help="feature 'fires' if activation > tau")
p.add_argument("--bio-table", default=f"{DFULL}/feature_biology_table.json")
p.add_argument("--out", default=f"{DFULL}/domain_f1_l30.json")
a = p.parse_args(); dev = "cuda"

# ---- residue-band rows + within-protein residue index (reused from residue_domain_probe.py) ----
tl = pq.read_table(f"{a.store}/token_labels.parquet")
band = np.array(tl.column("position_type").to_pylist(), dtype=object)
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64)
N = len(band); prot = band == "protein"
df = pd.DataFrame({"pid": rpid[prot].astype(str), "tidx": rtidx[prot], "row": np.arange(N)[prot]})
df = df.sort_values(["pid", "tidx"]).reset_index(drop=True)
df["res"] = df.groupby("pid").cumcount()                       # 0-indexed residue position within protein

# subsample WHOLE proteins so domain regions stay intact
pids_all = df["pid"].unique()
rng = np.random.default_rng(0)
if len(pids_all) > a.max_prot:
    keep_pids = set(rng.choice(pids_all, a.max_prot, replace=False).tolist())
    df = df[df["pid"].isin(keep_pids)].reset_index(drop=True)
print(f"[domF1] {len(df):,} residues across {df['pid'].nunique()} proteins", flush=True)

# ---- interpro_location spans ----
tr, _, _ = brp_data.load_reasoning_splits(max_length_protein=2000)
loc_of = {}
for pid, loc in zip(tr["protein_id"], tr["interpro_location"]):
    if not loc: continue
    try: loc_of[str(pid)] = json.loads(loc) if isinstance(loc, str) else loc
    except Exception: pass

def spans_for(pid, dom):
    """list of (start,end) 1-indexed regions of domain `dom` in protein `pid`."""
    span = loc_of.get(pid, {}).get(dom)
    if not span: return []
    return [tuple(s) for s in span] if isinstance(span[0], list) else [tuple(span)]

# ---- candidate (feature, domain) from the bio table (InterPro rows only) ----
tab = json.load(open(a.bio_table))
feat_dom = {}                                                   # feature -> IPR id
for r in sorted(tab, key=lambda x: -x["auroc"]):
    term = r["term"]
    if not term.startswith("IPR:"): continue
    ipr = term.split(":", 1)[1]
    feat_dom.setdefault(int(r["feature"]), (ipr, r["name"], r["auroc"]))
cand = sorted(feat_dom)
col_of = {f: i for i, f in enumerate(cand)}
print(f"[domF1] {len(cand)} candidate structural features (AUROC-best per InterPro domain)", flush=True)

# ---- gather per-residue activations for candidate feature columns only ----
ck = torch.load(a.sae, map_location="cpu"); cfg = ck["model_config"]
sae = TopKSAE(**cfg).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
cand_t = torch.tensor(cand, device=dev)
keep_rows = df["row"].to_numpy()
row_to_i = {int(r): i for i, r in enumerate(keep_rows)}
ACT = np.zeros((len(df), len(cand)), np.float32); got = np.zeros(len(df), bool)
keep_sorted = np.sort(keep_rows); row0 = 0
order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si, sp in enumerate(order):
        n = pq.read_metadata(sp).num_rows
        lo, hi = row0, row0 + n
        loc_rows = keep_sorted[(keep_sorted >= lo) & (keep_sorted < hi)]
        if len(loc_rows):
            X = shard_table_to_array(pq.read_table(sp))
            xb = torch.from_numpy(np.ascontiguousarray(X[loc_rows - lo])).to(dev)
            enc = sae.encode(xb)[:, cand_t].cpu().numpy()
            for j, gr in enumerate(loc_rows):
                i = row_to_i[int(gr)]; ACT[i] = enc[j]; got[i] = True
        row0 += n
        if si % 20 == 0: print(f"  shard {si}/{len(order)}", flush=True)
df = df[got].reset_index(drop=True); ACT = ACT[got]
print(f"[domF1] gathered {int(got.sum()):,} residues", flush=True)

# per-protein residue->row-in-ACT index, for region recall
by_pid = {pid: g for pid, g in df.groupby("pid", sort=False)}

# ---- compute per-feature domain-F1 + AUROC ----
from scipy.stats import rankdata
res_pid = df["pid"].to_numpy(); res_pos = df["res"].to_numpy() + 1   # 1-indexed residue
out = []
for f, (ipr, name, auroc) in feat_dom.items():
    col = ACT[:, col_of[f]]
    fires = col > a.tau
    # per-position precision: firing residues inside ANY region of this domain
    in_dom = np.array([1 if any(s <= res_pos[i] <= e for (s, e) in spans_for(res_pid[i], ipr)) else 0
                       for i in np.where(fires)[0]], dtype=np.int8) if fires.any() else np.array([])
    n_fire = int(fires.sum())
    precision = float(in_dom.mean()) if len(in_dom) else 0.0
    # per-region recall: fraction of this domain's regions with >=1 firing residue
    regions = 0; hit = 0
    for pid, g in by_pid.items():
        sps = spans_for(pid, ipr)
        if not sps: continue
        pos = g["res"].to_numpy() + 1
        firecol = fires[g.index.to_numpy()]
        fpos = set(pos[firecol].tolist())
        for (s, e) in sps:
            regions += 1
            if any(x in fpos for x in range(s, e + 1)): hit += 1
    recall = hit / regions if regions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # per-position AUROC (both sides per-position) for contrast
    y = np.array([1 if any(s <= res_pos[i] <= e for (s, e) in spans_for(res_pid[i], ipr)) else 0
                  for i in range(len(col))], dtype=np.int8)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos and nneg:
        rk = rankdata(col); a_uroc = (rk[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    else:
        a_uroc = float("nan")
    out.append({"feature": f, "domain": ipr, "name": name, "n_regions": regions, "n_fire": n_fire,
                "precision_perpos": round(precision, 3), "recall_perregion": round(recall, 3),
                "domain_f1": round(f1, 3), "auroc": round(auroc, 3), "auroc_recheck": round(float(a_uroc), 3)})

out.sort(key=lambda r: -r["domain_f1"])
print(f"\n{'feat':>7} {'F1':>5} {'prec':>5} {'recall':>6} {'AUROC':>6}  domain")
for r in out[:25]:
    print(f"  F{r['feature']:<6}{r['domain_f1']:>5} {r['precision_perpos']:>5} {r['recall_perregion']:>6} "
          f"{r['auroc']:>6}  {r['name'][:40]}")
good = [r for r in out if r["domain_f1"] >= 0.5]
print(f"\n[domF1] {len(good)}/{len(out)} features with domain-F1 >= 0.5")
json.dump({"layer": a.layer, "tau": a.tau, "n_proteins": int(df['pid'].nunique()),
           "metric": "per-position precision, per-region recall (Polina)", "features": out},
          open(a.out, "w"), indent=2)
print(f"[wrote] {a.out}")
