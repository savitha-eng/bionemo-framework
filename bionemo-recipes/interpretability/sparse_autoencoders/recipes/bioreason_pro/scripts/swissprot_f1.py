# Residue-level Swiss-Prot feature F1 (Jared's esm2/codonfm methodology) for the protein band.
#
# Our protein band is residue-resolved (1 token = 1 residue), so a UniProt annotation at residue p
# maps to protein-band token (band_min(protein)+1 + (p-1)) — band = ESM3-BOS + residues + EOS at abs idx 7.
# For each feature x concept (ACT_SITE, BINDING, DOMAIN, ...) we threshold the activation and score
# precision + per-residue recall -> best-threshold F1. MECHANISTIC bio metric: a feature firing on the
# actual catalytic/binding residues scores high; a composition (e.g. valine) counter scores ~0 on
# ACT_SITE/BINDING even if its coarse GO-AUC looks high.
#
# FULLY VECTORIZED: per-concept residue labels precomputed ONCE (searchsorted), then GPU accumulation.
#
#   python scripts/swissprot_f1.py --sae <ckpt> --store <store> --layer 28 --swissprot <tsv.gz> --out <json>
import argparse, glob, gzip, json, re
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

COL_TAG = {"Active site": "ACT_SITE", "Binding site": "BINDING", "Disulfide bond": "DISULFID",
           "Glycosylation": "CARBOHYD", "Lipidation": "LIPID", "Modified residue": "MOD_RES",
           "Signal peptide": "SIGNAL", "Transit peptide": "TRANSIT", "Domain [FT]": "DOMAIN",
           "Motif": "MOTIF", "Region": "REGION", "Zinc finger": "ZN_FING", "Compositional bias": "COMPBIAS"}
SITE_PREFIXES = ("ACT_SITE", "BINDING", "MOD_RES", "DISULFID", "CARBOHYD", "LIPID")
POS_RE = re.compile(r"(\d+)(?:\.\.(\d+))?")
MAXLEN = 100000  # max residues/protein for the (protein,residue) composite key


def parse_features(cell, tag):
    if not cell:
        return []
    out = []
    for m in re.finditer(rf"{tag}\s+(\d+(?:\.\.\d+)?)", cell):
        pm = POS_RE.match(m.group(1))
        if pm:
            s = int(pm.group(1)); e = int(pm.group(2)) if pm.group(2) else s
            out.append((s, e))
    return out


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True); ap.add_argument("--store", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--swissprot", required=True)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--thresholds", default="0.0,0.15,0.5,0.6,0.8")
    ap.add_argument("--min-positives", type=int, default=10); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"; thr = [float(t) for t in a.thresholds.split(",")]; nT = len(thr)
    thr_t = torch.tensor(thr, device=dev)

    # ---- parse swissprot TSV -> acc -> {concept: [(s,e)]} ----
    op = gzip.open if a.swissprot.endswith(".gz") else open
    with op(a.swissprot, "rt") as f:
        header = f.readline().rstrip("\n").split("\t"); ci = {c: i for i, c in enumerate(header)}
        ftc = [c for c in header if c in COL_TAG]; ann = {}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(header):
                continue
            d = {COL_TAG[c]: sp for c in ftc if (sp := parse_features(p[ci[c]], COL_TAG[c]))}
            if d:
                ann[p[ci["Entry"]]] = d
    print(f"[swissprot] parsed {len(ann)} annotated proteins", flush=True)

    # ---- protein-band rows -> per-protein residue index ----
    tl = pq.read_table(f"{a.store}/token_labels.parquet")
    pos = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    rpid = np.asarray(tl.column("protein_id").to_pylist(), dtype=object)
    rtidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    is_prot = pos == "protein"
    prot_global = np.where(is_prot)[0]
    pp, tt = rpid[is_prot], rtidx[is_prot]
    uniq, inv = np.unique(pp, return_inverse=True)
    bmin = np.full(len(uniq), 2**31, dtype=np.int64); np.minimum.at(bmin, inv, tt)
    res = tt - bmin[inv] - 1 - a.offset                      # 0-based residue per protein-band row
    acc_to_int = {acc: i for i, acc in enumerate(uniq)}
    # composite key (proteinIdx, residue) -> sorted for searchsorted lookup of annotations
    keys = inv.astype(np.int64) * MAXLEN + res
    order = np.argsort(keys); keys_s = keys[order]; rows_s = prot_global[order]
    print(f"[swissprot] {is_prot.sum()} protein-band rows, band starts idx {int(bmin.min())}..{int(bmin.max())}", flush=True)

    # ---- precompute per-concept residue label (bool over ALL rows) via vectorized searchsorted ----
    concepts_all = sorted({c for d in ann.values() for c in d})
    lab_global, concept_pos = {}, {}
    for c in concepts_all:
        ak = []
        for acc, d in ann.items():
            ai = acc_to_int.get(acc)
            if ai is None or c not in d:
                continue
            for s, e in d[c]:
                ak.extend(range(ai * MAXLEN + (s - 1), ai * MAXLEN + e))
        if not ak:
            continue
        ak = np.asarray(ak, dtype=np.int64)
        idx = np.searchsorted(keys_s, ak); idx = np.clip(idx, 0, len(keys_s) - 1)
        hit = keys_s[idx] == ak
        lab = np.zeros(len(rpid), dtype=bool); lab[rows_s[idx[hit]]] = True
        n = int(lab.sum())
        if n >= a.min_positives:
            lab_global[c] = lab; concept_pos[c] = n
    concepts = list(lab_global)
    is_site = {c: any(c.startswith(p) for p in SITE_PREFIXES) for c in concepts}
    print(f"[swissprot] {len(concepts)} concepts: " + ", ".join(f"{c}({concept_pos[c]})"
          for c in sorted(concepts, key=lambda x: -concept_pos[x])), flush=True)

    # ---- SAE + GPU accumulation ----
    ck = torch.load(a.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval(); sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()})
    H = sae.hidden_dim
    lab_t = {c: torch.from_numpy(lab_global[c]).to(dev) for c in concepts}
    tp = {c: torch.zeros(H, nT, device=dev) for c in concepts}
    fp = {c: torch.zeros(H, nT, device=dev) for c in concepts}
    shards = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
    is_prot_t = torch.from_numpy(is_prot).to(dev)

    # PASS 0: per-feature max over protein-band tokens (so thresholds are FRACTIONS of the feature's
    # peak — InterPLM/Jared normalization). Without this, raw thresholds 0-0.8 vs activations 0-17 make
    # every active feature "fire everywhere" (recall=1) and selectivity is invisible.
    feat_max = torch.zeros(H, device=dev)
    row0 = 0
    with torch.no_grad():
        for si, sp in enumerate(shards):
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            for s in range(0, n, 8192):
                e = min(n, s + 8192); gp = is_prot_t[row0 + s:row0 + e]
                if gp.any():
                    feat_max = torch.maximum(feat_max, sae.encode(torch.from_numpy(X[s:e]).to(dev))[gp].amax(0))
            row0 += n
            if si % 50 == 0:
                print(f"  [max pass] shard {si}/{len(shards)}", flush=True)
    feat_max = feat_max.clamp(min=1e-6)

    # PASS 1: normalized thresholds
    row0 = 0
    with torch.no_grad():
        for si, sp in enumerate(shards):
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            for s in range(0, n, 8192):
                e = min(n, s + 8192); gp = is_prot_t[row0 + s:row0 + e]
                if not gp.any():
                    continue
                acts = sae.encode(torch.from_numpy(X[s:e]).to(dev))[gp] / feat_max   # (m, H) normalized 0..1
                pred = (acts.unsqueeze(-1) > thr_t).float()                          # (m, H, nT)
                gabs = torch.arange(row0 + s, row0 + e, device=dev)[gp]
                for c in concepts:
                    lab = lab_t[c][gabs]
                    if lab.any():
                        tp[c] += pred[lab].sum(0); fp[c] += pred[~lab].sum(0)
            row0 += n
            if si % 30 == 0:
                print(f"  [f1 pass] shard {si}/{len(shards)}", flush=True)

    # ---- F1 (per-residue recall for all concepts) ----
    results = []
    for c in concepts:
        TP = tp[c].cpu().numpy(); FP = fp[c].cpu().numpy()
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0); rec = TP / concept_pos[c]
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        bt = f1.argmax(1)
        for fi in range(H):
            if f1[fi, bt[fi]] > 0:
                results.append({"feature": int(fi), "concept": c, "f1": round(float(f1[fi, bt[fi]]), 3),
                                "precision": round(float(prec[fi, bt[fi]]), 3), "recall": round(float(rec[fi, bt[fi]]), 3),
                                "threshold": float(thr[bt[fi]]), "site": is_site[c]})
    results.sort(key=lambda r: -r["f1"]); json.dump(results, open(a.out, "w"), indent=2)
    nm = len({r["feature"] for r in results if r["site"] and r["f1"] > 0.5})
    print(f"[swissprot] {len({r['feature'] for r in results if r['f1']>0.5})} features F1>0.5; {nm} MECHANISTIC (site F1>0.5). top:", flush=True)
    for r in results[:12]:
        print(f"  F{r['feature']} {r['concept']:10s} F1={r['f1']} P={r['precision']} R={r['recall']}", flush=True)
    print(f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
