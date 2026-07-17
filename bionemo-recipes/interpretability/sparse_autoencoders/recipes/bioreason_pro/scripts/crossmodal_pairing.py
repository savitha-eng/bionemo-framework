# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
"""CROSS-MODAL PAIRING (Sense A): does a protein/residue-band bio feature co-vary, ACROSS PROTEINS,
with a *different* reasoning-band feature? For a bio feature f, take its per-protein activation vector
SP[:,f] (pooled over that protein's PROTEIN-band tokens) and Pearson-correlate it, over the 8k proteins,
against every reasoning feature's per-protein vector SR[:,k] (pooled over REASONING-band tokens). High r
= the proteins where the residue-detector fires are the proteins where the reasoning-detector fires.

IMPORTANT — this is a POPULATION correlation, NOT feature-level fusion and NOT causal. Both features are
independent descendants of a fact ("this protein is a kinase") the model got from the PROMPT annotation.
Contrast with same-feature co-firing (crossmodal_coactivation.py / crossmodal.py), whose null HOLDS.

Usage: crossmodal_pairing.py [--bio-table feature_biology_table.json] [--n-bio 20] [--top 3] [--out ...]
"""
import argparse, json
from pathlib import Path
import numpy as np

DFULL = "/data/savithas/phase3_full"
p = argparse.ArgumentParser()
p.add_argument("--pooled-protein", default=f"{DFULL}/pooled_protein_l30_00e1d8.npz")
p.add_argument("--pooled-reasoning", default=f"{DFULL}/pooled_reasoning_l30_00e1d8.npz")
p.add_argument("--bio-table", default=f"{DFULL}/feature_biology_table.json")
p.add_argument("--bio-features", default="", help="comma-sep feature ids to override the bio-table pick")
p.add_argument("--n-bio", type=int, default=20, help="how many bio features from the table (by AUROC)")
p.add_argument("--top", type=int, default=3, help="top-N reasoning partners per bio feature")
p.add_argument("--min-active", type=int, default=20, help="skip bio features firing on < this many proteins")
p.add_argument("--words", nargs="*", default=[f"{DFULL}/synthesis_word_l30.json",
               f"{DFULL}/echo_synthesis_l30.json", f"{DFULL}/autointerp_l30.json"],
               help="JSONs to harvest per-feature words/labels to annotate reasoning partners")
p.add_argument("--out", default=f"{DFULL}/crossmodal_pairing_l30.json")
a = p.parse_args()

# ---- per-feature word/label lookup (best-effort, from whatever interpretability JSONs exist) ----
def harvest_words(paths):
    lut = {}
    def add(fid, words=None, label=None):
        fid = int(fid); e = lut.setdefault(fid, {"words": [], "label": ""})
        if words: e["words"] = (e["words"] + [w for w in words if w])[:10]
        if label and not e["label"]: e["label"] = label
    for pth in paths:
        if not Path(pth).exists(): continue
        d = json.load(open(pth))
        # synthesis_word_l30.json: {"features": {id: {"words"/"top_words":..., "label":...}}}
        feats = d.get("features") if isinstance(d, dict) else None
        if isinstance(feats, dict):
            for k, v in feats.items():
                if not isinstance(v, dict): continue
                try: kk = int(k)
                except (ValueError, TypeError): continue
                add(kk, v.get("words") or v.get("top_words"), v.get("label") or v.get("name"))
        # echo_synthesis: {"synthesis_features":{id:{words}}, "echo_features":{...}}
        for grp in ("synthesis_features", "echo_features"):
            for k, v in (d.get(grp, {}) or {}).items():
                add(k, v.get("words"), grp.replace("_features", ""))
        # autointerp_l30.json: {id: {"label"/"name":...}} or {"features":{...}}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and (v.get("label") or v.get("name")):
                    try: add(int(k), v.get("words") or v.get("top_words"), v.get("label") or v.get("name"))
                    except (ValueError, TypeError): pass
    return lut
WORDS = harvest_words(a.words)
def annot(fid):
    e = WORDS.get(int(fid), {}); return e.get("label", ""), e.get("words", [])[:6]

# ---- load pooled per-protein SAE activations for both bands ----
P = np.load(a.pooled_protein); R = np.load(a.pooled_reasoning)
SP, SR = P["SAE"], R["SAE"]                       # [n_prot x n_feat] each
keep = P["keep"] & R["keep"]
SP, SR = SP[keep], SR[keep]
n_prot, n_feat = SP.shape
print(f"[pairing] {n_prot} proteins x {n_feat} features", flush=True)

# pre-center + norm the reasoning matrix once (columns = features)
SRc = SR - SR.mean(0)
SRnorm = np.linalg.norm(SRc, axis=0) + 1e-9
# NULL: permute proteins in the reasoning matrix -> destroys real bio<->reasoning alignment but keeps each
# feature's marginal distribution. max-r vs this null = the winner's-curse floor (best of 40,960 by chance).
rng = np.random.default_rng(0)
perm = rng.permutation(SP.shape[0])
SRc_null = SRc[perm]

# ---- pick bio features ----
if a.bio_features:
    bio = [int(x) for x in a.bio_features.split(",")]
    biolabel = {int(x): ("", "") for x in bio}
else:
    tab = json.load(open(a.bio_table))
    tab = sorted(tab, key=lambda r: -r["auroc"])
    bio, biolabel, seen = [], {}, set()
    for r in tab:
        f = int(r["feature"])
        if f in seen: continue
        seen.add(f); bio.append(f); biolabel[f] = (r["name"], r["term"])
        if len(bio) >= a.n_bio: break

# ---- correlate each bio feature vs all reasoning features ----
results = []
for f in bio:
    sp = SP[:, f]
    n_fire = int((sp > 0).sum())
    if n_fire < a.min_active:
        results.append({"bio_feature": f, "bio_name": biolabel[f][0], "skipped": f"fires on {n_fire} prot"})
        continue
    spc = sp - sp.mean()
    denom = (np.linalg.norm(spc) * SRnorm)
    r = (SRc * spc[:, None]).sum(0) / denom       # Pearson r vs each reasoning feature
    r[f] = -2.0                                     # exclude self-pairing
    null_max = float(((SRc_null * spc[:, None]).sum(0) / denom).max())  # best-of-40960 by chance
    order = np.argsort(-r)[:a.top]
    partners = []
    for k in order:
        lab, ws = annot(k)
        partners.append({"reasoning_feature": int(k), "r": round(float(r[k]), 3),
                         "label": lab, "words": ws})
    bname, bterm = biolabel[f]
    results.append({"bio_feature": f, "bio_name": bname, "bio_term": bterm,
                    "n_proteins_firing": n_fire, "null_max_r": round(null_max, 3), "partners": partners})
    top = partners[0]
    print(f"  bio F{f:<6} {bname[:30]:30} -> reasoning F{top['reasoning_feature']:<6} "
          f"r={top['r']:+.3f} (null {null_max:+.3f})  {top['label'] or ' '.join(top['words'][:4])}", flush=True)

# ---- summary: how many bio features have a partner above thresholds ----
scored = [r for r in results if "partners" in r]
for thr in (0.4, 0.6, 0.8):
    n = sum(1 for r in scored if r["partners"][0]["r"] >= thr)
    print(f"[pairing] bio features with a reasoning partner at r>={thr}: {n}/{len(scored)}")
out = {"n_proteins": n_prot, "n_bio": len(scored),
       "n_partner_ge_0.4": sum(1 for r in scored if r["partners"][0]["r"] >= 0.4),
       "results": results}
json.dump(out, open(a.out, "w"), indent=2)
print(f"[wrote] {a.out}")
