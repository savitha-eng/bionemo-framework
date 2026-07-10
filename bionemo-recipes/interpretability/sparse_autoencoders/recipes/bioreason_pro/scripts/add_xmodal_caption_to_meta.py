#!/usr/bin/env python
"""Add a cross-modal reasoning caption to each strong protein feature's dashboard metadata.

For every feature with a leakage-free held-out protein-band GO-AUROC (go_auc_protein, written by
protein_band_auc_to_meta.py), we look at the model's REASONING text about the proteins that feature
fires on (reasoning band only, GO/InterPro accessions stripped, BPE spaces collapsed) and record the
biological concepts that are over-represented vs background. This is the "use the text modality to
interpret the protein feature" caption: the GO term is the coarse anchor, the reasoning concepts
refine it (e.g. GO 'catalytic activity' -> phosphatase / transferase / redox / mitochondrion).

Writes a new string column `xmodal_caption` into feature_metadata.parquet (backup kept). Additive:
existing columns/behaviour are untouched, so it does not change any current dashboard functionality.

Usage: add_xmodal_caption_to_meta.py <public/model_dir>   e.g. .../public/l16_balanced
"""
import re, shutil, sys
from collections import defaultdict, Counter
from pathlib import Path

import pyarrow as pa, pyarrow.parquet as pq

# BPE-robust concept stems, matched as substrings against the space-collapsed reasoning text.
CONCEPTS = ["membrane", "transmembrane", "mitochond", "transport", "channel", "receptor", "kinase",
 "phosphat", "nuclear", "nucleus", "ribosom", "transcript", "translat", "apopto", "signal", "immune",
 "metabol", "catalyt", "oxidas", "reductas", "hydroxylas", "transferas", "protease", "peptidas",
 "dnabinding", "rnabinding", "chromat", "cytoskelet", "actin", "tubulin", "collagen", "extracellular",
 "secret", "golgi", "lysosom", "peroxisom", "endoplasm", "vesicl", "synap", "neuron", "muscle",
 "develop", "embryo", "differenti", "prolifer", "adhesion", "junction", "cilium", "flagell", "mitosis",
 "meiosis", "ubiquitin", "glycos", "lipid", "hormone", "antigen", "helicase", "polymerase", "atpase",
 "gtpase", "zincfinger", "transitpeptide", "chloride", "calcium", "sodium", "potassium", "proton",
 "electron", "redox", "fold", "chaperone", "matrix", "cortex", "axon", "dendrit"]
# stem -> human-readable label for the card
NAME = {"mitochond": "mitochondrion", "phosphat": "phosphatase", "transferas": "transferase",
 "reductas": "reductase", "oxidas": "oxidase", "metabol": "metabolism", "catalyt": "catalytic",
 "transcript": "transcription", "translat": "translation", "dnabinding": "DNA-binding",
 "rnabinding": "RNA-binding", "cytoskelet": "cytoskeleton", "extracellular": "extracellular",
 "endoplasm": "endoplasmic reticulum", "peptidas": "peptidase", "differenti": "differentiation",
 "prolifer": "proliferation", "zincfinger": "zinc-finger", "transitpeptide": "transit-peptide"}
pretty = lambda c: NAME.get(c, c)


def clean(s):
    s = re.sub(r"go\s*:\s*[\d\s]+", "", str(s).lower())   # strip GO: accessions
    s = re.sub(r"i\s*pr\s*[\d\s]+", "", s)                # strip InterPro accessions
    return s.replace(" ", "")                             # collapse BPE subword spaces


def main():
    d = Path(sys.argv[1])
    mp = d / "feature_metadata.parquet"
    ep = d / "feature_examples.parquet"
    ex = pq.read_table(ep, columns=["feature_id", "band", "protein_id", "sequence", "max_activation"]).to_pandas()

    # per-protein reasoning concept sets (over the whole reasoning corpus) + background rates
    corpus = defaultdict(set)
    for _, r in ex[ex.band == "reasoning"].iterrows():
        corpus[r.protein_id].add(clean(r.sequence))
    prot_concepts = {p: {c for c in CONCEPTS if c in " ".join(v)} for p, v in corpus.items()}
    bg = Counter()
    for p in prot_concepts:
        bg.update(prot_concepts[p])
    nBG = max(len(prot_concepts), 1)

    # per feature: proteins it fires on (protein band), ranked by activation
    byf = defaultdict(list)
    for _, r in ex[ex.band == "protein"].iterrows():
        byf[r.feature_id].append((r.max_activation, r.protein_id))

    meta = pq.read_table(mp)
    fids = meta.column("feature_id").to_pylist()
    auc = dict(zip(fids, meta.column("go_auc_protein").to_pylist())) if "go_auc_protein" in meta.column_names else {}

    def caption(fid, topn=25):
        if not auc.get(fid) or auc[fid] <= 0.65:
            return ""
        prs = list(dict.fromkeys(p for _, p in sorted(byf.get(fid, []), reverse=True)))
        prs = [p for p in prs[:topn] if p in prot_concepts]
        if len(prs) < 3:
            return ""
        cnt = Counter()
        for p in prs:
            cnt.update(prot_concepts[p])
        n = len(prs)
        scored = [(k / n, bg[c] / nBG, c) for c, k in cnt.items()]
        scored = [(fr, bgr, c) for fr, bgr, c in scored if fr >= 0.3 and fr > 1.8 * bgr]
        scored.sort(reverse=True)
        return " · ".join(pretty(c) for _, _, c in scored[:6])

    caps = [caption(f) for f in fids]
    n_capped = sum(1 for c in caps if c)

    if "xmodal_caption" in meta.column_names:
        meta = meta.drop(["xmodal_caption"])
    meta = meta.append_column("xmodal_caption", pa.array(caps, pa.string()))
    shutil.copy(mp, mp.with_suffix(".parquet.pre_xmodal.bak"))
    pq.write_table(meta, mp)
    print(f"{d.name}: captioned {n_capped}/{len(fids)} features -> xmodal_caption column added")


if __name__ == "__main__":
    main()
