#!/usr/bin/env python
"""Precompute per-(protein-feature) REASONING EVIDENCE for the dashboard.

For each captioned protein feature, gather the model's own reasoning text about the top proteins it
fires on, and mark the biological concept words (nucleus, transcription, mitochondrion, kinase, …) so
the dashboard can highlight *what the model says these proteins do*. This is the evidence behind the
🧬🔀 caption — the actual reasoning, not a summary. Concept words are wrapped in «…» for the frontend.

Writes public/<model>/reasoning_evidence.json  = { feature_id: [ {protein_id, snippet}, ... ] }.
Usage: add_reasoning_evidence.py <public/model_dir>
"""
import json, re, sys
from pathlib import Path
import pyarrow.parquet as pq

CONCEPTS = ["transmembrane", "mitochondrion|mitochond", "transcription|transcript", "nucleus|nuclear",
    "chromatin|chromat", "polymerase", "ribosom", "translation|translat", "kinase", "phosphat",
    "membrane", "transport", "channel", "receptor", "signal", "immune", "apopto", "oxidoreductase|oxidas|reductas",
    "transferase|transferas", "hydrolase", "protease|peptidas", "DNA", "RNA", "helicase", "zinc.?finger",
    "chaperone", "cytoskelet|actin|tubulin", "collagen", "extracellular", "secret", "golgi", "lysosom",
    "vesicl", "synap", "neuron|dendrit|axon", "muscle", "develop", "embryo", "sperm|oocyte|meiosis|gamete",
    "bacter|fungal|antimicrob", "metabol", "catalytic", "redox", "ATPase|GTPase", "cilia|flagell",
    "chromosome|segregat", "gene expression", "repair", "replicat", "histone", "binding"]
PAT = re.compile("(" + "|".join(CONCEPTS) + ")", re.I)


def clean(s):  # collapse BPE subword spaces + strip accessions -> readable text
    s = re.sub(r"go\s*:\s*[\d\s]+", "", str(s)); s = re.sub(r"IPR\s*[\d\s]+", "", s, flags=re.I)
    return s.replace(" ", "")


def snippet(txt, width=220):
    m = PAT.search(txt)
    if not m:
        return None
    lo = max(0, m.start() - 60); hi = min(len(txt), lo + width)
    seg = txt[lo:hi]
    seg = PAT.sub(lambda x: f"«{x.group(0)}»", seg)     # mark every concept word
    return ("…" if lo > 0 else "") + seg + ("…" if hi < len(txt) else "")


def main():
    pub = Path(sys.argv[1])
    meta = pq.read_table(pub / "feature_metadata.parquet").to_pandas()
    capped = meta[(meta.xmodal_caption.astype(str) != "") & (meta.xmodal_caption.notna())].feature_id.tolist()
    ex = pq.read_table(pub / "feature_examples.parquet",
                       columns=["feature_id", "band", "protein_id", "sequence", "max_activation"]).to_pandas()
    prot = ex[ex.band == "protein"]; reas = ex[ex.band == "reasoning"]
    # one reasoning blob per protein (concat its reasoning windows, cleaned)
    rtext = {}
    for pid, g in reas.groupby("protein_id"):
        rtext[pid] = " ".join(clean(s) for s in g.sequence)
    out = {}
    for fid in capped:
        tops = list(dict.fromkeys(prot[prot.feature_id == fid].sort_values("max_activation", ascending=False)
                                  .protein_id.tolist()))
        snips = []
        for pid in tops:
            if pid in rtext:
                s = snippet(rtext[pid])
                if s:
                    snips.append({"protein_id": pid, "snippet": s})
            if len(snips) >= 4:
                break
        if snips:
            out[str(int(fid))] = snips
    (pub / "reasoning_evidence.json").write_text(json.dumps(out))
    print(f"wrote {pub/'reasoning_evidence.json'}: reasoning evidence for {len(out)} protein features")


if __name__ == "__main__":
    main()
