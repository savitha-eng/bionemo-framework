#!/usr/bin/env python
"""A/B comparison: Matryoshka SAE vs flat TopK, on the SAME balanced L24 data.

Reads two built dashboards' feature_metadata.parquet and reports the metrics that test whether Matryoshka
reduced feature absorption / produced cleaner features:
  - dead-everywhere % and live-feature count
  - "always-on" absorption artifacts (freq>0.5 AND ~equal band split = the 0.33/0.33/0.33 junk)
  - band-mass "cross-modal" feature count (protein_frac>0.2 AND text_frac>0.2) -- fewer = less noise
  - GO-AUC distribution for protein features (cleaner concepts -> higher AUC)
  - activation-frequency + max-activation distributions
Cross-modal cosine (the real test) is compared separately via crossmodal_cooccur.py --dump-json on each.
Usage: matryoshka_ab.py --matryoshka <pub/l24_matryoshka> --flat <pub/l24_balanced> [--out ab.json]
"""
import argparse, json
import numpy as np, pyarrow.parquet as pq


def summarize(pub):
    m = pq.read_table(f"{pub}/feature_metadata.parquet").to_pandas()
    freq = m["activation_frequency"].fillna(m["activation_freq"]) if "activation_frequency" in m else m["activation_freq"]
    pf, tf = m["protein_frac"].fillna(0), m["text_frac"].fillna(0)
    span = m["max_span"].fillna(0) if "max_span" in m else 0 * pf
    auc = m["go_auc"].fillna(0) if "go_auc" in m else 0 * pf
    always_on = ((freq > 0.5) & (pf.between(0.25, 0.45)) & (tf.between(0.25, 0.45))).sum()  # 0.33/0.33/0.33 junk
    bandmass = ((pf > 0.2) & (tf > 0.2)).sum()
    prot = pf > 0.5
    return {
        "n_features": int(len(m)),
        "median_freq_pct": round(float(freq.median() * 100), 3),
        "median_max_act": round(float(m["max_activation"].median()), 2),
        "always_on_artifacts": int(always_on),
        "bandmass_crossmodal": int(bandmass),
        "n_protein_features(pf>0.5)": int(prot.sum()),
        "median_GOAUC_protein_feats": round(float(auc[prot].median()) if prot.sum() else 0, 3),
        "multitoken_feats(span>=3)": int((span >= 3).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matryoshka", required=True); p.add_argument("--flat", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    A = summarize(a.matryoshka); B = summarize(a.flat)
    keys = list(A)
    print(f"{'metric':32} {'Matryoshka':>14} {'flat TopK':>14}")
    for k in keys:
        print(f"  {k:30} {str(A[k]):>14} {str(B[k]):>14}")
    print("\nInterpretation: Matryoshka should show FEWER always_on_artifacts + bandmass_crossmodal (less "
          "absorption), HIGHER median_GOAUC_protein_feats + multitoken_feats (cleaner concepts). "
          "Cross-modal ALIGNMENT (the real test) = compare crossmodal_cooccur --dump-json n_aligned for each.")
    if a.out:
        json.dump({"matryoshka": A, "flat": B}, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
