#!/usr/bin/env python
"""Compare SAE approaches by FEATURE QUALITY = high max-activation + low firing-frequency, per modality.

The interpretability ideal is a feature that fires RARELY but STRONGLY (specific, confident concept
detector). This is NIM-free (reads feature_metadata only) — robust, unlike autointerp det-F1.
Splits by modality (bio=protein-dominant, text=text-dominant) because bio features are ~10x weaker
here, so a good approach for us would raise the count of high-quality BIO features (or sophisticated
low-freq TEXT features). Usage: quality_frontier.py DIR1 DIR2 ... [--act-min 10 --freq-max 0.005 --top 12]
"""
import argparse, sys
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--act-min", type=float, default=8.0, help="max_activation threshold for 'strong'")
    ap.add_argument("--freq-max", type=float, default=0.005, help="activation_frequency threshold for 'rare/specific'")
    ap.add_argument("--span-min", type=int, default=5, help="max_span threshold for 'lights up many tokens'")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    # LOCKED quality def: modality by strongest-band PEAK; HQ = strong-ON-modality + rare (low freq) + wide span.
    print(f"HIGH-QUALITY = max_activation > {args.act_min} AND max_span >= {args.span_min} AND activation_frequency < {args.freq_max}\n")
    print(f"{'model':<22} {'mod':<5} {'n':>7} {'medAct':>7} {'HQ':>5}")
    TEXT = {"prompt", "reasoning", "answer", "text"}
    rows = {}
    for d in args.dirs:
        m = pq.read_table(f"{d}/feature_metadata.parquet").to_pandas().set_index("feature_id")
        ex = pq.read_table(f"{d}/feature_examples.parquet").to_pandas()
        name = d.rstrip("/").split("/")[-1]
        # MODALITY = which band the feature fires STRONGEST on (per-band PEAK activation), NOT firing
        # frequency (protein_frac mislabels weak-but-frequent-on-protein features whose real strength is text).
        ex["mod"] = ex["band"].map(lambda b: "protein" if b == "protein" else ("text" if b in TEXT else "go"))
        pk = ex.groupby(["feature_id", "mod"])["max_activation"].max().unstack(fill_value=0)
        ppk = pk["protein"] if "protein" in pk else pk.get("protein", 0) * 0
        tpk = pk["text"] if "text" in pk else ppk * 0
        af = m["activation_frequency"].reindex(pk.index).fillna(1.0)
        sp = m["max_span"].reindex(pk.index).fillna(0)
        for mod in ("bio", "text"):
            strength = ppk if mod == "bio" else tpk
            other = tpk if mod == "bio" else ppk
            is_mod = strength > other                                   # fires strongest on this modality
            hq_mask = is_mod & (strength > args.act_min) & (sp >= args.span_min) & (af < args.freq_max)
            med = float(strength[is_mod].median()) if int(is_mod.sum()) else 0.0
            print(f"{name:<22} {mod:<5} {int(is_mod.sum()):>7} {med:>7.2f} {int(hq_mask.sum()):>5}")
            rows[(name, mod)] = (pk.index[hq_mask], strength[hq_mask], af[hq_mask], sp[hq_mask])
        print()

    print("=== top HIGH-QUALITY features (strong-ON-modality + rare + wide span) — inspect these ===")
    for mod in ("bio", "text"):
        print(f"-- {mod} --")
        for d in args.dirs:
            name = d.rstrip("/").split("/")[-1]
            ids_, st_, af_, sp_ = rows.get((name, mod), ([], None, None, None))
            if len(ids_) == 0:
                print(f"  {name}: none"); continue
            order = st_.sort_values(ascending=False).index[: args.top]
            s = ", ".join(f"F{int(i)}(act{st_[i]:.0f}/f{af_[i] * 100:.2f}%/s{int(sp_[i])})" for i in order)
            print(f"  {name}: {s}")


if __name__ == "__main__":
    main()
