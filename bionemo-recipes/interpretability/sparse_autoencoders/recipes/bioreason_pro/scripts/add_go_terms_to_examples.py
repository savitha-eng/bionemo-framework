#!/usr/bin/env python
"""Resolve GO accessions in feature_examples windows to term NAMES (via go-basic.obo).

GO accessions render digit-split in the example windows ("GO : 0 0 0 6 8 8 7"). This reconstructs
each accession, looks up its name, and writes a `go_terms` column (e.g.
"GO:0006887 exocytosis; GO:0048661 positive regulation of smooth muscle cell proliferation")
so the dashboard can label GO examples directly instead of you reading the term out of the text.
Re-run after rebuilding feature_examples.parquet. See [[go-graph-slots-not-resolvable]].
"""
import argparse
import re

import pyarrow as pa
import pyarrow.parquet as pq

OBO = "/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo"
GO_RE = re.compile(r"GO\s*:\s*((?:\d\s*){7})")        # "GO : 0 0 0 6 8 8 7" (7 digit-tokens)


def load_go_names(path):
    names, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith("id: GO:"):
            cur = line[4:]
        elif line.startswith("name:") and cur:
            names[cur] = line[6:]
            cur = None
    return names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--max-terms", type=int, default=6, help="cap terms listed per example window")
    args = p.parse_args()

    names = load_go_names(OBO)
    t = pq.read_table(f"{args.dir}/feature_examples.parquet")
    seqs = t.column("sequence").to_pylist()

    go_terms = []
    n_with = 0
    for s in seqs:
        seen, terms = set(), []
        for m in GO_RE.finditer(s):
            acc = "GO:" + re.sub(r"\s", "", m.group(1))[:7]
            if acc in names and acc not in seen:
                seen.add(acc)
                terms.append(f"{acc} {names[acc]}")
                if len(terms) >= args.max_terms:
                    break
        go_terms.append("; ".join(terms))
        if terms:
            n_with += 1

    keep = [c for c in t.column_names if c != "go_terms"]
    t = t.select(keep).append_column("go_terms", pa.array(go_terms, pa.string()))
    pq.write_table(t, f"{args.dir}/feature_examples.parquet")
    print(f"[go-terms] resolved GO names for {n_with:,}/{len(seqs):,} example windows "
          f"(obo had {len(names):,} terms)")


if __name__ == "__main__":
    main()
