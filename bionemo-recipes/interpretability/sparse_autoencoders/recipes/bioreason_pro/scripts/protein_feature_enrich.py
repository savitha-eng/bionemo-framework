#!/usr/bin/env python
"""Enrichment-grounded interpretation of PROTEIN SAE features (no-LLM-hallucination core).

For a protein feature, take the proteins it fires strongest on, and test which annotation is
statistically OVER-REPRESENTED vs a background protein set (Fisher exact). Two annotation sources:
  - GO terms: local per-protein go_ids (proteins.parquet) — instant, no network, full 117k background.
  - UniProt keywords / domains: fetched from rest.uniprot.org (cached to disk), richer function/structure.

The enrichment p-value is the reliable signal; an LLM (optional, --llm) only phrases the verified result.
Usage: protein_feature_enrich.py --model l16_balanced --features 13056,7856 [--uniprot] [--n-proteins 20]
"""
import argparse, json, os, urllib.request, urllib.error
from collections import Counter
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
try:
    from scipy.stats import fisher_exact
except Exception:  # noqa: BLE001
    fisher_exact = None

PUB = "/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro/multimodal_dashboard/public"
OBO = "/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo"


def load_go_names():
    names, cur = {}, None
    if not Path(OBO).exists():
        return names
    for line in open(OBO):
        line = line.strip()
        if line == "[Term]":
            cur = {}
        elif line.startswith("id: GO:") and cur is not None:
            cur["id"] = line[4:]
        elif line.startswith("name:") and cur is not None and "id" in cur:
            names[cur["id"]] = line[6:]
    return names
PROTEINS = "/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro/cache_dir/activations/train_full_L16_L18_L20_L22/proteins.parquet"
UCACHE = Path("/data/savithas/phase3_full/uniprot_cache.json")


def _fisher(k, n, K, N):
    """k of n feature-proteins have the term; K of N background do. Right-tailed enrichment p-value."""
    if fisher_exact is None:
        # hypergeometric-ish fallback via normal approx is unreliable; require scipy
        return float("nan")
    table = [[k, n - k], [K - k, (N - n) - (K - k)]]
    try:
        return float(fisher_exact(table, alternative="greater")[1])
    except Exception:  # noqa: BLE001
        return float("nan")


def load_uniprot_cache():
    if UCACHE.exists():
        return json.load(open(UCACHE))
    return {}


def fetch_uniprot(acc, cache):
    if acc in cache:
        return cache[acc]
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json?fields=keyword,ft_domain,protein_name"
    out = {"keywords": [], "domains": [], "name": ""}
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        out["name"] = d.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
        out["keywords"] = [k.get("name", "") for k in d.get("keywords", [])]
        out["domains"] = sorted({f["description"] for f in d.get("features", [])
                                 if f.get("type") == "Domain" and f.get("description")})
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError):
        pass
    cache[acc] = out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="l16_balanced"); p.add_argument("--features", required=True)
    p.add_argument("--n-proteins", type=int, default=20, help="top proteins per feature (by activation)")
    p.add_argument("--background", type=int, default=2000, help="random background proteins for enrichment")
    p.add_argument("--uniprot", action="store_true", help="also fetch UniProt keyword/domain enrichment")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    ex = pq.read_table(f"{PUB}/{a.model}/feature_examples.parquet").to_pandas()
    prot = pq.read_table(PROTEINS).to_pandas()
    go_names = load_go_names()
    go = {r.protein_id: (json.loads(r.go_ids) if isinstance(r.go_ids, str) else list(r.go_ids or []))
          for r in prot.itertuples()}
    all_pids = list(go.keys())
    bg = list(rng.choice(all_pids, size=min(a.background, len(all_pids)), replace=False))
    # background GO frequencies
    bg_go = Counter(t for pid in bg for t in go.get(pid, []))
    Nbg = len(bg)
    ucache = load_uniprot_cache() if a.uniprot else {}
    if a.uniprot:
        bg_kw = Counter(kw for pid in bg for kw in fetch_uniprot(pid, ucache)["keywords"])

    for fid in [int(x) for x in a.features.split(",")]:
        pb = ex[(ex.feature_id == fid) & (ex.band == "protein")].nlargest(a.n_proteins, "max_activation")
        pids = list(dict.fromkeys(pb.protein_id.tolist()))  # unique, keep order
        print(f"\n=== F{fid}: fires on {len(pids)} proteins (top by activation) ===")
        if not pids:
            print("  (no protein-band examples)"); continue
        # GO enrichment
        fg = Counter(t for pid in pids for t in go.get(pid, []))
        n = len(pids)
        rows = []
        for term, k in fg.items():
            if k < 2:
                continue
            K = bg_go.get(term, 0) + k  # ensure term seen in bg count space
            rows.append((term, k, n, _fisher(k, n, max(bg_go.get(term, 1), k), Nbg)))
        rows.sort(key=lambda r: r[3])
        print("  top enriched GO terms (Fisher p):")
        for term, k, n_, pv in rows[:5]:
            print(f"    {term} {go_names.get(term, '?')}  ({k}/{n_} proteins, p={pv:.2e})")
        if a.uniprot:
            fkw = Counter(kw for pid in pids for kw in fetch_uniprot(pid, ucache)["keywords"])
            krows = [(kw, k, _fisher(k, n, max(bg_kw.get(kw, 1), k), Nbg)) for kw, k in fkw.items() if k >= 2]
            krows.sort(key=lambda r: r[2])
            print("  top enriched UniProt keywords (Fisher p):")
            for kw, k, pv in krows[:5]:
                print(f"    {kw}  ({k}/{n} proteins, p={pv:.2e})")
            names = [fetch_uniprot(pid, ucache)["name"] for pid in pids[:6]]
            print("  example protein names:", "; ".join(x for x in names if x)[:200])
        json.dump(ucache, open(UCACHE, "w")) if a.uniprot else None


if __name__ == "__main__":
    main()
