#!/usr/bin/env python
"""Download AlphaFold structures (v6 CIF) for the store's proteins, for the contact-residue probe.

AFDB moved to v6 (the dataset's structure_path says v4 = obsolete -> 404). Some accessions aren't in AFDB
(obsolete/merged) -> those 404 and are skipped (logged). Concurrent + resumable (skips existing files).

Usage: download_alphafold.py <store> <out_dir> [--workers 16] [--limit N]
"""
import sys, os, argparse, concurrent.futures as cf
import urllib.request, urllib.error
import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument("store"); p.add_argument("out_dir")
p.add_argument("--workers", type=int, default=16)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

pids = [str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
if a.limit: pids = pids[:a.limit]
print(f"[af] {len(pids)} proteins to fetch -> {a.out_dir}", flush=True)
URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.cif"


def fetch(acc):
    dst = os.path.join(a.out_dir, f"{acc}.cif")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return (acc, "cached")
    try:
        with urllib.request.urlopen(URL.format(acc), timeout=30) as r:
            data = r.read()
        with open(dst, "wb") as fh:
            fh.write(data)
        return (acc, "ok")
    except urllib.error.HTTPError as e:
        return (acc, f"http{e.code}")           # 404 = not in AFDB (obsolete accession)
    except Exception as e:
        return (acc, f"err:{type(e).__name__}")


ok = miss = err = 0
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for i, (acc, st) in enumerate(ex.map(fetch, pids)):
        if st in ("ok", "cached"): ok += 1
        elif st.startswith("http404"): miss += 1
        else: err += 1
        if i % 500 == 0:
            print(f"  {i}/{len(pids)}  ok={ok} miss(404)={miss} err={err}", flush=True)
print(f"[af] DONE: {ok} downloaded/cached, {miss} not-in-AFDB(404), {err} errors "
      f"({100*ok/len(pids):.0f}% coverage)", flush=True)
