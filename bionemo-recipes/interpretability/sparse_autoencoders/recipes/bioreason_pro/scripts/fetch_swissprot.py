# Fetch per-residue Swiss-Prot annotations (active sites, binding, domains, ...) from UniProt
# for the EXACT protein set in our activation store, so we can run residue-level feature F1
# (Jared's esm2/codonfm methodology) instead of only coarse protein-level GO-AUC.
#
# Network-only (UniProt REST), no GPU. Batches accessions, retries, resumable (skips if out exists).
#
#   python scripts/fetch_swissprot.py --store <store> --out <dir>/swissprot.tsv
import argparse, gzip, io, sys, time
from pathlib import Path
import pyarrow.parquet as pq
import requests

# Same UniProt feature fields the esm2/codonfm pipelines use.
FIELDS = ["accession", "sequence", "length",
          "ft_act_site", "ft_binding", "ft_disulfid", "ft_carbohyd", "ft_lipid",
          "ft_mod_res", "ft_signal", "ft_transit", "ft_domain", "ft_motif",
          "ft_region", "ft_zn_fing", "ft_compbias"]
STREAM = "https://rest.uniprot.org/uniprotkb/stream"


def fetch_batch(accs, retries=5):
    """One UniProt stream query for a list of accessions -> list of TSV lines (no header)."""
    query = " OR ".join(f"accession:{a}" for a in accs)
    params = {"query": query, "format": "tsv", "fields": ",".join(FIELDS)}
    for attempt in range(retries):
        try:
            r = requests.get(STREAM, params=params, timeout=120)
            r.raise_for_status()
            lines = r.text.strip("\n").split("\n")
            return lines[0], lines[1:]  # header, rows
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  batch failed ({len(accs)} accs): {type(e).__name__}", flush=True)
                return None, []
            time.sleep(3 * (attempt + 1))


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"{out} exists — delete to refetch"); return
    accs = [str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
    print(f"fetching Swiss-Prot annotations for {len(accs)} accessions in batches of {a.batch}", flush=True)
    header, rows = None, []
    for i in range(0, len(accs), a.batch):
        h, rs = fetch_batch(accs[i:i + a.batch])
        if h and header is None:
            header = h
        rows.extend(rs)
        print(f"  {i + a.batch}/{len(accs)}  (+{len(rs)} rows, {len(rows)} total)", flush=True)
        time.sleep(0.3)  # be polite to UniProt
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") if str(out).endswith(".gz") else open(out, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)}/{len(accs)} annotated proteins -> {out}", flush=True)


if __name__ == "__main__":
    main()
