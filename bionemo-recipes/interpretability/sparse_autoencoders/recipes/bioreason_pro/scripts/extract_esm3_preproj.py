#!/usr/bin/env python
"""Extract ESM3 PRE-PROJECTION per-residue embeddings (the frozen-encoder representation).

BioReason-Pro reads protein embeddings from ESM3 (esm3_sm_open_v1) internal layer 37 -> 1536-d per
residue, then `protein_projection` maps 1536->2560->2560 into the LLM slots. This script captures the
1536-d layer-37 output DIRECTLY (via the authors' own `protein_encoder.encode_sequences`), BEFORE any
projection or LLM layer. Running the InterPro / burial / per-feature probes on this store answers the
paper's frozen-encoder control (§4.5.2/§4.5.4): is protein structure already in the frozen encoder,
i.e. does the LLM add any structural decodability? It extends our L16->L32 layer curve back to "layer 0".

Writes a standard (fast-format) ActivationStore at <out>/layer37/ + token_labels.parquet + proteins.parquet,
using the SAME protein_ids/residue order as an existing LLM-residual store (--match-store) so the probes'
protein_id-keyed labels line up for a head-to-head comparison.

Usage: extract_esm3_preproj.py --out <dir> [--match-store <L30 store>] [--num-proteins N]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pyarrow as pq_mod, pyarrow.parquet as pq, pyarrow as pa, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--match-store", default="/data/savithas/phase3_subset_8k/L30_subset8k",
               help="existing store whose protein_ids/order to reuse (for head-to-head probe alignment)")
p.add_argument("--num-proteins", type=int, default=0, help="0 = all proteins in --match-store")
p.add_argument("--shard-size", type=int, default=200_000)
p.add_argument("--device", default="cuda")
a = p.parse_args()

from bioreason_pro_sae.model_loader import load_bioreason_pro_sft
import bioreason_pro_sae.data as brp_data
from sae.activation_store import ActivationStore, ActivationStoreConfig

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")

# --- protein set + sequences: reuse the match-store's protein order so labels align head-to-head ---
want_ids = [str(x) for x in pq.read_table(f"{a.match_store}/proteins.parquet").column("protein_id").to_pylist()]
if a.num_proteins:
    want_ids = want_ids[:a.num_proteins]
want = set(want_ids)
print(f"[esm3] target {len(want)} proteins from {a.match_store}", flush=True)

tr, _, _ = brp_data.load_reasoning_splits(max_length_protein=2000)
seq_col = "sequence" if "sequence" in tr.column_names else "protein_sequences"
seq_of = {}
for pid, seq in zip(tr["protein_id"], tr[seq_col]):
    s = str(pid)
    if s in want and s not in seq_of:
        seq_of[s] = seq if isinstance(seq, str) else (seq[0] if seq else "")
order = [pid for pid in want_ids if seq_of.get(pid)]
print(f"[esm3] resolved sequences for {len(order)}/{len(want)} proteins", flush=True)

# --- load model (need the ESM3 protein_encoder on GPU) ---
print("[esm3] loading BioReason-Pro (for its ESM3 encoder)...", flush=True)
t0 = time.time()
model = load_bioreason_pro_sft(ckpt_dir=DEFAULT_CKPT, bioreason_pro_root="/data/savithas/bioreason-pro",
                               device=a.device)
enc = model.protein_encoder
print(f"[esm3] loaded in {time.time()-t0:.0f}s; embedding_layer={getattr(enc,'embedding_layer','?')}, "
      f"dim={getattr(enc,'embedding_dim','?')}", flush=True)

out = Path(a.out); (out / "layer37").mkdir(parents=True, exist_ok=True)
store = ActivationStore(out / "layer37", ActivationStoreConfig(shard_size=a.shard_size))
pid_col, idx_col, pt_col = [], [], []
proteins_meta = []
buf, buf_rows = [], 0
n_res_total = 0
t_start = time.time()

with torch.no_grad():
    for k, pid in enumerate(order):
        seq = seq_of[pid]
        try:
            emb = enc.encode_sequences([seq], batch_idx_map=[0], batch_size=1)[0]  # (n_res, 1536)
        except Exception as e:
            print(f"  [skip] {pid}: {e}", flush=True); continue
        if emb is None or emb.numel() == 0:
            continue
        arr = emb.float().cpu().numpy()
        nres = arr.shape[0]
        buf.append(arr); buf_rows += nres
        pid_col.extend([pid] * nres); idx_col.extend(range(nres)); pt_col.extend(["protein"] * nres)
        proteins_meta.append({"protein_id": pid, "n_protein": nres, "n_go": 0, "n_text": 0})
        n_res_total += nres
        if buf_rows >= a.shard_size:
            block = np.concatenate(buf, axis=0)
            nfull = (block.shape[0] // a.shard_size) * a.shard_size
            store.append(np.ascontiguousarray(block[:nfull]))
            rem = block[nfull:]; buf = [rem.copy()] if rem.shape[0] else []; buf_rows = rem.shape[0]
        if k % 200 == 0:
            print(f"  {k}/{len(order)} proteins, {n_res_total:,} residues, {time.time()-t_start:.0f}s", flush=True)

if buf_rows:
    store.append(np.ascontiguousarray(np.concatenate(buf, axis=0)))
store.finalize(metadata={"model": "esm3_sm_open_v1-layer37-preprojection", "layer": 37,
                         "hidden_dim": int(arr.shape[1]), "n_proteins": len(proteins_meta),
                         "position_types": "protein", "hook": "protein_encoder layer_outputs[37] (pre-projection)"})
# token_labels + proteins tables (same layout the probes read)
pq.write_table(pa.table({"protein_id": pa.array(pid_col, pa.string()),
                         "token_index": pa.array(idx_col, pa.int32()),
                         "position_type": pa.array(pt_col, pa.string())}), str(out / "token_labels.parquet"))
pq.write_table(pa.Table.from_pylist(proteins_meta), str(out / "proteins.parquet"))
print(f"[esm3] DONE: {len(proteins_meta)} proteins, {n_res_total:,} residues, "
      f"{time.time()-t_start:.0f}s -> {out}", flush=True)
