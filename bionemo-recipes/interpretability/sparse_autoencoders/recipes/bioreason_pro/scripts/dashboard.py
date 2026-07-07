# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Generate dashboard activation-highlights for a BioReason-Pro SAE (aligned to codonfm/dashboard.py).

Mirrors Jared's ``codonfm/scripts/dashboard.py`` contract: per feature, the top-N examples with the
decoded ``sequence`` + a per-token ``activations`` list + ``max_activation``, written to
``feature_examples.parquet`` (the exact columns the React app reads: see App.jsx — sequence/activations).
The React app does the highlighting by coloring each token by its activation.

Differences from codonfm (multimodal): tokens are subword text pieces (rendered as space-joined pieces
so the front-end splits per token), and protein/GO placeholder slots render as ``<protein>``/``<go>``.
We reuse the already-extracted activation store (no model forward) — only the tokenizer is needed to
decode token windows — so this runs on CPU without a GPU.

    python scripts/dashboard.py --sae <ckpt> --store <val300_L28> --layer 28 \
        --out-dir multimodal_dashboard/public --device cpu

Keeps the existing feature_metadata.parquet / features_atlas.parquet (GO labels, band, UMAP) unless
``--rebuild-metadata`` is passed; only feature_examples.parquet is (re)written with real highlights.
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = (
    "/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
    "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b"
)
DEFAULT_ROOT = "/data/savithas/bioreason-pro"


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Build BioReason-Pro SAE dashboard activation highlights")
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True, help="Activation store with token_labels.parquet (e.g. val300_L28)")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out-dir", required=True, help="Dashboard public dir (writes feature_examples.parquet)")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default=DEFAULT_ROOT)
    p.add_argument("--split", default="validation")
    p.add_argument("--num-proteins", type=int, default=300)
    p.add_argument("--n-examples", type=int, default=6, help="Top examples per feature")
    p.add_argument("--drop-go", action="store_true", help="exclude <go> graph-slot tokens (ablation shows they're unused)")
    p.add_argument("--window", type=int, default=48, help="tokens of context each side of the max-activating token")
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    p.add_argument("--encode-batch", type=int, default=8192)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bands", default="protein,go,text",
                   help="position_type bands. This window-decoder is protein/text specific; for the DNA "
                        "store (dna,text) the DNA tokens are opaque evo2 embeddings with no per-token "
                        "text, so build_dashboard.py already emits the (band-tagged) feature_examples.")
    return p.parse_args()


def main():  # noqa: D103
    args = parse_args()
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    # This window-decoder reconstructs protein-token/text windows via the bioreason_pro SFT tokenizer +
    # protein sequences. The DNA store has no protein sequences and its DNA tokens are opaque evo2
    # embeddings (no per-token text), so decoding windows is undefined there. build_dashboard.py already
    # writes a band-tagged feature_examples.parquet for the DNA dashboard; use that instead.
    if "dna" in bands:
        raise SystemExit(
            "[dashboard] DNA store detected (--bands includes 'dna'). This protein/text window decoder "
            "does not apply to DNA (opaque evo2-embedding tokens). Use build_dashboard.py's "
            "feature_examples.parquet output for the DNA dashboard.")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import _install_unsloth_stub

    # bioreason2's collate chain does a top-level `from unsloth import FastLanguageModel`; unsloth
    # isn't installed in Env A and is only used on the training path. Register the dummy stub so the
    # tokenizer-only collate import succeeds (same trick load_bioreason_pro_sft uses).
    _install_unsloth_stub()

    from sae.activation_store import shard_table_to_array
    from sae.architectures import TopKSAE
    from transformers import AutoTokenizer

    dev = args.device if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- tokenizer + special ids ONLY (no ESM3/Qwen load → pure CPU, no GPU) ----
    # make_collate_fn needs only the tokenizer; the protein/GO placeholder ids are special tokens.
    print("[dashboard] loading tokenizer only (no model on GPU)")
    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    pid_tok = tok.convert_tokens_to_ids("<|protein_pad|>")
    gid_tok = tok.convert_tokens_to_ids("<|go_graph_pad|>")
    pad_tok = tok.pad_token_id
    print(f"[dashboard] protein_id={pid_tok} go_id={gid_tok} pad_id={pad_tok}")

    train_ds, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=args.max_length_protein)
    splits = {"train": train_ds, "validation": val_ds, "test": test_ds}
    # Match the store's ACTUAL proteins (extraction may have shuffled), else a range-select mismatches
    # the store rows and every context window fails the length check -> 0 examples. The store may come
    # from any split, so search ALL splits and pick the best match (don't trust --split).
    _store_pp = Path(args.store) / "proteins.parquet"
    if _store_pp.exists():
        store_pids = [str(x) for x in pq.read_table(_store_pp).column("protein_id").to_pylist()]
        if args.num_proteins and args.num_proteins < len(store_pids):
            store_pids = store_pids[:args.num_proteins]  # CAP for speed: first N proteins only
        best, best_sel, best_name = -1, None, args.split
        for name, sds in splits.items():
            idx = {str(p): i for i, p in enumerate(sds["protein_id"])}
            sel = [idx[p] for p in store_pids if p in idx]
            print(f"[dashboard] '{name}': matched {len(sel)}/{len(store_pids)} store proteins")
            if len(sel) > best:
                best, best_sel, best_name = len(sel), sel, name
        ds = splits[best_name].select(best_sel)
        print(f"[dashboard] using split '{best_name}' ({best}/{len(store_pids)} matched)")
        if best == 0:
            raise SystemExit("[dashboard] FATAL: store proteins matched 0 rows in any split — wrong store/dataset?")
    else:
        ds = splits[args.split].select(range(min(args.num_proteins, len(splits[args.split]))))
    collate_fn = brp_data.make_collate_fn(tok, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # reproduce per-protein unpadded token ids (same order as token_index in the store) + AA sequence
    pid_to_ids, pid_to_seq = {}, {}
    for b_i, batch in enumerate(loader):
        row = batch["input_ids"][0]
        keep = row != pad_tok
        pid = str(ds[b_i].get("protein_id", f"row{b_i}"))
        pid_to_ids[pid] = row[keep].tolist()
        pid_to_seq[pid] = str(ds[b_i].get("sequence", "") or "")
    print(f"[dashboard] reproduced token ids for {len(pid_to_ids)} proteins")

    # ---- load store: activations + per-token labels ----
    # NOTE: to_numpy >> to_pylist on tens-of-M rows, and cap to first-N proteins via CONTIGUITY
    # (proteins are contiguous in store order) instead of np.isin on a 29M-element object array (minutes).
    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    row_pid_full = tl.column("protein_id").to_numpy(zero_copy_only=False)
    n_rows = len(row_pid_full)
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    if args.num_proteins:  # R = first row of the (num_proteins)-th distinct protein (leading-contiguous)
        change = np.ones(n_rows, dtype=bool); change[1:] = row_pid_full[1:] != row_pid_full[:-1]
        starts = np.flatnonzero(change)
        R = int(starts[args.num_proteins]) if args.num_proteins < len(starts) else n_rows
        n_rows = R
        _ss = int(pq.read_metadata(shards[0]).num_rows)
        shards = shards[: (R // _ss) + 2]
    row_pid = row_pid_full[:n_rows]
    row_tidx = tl.column("token_index").to_numpy(zero_copy_only=False)[:n_rows].astype(np.int64)
    row_band = tl.column("position_type").to_numpy(zero_copy_only=False)[:n_rows]
    X = np.concatenate([shard_table_to_array(pq.read_table(sp)) for sp in shards], axis=0)[:n_rows]
    assert X.shape[0] == n_rows, f"store rows {X.shape[0]} != labels {n_rows}"

    # keep only rows for proteins we re-collated (we can only render those); also bounds CPU pass-1
    have = set(pid_to_ids)
    keep_mask = np.fromiter((p in have for p in row_pid), dtype=bool, count=n_rows)
    # NOTE: do NOT drop go-rows from the store here — window() needs the FULL token sequence to align
    # with pid_to_ids (len check). --drop-go is handled by excluding 'go' from BANDS_EX below.
    if not keep_mask.all():
        X, row_pid, row_tidx, row_band = X[keep_mask], row_pid[keep_mask], row_tidx[keep_mask], row_band[keep_mask]
        n_rows = X.shape[0]
    print(f"[dashboard] loaded {n_rows:,} token activations ({len(have)} proteins), dim={X.shape[1]}")

    # ---- refine the 'text' band into prompt / reasoning / answer ----
    # The assistant turn is wrapped <think>\n{reasoning}\n</think>\n\n{answer}. So text tokens before
    # <think> = prompt (user instructions + go_pred), between markers = reasoning, after = answer.
    # Lets us see whether a (text or cross-modal) feature is reasoning- vs answer-specific.
    THINK_OPEN = tok.convert_tokens_to_ids("<think>")
    THINK_CLOSE = tok.convert_tokens_to_ids("</think>")
    pid_marks = {}
    for _pid, _ids in pid_to_ids.items():
        pid_marks[_pid] = (_ids.index(THINK_OPEN) if THINK_OPEN in _ids else -1,
                           _ids.index(THINK_CLOSE) if THINK_CLOSE in _ids else -1)
    _nb = row_band.copy()
    for r in np.nonzero(row_band == "text")[0]:
        o, c = pid_marks.get(row_pid[r], (-1, -1))
        if c < 0:
            continue                                  # no markers found -> leave as generic 'text'
        ti = row_tidx[r]
        _nb[r] = "prompt" if (o >= 0 and ti < o) else ("reasoning" if ti < c else "answer")
    row_band = _nb
    import collections as _c
    print(f"[dashboard] text band split -> {dict(_c.Counter(row_band.tolist()))}")

    # protein -> contiguous row indices in token_index order
    order = np.argsort(row_tidx, kind="stable")
    pid_rows = {}
    for r in order:
        pid_rows.setdefault(row_pid[r], []).append(int(r))

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()}, strict=False)
    H = sae.hidden_dim

    # ---- Pass 1: per-(protein, feature) max activation ----
    pids = list(pid_rows.keys())
    pidx = {p: i for i, p in enumerate(pids)}
    row_pidx = np.array([pidx[p] for p in row_pid], dtype=np.int64)  # protein index per store row
    # per-BAND per-protein max, so cross-modal features get examples from BOTH bands (not just the
    # dominant one). max_acts[band][protein, feature] = max activation over that protein's band-tokens.
    BANDS_EX = ["protein", "go", "prompt", "reasoning", "answer", "text"]  # text = fallback (no markers)
    if args.drop_go:  # emit no go-CENTERED examples (go tokens stay in the store for window alignment)
        BANDS_EX = [b for b in BANDS_EX if b != "go"]
        print("[dashboard] --drop-go: excluding go from example bands (keeping go tokens for alignment)")
    max_acts = {b: np.zeros((len(pids), H), dtype=np.float32) for b in BANDS_EX}
    print(f"[dashboard] pass 1: encoding {n_rows:,} tokens for per-BAND per-protein max over {H:,} features")
    with torch.no_grad():
        for s in range(0, n_rows, args.encode_batch):
            e = min(n_rows, s + args.encode_batch)
            codes = sae.encode(torch.from_numpy(X[s:e]).to(dev)).cpu().numpy()  # (nb, H)
            rbs = row_band[s:e]
            for b in BANDS_EX:
                bm = (rbs == b)
                if bm.any():
                    np.maximum.at(max_acts[b], row_pidx[s:e][bm], codes[bm])
            if s % (args.encode_batch * 20) == 0:
                print(f"  {e:,}/{n_rows:,}", flush=True)
    per_band_n = args.n_examples  # top-N example proteins PER BAND the feature fires on

    # ---- Pass 2: per top (feature, protein), decode window + per-token activations ----
    print(f"[dashboard] pass 2: decoding windows for top-{per_band_n} examples/band of {H:,} features")

    res_cache = {}  # pid -> per-token residue index (protein-slot tokens map 1:1 to AA residues)

    def resmap(pid, ids):
        if pid not in res_cache:
            m, c = [None] * len(ids), 0
            for j, t in enumerate(ids):
                if t == pid_tok:
                    m[j] = c
                    c += 1
            res_cache[pid] = m
        return res_cache[pid]

    def window(pid, rows_sorted, jmax):
        """Return (space-joined decoded pieces, lo, hi) for the +/-window around token jmax.

        Protein-slot tokens render as their amino-acid residue (so bio features are readable);
        GO slots render as <go> (200 graph-memory slots, no per-slot text); text tokens decode normally.
        """
        ids = pid_to_ids.get(pid)
        if ids is None or len(ids) != len(rows_sorted):
            return None, None, None
        lo, hi = max(0, jmax - args.window), min(len(ids), jmax + args.window + 1)
        rmap, seq = resmap(pid, ids), pid_to_seq.get(pid, "")
        pieces = []
        for j in range(lo, hi):
            tid = ids[j]
            if tid == pid_tok:
                ri = rmap[j]
                piece = seq[ri] if (ri is not None and ri < len(seq)) else "<protein>"
            elif tid == gid_tok:
                piece = "<go>"
            else:
                piece = tok.decode([tid]).strip()
            pieces.append(piece or "·")
        return " ".join(pieces), lo, hi

    cache = {}  # pid -> codes over its rows (n_kept, H), rows in token_index order

    def codes_for(rows_sorted_key, rows_sorted):
        if rows_sorted_key not in cache:
            with torch.no_grad():
                cache[rows_sorted_key] = sae.encode(torch.from_numpy(X[rows_sorted]).to(dev)).cpu().numpy()
        return cache[rows_sorted_key]

    # precompute per-protein band membership masks (over its token_index-sorted rows)
    band_mask_cache = {}
    def band_mask(pid, rows_sorted, b):
        key = (pid, b)
        if key not in band_mask_cache:
            band_mask_cache[key] = np.fromiter((row_band[r] == b for r in rows_sorted), dtype=bool, count=len(rows_sorted))
        return band_mask_cache[key]

    rows_out = []
    for f in range(H):
        rank = 0
        for b in BANDS_EX:                              # emit top examples PER BAND -> both sides of cross-modal
            col = max_acts[b][:, f]
            if col.max() <= 0:
                continue                                # feature never fires on this band
            for pi in np.argsort(-col)[:per_band_n]:
                if col[pi] <= 0:
                    continue
                pid = pids[pi]; rows_sorted = pid_rows[pid]
                c = codes_for(pid, rows_sorted)[:, f]
                bm = band_mask(pid, rows_sorted, b)
                if not bm.any():
                    continue
                jmax = int(np.argmax(np.where(bm, c, -1e30)))   # peak token WITHIN band b
                seq, lo, hi = window(pid, rows_sorted, jmax)
                if seq is None:
                    continue
                rows_out.append({
                    "feature_id": f, "example_rank": rank, "protein_id": pid, "band": b,
                    "sequence": seq, "activations": [float(x) for x in c[lo:hi]],
                    "max_activation": float(col[pi]),
                })
                rank += 1
        if f % 2000 == 0:
            print(f"  feature {f:,}/{H:,}  (examples so far {len(rows_out):,})", flush=True)

    tbl = pa.table({
        "feature_id": pa.array([r["feature_id"] for r in rows_out], pa.int32()),
        "example_rank": pa.array([r["example_rank"] for r in rows_out], pa.int16()),  # int8 overflows past 127 examples
        "protein_id": pa.array([r["protein_id"] for r in rows_out]),
        "band": pa.array([r["band"] for r in rows_out]),
        "sequence": pa.array([r["sequence"] for r in rows_out]),
        "activations": pa.array([r["activations"] for r in rows_out], pa.list_(pa.float32())),
        "max_activation": pa.array([r["max_activation"] for r in rows_out], pa.float32()),
    })
    pq.write_table(tbl, out / "feature_examples.parquet", row_group_size=args.n_examples * 100, compression="snappy")
    n_feat_with_ex = len({r["feature_id"] for r in rows_out})
    print(f"\n[dashboard] wrote {out/'feature_examples.parquet'}: {len(rows_out):,} examples over "
          f"{n_feat_with_ex:,} features in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
