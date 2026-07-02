#!/usr/bin/env python
"""Dump FULL protein sequences + per-token activations for genuine cross-modal features, for a viewer.

Unlike dashboard.py (±48-token windows centered on one peak -> can't show bio AND text firing together),
this emits the WHOLE sequence per (feature, protein), with each token's activation + band -> so you can SEE
where a feature fires across BOTH the protein region and the text region. Selects "genuine" cross-modal
features by BAND-MASS (protein_frac>thr AND text_frac>thr), and for each picks the protein where it co-fires
hardest on both bands. Output: crossmodal_fullseq.json -> rendered by crossmodal_viewer.html.
Usage: crossmodal_fullseq.py --sae <ckpt> --store <store> --layer L --out-dir <public/dir> [--min-frac 0.15]
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True); p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--min-frac", type=float, default=0.15, help="min band-mass frac in BOTH protein & text")
    p.add_argument("--num-proteins", type=int, default=400)
    p.add_argument("--max-features", type=int, default=40, help="cap # cross-modal features to dump")
    p.add_argument("--max-length-text", type=int, default=10000); p.add_argument("--max-length-protein", type=int, default=2000)
    args = p.parse_args()
    dev = "cuda"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp
    from bioreason_pro_sae.model_loader import _install_unsloth_stub
    _install_unsloth_stub()
    from transformers import AutoTokenizer
    from sae.architectures import TopKSAE
    from sae.activation_store import shard_table_to_array

    tok = AutoTokenizer.from_pretrained(args.ckpt_dir, trust_remote_code=True)
    pid_tok = tok.convert_tokens_to_ids("<|protein_pad|>"); gid_tok = tok.convert_tokens_to_ids("<|go_graph_pad|>")
    pad_tok = tok.pad_token_id
    THINK_O = tok.convert_tokens_to_ids("<think>"); THINK_C = tok.convert_tokens_to_ids("</think>")

    train_ds, val_ds, test_ds = brp.load_reasoning_splits(max_length_protein=args.max_length_protein)
    splits = {"train": train_ds, "validation": val_ds, "test": test_ds}
    store_pids = [str(x) for x in pq.read_table(Path(args.store) / "proteins.parquet").column("protein_id").to_pylist()][:args.num_proteins]
    best, bestsel, bestname = -1, None, "train"
    for nm, sds in splits.items():
        idx = {str(pp): i for i, pp in enumerate(sds["protein_id"])}
        sel = [idx[pp] for pp in store_pids if pp in idx]
        if len(sel) > best: best, bestsel, bestname = len(sel), sel, nm
    ds = splits[bestname].select(bestsel)
    collate = brp.make_collate_fn(tok, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    pid_to_ids, pid_to_seq = {}, {}
    for bi, b in enumerate(loader):
        r = b["input_ids"][0]; keep = r != pad_tok
        pp = str(ds[bi].get("protein_id", f"row{bi}"))
        pid_to_ids[pp] = r[keep].tolist(); pid_to_seq[pp] = str(ds[bi].get("sequence", "") or "")
    print(f"[fullseq] reproduced {len(pid_to_ids)} proteins", flush=True)

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    row_pid = np.asarray(tl.column("protein_id").to_pylist(), dtype=object)
    row_tidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    pos = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    have = set(pid_to_ids); keep = np.fromiter((x in have for x in row_pid), bool, len(row_pid))
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")), key=lambda q: int(Path(q).stem.split("_")[1]))
    # load ALL shards (small subset store) then mask — X rows must align 1:1 with token_labels before keep
    X = np.concatenate([shard_table_to_array(pq.read_table(s)) for s in shards], axis=0)[:len(row_pid)][keep]
    row_pid, row_tidx, pos = row_pid[keep], row_tidx[keep], pos[keep]

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim
    pid_rows = {}
    for i, pp in enumerate(row_pid): pid_rows.setdefault(pp, []).append(i)

    # encode per protein; per-band mass per feature -> pick cross-modal (band-mass) features
    pids = list(pid_rows.keys())
    pmass = {b: np.zeros((len(pids), H), np.float32) for b in ("protein", "text")}
    codes_cache = {}
    with torch.no_grad():
        for pi, pp in enumerate(pids):
            rows = pid_rows[pp]
            c = sae.encode(torch.from_numpy(X[rows]).to(dev)).cpu().numpy()
            codes_cache[pp] = c
            band = pos[rows]
            for b in ("protein", "text"):
                m = band == b
                if m.any(): pmass[b][pi] = c[m].sum(0)
    tot = pmass["protein"] + pmass["text"] + 1e-9
    pfrac = pmass["protein"].sum(0) / (pmass["protein"].sum(0) + pmass["text"].sum(0) + 1e-9)  # global frac per feature
    # per-feature global band fractions
    gp = pmass["protein"].sum(0); gt = pmass["text"].sum(0); gtot = gp + gt + 1e-9
    cross = [f for f in range(H) if gp[f] / gtot[f] > args.min_frac and gt[f] / gtot[f] > args.min_frac]
    print(f"[fullseq] {len(cross)} cross-modal (band-mass>{args.min_frac} both) features", flush=True)

    def role_of(pp, ti):
        ids = pid_to_ids[pp]
        o = ids.index(THINK_O) if THINK_O in ids else -1; cc = ids.index(THINK_C) if THINK_C in ids else -1
        if pos_lookup.get((pp, ti)) != "text" or cc < 0: return pos_lookup.get((pp, ti))
        return "prompt" if (o >= 0 and ti < o) else ("reasoning" if ti < cc else "answer")
    pos_lookup = {(row_pid[i], int(row_tidx[i])): pos[i] for i in range(len(row_pid))}

    out = []
    for f in cross[:args.max_features]:
        # protein where it co-fires hardest on both bands
        score = np.minimum(pmass["protein"][:, f], pmass["text"][:, f])
        pi = int(np.argmax(score))
        pp = pids[pi]
        if score[pi] <= 0: continue
        rows = pid_rows[pp]; c = codes_cache[pp][:, f]
        ids = pid_to_ids[pp]; seq = pid_to_seq.get(pp, "")
        # build full token list (decoded) + activation + band, ordered by token_index
        order = sorted(range(len(rows)), key=lambda j: int(row_tidx[rows[j]]))
        rmap, rc = {}, 0
        toks, acts, bands = [], [], []
        for j in order:
            ti = int(row_tidx[rows[j]]); tid = ids[ti] if ti < len(ids) else pad_tok
            if tid == pid_tok:
                piece = seq[rc] if rc < len(seq) else "<protein>"; rc += 1; band = "protein"
            elif tid == gid_tok:
                piece = "<go>"; band = "go"
            else:
                piece = tok.decode([tid]).strip() or "·"
                band = role_of(pp, ti) or "text"
            toks.append(piece); acts.append(round(float(c[j]), 2)); bands.append(band)
        out.append({"feature_id": int(f), "protein_id": pp,
                    "protein_frac": round(float(gp[f] / gtot[f]), 2), "text_frac": round(float(gt[f] / gtot[f]), 2),
                    "max_protein_act": round(float(pmass["protein"][pi, f]), 1), "max_text_act": round(float(pmass["text"][pi, f]), 1),
                    "tokens": toks, "activations": acts, "bands": bands})
        if len(out) % 10 == 0: print(f"  dumped {len(out)}", flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out_dir) / "crossmodal_fullseq.json", "w"))
    print(f"[fullseq] wrote {len(out)} cross-modal full-sequence examples -> {args.out_dir}/crossmodal_fullseq.json")


if __name__ == "__main__":
    main()
