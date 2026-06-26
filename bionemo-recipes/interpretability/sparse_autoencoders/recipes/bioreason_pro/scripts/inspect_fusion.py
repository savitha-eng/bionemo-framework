# Manual-inspection SAE-V: find features that fire on BOTH the protein/GO embedding band AND
# the model's own REASONING text (response, not prompt boilerplate), then dump side-by-side
# evidence so a human can judge: shared concept (real fusion) vs. boundary/coincidence.
#
# Two passes over a shard SAMPLE (GPU):
#   pass 1: per-feature activation mass per band -> rank co-firing candidates
#   pass 2: for the top candidates, collect top-activating tokens in protein-band & reasoning,
#           then decode (residue for protein tokens, words for reasoning) via tokenizer re-collate.
#
#   python scripts/inspect_fusion.py --sae <ckpt> --store <store> --layer 28 --max-shards 30 --out out.json
import argparse, glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np, pyarrow.parquet as pq, torch
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ckpt-dir", default="/data/savithas/scratch/hf-cache/hub/"
                    "models--wanglab--bioreason-pro-sft/snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")
    ap.add_argument("--max-shards", type=int, default=30, help="shard sample for the ranking (speed)")
    ap.add_argument("--n-cand", type=int, default=40, help="top co-firing features to dump")
    ap.add_argument("--topk", type=int, default=8, help="examples per band per feature")
    ap.add_argument("--min-react", type=float, default=0.05, help="min reasoning mass fraction to be 'co-firing'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"

    # ---- row-aligned band/role labels ----
    tl = pq.read_table(f"{a.store}/token_labels_with_role.parquet")
    pos = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    role = np.asarray(tl.column("role").to_pylist(), dtype=object)
    rpid = np.asarray(tl.column("protein_id").to_pylist(), dtype=object)
    rtidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    is_prot = torch.from_numpy(pos == "protein").to(dev)
    is_react = torch.from_numpy((pos == "text") & (role == "response")).to(dev)  # reasoning, NOT prompt
    is_go = torch.from_numpy(pos == "go").to(dev)

    ck = torch.load(a.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval(); sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()})
    H = sae.hidden_dim

    shards = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"),
                    key=lambda q: int(Path(q).stem.split("_")[1]))[:a.max_shards]
    # ---- pass 1: per-feature mass per band ----
    mass_p = torch.zeros(H, device=dev); mass_r = torch.zeros(H, device=dev); mass_g = torch.zeros(H, device=dev)
    row0 = 0
    with torch.no_grad():
        for sp in shards:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            for s in range(0, n, 8192):
                e = min(n, s + 8192); c = sae.encode(torch.from_numpy(X[s:e]).to(dev))
                gp, gr, gg = is_prot[row0 + s:row0 + e], is_react[row0 + s:row0 + e], is_go[row0 + s:row0 + e]
                if gp.any(): mass_p += c[gp].sum(0)
                if gr.any(): mass_r += c[gr].sum(0)
                if gg.any(): mass_g += c[gg].sum(0)
            row0 += n
    mp, mr, mg = mass_p.cpu().numpy(), mass_r.cpu().numpy(), mass_g.cpu().numpy()
    total = mp + mr + mg + 1e-9
    react_frac = mr / total; prot_frac = mp / total
    # co-firing = meaningful mass in BOTH protein band and reasoning
    cand = np.where((prot_frac > 0.15) & (react_frac > a.min_react) & (mp > 0) & (mr > 0))[0]
    cand = cand[np.argsort(-(prot_frac[cand] * react_frac[cand]))][:a.n_cand]
    print(f"[fusion] {len(cand)} co-firing candidates (protein>15% AND reasoning>{a.min_react*100:.0f}%)", flush=True)

    # ---- pass 2: top tokens per band for candidates ----
    cset = torch.tensor(cand, device=dev)
    # buffer per (feature, band): list of (acts_array, rows_array) appended PER BATCH (not per token)
    buf = {int(f): {"prot": [], "react": []} for f in cand}
    row0 = 0
    with torch.no_grad():
        for sp in shards:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            for s in range(0, n, 8192):
                e = min(n, s + 8192); c = sae.encode(torch.from_numpy(X[s:e]).to(dev))[:, cset]  # (b, n_cand)
                for bi, band in [("prot", is_prot[row0 + s:row0 + e]), ("react", is_react[row0 + s:row0 + e])]:
                    if not band.any(): continue
                    sub = c[band].cpu().numpy()                                          # (mb, n_cand)
                    rws = torch.arange(row0 + s, row0 + e, device=dev)[band].cpu().numpy()
                    for ci, f in enumerate(cand):
                        nz = np.where(sub[:, ci] > 0)[0]
                        if len(nz):
                            buf[int(f)][bi].append((sub[nz, ci], rws[nz]))
            row0 += n
    heaps = {int(f): {"prot": [], "react": []} for f in cand}
    for f in buf:
        for b in buf[f]:
            if buf[f][b]:
                av = np.concatenate([x[0] for x in buf[f][b]]); rv = np.concatenate([x[1] for x in buf[f][b]])
                ordr = np.argsort(-av)[:a.topk]
                heaps[f][b] = list(zip(av[ordr].tolist(), rv[ordr].tolist()))

    # ---- decode: residues for protein tokens, words for reasoning ----
    print("[fusion] loading tokenizer + dataset for decode...", flush=True)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from bioreason_pro_sae.model_loader import _install_unsloth_stub
    _install_unsloth_stub()  # collate import chain pulls in unsloth (not installed) — stub it like dashboard.py
    import bioreason_pro_sae.data as brp_data  # noqa
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tok = AutoTokenizer.from_pretrained(a.ckpt_dir, trust_remote_code=True)
    pad = tok.pad_token_id
    tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
    splits = {"train": tr, "validation": va, "test": te}
    store_pids = [str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
    # which proteins do we actually need to decode?
    need = set()
    for f in heaps:
        for b in heaps[f]:
            for _act, gr in heaps[f][b]:
                need.add(str(rpid[gr]))
    # pick split + collate only the needed proteins
    best = max(splits, key=lambda nm: sum(p in {str(x) for x in splits[nm]["protein_id"]} for p in store_pids[:200]))
    ds = splits[best]; idx = {str(p): i for i, p in enumerate(ds["protein_id"])}
    need_idx = [idx[p] for p in need if p in idx]
    sub = ds.select(need_idx)
    cf = brp_data.make_collate_fn(tok, 10000, 2000)
    loader = DataLoader(sub, batch_size=1, shuffle=False, collate_fn=cf)
    pid_ids, pid_seq = {}, {}
    for bi, batch in enumerate(loader):
        row = batch["input_ids"][0]; keep = row != pad
        pid = str(sub[bi].get("protein_id", "")); pid_ids[pid] = row[keep].tolist()
        pid_seq[pid] = str(sub[bi].get("sequence", "") or "")
    print(f"[fusion] decoded {len(pid_ids)} proteins (split={best})", flush=True)

    def ctx(gr):
        pid = str(rpid[gr]); t = int(rtidx[gr]); p = pos[gr]
        if p == "protein":
            seq = pid_seq.get(pid, "")
            # protein-band token_index maps to residue position within the band
            return {"protein": pid, "pos": t, "residue": (seq[t] if t < len(seq) else "?"),
                    "around": seq[max(0, t - 6):t + 7]}
        ids = pid_ids.get(pid, [])
        seg = ids[max(0, t - 10):t + 11]
        return {"protein": pid, "pos": t, "text": tok.decode(seg).replace("\n", " ")[:200]}

    out = []
    for f in heaps:
        out.append({"feature": int(f),
                    "prot_frac": round(float(prot_frac[f]), 3), "react_frac": round(float(react_frac[f]), 3),
                    "protein_examples": [{"act": round(a_, 2), **ctx(g)} for a_, g in heaps[f]["prot"]],
                    "reasoning_examples": [{"act": round(a_, 2), **ctx(g)} for a_, g in heaps[f]["react"]]})
    out.sort(key=lambda x: -(x["prot_frac"] * x["react_frac"]))
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[fusion] wrote {len(out)} co-firing features -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
