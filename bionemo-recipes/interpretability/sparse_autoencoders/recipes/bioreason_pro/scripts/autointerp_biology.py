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

r"""Biology-grounded autointerp for a BioReason-Pro SAE (Env B — no model needed).

Standard autointerp shows an LLM the *text* around a feature's top-activating tokens. Here the
strongest, model-free signal is biological: each SAE feature can be characterised by *what the
proteins it fires on have in common*. For the most-active features we ask: which GO term does this
feature most selectively predict (per-protein, ROC-AUC), and how enriched is it?

This is the feature -> concept direction (complement to eval.py's concept -> best-feature). It uses
only the activation store + sidecar (protein_id / position_type) + per-protein GO labels + go-basic.obo
for human-readable names. AUC is per-protein (feature's max activation over the protein's tokens vs
whether the protein carries the term), so it is not inflated by token counts.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from sae.architectures import TopKSAE

BANDS = ["protein", "go", "text"]


def parse_obo(path):
    """Minimal go-basic.obo parser -> {GO:id: name}."""
    names, cur = {}, None
    if not Path(path).exists():
        return names
    for line in open(path):
        line = line.strip()
        if line == "[Term]":
            cur = {}
        elif line.startswith("id: GO:") and cur is not None:
            cur["id"] = line[4:]
        elif line.startswith("name:") and cur is not None:
            cur["name"] = line[6:]
            if "id" in cur:
                names[cur["id"]] = cur["name"]
    return names


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--obo", default="/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo")
    p.add_argument("--top-features", type=int, default=40, help="# most-active features to interpret")
    p.add_argument("--go-min-prev", type=float, default=0.02)
    p.add_argument("--go-max-prev", type=float, default=0.5)
    p.add_argument("--max-go-terms", type=int, default=60, help="candidate informative GO terms")
    p.add_argument("--encode-batch", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict(ck["model_state_dict"])
    sae = sae.to(dev).eval()
    H = sae.hidden_dim
    obo = parse_obo(args.obo)

    layer_dir = Path(args.store) / f"layer{args.layer}"
    prot = pq.read_table(Path(args.store) / "proteins.parquet")
    pid_list = prot.column("protein_id").to_pylist()
    go_ids = [json.loads(s) for s in prot.column("go_ids").to_pylist()]  # stored as JSON strings
    pid_to_row = {pid: i for i, pid in enumerate(pid_list)}
    n_prot = len(pid_list)
    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    tok_pid = tl.column("protein_id").to_pylist()
    protein_index = np.array([pid_to_row[x] for x in tok_pid], dtype=np.int64)
    print(f"[interp] SAE H={H}, {n_prot} proteins, {len(tok_pid)} tokens")

    # ---- per-protein MAX feature activation (GPU scatter_reduce amax), + per-feature fire count ----
    pmax = torch.zeros(n_prot, H, device=dev)
    fire_count = torch.zeros(H, device=dev)
    shards = sorted(layer_dir.glob("shard_*.parquet"), key=lambda q: int(q.stem.split("_")[1]))
    row0 = 0
    with torch.no_grad():
        for sp in shards:
            t = pq.read_table(sp)
            cols = sorted([c for c in t.column_names if c.startswith("dim_")], key=lambda c: int(c.split("_")[1]))
            acts = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)
            n = acts.shape[0]
            pidx = torch.from_numpy(protein_index[row0:row0 + n]).to(dev)
            for s in range(0, n, args.encode_batch):
                e = min(n, s + args.encode_batch)
                x = torch.from_numpy(acts[s:e]).to(dev)
                codes = sae.encode(x)
                fire_count += (codes > 0).sum(dim=0)
                idx = pidx[s:e].unsqueeze(1).expand(-1, H)
                pmax.scatter_reduce_(0, idx, codes, reduce="amax", include_self=True)
            row0 += n
    pmax = pmax.cpu().numpy()
    fire_count = fire_count.cpu().numpy()

    # ---- informative GO terms (prevalence band, drop roots/rare) ----
    sets = [set(g) for g in go_ids]
    freq = Counter(t for g in go_ids for t in g)
    terms = []
    for t, _ in freq.most_common():
        prev = sum(t in s for s in sets) / n_prot
        if args.go_min_prev <= prev <= args.go_max_prev:
            terms.append(t)
        if len(terms) >= args.max_go_terms:
            break
    Y = {t: np.array([1 if t in s else 0 for s in sets]) for t in terms}

    # ---- per-feature best-GO-AUC over ALL active features, VECTORIZED via rank-AUC ----
    # AUC(feature f, term t) = (sum of ranks of positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    # We rank each feature's per-protein activations once, then matmul each term's positive mask.
    # Ranking pmax breaks the "highest-firing feature wins" bias: a feature that fires on ~all
    # proteins has near-tied ranks -> AUC ~0.5, correctly flagged as non-selective.
    active = np.where(fire_count > 0)[0]
    A = pmax[:, active]  # [n_prot, n_active]
    # average ranks along proteins (ties -> mean rank), per feature
    order_idx = np.argsort(A, axis=0)
    ranks = np.empty_like(A)
    ar = np.arange(1, n_prot + 1)
    for j in range(A.shape[1]):
        ranks[order_idx[:, j], j] = ar
    Ymat = np.stack([Y[t] for t in terms]).astype(np.float64)  # [n_terms, n_prot]
    npos = Ymat.sum(1)  # [n_terms]
    valid_t = (npos >= 5) & (npos <= n_prot - 5)
    sum_ranks = Ymat @ ranks  # [n_terms, n_active]
    auc_mat = (sum_ranks - (npos * (npos + 1) / 2)[:, None]) / (npos[:, None] * (n_prot - npos)[:, None])
    auc_mat[~valid_t] = 0.5
    best_t_idx = np.argmax(auc_mat, axis=0)
    best_auc = auc_mat[best_t_idx, np.arange(A.shape[1])]

    # winner's-curse control: same best-of-terms on random features
    rng = np.random.default_rng(args.seed)
    Rr = np.empty((n_prot, 200))
    rand = rng.standard_normal((n_prot, 200))
    oi = np.argsort(rand, axis=0)
    for j in range(200):
        Rr[oi[:, j], j] = ar
    rauc = (Ymat @ Rr - (npos * (npos + 1) / 2)[:, None]) / (npos[:, None] * (n_prot - npos)[:, None])
    rauc[~valid_t] = 0.5
    rand_baseline = round(float(np.max(rauc, axis=0).mean()), 3)

    # De-bias the reported features: select their best term on a TRAIN protein split, report AUC on a
    # held-out TEST split (kills winner's-curse from picking best-of-terms in-sample).
    tr = rng.permutation(n_prot)
    half = n_prot // 2
    tr_idx, te_idx = tr[:half], tr[half:]

    def heldout_auc(f):
        fcol = pmax[:, f]
        best, bt = 0.5, None
        for t in terms:
            y = Y[t]
            if y[tr_idx].sum() < 3 or (1 - y[tr_idx]).sum() < 3:
                continue
            try:
                a_tr = roc_auc_score(y[tr_idx], fcol[tr_idx])
            except ValueError:
                continue
            if a_tr > best:
                best, bt = a_tr, t
        if bt is None or Y[bt][te_idx].sum() < 3 or (1 - Y[bt][te_idx]).sum() < 3:
            return None, 0.5
        try:
            return bt, roc_auc_score(Y[bt][te_idx], fcol[te_idx])
        except ValueError:
            return None, 0.5

    # rank features by in-sample selectivity; report the most selective (interpretable) ones
    sel = np.argsort(-best_auc)[: args.top_features]
    results = []
    for j in sel:
        f = int(active[j])
        t = terms[best_t_idx[j]]
        y = Y[t]
        fcol = pmax[:, f]
        pos_mean = float(fcol[y == 1].mean())
        neg_mean = float(fcol[y == 0].mean() + 1e-9)
        ht, hauc = heldout_auc(f)
        results.append({
            "feature": f,
            "fire_frac": round(float(fire_count[f]) / len(tok_pid), 4),
            "best_go": t,
            "go_name": obo.get(t, "?"),
            "auc_insample": round(float(best_auc[j]), 3),
            "auc_heldout": round(float(hauc), 3),
            "heldout_go": ht,
            "heldout_go_name": obo.get(ht, "?") if ht else "?",
            "enrichment_x": round(pos_mean / neg_mean, 2),
            "n_pos_proteins": int(y.sum()),
        })
    n_selective = int((best_auc > rand_baseline + 0.05).sum())

    out = {
        "sae": args.sae, "layer": args.layer,
        "n_active_features": int(len(active)),
        "n_candidate_go_terms": int(valid_t.sum()),
        "random_best_of_terms_auc_baseline": rand_baseline,
        "n_selective_features_auc_gt_baseline+0.05": n_selective,
        "top_features": results,
    }
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"\n[interp] {len(active)} active features, {int(valid_t.sum())} candidate GO terms")
    print(f"[interp] random best-of-terms AUC baseline = {rand_baseline}; "
          f"{n_selective} features clear baseline+0.05 (selective/interpretable)\n")
    print(f"{'feat':>6} {'fire%':>6} {'AUCin':>6} {'AUCho':>6} {'enr':>5} {'nP':>4}  GO term -> name")
    for r in results:
        print(f"{r['feature']:>6} {100*r['fire_frac']:>5.1f} {r['auc_insample']:>6.2f} {r['auc_heldout']:>6.2f} "
              f"{r['enrichment_x']:>4.1f}x {r['n_pos_proteins']:>4}  {r['best_go']} -> {r['go_name']}")


if __name__ == "__main__":
    main()
