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

r"""Build the BioReason-Pro multimodal SAE dashboard data (Env B — no model).

Emits the same parquet contract as the ESM2 dashboard (feature_metadata / features_atlas /
feature_examples) so Jared's React dashboard renders it, PLUS multimodal columns that power a
dedicated multimodal view:

  * per-band fire RATE -> protein_frac / go_frac / text_frac  (rate, not raw count: text is ~85% of
    tokens, so raw counts make everything look text-heavy; rate = fires_in_band / tokens_in_band)
  * band_class: protein-heavy / go-heavy / text-heavy / cross-modal / mixed
  * crossmodal_omega: SAE-V (arXiv:2502.17514) paired-cosine across band pairs, minus random baseline
    (>0 => the feature's top tokens point the same way across modalities = genuine fusion)
  * go_label / go_auc: best GO concept this feature predicts (per-protein, held-out de-biased AUC)
  * umap_x / umap_y: 2D layout of decoder feature directions (atlas)

The string columns (band_class, go_label) are auto-detected by the dashboard as color categories; the
numeric ones (protein_frac, crossmodal_omega, ...) as histogram metrics.
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import roc_auc_score

from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

DEFAULT_BANDS = ["protein", "go", "text"]


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--bands", default=",".join(DEFAULT_BANDS),
                   help="comma-separated position_type bands (default protein,go,text; "
                        "DNA store: dna,text). The per-band fire-rate columns are named <band>_frac.")
    p.add_argument("--id-col", default="protein_id",
                   help="row/sample id column in token_labels + examples table (DNA store: sequence_id)")
    p.add_argument("--examples-file", default="proteins.parquet",
                   help="per-sample metadata table with the id column (DNA store: examples.parquet). "
                        "GO-AUC is computed only if it also has a go_ids column.")
    p.add_argument("--token-labels", default="token_labels.parquet",
                   help="per-token band sidecar (override to use a reconstructed one)")
    p.add_argument("--obo", default="/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo")
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--topk-omega", type=int, default=16)
    p.add_argument("--heavy-threshold", type=float, default=0.6, help="band frac to call '<band>-heavy'")
    p.add_argument("--crossmodal-min-frac", type=float, default=0.2, help="min 2nd-band frac for cross-modal")
    p.add_argument("--drop-go", action="store_true",
                   help="exclude <go> tokens from ALL stats (balanced models never trained on them; "
                        "running the SAE on go yields OOD-artifact 'go features'). Zeros go activations at encode.")
    p.add_argument("--skip-omega", action="store_true",
                   help="skip the SAE-V omega metric (loads full residuals to GPU -> OOM on big stores; "
                        "artifact-prone, unused). Lets the dashboard build on the FULL store with reliable stats.")
    p.add_argument("--go-min-prev", type=float, default=0.02)
    p.add_argument("--go-max-prev", type=float, default=0.5)
    p.add_argument("--max-go-terms", type=int, default=60)
    p.add_argument("--encode-batch", type=int, default=8192)
    p.add_argument("--max-shards", type=int, default=0,
                   help="0=all; cap shards (Z residuals are held in memory — avoid OOM on big stores)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    BANDS = [b.strip() for b in args.bands.split(",") if b.strip()]
    idc = args.id_col
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"])
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k):v for k,v in ck["model_state_dict"].items()})
    sae = sae.to(dev).eval()
    H = sae.hidden_dim

    # ---- row meta + (optional) GO labels ----
    # The per-sample table is proteins.parquet (protein store) or examples.parquet (DNA store).
    # GO-AUC is only meaningful when the table carries a go_ids column; otherwise it is skipped.
    prot = pq.read_table(Path(args.store) / args.examples_file)
    pid_list = prot.column(idc).to_pylist()
    has_go = "go_ids" in prot.column_names and "go" in BANDS
    go_ids = [json.loads(s) for s in prot.column("go_ids").to_pylist()] if has_go else [[] for _ in pid_list]
    if not has_go:
        print(f"[dash] no go_ids in {args.examples_file} (or 'go' not in bands) -> skipping GO-AUC")
    pid_to_row = {pid: i for i, pid in enumerate(pid_list)}
    n_prot = len(pid_list)
    tl = pq.read_table(Path(args.store) / args.token_labels)
    # truncate the row-aligned token_labels to the capped-shard row-range BEFORE the expensive
    # Python-level to_pylist()/pid_to_row loops (else we churn the full corpus even when --max-shards).
    if args.max_shards:
        _sh = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                     key=lambda q: int(Path(q).stem.split("_")[1]))[:args.max_shards]
        R = int(sum(pq.read_metadata(s).num_rows for s in _sh))
        tl = tl.slice(0, R)
    tok_pid = tl.column(idc).to_pylist()
    tok_tidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    pos_type = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    protein_index = np.array([pid_to_row[x] for x in tok_pid], dtype=np.int64)
    band_code = np.select([pos_type == b for b in BANDS], list(range(len(BANDS))), default=-1).astype(np.int64)
    band_tokens = np.array([(band_code == i).sum() for i in range(len(BANDS))], dtype=np.float64)  # corpus tot/band
    N = len(tok_pid)
    if args.max_shards:
        print(f"[dash] capped to {args.max_shards} shards = {N} tokens")
    print(f"[dash] H={H}, {n_prot} proteins, {N} tokens, band token totals={dict(zip(BANDS, band_tokens.astype(int)))}")

    obo = {}
    if Path(args.obo).exists():
        cur = None
        for line in open(args.obo):
            line = line.strip()
            if line == "[Term]":
                cur = {}
            elif line.startswith("id: GO:") and cur is not None:
                cur["id"] = line[4:]
            elif line.startswith("name:") and cur is not None and "id" in cur:
                obo[cur["id"]] = line[6:]

    # ---- streaming pass: per-feature stats, per-band fire counts, per-protein max, top examples ----
    Z = []  # keep residuals for omega cosine (val set is ~1M tokens x 2560 ~ 11GB; ok)
    fire_total = torch.zeros(H, device=dev)
    fire_band = torch.zeros(len(BANDS), H, device=dev)
    act_sum = torch.zeros(H, device=dev)
    act_max = torch.zeros(H, device=dev)
    pmax = torch.zeros(n_prot, H, device=dev)             # per-protein max activation (for GO AUC)
    NEG = -1e30
    K = args.topk_omega
    topv_band = torch.full((len(BANDS), H, K), NEG, device=dev)   # top-K activation per (band, feature)
    topi_band = torch.full((len(BANDS), H, K), -1, dtype=torch.long, device=dev)
    nex = args.n_examples
    topv_ex = torch.full((H, nex), NEG, device=dev)              # top-K examples per feature (any band)
    topi_ex = torch.full((H, nex), -1, dtype=torch.long, device=dev)
    # max_span: longest contiguous run of tokens on which a feature fires within a single sample.
    max_span = torch.zeros(H, device=dev)

    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    if args.max_shards:   # cap tokens (Z is held in memory) — dashboard needs only a representative subset
        shards = shards[:args.max_shards]
    bcode_t = torch.from_numpy(band_code).to(dev)
    pidx_t = torch.from_numpy(protein_index).to(dev)
    row0 = 0
    with torch.no_grad():
        for sp in shards:
            X = shard_table_to_array(pq.read_table(sp)).astype(np.float32)  # handles dim_ + FixedSizeList 'act'
            if not args.skip_omega:
                Z.append(X)
            n = X.shape[0]
            for s in range(0, n, args.encode_batch):
                e = min(n, s + args.encode_batch)
                g = torch.arange(row0 + s, row0 + e, device=dev)
                x = torch.from_numpy(X[s:e]).to(dev)
                codes = sae.encode(x)                                  # (nb, H)
                if args.drop_go and "go" in BANDS:   # balanced models never trained on go -> zero its OOD activations
                    codes[bcode_t[row0 + s: row0 + e] == BANDS.index("go")] = 0.0
                active = codes > 0
                fire_total += active.sum(0).float()
                act_sum += codes.sum(0)
                act_max = torch.maximum(act_max, codes.max(0).values)
                # ---- max_span: longest contiguous fire-run per feature, vectorized (nb, H) ----
                # For each column, run length at token t = t - (last position that was NOT a run
                # member). A position breaks the run if the feature is inactive OR the sample id
                # changed vs the previous token. Runs reset at batch boundaries (encode_batch >> the
                # ~4.2k tokens/sample here, so this is a tight lower bound on the true span).
                nb_ = active.shape[0]
                pidx_b = pidx_t[row0 + s: row0 + e]                     # (nb,)
                # break_at[t] = last index <= t that ENDS a run (exclusive lower bound for the run).
                # A token breaks at its own position if it is inactive. A sample boundary breaks the
                # run BETWEEN samples, i.e. at position t-1 (the token before the new sample starts) —
                # so an active first-token of a new sample gets run length 1, not 0.
                pos = torch.arange(nb_, device=dev).unsqueeze(1)        # (nb,1)
                brk_pos = torch.where(active, torch.full_like(pos, -1), pos)  # inactive -> break at self
                sb = torch.full((nb_,), -1, dtype=torch.long, device=dev)
                sb[1:] = torch.where(pidx_b[1:] != pidx_b[:-1],
                                     torch.arange(nb_, device=dev)[1:] - 1,
                                     torch.full((nb_ - 1,), -1, device=dev))
                brk_pos = torch.maximum(brk_pos, sb.unsqueeze(1))       # fold in sample-boundary breaks
                last_brk = torch.cummax(brk_pos, dim=0).values
                run_len = torch.where(active, (pos - last_brk).float(), torch.zeros((), device=dev))
                if run_len.numel():
                    max_span = torch.maximum(max_span, run_len.max(0).values)
                bc = bcode_t[row0 + s: row0 + e]
                for bi in range(len(BANDS)):
                    m = bc == bi
                    if m.any():
                        fire_band[bi] += active[m].sum(0).float()
                # per-protein max
                idx = pidx_t[row0 + s: row0 + e].unsqueeze(1).expand(-1, H)
                pmax.scatter_reduce_(0, idx, codes, reduce="amax", include_self=True)
                # top-K examples (any band) + per-band top-K (for omega)
                gg = g.unsqueeze(0).expand(H, -1)                      # (H, nb)
                ct = codes.T                                           # (H, nb)
                allv = torch.cat([topv_ex, ct], 1); alli = torch.cat([topi_ex, gg], 1)
                topv_ex, o = torch.topk(allv, nex, dim=1); topi_ex = torch.gather(alli, 1, o)
                for bi in range(len(BANDS)):
                    m = bc == bi
                    if m.any():
                        cb = ct[:, m]; ib = g[m].unsqueeze(0).expand(H, -1)
                        av = torch.cat([topv_band[bi], cb], 1); ai = torch.cat([topi_band[bi], ib], 1)
                        topv_band[bi], o = torch.topk(av, K, dim=1); topi_band[bi] = torch.gather(ai, 1, o)
            row0 += n
    Z = np.concatenate(Z) if Z else None
    fire_total_np = fire_total.cpu().numpy()
    fire_band_np = fire_band.cpu().numpy()              # (3, H)
    max_span_np = max_span.cpu().numpy()
    act_mean_np = (act_sum / fire_total.clamp(min=1)).cpu().numpy()
    act_max_np = act_max.cpu().numpy()
    pmax_np = pmax.cpu().numpy()

    # ---- band composition by RATE (corrects for text dominating token counts) ----
    rate = fire_band_np / band_tokens[:, None].clip(min=1)             # (3, H) fires-per-token-in-band
    rate_sum = rate.sum(0)
    fracs = np.where(rate_sum > 0, rate / rate_sum.clip(min=1e-12), 0.0)  # (nb, H), columns sum to 1
    band_frac = {f"{b}_frac": fracs[i] for i, b in enumerate(BANDS)}   # e.g. protein_frac/go_frac/text_frac or dna_frac/text_frac

    rng = np.random.default_rng(args.seed)   # used by omega AND the later train/test split -> define unconditionally
    fires_band = topv_band[:, :, -1] > 0                               # (nb, H): has >=K activations in band
    fires_band_bool = fires_band.cpu().numpy()
    # omega (SAE-V paired cosine) is wired for the 3-band protein/go/text taxonomy only; for any other
    # band set (e.g. dna/text) it is undefined, so treat it exactly like --skip-omega.
    if args.skip_omega or len(BANDS) != 3:
        # omega loads the FULL residual matrix to GPU (276 GiB on the 8k store -> OOM) and is an
        # artifact-prone metric we no longer use (band-mass + the per-sample cross-modal sweep replace it).
        # Skipping it lets the dashboard build on the FULL store cheaply, with reliable per-feature stats.
        omega = np.zeros(H, dtype=np.float32)
        crossmodal_pair = np.full(H, "none", dtype=object)
        fusion_class = np.full(H, "unimodal", dtype=object)
    else:
        # ---- SAE-V omega per feature (paired cosine across band pairs) - random baseline ----
        Znorm = torch.nn.functional.normalize(torch.from_numpy(Z).to(dev), dim=1)
        rng = np.random.default_rng(args.seed)
        band_tok_idx = {bi: np.where(band_code == bi)[0] for bi in range(len(BANDS))}
        base = {}
        for a in range(3):
            for b in range(a + 1, 3):
                ia = band_tok_idx[a]; ib = band_tok_idx[b]
                if len(ia) and len(ib):
                    ra = torch.from_numpy(rng.choice(ia, 8000)).to(dev)
                    rb = torch.from_numpy(rng.choice(ib, 8000)).to(dev)
                    base[(a, b)] = float((Znorm[ra] * Znorm[rb]).sum(1).mean())
                else:
                    base[(a, b)] = 0.0
        # Vectorized paired cosine per band pair over ALL features at once (gather (H,K,d), mean cos over K).
        pair_names = []
        om_pairs = []
        for a in range(3):
            for b in range(a + 1, 3):
                za = Znorm[topi_band[a].clamp(min=0)]                      # (H, K, d)
                zb = Znorm[topi_band[b].clamp(min=0)]
                cos = (za * zb).sum(-1).mean(1) - base[(a, b)]             # (H,)
                valid = fires_band[a] & fires_band[b]
                om_pairs.append(torch.where(valid, cos, torch.full_like(cos, -1e9)))
                pair_names.append(f"{BANDS[a]}-{BANDS[b]}")
                del za, zb
        om_np = torch.stack(om_pairs).cpu().numpy()                        # (n_pairs, H); pairs: pg, pt, gt
        pmap = {(0, 1): 0, (0, 2): 1, (1, 2): 2}                           # (band_a,band_b) -> row in om_np
        top2 = np.sort(np.argsort(-fracs, axis=0)[:2], axis=0)             # (2, H) two dominant bands
        pair_idx = np.array([pmap[(int(top2[0, f]), int(top2[1, f]))] for f in range(H)])
        omega = om_np[pair_idx, np.arange(H)]
        omega[omega < -1e8] = 0.0                                          # feature lacks >=K acts in a top-2 band
        crossmodal_pair = np.array([pair_names[pair_idx[f]] for f in range(H)], dtype=object)
        crossmodal_pair[omega <= 0.05] = "none"
        fusion_class = np.where(
            np.array([("text" in crossmodal_pair[f]) and omega[f] > 0.1 for f in range(H)]),
            np.array([f"cross-modal:{crossmodal_pair[f]}" for f in range(H)], dtype=object),
            "unimodal",
        )

    # ---- band_class ----
    band_class = np.empty(H, dtype=object)
    top_band = np.argmax(fracs, axis=0)
    second = np.sort(fracs, axis=0)[-2]
    for f in range(H):
        if fire_total_np[f] == 0:
            band_class[f] = "dead"
        elif fracs[:, f].max() >= args.heavy_threshold:
            band_class[f] = f"{BANDS[top_band[f]]}-heavy"
        elif second[f] >= args.crossmodal_min_frac and omega[f] > 0.05:
            # Genuine multimodal-reasoning fusion involves TEXT; protein-go alone is shared
            # injected-embedding structure (SAE-V baseline showed it's not special).
            band_class[f] = "cross-modal" if "text" in crossmodal_pair[f] else "protein-go-shared"
        else:
            band_class[f] = "mixed"

    # ---- GO label per feature: best informative term by held-out de-biased per-protein AUC ----
    # Skipped gracefully when the sample table carries no go_ids (e.g. the DNA store): go_label/go_auc
    # stay at their "none"/0 defaults so the parquet contract is preserved.
    active_feats = np.where(fire_total_np > 0)[0]
    go_label = np.array(["none"] * H, dtype=object)
    go_auc = np.zeros(H)
    sets = [set(g) for g in go_ids]
    freq = Counter(t for g in go_ids for t in g)
    terms = []
    for t, _ in freq.most_common():
        prev = sum(t in s for s in sets) / n_prot
        if args.go_min_prev <= prev <= args.go_max_prev:
            terms.append(t)
        if len(terms) >= args.max_go_terms:
            break
    if not has_go or not terms:
        print("[dash] GO-AUC skipped (no go_ids / no informative terms)")
    else:
        Ymat = np.stack([np.array([1 if t in s else 0 for s in sets]) for t in terms]).astype(np.float64)  # (T, n_prot)
        tr = rng.permutation(n_prot); half = n_prot // 2
        tr_idx, te_idx = tr[:half], tr[half:]
        Aact = pmax_np[:, active_feats]                                    # (n_prot, n_active)

        def _rank_auc(rows, terms_y):
            """Vectorized rank-AUC: AUC[t, f] for all terms x all active features over `rows` proteins."""
            sub = Aact[rows]                                              # (m, n_active)
            m = sub.shape[0]
            oi = np.argsort(sub, axis=0)
            R = np.empty_like(sub); ar = np.arange(1, m + 1)
            for j in range(sub.shape[1]):
                R[oi[:, j], j] = ar
            npos = terms_y.sum(1)                                        # (T,)
            sr = terms_y @ R                                             # (T, n_active)
            return (sr - (npos * (npos + 1) / 2)[:, None]) / (npos[:, None] * (m - npos)[:, None] + 1e-9)

        auc_tr = _rank_auc(tr_idx, Ymat[:, tr_idx])
        auc_te = _rank_auc(te_idx, Ymat[:, te_idx])
        valid_t = ((Ymat[:, tr_idx].sum(1) >= 3) & ((1 - Ymat[:, tr_idx]).sum(1) >= 3)
                   & (Ymat[:, te_idx].sum(1) >= 3) & ((1 - Ymat[:, te_idx]).sum(1) >= 3))
        auc_tr[~valid_t] = 0.5
        best_t = np.argmax(auc_tr, axis=0)                                # (n_active,) best term per feature on TRAIN
        heldout = auc_te[best_t, np.arange(len(active_feats))]            # scored on TEST (de-biased)
        for jj, f in enumerate(active_feats):
            if heldout[jj] > 0.65:
                go_label[f] = obo.get(terms[best_t[jj]], terms[best_t[jj]])
                go_auc[f] = heldout[jj]

    # ---- UMAP of decoder feature directions ----
    W = sae.decoder.weight.detach().cpu().numpy()       # (d_model, H) -> each col is a feature dir
    feat_vecs = W.T                                     # (H, d_model)
    try:
        import umap
        xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(
            feat_vecs[active_feats])
        umap_x = np.full(H, np.nan); umap_y = np.full(H, np.nan)
        umap_x[active_feats] = xy[:, 0]; umap_y[active_feats] = xy[:, 1]
    except Exception as ex:
        print(f"[dash] UMAP skipped ({ex}); using zeros")
        umap_x = np.zeros(H); umap_y = np.zeros(H)

    # ---- write feature_metadata + features_atlas ----
    freqv = fire_total_np / max(1, N)
    n_prot_active = (pmax_np > 0).sum(0)
    feat_ids = np.arange(H)
    # Display label/description (dashboard contract): GO concept if labeled, else composition class.
    descr = np.array([
        (f"{go_label[f]} (AUC {go_auc[f]:.2f})" if go_label[f] != "none" else band_class[f])
        for f in range(H)], dtype=object)
    label = np.array([f"F{i}: {descr[i]}" for i in range(H)], dtype=object)
    meta = {
        "feature_id": feat_ids,
        "label": label.astype(str),
        "description": descr.astype(str),
        "activation_freq": freqv,                 # dashboard contract name
        "activation_frequency": freqv,
        "log_frequency": np.log10(freqv + 1e-9),
        "mean_activation": np.nan_to_num(act_mean_np),
        "max_activation": act_max_np,
        "max_span": max_span_np,                               # longest contiguous fire-run within a sample
        "n_proteins_active": n_prot_active.astype(np.int64),   # dashboard contract name (= n samples firing)
        "n_samples_active": n_prot_active.astype(np.int64),
        **band_frac,                                            # <band>_frac per band (protein/go/text or dna/text)
        "band_class": band_class.astype(str),
        "crossmodal_omega": omega,
        "crossmodal_pair": crossmodal_pair.astype(str),
        "fusion_class": fusion_class.astype(str),
        "go_label": go_label.astype(str),
        "go_auc": go_auc,
    }
    live = freqv > 0
    pq.write_table(pa.table({k: v[live] if hasattr(v, "__len__") else v for k, v in meta.items()}),
                   str(out / "feature_metadata.parquet"))
    atlas = dict(meta); atlas["x"] = umap_x; atlas["y"] = umap_y       # atlas uses x/y (decoder UMAP)
    pq.write_table(pa.table({k: v[live] for k, v in atlas.items()}), str(out / "features_atlas.parquet"))

    # ---- feature_examples (band-tagged; text decoding added by interp_text_contexts.py later) ----
    ex = {k: [] for k in ("feature_id", "protein_id", "band", "activation_value", "example_rank",
                           "token_index", "residue_idx", "window_start", "sequence_window", "highlight_values")}
    topi_ex_np = topi_ex.cpu().numpy(); topv_ex_np = topv_ex.cpu().numpy()
    for f in active_feats:
        rank = 0
        for j in range(nex):
            r = topi_ex_np[f, j]
            if r < 0 or topv_ex_np[f, j] <= 0:
                continue
            band = str(pos_type[r]); act = float(topv_ex_np[f, j]); ti = int(tok_tidx[r])
            ex["feature_id"].append(int(f)); ex["protein_id"].append(tok_pid[r]); ex["band"].append(band)
            ex["activation_value"].append(act); ex["example_rank"].append(rank)
            ex["token_index"].append(ti); ex["residue_idx"].append(0); ex["window_start"].append(ti)
            # Placeholder window; interp_text_contexts.py (Env A) overwrites text-band rows with the
            # decoded reasoning window. Keeps the card renderable in the meantime.
            ex["sequence_window"].append(f"[{band}] {tok_pid[r]} @tok{ti}")
            ex["highlight_values"].append([act])
            rank += 1
    pq.write_table(pa.table(ex), str(out / "feature_examples.parquet"))

    # ---- summary ----
    print(f"[dash] wrote {live.sum()} live features to {out}")
    print(f"[dash] band_class (composition): {dict(Counter(band_class[live]))}")
    print(f"[dash] fusion_class (truly cross-modal): {dict(Counter(fusion_class[live]))}")
    print(f"[dash] features with GO label: {int((go_label[live]!='none').sum())}")


if __name__ == "__main__":
    main()
