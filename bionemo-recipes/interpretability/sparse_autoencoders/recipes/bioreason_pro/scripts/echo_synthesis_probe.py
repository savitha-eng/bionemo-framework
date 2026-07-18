#!/usr/bin/env python
"""ECHO-vs-SYNTHESIS probe (NON-CIRCULAR): which SAE features fire when the model reasons BEYOND the given
annotations (synthesis) vs when it restates them (echo)? The prompt GIVES InterPro domains + go_pred; a
reasoning token is ECHO if its word is in the prompt, SYNTHESIS if novel. The label is a property of the
REASONING PROCESS, not the GO answer -> non-circular (unlike GO-term probing). Adapts Goodfire's
reasoning-feature discovery: find 'synthesis features' by their firing at synthesis vs echo moments.

Per SAE feature: rank-sum AUROC(activation, synthesis-token vs echo-token) over response tokens. Top =
synthesis features (fire on novel inference); bottom = echo features (fire on restatement). + label-shuffle null.
Usage: echo_synthesis_probe.py <sae.pt> <store> <layer> [--nprot 2500] [--shard-stride 2] [--max-tok 60000]
"""
import argparse, glob, sys, json
from pathlib import Path
from collections import Counter
import numpy as np, torch, pyarrow.parquet as pq, pandas as pd
from scipy.stats import rankdata
sys.path.insert(0, "src")
from bioreason_pro_sae.model_loader import _install_unsloth_stub; _install_unsloth_stub()  # official stub (protein_llm imports unsloth)
from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array
from transformers import AutoTokenizer
import bioreason_pro_sae.data as brp
DEFAULT_CKPT = ("/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
                "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")
p = argparse.ArgumentParser()
p.add_argument("sae"); p.add_argument("store"); p.add_argument("layer", type=int)
p.add_argument("--nprot", type=int, default=2500); p.add_argument("--shard-stride", type=int, default=2)
p.add_argument("--max-tok", type=int, default=60000); p.add_argument("--ckpt", default=DEFAULT_CKPT)
p.add_argument("--out", default="/data/savithas/phase3_full/echo_synthesis_l30.json")
p.add_argument("--freq-file", default="/data/savithas/phase3_full/per_token_freq_l30.npy",
               help="per-feature firing frequency; features firing on >freq-max of tokens are near-dense "
                    "(always-on) and are EXCLUDED from the synthesis/echo lists (else they contaminate the top)")
p.add_argument("--freq-max", type=float, default=0.02, help="max per-token firing freq for a CLEAN feature")
a = p.parse_args(); dev = "cuda"

def norm(s): return "".join(c for c in s.lower() if c.isalnum())
tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
tr, _, _ = brp.load_reasoning_splits(max_length_protein=2000)
# use the SAME prompt build as the store: prompt includes interpro_formatted + go_pred + names + function
pids_ds = [str(x) for x in tr["protein_id"]]
sub = pids_ds[:a.nprot]
# reconstruct per-protein: word per token, response mask, prompt-word set (the GIVEN annotations)
word_by_pid = {}; role_by_pid = {}; prompt_words = {}; word_doc = Counter()
coll = brp.make_collate_fn(tok, 10000, 2000)
from torch.utils.data import DataLoader
idx_of = {p: i for i, p in enumerate(pids_ds)}
dsub = tr.select([idx_of[p] for p in sub])
for k in range(len(dsub)):
    row = dsub[k]; pid = str(row["protein_id"])
    b = coll([row]); ids = b["input_ids"][0].tolist(); labels = b["labels"][0].tolist()
    idk = [t for t in ids if t != tok.pad_token_id]
    isresp = np.array([labels[j] != -100 for j in range(len(ids)) if ids[j] != tok.pad_token_id], bool)
    pieces = tok.convert_ids_to_tokens(idk)
    starts = [1 if (j == 0 or pi.startswith("Ġ") or pi.startswith("Ċ")) else 0 for j, pi in enumerate(pieces)]
    wid = np.cumsum(starts) - 1; wp = {}
    for j, pi in enumerate(pieces): wp[wid[j]] = wp.get(wid[j], "") + pi.replace("Ġ", "").replace("Ċ", "")
    wordstr = {w: norm(s) for w, s in wp.items()}
    tw = np.array([wordstr[wid[j]] for j in range(len(idk))], dtype=object)
    word_by_pid[pid] = tw; role_by_pid[pid] = isresp
    prompt_words[pid] = {tw[j] for j in range(len(idk)) if not isresp[j] and len(tw[j]) >= 3}
    for w in set(tw):
        if len(w) >= 4: word_doc[w] += 1
    if k % 500 == 0: print(f"  reconstruct {k}/{len(sub)}", flush=True)
realword = {w for w, c in word_doc.items() if c >= 8}
print(f"[echo-synth] {len(sub)} proteins, {len(realword)} recurring content words", flush=True)

# align store rows -> response mask + synthesis(novel) label + word
tl = pq.read_table(f"{a.store}/token_labels.parquet")
rpid = np.array(tl.column("protein_id").to_pylist(), dtype=object)
rtidx = np.array(tl.column("token_index").to_pylist(), dtype=np.int64); N = len(rpid)
resp = np.zeros(N, bool); novel = np.zeros(N, bool); word_row = np.empty(N, dtype=object); word_row[:] = ""
df = pd.DataFrame({"pid": rpid.astype(str), "j": rtidx, "row": np.arange(N)})
for pid, g in df.groupby("pid", sort=False):
    tw = word_by_pid.get(pid); r = role_by_pid.get(pid)
    if tw is None: continue
    rows = g["row"].to_numpy(); js = g["j"].to_numpy(); ok = js < len(tw); rows, js = rows[ok], js[ok]
    isr = r[js]; resp[rows] = isr; w = tw[js]; word_row[rows] = w
    pw = prompt_words[pid]
    novel[rows] = np.array([isr[i] and len(w[i]) >= 4 and (w[i] not in pw) for i in range(len(w))], bool)
# a response token is ECHO if its (>=4char) word IS in the prompt annotations
echo = resp & (~novel) & np.array([len(str(x)) >= 4 for x in word_row])
synth = resp & novel
print(f"[echo-synth] response tokens: {int(resp.sum()):,}; synthesis {int(synth.sum()):,}; echo {int(echo.sum()):,}", flush=True)

# gather ALL-feature SAE activations on a SAMPLE of echo+synth response tokens
cand = np.where(echo | synth)[0]
rng = np.random.default_rng(0)
if len(cand) > a.max_tok: cand = np.sort(rng.choice(cand, a.max_tok, replace=False))
candset = set(cand.tolist()); y = synth[cand].astype(np.int8)  # 1=synthesis, 0=echo
ck = torch.load(a.sae, map_location="cpu"); sae = TopKSAE(**ck["model_config"]).to(dev).eval()
sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()}, strict=False)
H = sae.hidden_dim; Z = np.zeros((len(cand), H), np.float16); pos = {int(r): i for i, r in enumerate(cand)}
row0 = 0; got = np.zeros(len(cand), bool)
order = sorted(glob.glob(f"{a.store}/layer{a.layer}/shard_*.parquet"), key=lambda q: int(Path(q).stem.split("_")[1]))
with torch.no_grad():
    for si, sp in enumerate(order):
        nrows = pq.read_metadata(sp).num_rows
        if si % a.shard_stride == 0:
            X = shard_table_to_array(pq.read_table(sp)); n = X.shape[0]
            lr = cand[(cand >= row0) & (cand < row0 + n)]
            if len(lr):
                enc = sae.encode(torch.from_numpy(np.ascontiguousarray(X[lr - row0])).to(dev)).half().cpu().numpy()
                for j, gr in enumerate(lr): Z[pos[int(gr)]] = enc[j]; got[pos[int(gr)]] = True
        row0 += nrows
        if si % 30 == 0: print(f"  gather {si}/{len(order)}", flush=True)
Zg = Z[got].astype(np.float32); yg = y[got]
npos, nneg = int(yg.sum()), int((1 - yg).sum())
print(f"[echo-synth] gathered {int(got.sum()):,} tokens ({npos} synth / {nneg} echo) x {H}; ranking...", flush=True)

# per-feature rank-sum AUROC: synthesis vs echo
r = rankdata(Zg, axis=0); Rpos = r[yg == 1].sum(0)
auc = (Rpos - npos * (npos + 1) / 2) / (npos * nneg)          # >0.5 => fires on SYNTHESIS; <0.5 => on ECHO
# shuffle null
yp = rng.permutation(yg); Rp = rankdata(Zg, axis=0)[yp == 1].sum(0)
aucn = (Rp - npos * (npos + 1) / 2) / (npos * nneg)
def topwords(fi, want_syn):
    col = Zg[:, fi]; nz = col > 0
    top = np.where(got)[0][nz][np.argsort(-col[nz])[:80]]  # indices into cand
    ws = [str(word_row[cand[t]]) for t in top]
    return list(dict.fromkeys([w for w in ws if len(w) >= 4]))[:10]
# FREQUENCY FILTER: exclude near-dense (always-on) features. Ranking synthesis features by AUROC alone lets
# ~100%-firing features top the list on a weak magnitude bias -- they are NOT clean interpretable features.
# We want sparse, dashboard-browsable features that fire MORE on synthesis than echo, so mask on firing freq.
try:
    freq = np.load(a.freq_file)
    active_cnt = (Zg > 0).sum(0)                               # fires on enough sampled tokens to be real
    clean = (freq[:H] <= a.freq_max) & (active_cnt >= 30)
    ndense = int((freq[:H] > a.freq_max).sum())
    print(f"[echo-synth] freq filter <= {a.freq_max:.0%}: {int(clean.sum())} clean features "
          f"({ndense} near-dense excluded)")
except FileNotFoundError:
    clean = np.ones(H, bool); print("[echo-synth] no freq file -> NOT frequency-filtered (list may be contaminated)")
auc_masked = np.where(clean, auc, 0.5)                         # dense features -> neutral 0.5, drop from both ends
synth_f = np.argsort(-auc_masked)[:12]; echo_f = np.argsort(auc_masked)[:12]
print(f"\nnull AUROC (shuffled): max={aucn.max():.3f} min={aucn.min():.3f} (real should exceed)")
print("=== CLEAN SYNTHESIS features (sparse; fire MORE on novel elaboration than on restatement) ===")
for f in synth_f: print(f"  F{f}: AUROC={auc[f]:.3f} freq={freq[f]*100:.2f}%  words={topwords(f,1)}")
print("=== CLEAN ECHO features (sparse; fire MORE on restated InterPro/GO IDs) ===")
for f in echo_f: print(f"  F{f}: AUROC={auc[f]:.3f} freq={freq[f]*100:.2f}%  words={topwords(f,0)}")
# save gathered activations so the probe can be re-fit without re-streaming shards
np.savez(a.out.replace(".json", "_acts.npz"), Zg=Zg.astype(np.float16), yg=yg.astype(np.int8),
         feat=np.arange(H, dtype=np.int32))
print(f"[wrote] {a.out.replace('.json','_acts.npz')} (Zg for probe re-fit)", flush=True)

# ---- TRAINED LINEAR PROBE (not single-feature AUROC): is echo-vs-synthesis linearly DECODABLE from
# the SAE feature vector, and WHICH features are load-bearing? L1-logistic -> nonzero coefs = selected
# features; sign(+)=elaboration/synthesis, sign(-)=echo/restatement. Held-out 5-fold CV AUROC. ----
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD
active = np.where(Zg.std(0) > 1e-6)[0]           # drop dead features
Xa = StandardScaler().fit_transform(Zg[:, active])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
# (1) L1-sparse probe: interpretable feature selection
clfL1 = LogisticRegression(penalty="l1", solver="saga", C=0.05, max_iter=400)
pL1 = cross_val_predict(clfL1, Xa, yg, cv=cv, method="decision_function", n_jobs=5)
aucL1 = roc_auc_score(yg, pL1)
clfL1.fit(Xa, yg); coef = clfL1.coef_[0]; nz = int((coef != 0).sum())
elab = active[np.argsort(-coef)[:12]]; rest = active[np.argsort(coef)[:12]]  # +coef synth / -coef echo
# (2) dense L2 probe on SVD-256: distributed-decodability ceiling
svd = TruncatedSVD(256, random_state=0).fit_transform(StandardScaler().fit_transform(Zg))
aucDense = roc_auc_score(yg, cross_val_predict(LogisticRegression(C=1.0, max_iter=400),
                         svd, yg, cv=cv, method="decision_function", n_jobs=5))
# (3) shuffle-label null for the trained probe
aucNull = roc_auc_score(rng.permutation(yg),
                        cross_val_predict(LogisticRegression(C=1.0, max_iter=200),
                        svd, rng.permutation(yg), cv=cv, method="decision_function", n_jobs=5))
best_single = float(max(auc.max(), 1 - auc.min()))
print(f"\n=== TRAINED echo-vs-synthesis probe (held-out 5-fold CV AUROC) ===")
print(f"  L1-sparse  : {aucL1:.3f}  ({nz} features selected)")
print(f"  dense SVD256: {aucDense:.3f}   shuffle-null: {aucNull:.3f}   best-single-feature: {best_single:.3f}")
print(f"  gap (dense - best-single) = {aucDense - best_single:+.3f}  (large => DISTRIBUTED, not monosemantic)")
print("  ELABORATION features (+coef, model reasons beyond prompt):")
for f in elab: print(f"    F{f}: coef={coef[list(active).index(f)]:+.2f} {topwords(f,1)[:5]}")
print("  RESTATEMENT features (-coef, model restates given IDs):")
for f in rest: print(f"    F{f}: coef={coef[list(active).index(f)]:+.2f} {topwords(f,0)[:5]}")
probe = {"cv_auroc_l1_sparse": round(float(aucL1), 3), "n_features_selected": nz,
         "cv_auroc_dense_svd256": round(float(aucDense), 3), "cv_auroc_shuffle_null": round(float(aucNull), 3),
         "best_single_feature_auroc": round(best_single, 3),
         "gap_dense_minus_single": round(float(aucDense - best_single), 3),
         "elaboration_features": {int(f): {"coef": round(float(coef[list(active).index(f)]), 3),
                                           "words": topwords(f, 1)} for f in elab},
         "restatement_features": {int(f): {"coef": round(float(coef[list(active).index(f)]), 3),
                                           "words": topwords(f, 0)} for f in rest}}
def ffreq(f):
    try: return round(float(freq[f] * 100), 3)
    except Exception: return None
out = {"n_synth": npos, "n_echo": nneg, "null_auc_max": round(float(aucn.max()), 3),
       "freq_max_pct": a.freq_max * 100, "note": "synthesis/echo lists are FREQUENCY-FILTERED (clean, sparse); "
       "AUROC ranks how synthesis-vs-echo-leaning a feature is, NOT a claim any feature 'is synthesis'",
       "synthesis_features": {int(f): {"auroc": round(float(auc[f]), 3), "freq_pct": ffreq(f), "words": topwords(f, 1)} for f in synth_f},
       "echo_features": {int(f): {"auroc": round(float(auc[f]), 3), "freq_pct": ffreq(f), "words": topwords(f, 0)} for f in echo_f},
       "trained_probe": probe}
json.dump(out, open(a.out, "w"), indent=2); print(f"[wrote] {a.out}")
