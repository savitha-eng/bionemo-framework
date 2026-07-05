#!/usr/bin/env python
"""On-demand autointerp backend for the dashboard 'Auto-interpret' button.
POST /autointerp {"model": "l16_balanced", "feature_id": 123, "bands": true}
 -> {"label": "...", "band_labels": {"reasoning": {"label", "peak_activation", ...},
                                     "answer": {...}, "protein": {...}}}
    (overall label + a SEPARATE interpretation per band the feature fires on, so protein dashboards
     give reasoning-vs-answer and DNA gives dna-vs-text). Labels a feature's top windows via NIM, live.
Keeps the NIM key server-side. Run: source /data/savithas/.nim_env && python autointerp_server.py [port]
"""
import json, os, sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pyarrow.parquet as pq
from openai import OpenAI

PUB = "/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/bioreason_pro/multimodal_dashboard/public"
MODEL = "meta/llama-3.1-70b-instruct"
TEXT = {"prompt", "reasoning", "answer", "text"}
_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NIM_API_KEY"], timeout=60)
_cache = {}


def _examples(model):
    if model not in _cache:
        _cache[model] = pq.read_table(f"{PUB}/{model}/feature_examples.parquet").to_pandas()
    return _cache[model]


# ---- protein-feature ENRICHMENT (reliable, no-LLM): which GO term is statistically over-represented
# among the proteins a feature fires on, vs a random background (Fisher exact). Local go_ids, instant. ----
_PROTEINS = ("/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/"
             "bioreason_pro/cache_dir/activations/train_full_L16_L18_L20_L22/proteins.parquet")
_OBO = "/data/savithas/bioreason-pro/bioreason2/dataset/go-basic.obo"
_go_state = {}


def _protein_go():
    if not _go_state:
        import numpy as _np
        from collections import Counter as _C
        prot = pq.read_table(_PROTEINS).to_pandas()
        gmap = {r.protein_id: (json.loads(r.go_ids) if isinstance(r.go_ids, str) else list(r.go_ids or []))
                for r in prot.itertuples()}
        names, cur = {}, None
        if os.path.exists(_OBO):
            for line in open(_OBO):
                line = line.strip()
                if line == "[Term]":
                    cur = {}
                elif line.startswith("id: GO:") and cur is not None:
                    cur["id"] = line[4:]
                elif line.startswith("name:") and cur is not None and "id" in cur:
                    names[cur["id"]] = line[6:]
        bg = list(_np.random.default_rng(0).choice(list(gmap), size=min(2000, len(gmap)), replace=False))
        _go_state.update(gmap=gmap, names=names, bg=_C(t for pid in bg for t in gmap.get(pid, [])), nbg=len(bg))
    return _go_state


def _protein_enrichment(pids, top=3, min_k=2):
    from collections import Counter
    from scipy.stats import fisher_exact
    st = _protein_go(); n, N = len(pids), st["nbg"]
    fg = Counter(t for pid in pids for t in st["gmap"].get(pid, []))
    out = []
    for term, k in fg.items():
        if k < min_k:
            continue
        K = max(st["bg"].get(term, 1), k)  # background count (+guard so table is valid)
        p = float(fisher_exact([[k, n - k], [K - k, (N - n) - (K - k)]], alternative="greater")[1])
        out.append({"go": term, "name": st["names"].get(term, ""), "k": k, "n": n, "p": p})
    out.sort(key=lambda d: d["p"])
    return out[:top]


# ---- UniProt keyword/domain enrichment (richer/structural vocab; reads a pre-warmed cache) ----
_UCACHE = "/data/savithas/phase3_full/uniprot_cache.json"
_up_state = {}


def _uniprot_bg():
    """Load the UniProt cache + background keyword/domain frequencies over the same 2000-protein GO background."""
    if not _up_state:
        import numpy as _np
        from collections import Counter as _C
        cache = json.load(open(_UCACHE)) if os.path.exists(_UCACHE) else {}
        st = _protein_go()
        bg = list(_np.random.default_rng(0).choice(list(st["gmap"]), size=min(2000, len(st["gmap"])), replace=False))
        bgkw = _C(kw for p in bg for kw in cache.get(p, {}).get("keywords", []))
        bgdm = _C(dm for p in bg for dm in cache.get(p, {}).get("domains", []))
        _up_state.update(cache=cache, bgkw=bgkw, bgdm=bgdm, nbg=len(bg))
    return _up_state


def _uniprot_enrichment(pids, top=3, min_k=2):
    from collections import Counter
    from scipy.stats import fisher_exact
    st = _uniprot_bg(); cache = st["cache"]
    pids = [p for p in pids if p in cache]  # only cached proteins (pre-warmed)
    n, N = len(pids), st["nbg"]
    if n == 0:
        return []
    def enrich(field, bg):
        fg = Counter(v for p in pids for v in cache[p].get(field, []))
        rows = []
        for term, k in fg.items():
            if k < min_k or not term:
                continue
            K = max(bg.get(term, 1), k)
            pv = float(fisher_exact([[k, n - k], [K - k, (N - n) - (K - k)]], alternative="greater")[1])
            rows.append({"term": term, "k": k, "n": n, "p": pv})
        rows.sort(key=lambda d: d["p"]); return rows[:top]
    return {"keywords": enrich("keywords", st["bgkw"]), "domains": enrich("domains", st["bgdm"])}


def _win(seq, acts, ctx=8):
    """Return (window_with_peak_marked, peak_token)."""
    toks = seq.split(" ") if isinstance(seq, str) else list(seq)
    a = (json.loads(acts) if isinstance(acts, str) else list(acts))[:len(toks)]  # activations stored as JSON string
    if not a:
        return (seq or "")[:160], ""
    pk = max(range(len(a)), key=lambda i: a[i])
    peak_tok = toks[pk] if pk < len(toks) else ""
    lo, hi = max(0, pk - ctx), min(len(toks), pk + ctx + 1)
    out = toks[lo:hi]; r = pk - lo
    if 0 <= r < len(out):
        out[r] = f"«{out[r]}»"
    return " ".join(out), peak_tok


# The system prompt + KIND taxonomy are domain-specific. A DNA model has nucleotide/variant/gene tokens,
# NOT GO terms or protein residues, so it gets its own prompt (otherwise DNA features get force-fit into the
# protein KINDs, e.g. everything -> STRUCTURE). prompts_for() picks by model name.
SYS_PROTEIN = (
    "You interpret a sparse-autoencoder feature of a protein-reasoning LLM. Its reasoning/answer text "
    "frequently PRINTS Gene-Ontology term names (e.g. 'cytosol', 'protein binding') and GO accessions "
    "(e.g. 'GO:0005737'). «token» marks where the feature fires HARDEST. Do NOT give a generic biological "
    "category — name the SPECIFIC token/pattern, and judge whether it is merely firing on a printed GO "
    "term/accession (label-reading) vs genuine reasoning.")
KIND_PROTEIN = (
    "KIND: <one of GO-TERM-TEXT | ACCESSION | REASONING | STRUCTURE | PROTEIN | SYNTACTIC | POLYSEMANTIC>  "
    "(GO-TERM-TEXT/ACCESSION = it's just reading a printed GO term/id = label-reading; "
    "REASONING = genuine reasoning content; STRUCTURE = formatting/position; PROTEIN = residues; "
    "SYNTACTIC = fires on a grammatical position like conjunctions/punctuation; "
    "POLYSEMANTIC = no single consistent concept = not cleanly interpretable)")
SYS_DNA = (
    "You interpret a sparse-autoencoder feature of a DNA variant-effect-prediction LLM (Evo2 DNA embeddings "
    "fed into Qwen). The DNA side is nucleotide tokens (A/C/G/T k-mers, incl. ⟦S⟧/⟦E⟧ segment markers); the "
    "text side is a variant QUESTION (e.g. '...chromosome 1 position 1040717, gene AGRN: benign or "
    "pathogenic?') and a short ANSWER (e.g. 'Answer: pathogenic; Congenital myasthenic syndrome'). «token» "
    "marks where the feature fires HARDEST. Name the SPECIFIC token/pattern; do NOT give a generic category.")
KIND_DNA = (
    "KIND: <one of NUCLEOTIDE-MOTIF | GENE-NAME | VARIANT-COORD | VERDICT | DISEASE-NAME | STRUCTURE | "
    "SYNTACTIC | POLYSEMANTIC>  "
    "(NUCLEOTIDE-MOTIF = a DNA k-mer/sequence pattern; GENE-NAME = a gene symbol in text; "
    "VARIANT-COORD = chromosome/position tokens; VERDICT = benign/pathogenic; DISEASE-NAME = a disease term; "
    "STRUCTURE = formatting/boundary markers like ⟦E⟧ or <|im_start|>; "
    "SYNTACTIC = fires on a grammatical position like conjunctions/punctuation; "
    "POLYSEMANTIC = no single consistent concept = not cleanly interpretable)")


def prompts_for(model):
    return (SYS_DNA, KIND_DNA) if "dna" in (model or "").lower() else (SYS_PROTEIN, KIND_PROTEIN)


# high-frequency function words / punctuation / chat markers. A feature whose peaks are dominated by these
# is usually SYNTACTIC or POLYSEMANTIC, not a clean concept — the labeler should say so, not invent meaning.
STOPWORDS = {"and", "or", "of", "the", "a", "an", "in", "on", "to", "for", "with", "by", "is", "are",
             "as", "at", "that", "this", "it", "its", "be", "been", "which", "from", ",", ".", ";", ":",
             "(", ")", "-", "'", "via", "<|im_start|>", "<|im_end|>", "assistant", "user"}


def _label(windows, peak_tokens, sys=SYS_PROTEIN, kind_line=KIND_PROTEIN):
    body = "\n".join("  - " + w for w in windows[:50])
    toks = [t for t in peak_tokens if t and t.strip()]
    pk = Counter(toks).most_common(8)
    pk_str = ", ".join(f"'{t}'×{n}" for t, n in pk) or "(n/a)"
    stop_frac = (sum(1 for t in toks if t.strip().lower() in STOPWORDS) / len(toks)) if toks else 0.0
    hint = ""
    if stop_frac >= 0.5:
        hint = (f"\nNOTE: {stop_frac:.0%} of this feature's peak tokens are high-frequency function words / "
                f"punctuation / chat markers. That usually means the feature is SYNTACTIC (fires on a "
                f"grammatical position) or POLYSEMANTIC (no single concept). Do NOT invent a concept for a "
                f"stopword. Look across ALL windows for a genuinely consistent CONTENT context; if there "
                f"isn't one, say KIND: POLYSEMANTIC and say plainly it is not a clean, interpretable feature.")
    usr = (f"This feature's PEAK token (what it fires hardest on) across the windows: {pk_str}.{hint}\n"
           f"Windows (« » = peak):\n{body}\n\n"
           f"Reply in EXACTLY this format:\n"
           f"TRIGGER: <the specific token or short pattern it fires on>\n"
           f"{kind_line}\n"
           f"MEANING: <ONE precise, non-generic sentence; if polysemantic/syntactic, SAY SO plainly>")
    r = _client.chat.completions.create(model=MODEL, temperature=0.1, max_tokens=130,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}])
    return r.choices[0].message.content.strip(), pk_str


# candidate per-band split, in display order. Only bands actually present for the feature are labeled,
# so protein dashboards yield reasoning/answer(/prompt/protein) and DNA yields dna/text automatically.
BAND_ORDER = ["reasoning", "answer", "prompt", "question", "protein", "dna_ref", "dna_variant", "dna", "text", "go"]


def _interp(model, fid, bands):
    ex = _examples(model)
    sys, kind_line = prompts_for(model)  # DNA vs protein taxonomy
    sub = ex[ex.feature_id == int(fid)].sort_values("max_activation", ascending=False)
    if not len(sub):
        return {"error": f"feature {fid} not found in {model}"}
    ws = [_win(r.sequence, r.activations) for _, r in sub.head(50).iterrows()]
    label, pk_str = _label([w for w, _ in ws], [t for _, t in ws], sys, kind_line)
    out = {"feature_id": int(fid), "label": label, "peak_tokens": pk_str,
           "windows": [w for w, _ in ws[:8]]}  # show the actual highlighted phrases
    if bands:  # label each band the feature fires on SEPARATELY (reasoning vs answer, etc.)
        present = [b for b in BAND_ORDER if (sub.band == b).any()]
        band_labels = {}
        for b in present:
            bb = sub[sub.band == b].head(12)
            bws = [_win(r.sequence, r.activations) for _, r in bb.iterrows()]
            # per-band peak activation, so the UI can flag a band the feature barely touches
            peak = float(bb["max_activation"].max()) if "max_activation" in bb else 0.0
            lab, bpk = _label([w for w, _ in bws], [t for _, t in bws], sys, kind_line)
            band_labels[b] = {"label": lab, "peak_activation": round(peak, 2),
                              "n_examples": int(len(bb)), "peak_tokens": bpk}
        out["band_labels"] = band_labels
        # protein-band features get an ENRICHMENT-grounded label (which GO term is over-represented among
        # the proteins it fires on) — reliable, no-hallucination, complements the raw-AA LLM guess.
        # (a) protein-TOKEN enrichment: proteins whose PROTEIN tokens the feature fires on (protein features)
        if "protein" in present:
            pb = sub[sub.band == "protein"].sort_values("max_activation", ascending=False)
            pids = list(dict.fromkeys(pb.protein_id.tolist()))
            try:
                enr = _protein_enrichment(pids)
                if enr:
                    out["protein_enrichment"] = enr
            except Exception as e:  # noqa: BLE001
                out["protein_enrichment_error"] = f"{type(e).__name__}: {e}"
            try:
                up = _uniprot_enrichment(pids)  # keyword/domain enrichment (richer/structural)
                if up and (up.get("keywords") or up.get("domains")):
                    out["uniprot_enrichment"] = up
            except Exception as e:  # noqa: BLE001
                out["uniprot_enrichment_error"] = f"{type(e).__name__}: {e}"
        # (b) SAMPLE-level enrichment: GO terms of the proteins of the SAMPLES the feature fires on (ANY band).
        # For a TEXT/reasoning feature this reveals if it specializes by protein function (a sample-level
        # cross-modal association). Caveat: a feature that fires on printed GO-term text enriches trivially
        # (label-reading) — read alongside KIND (reasoning vs GO-TERM-TEXT).
        try:
            spids = list(dict.fromkeys(sub.sort_values("max_activation", ascending=False).protein_id.tolist()))
            senr = _protein_enrichment(spids, min_k=3)
            if senr:
                out["sample_enrichment"] = senr
        except Exception as e:  # noqa: BLE001
            out["sample_enrichment_error"] = f"{type(e).__name__}: {e}"
        # cross-band SYNTHESIS: is the SAME concept shared across bands (a shared/cross-modal feature),
        # or does each band mean something different? This is the "embryo is a shared concept" judgment.
        if len(band_labels) >= 2:
            per = "\n".join(f"  {b} (peak {v['peak_activation']}): {v['label']}" for b, v in band_labels.items())
            syn = _client.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=80,
                messages=[{"role": "system", "content": "You are a mechanistic-interpretability analyst."},
                          {"role": "user", "content":
                           f"A single SAE feature fires across these bands of one model:\n{per}\n\n"
                           f"Is there a SINGLE concept genuinely SHARED across these bands (a shared/cross-"
                           f"modal feature), or does it mean different things per band / is it syntactic? "
                           f"Reply EXACTLY as: SHARED: <concept in 1-4 words> | NOT-SHARED: <why in 1 clause>"}])
            out["shared_concept"] = syn.choices[0].message.content.strip()
    return out


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res = _interp(req["model"], req["feature_id"], req.get("bands", False))
            code = 200
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}; code = 500
        body = json.dumps(res).encode()
        self.send_response(code); self._cors()
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5199
    print(f"autointerp server on 0.0.0.0:{port} (model={MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
