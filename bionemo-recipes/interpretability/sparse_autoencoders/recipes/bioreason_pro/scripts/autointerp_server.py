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


SYS = ("You interpret a sparse-autoencoder feature of a protein-reasoning LLM. Its reasoning/answer text "
       "frequently PRINTS Gene-Ontology term names (e.g. 'cytosol', 'protein binding') and GO accessions "
       "(e.g. 'GO:0005737'). «token» marks where the feature fires HARDEST. Do NOT give a generic biological "
       "category — name the SPECIFIC token/pattern, and judge whether it is merely firing on a printed GO "
       "term/accession (label-reading) vs genuine reasoning.")


def _label(windows, peak_tokens):
    body = "\n".join("  - " + w for w in windows[:50])
    pk = Counter(t for t in peak_tokens if t and t.strip()).most_common(8)
    pk_str = ", ".join(f"'{t}'×{n}" for t, n in pk) or "(n/a)"
    usr = (f"This feature's PEAK token (what it fires hardest on) across the windows: {pk_str}.\n"
           f"Windows (« » = peak):\n{body}\n\n"
           f"Reply in EXACTLY this format:\n"
           f"TRIGGER: <the specific token or short pattern it fires on>\n"
           f"KIND: <one of GO-TERM-TEXT | ACCESSION | REASONING | STRUCTURE | PROTEIN>  "
           f"(GO-TERM-TEXT/ACCESSION = it's just reading a printed GO term/id = label-reading; "
           f"REASONING = genuine reasoning content; STRUCTURE = formatting/position; PROTEIN = residues)\n"
           f"MEANING: <ONE precise, non-generic sentence>")
    r = _client.chat.completions.create(model=MODEL, temperature=0.1, max_tokens=130,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": usr}])
    return r.choices[0].message.content.strip(), pk_str


# candidate per-band split, in display order. Only bands actually present for the feature are labeled,
# so protein dashboards yield reasoning/answer(/prompt/protein) and DNA yields dna/text automatically.
BAND_ORDER = ["reasoning", "answer", "prompt", "protein", "dna", "text", "go"]


def _interp(model, fid, bands):
    ex = _examples(model)
    sub = ex[ex.feature_id == int(fid)].sort_values("max_activation", ascending=False)
    if not len(sub):
        return {"error": f"feature {fid} not found in {model}"}
    ws = [_win(r.sequence, r.activations) for _, r in sub.head(50).iterrows()]
    label, pk_str = _label([w for w, _ in ws], [t for _, t in ws])
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
            lab, bpk = _label([w for w, _ in bws], [t for _, t in bws])
            band_labels[b] = {"label": lab, "peak_activation": round(peak, 2),
                              "n_examples": int(len(bb)), "peak_tokens": bpk}
        out["band_labels"] = band_labels
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
