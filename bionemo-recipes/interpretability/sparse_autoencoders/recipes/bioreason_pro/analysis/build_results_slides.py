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

"""Build the results+analysis reveal.js deck (feature analysis on the BioReason-Pro SAE).

Reads top_features markdown + an optional decoded-contexts JSON to populate example slides.
Run: python analysis/build_results_slides.py -> analysis/slides_feature_analysis.html
"""

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
CTX = Path("/scratch/savithas/phase3_subset/textinterp_l32_normloss.json")


def img(name):
    return f'<img src="data:image/png;base64,{base64.b64encode((FIG / name).read_bytes()).decode()}" />'


def context_rows():
    """Build example rows from decoded text contexts, if available."""
    if not CTX.exists():
        return "<p class='dim'>(decoded text snippets pending — see feature_contexts file)</p>"
    d = json.load(open(CTX))["features"]
    rows = []
    for fid, info in list(d.items())[:6]:
        exs = info.get("examples", [])
        if not exs:
            continue
        snip = exs[0]["context"].replace("<", "&lt;").replace(">", "&gt;")
        rows.append(f"<tr><td><b>F{fid}</b></td><td style='font-family:monospace;font-size:0.7em'>…{snip}…</td></tr>")
    return "<table class='ctx'>" + "".join(rows) + "</table>" if rows else "<p class='dim'>(no text-band examples)</p>"


SLIDES = [
    ("""<h1>BioReason-Pro SAE — feature analysis</h1>
        <h3>What the sparse dictionary learned</h3>
        <p class="dim">TopK SAE (20,480 features, top_k 128, normalize_input + normalize_loss) on the Qwen3
        residual stream. Analysis on a held-out 300-protein eval set (1.07M tokens).</p>""", ""),

    ("""<h2>The SAE is faithful & healthy (all layers, normalize_loss)</h2>""",
     f"""<div class="fig">{img('fig_layer_reversal.png')}</div>
        <p>Honest var-explained 0.81–0.86 · dead latents 4–5% · loss-recovered 0.98 (model keeps 98% of
        its fidelity when the SAE recon is substituted in) · GO-AUC ~0.84.
        <b>Steering → L28</b> (earliest, most faithful) · <b>richest atlas → L32</b>.</p>"""),

    ("""<h2>How the interpretation works</h2>""",
     """<p>The SAE gives each token a <b>sparse code</b> (128 of 20,480 features active). To interpret a
        feature, ask <b>what the tokens/proteins where it fires have in common</b> — four complementary views:</p>
        <ul>
        <li><b>Feature → biology:</b> max-pool a feature over each protein's tokens → score; find the GO
            term it best predicts by <b>per-protein ROC-AUC</b>, <b>de-biased</b> (pick term on a train
            split, score on a held-out split → kills winner's-curse). Label if held-out AUC &gt; 0.65.</li>
        <li><b>Band composition:</b> per-band fire <i>rate</i> (corrects for text being ~80% of tokens) →
            protein/go/text-heavy.</li>
        <li><b>Cross-modal (SAE-V):</b> paired cosine of a feature's top tokens across modality bands, minus
            a random baseline → genuine fusion vs coincidence.</li>
        <li><b>Text context:</b> decode the reasoning-text window around top-activating tokens → the trigger.</li>
        </ul>
        <p class="dim">Scripts: build_dashboard.py · autointerp_biology.py · crossmodal.py · interp_text_contexts.py</p>"""),

    ("""<h2>What the dictionary contains</h2>""",
     f"""<div class="fig">{img('fig_feature_analysis.png')}</div>
        <p>Of 19,389 live features: <b>4,703 carry a confident GO concept</b>, <b>95 are truly cross-modal</b>.
        Most features are about the reasoning <b>text</b> (the model integrates biology into its reasoning);
        a smaller set are protein/GO-specific.</p>"""),

    ("""<h2>Features map to concrete biology (held-out de-biased)</h2>""",
     """<table>
        <tr><th>concept (GO)</th><th>best feature held-out AUC</th></tr>
        <tr><td>catalytic activity</td><td>0.93</td></tr>
        <tr><td>nucleic acid binding</td><td>0.91</td></tr>
        <tr><td>cytosol</td><td>0.98</td></tr>
        <tr><td>plasma membrane</td><td>0.98</td></tr>
        <tr><td>protein-containing complex</td><td>0.94</td></tr>
        <tr><td>positive regulation of biological process</td><td>0.98</td></tr>
        </table>
        <p>Features are <b>sparse &amp; monosemantic</b> (fire on &lt;1% of tokens) and <b>feature-split</b>
        (e.g. ~8 distinct "cytosol" features) — healthy-SAE signatures. Full list:
        <code>analysis/feature_tables/features_L32_normloss.csv</code> (all 19,389).</p>"""),

    ("""<h2>Genuine multimodal fusion (SAE-V)</h2>""",
     """<p>Cross-modal cosine (omega), <b>relative to a random cross-band baseline</b>:</p>
        <table>
        <tr><th>band pair</th><th>lift over baseline</th><th>reading</th></tr>
        <tr><td><b>protein ↔ text</b></td><td><b>+0.59</b></td><td><b>strong, genuine fusion</b></td></tr>
        <tr><td>go ↔ text</td><td>+0.33</td><td>moderate fusion</td></tr>
        <tr><td>protein ↔ go</td><td>+0.02</td><td>not special (shared injected structure)</td></tr>
        </table>
        <p>So features like "catalytic activity" represent the concept in <b>both</b> the ESM3 protein
        embedding <b>and</b> the reasoning text, pointing the same direction — the model's shared
        cross-modal representation. (The baseline is essential — it flips which pair looks fused.)</p>"""),

    ("""<h2>What triggers a feature — real reasoning-text snippets</h2>""",
     f"""{context_rows()}
        <p class="dim">⟦ ⟧ marks the firing token. Top interpretable features at L32; full contexts in
        <code>feature_tables/feature_contexts_L32_normloss.md</code>.</p>"""),

    ("""<h2>Inspect it yourself</h2>""",
     """<ul>
        <li><b>Every feature → file:</b> <code>analysis/feature_tables/features_L32_normloss.csv</code>
            (19,389 features: concept, AUC, band, fusion, fire-rate, fractions). Sorted by GO-AUC.</li>
        <li><b>Readable top-100:</b> <code>top_features_L32_normloss.md</code>.</li>
        <li><b>Interactive dashboard:</b> <code>multimodal_dashboard/</code> — UMAP atlas + a Multimodal
            view (composition scatter, cross-modal ringed); <code>ssh -L 5174</code> then localhost:5174.</li>
        </ul>"""),

    ("""<h2>Takeaways</h2>""",
     """<ol>
        <li>The SAE is <b>faithful</b> (loss-recovered 0.98) and its sparse features are
            <b>human-readable biology</b> — ~4,700 GO-labeled, AUC up to 0.98 on held-out proteins.</li>
        <li>It captures <b>localization, molecular function, and process</b> concepts; mostly via the
            <b>text/reasoning</b> band, with a real <b>protein↔text fusion</b> subset.</li>
        <li>Validated on a subset — ready to scale, with the full feature table + dashboard for inspection.</li>
        </ol>"""),
]


def main():
    sections = "\n".join(f'<section><div class="wrap">{t}{b}</div></section>' for t, b in SLIDES)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BioReason-Pro SAE — feature analysis</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/white.css">
<style>
  :root {{ --nv:#76B900; }}
  .reveal {{ font-family:'NVIDIA Sans',Arial,sans-serif; }}
  .reveal h1 {{ font-size:1.7em; color:#1a1a1a; }}
  .reveal h2 {{ color:#1a1a1a; border-bottom:3px solid var(--nv); padding-bottom:8px; font-size:1.25em; }}
  .reveal h3 {{ color:var(--nv); }}
  .reveal section {{ text-align:left; }} .reveal .wrap {{ max-width:1100px; margin:0 auto; }}
  .reveal ul,.reveal ol {{ font-size:0.72em; line-height:1.5; }} .reveal p {{ font-size:0.72em; line-height:1.45; }}
  .reveal .dim {{ color:#666; }} .reveal code {{ color:#0046a4; background:#f3f3f3; padding:1px 5px; border-radius:4px; font-size:0.85em; }}
  .reveal .fig {{ text-align:center; }} .reveal .fig img {{ max-height:420px; border:1px solid #e0e0e0; border-radius:6px; box-shadow:0 2px 10px rgba(0,0,0,.08); }}
  .reveal table {{ font-size:0.72em; border-collapse:collapse; margin:8px 0; }}
  .reveal th,.reveal td {{ border:1px solid #ddd; padding:5px 12px; text-align:left; }}
  .reveal th {{ background:var(--nv); color:#000; }}
  .reveal table.ctx td {{ font-size:0.95em; }}
</style></head><body><div class="reveal"><div class="slides">
{sections}
</div></div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script>Reveal.initialize({{hash:true, slideNumber:'c/t', width:1280, height:760, margin:0.04}});</script>
</body></html>"""
    out = HERE / "slides_feature_analysis.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
