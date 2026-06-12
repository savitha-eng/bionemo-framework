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

"""Build a self-contained reveal.js HTML deck explaining the variance-explained bug & fix.

Figures are inlined as base64 so the single .html file is fully portable (reveal.js itself loads
from CDN). Run: python analysis/build_slides.py  ->  analysis/slides_variance_explained_fix.html
"""

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"


def img(name):
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


SLIDES = [
    # (title_html, body_html)
    ("""<h1>What "97% variance-explained" really meant</h1>
        <h3>Debugging the BioReason-Pro SAE</h3>
        <p class="dim">First multimodal decoder-LLM SAE in bionemo — TopK SAE on the Qwen3 residual
        stream (ESM3 protein + GO memory + reasoning text).</p>
        <p class="dim">Initial runs reported <b>var-explained ≈ 0.97</b> — which was hiding <b>two</b> bugs.</p>""",
     ""),

    ("""<h2>The symptom 🚩</h2>""",
     """<ul>
        <li><b>0.97 variance-explained</b> with only ~12% dead latents.</li>
        <li>That's a <b>red flag</b>, not a win — a near-identity <i>copy machine</i>, not an
            interpretable feature model. (Anthropic production SAEs sit ~0.5–0.7.)</li>
        <li>Two independent problems: a <b>metric</b> bug and a <b>loss</b> bug.</li>
        </ul>"""),

    ("""<h2>Problem 1 — the metric measured the wrong space</h2>""",
     f"""<div class="fig">{img('fig_metric_space.png')}</div>
        <p><code>normalize_input</code> standardizes each token, then <b>de-normalizes</b> the recon —
        reinserting each token's magnitude <b>for free</b>. Norms vary hugely (protein <b>2,339</b> vs
        text <b>321</b>), so raw var-explained is dominated by free magnitude.
        <b>Honest (normalized) = 0.76, not 0.97.</b></p>"""),

    ("""<h2>Why magnitude dominates — PCA of the residual stream</h2>""",
     f"""<div class="fig">{img('fig_pca_layers.png')}</div>
        <p><b>Raw:</b> a single PC explains <b>~80%</b> of variance at every layer (k90 = 2–3) — that PC
        <b>is the magnitude direction</b>, so the raw stream is effectively rank-1. Capture it → 80%
        "explained" for free, and the raw loss is dominated by it. <b>Normalized</b> (what the SAE models):
        magnitude removed, PC1 ~27–30%, &gt;300 PCs for 90% → the real target is high-dimensional.</p>"""),

    ("""<h2>Where the magnitude comes from — the BioReason-Pro code</h2>""",
     f"""<div class="fig">{img('fig_modality_norms.png')}</div>
        <p><b>Text</b> tokens use the pretrained Qwen <b>embedding table</b> (norm ≈ 1). <b>Protein/GO</b>
        tokens are spliced in via trained <code>Linear→GELU→Linear</code> projections
        (<code>protein_llm.py</code>) with <b>no output normalization</b> → unconstrained magnitude
        (protein ≈ <b>1,356×</b> text at input). RMSNorm fixes it per-block for the model, but the
        <b>raw residual stream keeps the gap</b> (~9× at L32) — what dominates raw-space SAE loss/metrics.</p>"""),

    ("""<h2>Ruled out — it was <i>not</i> sink tokens</h2>""",
     f"""<div class="fig">{img('fig_sinks.png')}</div>
        <p>The obvious culprit — a few architectural "sink" tokens — was tested and <b>cleared</b>:
        the same channels <b>[0, 4, 19, 21]</b> appear at both layers &amp; across seeds (→ wired/architectural),
        but removing every sink token moves var-explained by <b>~0.0001</b>. Good controls matter.</p>"""),

    ("""<h2>Problem 2 — the <i>loss</i> was raw-space too</h2>""",
     """<ul>
        <li>Fixing the metric didn't fix training: the SAE was <b>still optimizing a raw-space FVU</b>.</li>
        <li>A protein token (norm ~2,339) drives the MSE gradient <b>~50× more</b> than a text token (~321).</li>
        <li>So the SAE spent capacity reconstructing <b>magnitude that de-norm already supplies for free</b>
            → wasted capacity → dead latents.</li>
        <li><code>aggregate_loss</code> (prior fix) addressed per-token <i>ratio</i> starvation — but not the
            <b>space</b>.</li>
        </ul>"""),

    ("""<h2>The fix — <code>normalize_loss</code></h2>""",
     f"""<div class="fig">{img('fig_loss_fix.png')}</div>
        <p>Compute the FVU in <b>normalized space</b> → every token weighted equally; the objective now
        matches the honest metric. Opt-in (unimodal recipes unaffected).
        <b>Strictly better: dead latents halved, reconstruction + fidelity up.</b></p>"""),

    ("""<h2>The payoff — the layer choice flips</h2>""",
     f"""<div class="fig">{img('fig_layer_reversal.png')}</div>
        <p>L28's "collapse" (54.9% dead) was a <b>raw-loss artifact</b> — under <code>normalize_loss</code>
        it drops to <b>5.1%</b>. All layers healthy ⇒ choice is <b>use-case-driven</b>:
        <b>steering → L28</b> (earliest, most faithful); <b>richest atlas → L32</b> (highest rank).</p>"""),

    ("""<h2>Takeaways</h2>""",
     """<ol>
        <li><b>A suspiciously-high metric is a bug signal.</b> 0.97 var-explained = copy machine.</li>
        <li><b>Two bugs, one root:</b> raw-space <i>measurement</i> (inflated) <b>and</b> raw-space
            <i>objective</i> (over-weights high-magnitude tokens). Fix both → 0.76→0.81 var-exp, dead 12.8→5.3%.</li>
        <li><b>Controls earn trust:</b> the "obvious" sink culprit was ruled out by a one-line experiment.</li>
        <li><b>A wrong loss faked a layer conclusion:</b> it made L28 look broken; the right loss rescued it.</li>
        </ol>
        <p class="dim">Matters most where token norms vary (protein vs text); unimodal models
        (Evo2/ESM2) have ~uniform norms, so these are ~no-ops there.</p>"""),
]


def main():
    sections = "\n".join(
        f'<section><div class="wrap">{t}{b}</div></section>' for t, b in SLIDES
    )
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BioReason-Pro SAE — variance-explained bug & fix</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/theme/white.css">
<style>
  :root {{ --nv:#76B900; }}
  .reveal {{ font-family: 'NVIDIA Sans', Arial, sans-serif; }}
  .reveal h1 {{ font-size: 1.7em; color:#1a1a1a; }}
  .reveal h2 {{ color:#1a1a1a; border-bottom:3px solid var(--nv); padding-bottom:8px; font-size:1.3em; }}
  .reveal h3 {{ color: var(--nv); font-size:1.0em; }}
  .reveal section {{ text-align:left; }}
  .reveal .wrap {{ max-width:1100px; margin:0 auto; }}
  .reveal ul, .reveal ol {{ font-size:0.78em; line-height:1.5; }}
  .reveal p {{ font-size:0.74em; line-height:1.45; }}
  .reveal .dim {{ color:#666; }}
  .reveal code {{ color:#0046a4; background:#f3f3f3; padding:1px 5px; border-radius:4px; }}
  .reveal .fig {{ text-align:center; margin:6px 0; }}
  .reveal .fig img {{ max-height:430px; border:1px solid #e0e0e0; border-radius:6px; box-shadow:0 2px 10px rgba(0,0,0,.08); }}
  .reveal b {{ color:#111; }}
</style></head>
<body><div class="reveal"><div class="slides">
{sections}
</div></div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist/reveal.js"></script>
<script>Reveal.initialize({{hash:true, slideNumber:'c/t', width:1280, height:760, margin:0.04}});</script>
</body></html>
"""
    out = HERE / "slides_variance_explained_fix.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
