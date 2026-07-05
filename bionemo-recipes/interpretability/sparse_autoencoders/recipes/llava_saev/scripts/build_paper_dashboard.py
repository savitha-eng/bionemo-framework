#!/usr/bin/env python
"""Build a static HTML dashboard for the LLaVA SAE-V paper-metric run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--paper-json", required=True)
    p.add_argument("--control-json", default=None)
    p.add_argument("--store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top-features", type=int, default=100)
    p.add_argument("--top-samples", type=int, default=100)
    return p.parse_args()


def fmt(x, digits=3):
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.{digits}f}"
    return html.escape(str(x))


def row_labels(labels_path: Path, rows: set[int]) -> dict[int, dict]:
    if not rows:
        return {}
    wanted = sorted(rows)
    out = {}
    ptr = 0
    offset = 0
    pf = pq.ParquetFile(labels_path)
    for batch in pf.iter_batches(batch_size=200_000, columns=["protein_id", "token_index", "position_type"]):
        n = batch.num_rows
        start = ptr
        while ptr < len(wanted) and wanted[ptr] < offset + n:
            ptr += 1
        if ptr > start:
            ids = batch.column(0).to_pylist()
            token_idx = batch.column(1).to_pylist()
            bands = batch.column(2).to_pylist()
            for row in wanted[start:ptr]:
                local = row - offset
                out[row] = {
                    "sample_id": ids[local],
                    "token_index": int(token_idx[local]),
                    "position_type": bands[local],
                }
        if ptr >= len(wanted):
            break
        offset += n
    return out


def metric_cards(paper: dict, control: dict | None) -> str:
    cards = [
        ("Paper weighted features", paper["n_weighted_features"]),
        ("Weight > 0.3", paper["n_weight_gt_0_3"]),
        ("Weight > 0.5", paper["n_weight_gt_0_5"]),
        ("Mean sample score", paper["mean_cosine_similarity_score"]),
        ("Mean L0", paper["mean_l0"]),
        ("Mean co-occurrence", paper["mean_cooccurrence"]),
        ("Rows scored", paper["n_rows"]),
        ("Image / text rows", f"{paper['n_image_rows']:,} / {paper['n_text_rows']:,}"),
    ]
    if control:
        cards.extend(
            [
                ("Control cofire features", control.get("n_cofire", "n/a")),
                ("Control aligned > 0.3", control.get("n_aligned_0.3", "n/a")),
                ("Control aligned > 0.5", control.get("n_aligned_0.5", "n/a")),
            ]
        )
    return "\n".join(
        f'<section class="metric"><span>{html.escape(label)}</span><strong>{fmt(value)}</strong></section>'
        for label, value in cards
    )


def feature_table(paper: dict, labels: dict[int, dict], n: int) -> str:
    rows = []
    for feature in paper["features"][:n]:
        image_bits = []
        text_bits = []
        for row, act in zip(feature["top_image_rows"], feature["top_image_activations"], strict=True):
            lab = labels.get(row, {})
            image_bits.append(
                f"<li>row {row:,}, {html.escape(str(lab.get('sample_id', 'unknown')))}"
                f":tok {lab.get('token_index', '?')} act {act:.2f}</li>"
            )
        for row, act in zip(feature["top_text_rows"], feature["top_text_activations"], strict=True):
            lab = labels.get(row, {})
            text_bits.append(
                f"<li>row {row:,}, {html.escape(str(lab.get('sample_id', 'unknown')))}"
                f":tok {lab.get('token_index', '?')} act {act:.2f}</li>"
            )
        rows.append(
            "<tr>"
            f"<td class='id'>F{feature['feature_id']}</td>"
            f"<td>{feature['weight']:.3f}</td>"
            f"<td>{feature['active_samples']:,}</td>"
            f"<td>{feature['cooccur_samples']:,}</td>"
            f"<td><ol>{''.join(image_bits)}</ol></td>"
            f"<td><ol>{''.join(text_bits)}</ol></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Feature</th><th>Weight</th><th>Active Samples</th>"
        "<th>Cooccur Samples</th><th>Top Image Tokens</th><th>Top Text Tokens</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def sample_table(paper: dict, n: int) -> str:
    rows = []
    for sample in paper["samples"][:n]:
        rows.append(
            "<tr>"
            f"<td class='id'>{html.escape(sample['sample_id'])}</td>"
            f"<td>{sample['cosine_similarity_score']:.3f}</td>"
            f"<td>{sample['l0']:,}</td>"
            f"<td>{sample['image_l0']:,}</td>"
            f"<td>{sample['text_l0']:,}</td>"
            f"<td>{sample['cooccurrence']:,}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Sample</th><th>Cosine Score</th><th>L0</th>"
        "<th>Image L0</th><th>Text L0</th><th>Cooccurrence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def bars(paper: dict, n: int = 40) -> str:
    feats = paper["features"][:n]
    max_w = max((f["weight"] for f in feats), default=1.0)
    parts = []
    for f in feats:
        pct = 100 * f["weight"] / max_w if max_w else 0
        parts.append(
            "<div class='bar-row'>"
            f"<span>F{f['feature_id']}</span>"
            f"<div class='bar'><i style='width:{pct:.2f}%'></i></div>"
            f"<b>{f['weight']:.3f}</b>"
            "</div>"
        )
    return "".join(parts)


def main() -> None:
    args = parse_args()
    paper = json.loads(Path(args.paper_json).read_text())
    control = json.loads(Path(args.control_json).read_text()) if args.control_json else None

    rows = set()
    for feature in paper["features"][: args.top_features]:
        rows.update(int(r) for r in feature["top_image_rows"])
        rows.update(int(r) for r in feature["top_text_rows"])
    labels = row_labels(Path(args.store) / "token_labels.parquet", rows)

    caveat = (
        "This dashboard uses SAE-V paper Algorithm 1/Table 5 metric parameters "
        "(top-K=5, activation_bound=1, sample_data_size=1000) on our trained "
        "BioNeMo TopKSAE checkpoint. It is not an exact SAELens-V architecture "
        "reproduction of the paper checkpoint; the TopK model fixes per-token "
        "nonzeros at k=128, so its L0 is not directly comparable to the paper's "
        "reported LLaVA-NeXT-Mistral SAE-V L0=192.5. Source image URLs/text were "
        "not saved in the extractor sidecar; token rows are located by sample_id "
        "and token_index here."
    )

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLaVA SAE-V Paper Metric Dashboard</title>
  <style>
    :root {{ color-scheme: light; --ink:#15171a; --muted:#5d6673; --line:#d8dde6; --panel:#f7f8fa; --accent:#1b6f6a; }}
    body {{ margin:0; font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif; color:var(--ink); background:white; }}
    header {{ padding:28px 32px 20px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:28px 0 12px; font-size:18px; }}
    main {{ padding:0 32px 40px; }}
    .meta {{ color:var(--muted); max-width:980px; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin:18px 0 8px; }}
    .metric {{ border:1px solid var(--line); border-radius:6px; padding:12px; background:var(--panel); }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric strong {{ display:block; margin-top:4px; font-size:20px; }}
    table {{ border-collapse:collapse; width:100%; table-layout:fixed; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; vertical-align:top; text-align:left; }}
    th {{ font-size:12px; color:var(--muted); background:#fafbfc; position:sticky; top:0; }}
    td.id {{ font-weight:650; white-space:nowrap; }}
    ol {{ margin:0; padding-left:18px; }}
    li {{ margin:0 0 4px; overflow-wrap:anywhere; }}
    .bar-row {{ display:grid; grid-template-columns:90px 1fr 60px; align-items:center; gap:10px; margin:6px 0; }}
    .bar {{ height:10px; background:#e8ecef; border-radius:5px; overflow:hidden; }}
    .bar i {{ display:block; height:100%; background:var(--accent); }}
    .note {{ padding:12px 14px; background:#fff8e6; border:1px solid #ead9a6; border-radius:6px; max-width:1120px; }}
    @media (max-width: 820px) {{ main,header {{ padding-left:16px; padding-right:16px; }} table {{ font-size:12px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>LLaVA SAE-V Paper Metric Dashboard</h1>
    <div class="meta">
      Model: LLaVA-NeXT/Mistral-7B, layer {paper['layer']}, {paper['n_latents']:,} latents.
      Metric JSON: {html.escape(Path(args.paper_json).name)}.
    </div>
  </header>
  <main>
    <section class="cards">{metric_cards(paper, control)}</section>
    <p class="note">{html.escape(caveat)}</p>
    <h2>Top Feature Weights</h2>
    {bars(paper)}
    <h2>Top Features</h2>
    {feature_table(paper, labels, args.top_features)}
    <h2>Top Ranked Samples</h2>
    {sample_table(paper, args.top_samples)}
  </main>
</body>
</html>
"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(body)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
