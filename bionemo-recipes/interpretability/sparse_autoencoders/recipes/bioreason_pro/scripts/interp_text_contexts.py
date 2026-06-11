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

r"""Part-B autointerp: decode the reasoning-TEXT windows where SAE features fire (Env A — tokenizer).

The biology autointerp (autointerp_biology.py) labels features by the GO functions of the proteins
they fire on. This complements it with the *linguistic* view: for a given feature, show the actual
reasoning-text snippets around its top-activating **text-band** tokens, so we can read what concept
in the model's reasoning triggers it.

token_index in the store = position in the protein's unpadded token sequence (extract.py). We reproduce
each protein's tokens by re-running the SAME collate (no model forward) and matching by protein_id, then
decode a window around each firing token. Protein/GO placeholder tokens render as <protein>/<go>.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

DEFAULT_CKPT = (
    "/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/"
    "snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b"
)
DEFAULT_ROOT = "/data/savithas/bioreason-pro"


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--features", type=int, nargs="+", required=True)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--bioreason-root", default=DEFAULT_ROOT)
    p.add_argument("--split", default="validation")
    p.add_argument("--num-proteins", type=int, default=300)
    p.add_argument("--top-examples", type=int, default=6)
    p.add_argument("--window", type=int, default=12, help="tokens of context on each side")
    p.add_argument("--max-length-text", type=int, default=10000)
    p.add_argument("--max-length-protein", type=int, default=2000)
    p.add_argument("--encode-batch", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import bioreason_pro_sae.data as brp_data
    from bioreason_pro_sae.model_loader import load_bioreason_pro_sft

    from sae.architectures import TopKSAE

    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---- model (only for tokenizer + collate + special ids; NO forward) ----
    model = load_bioreason_pro_sft(
        ckpt_dir=args.ckpt_dir, bioreason_pro_root=args.bioreason_root, device=args.device,
        max_length_text=args.max_length_text, max_length_protein=args.max_length_protein,
    )
    tok = model.text_tokenizer
    pid_tok, gid_tok, pad_tok = model.protein_token_id, model.go_token_id, tok.pad_token_id

    train_ds, val_ds, test_ds = brp_data.load_reasoning_splits(max_length_protein=args.max_length_protein)
    ds = {"train": train_ds, "validation": val_ds, "test": test_ds}[args.split]
    ds = ds.select(range(min(args.num_proteins, len(ds))))
    collate_fn = brp_data.make_collate_fn(tok, args.max_length_text, args.max_length_protein)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # reproduce per-protein unpadded token ids (token_index order matches the store)
    pid_to_ids = {}
    for b_i, batch in enumerate(loader):
        row = batch["input_ids"][0]
        keep = row != pad_tok
        pid = str(ds[b_i].get("protein_id", f"row{b_i}"))
        pid_to_ids[pid] = row[keep].tolist()
    print(f"[textinterp] reproduced token ids for {len(pid_to_ids)} proteins")

    # ---- store row meta + activations for the target features ----
    layer_dir = Path(args.store) / f"layer{args.layer}"
    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    row_pid = tl.column("protein_id").to_pylist()
    row_tidx = np.asarray(tl.column("token_index").to_pylist(), dtype=np.int64)
    row_band = np.asarray(tl.column("position_type").to_pylist(), dtype=object)

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict(ck["model_state_dict"])
    feats = torch.tensor(args.features, device=dev)

    shards = sorted(glob.glob(str(layer_dir / "shard_*.parquet")), key=lambda q: int(Path(q).stem.split("_")[1]))
    acts_per_feat = {f: [] for f in args.features}  # (row, activation)
    row0 = 0
    with torch.no_grad():
        for sp in shards:
            t = pq.read_table(sp)
            cols = sorted([c for c in t.column_names if c.startswith("dim_")], key=lambda c: int(c.split("_")[1]))
            X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)
            n = X.shape[0]
            for s in range(0, n, args.encode_batch):
                e = min(n, s + args.encode_batch)
                codes = sae.encode(torch.from_numpy(X[s:e]).to(dev))[:, feats]  # (nb, nfeat)
                codes = codes.cpu().numpy()
                for jf, f in enumerate(args.features):
                    rows = np.arange(row0 + s, row0 + e)
                    for r, a in zip(rows, codes[:, jf]):
                        if a > 0:
                            acts_per_feat[f].append((int(r), float(a)))
            row0 += n

    def render(pid, tidx, mark):
        ids = pid_to_ids.get(pid)
        if ids is None:
            return None
        lo, hi = max(0, tidx - args.window), min(len(ids), tidx + args.window + 1)
        out = []
        for j in range(lo, hi):
            tid = ids[j]
            if tid == pid_tok:
                piece = "<protein>"
            elif tid == gid_tok:
                piece = "<go>"
            else:
                piece = tok.decode([tid])
            if j == tidx and mark:
                piece = f"⟦{piece.strip()}⟧"
            out.append(piece)
        return "".join(out).replace("\n", " ")

    report = {"sae": args.sae, "layer": args.layer, "features": {}}
    for f in args.features:
        items = acts_per_feat[f]
        # restrict to TEXT-band tokens, take top activations
        text_items = [(r, a) for (r, a) in items if row_band[r] == "text"]
        text_items.sort(key=lambda x: -x[1])
        examples = []
        seen_pid = set()
        for r, a in text_items:
            pid = row_pid[r]
            if pid in seen_pid:  # one example per protein for diversity
                continue
            seen_pid.add(pid)
            snip = render(pid, int(row_tidx[r]), mark=True)
            if snip:
                examples.append({"protein": pid, "act": round(a, 2), "context": snip})
            if len(examples) >= args.top_examples:
                break
        frac_text = round(len(text_items) / max(1, len(items)), 3)
        report["features"][f] = {"n_fire": len(items), "frac_in_text_band": frac_text, "examples": examples}
        print(f"\n========== FEATURE {f}  (fires {len(items)}x, {100*frac_text:.0f}% in text band) ==========")
        for ex in examples:
            print(f"  [{ex['act']:.1f}] {ex['protein']}: …{ex['context']}…")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\n[textinterp] wrote {args.out_json}")


if __name__ == "__main__":
    main()
