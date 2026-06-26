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

r"""Per-modality feature coverage for a BioReason-Pro SAE (Env B).

The global "dead %" hides the multimodal story: a feature can be alive on text yet never fire on a
protein token. This reports, per band, **how many features ever fire on that band's tokens** — i.e. how
much SAE capacity each modality actually gets. Motivates modality-balanced training (see
analysis/MODALITY_BALANCING.md).

    python scripts/per_band_coverage.py --sae <ckpt> --store <val300_L*> --layer L --out-json <out>
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from sae.architectures import TopKSAE
from sae.activation_store import shard_table_to_array

BANDS = ["protein", "go", "text"]


def main():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--sae", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--encode-batch", type=int, default=8192)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.sae, map_location="cpu")
    sae = TopKSAE(**ck["model_config"]).to(dev).eval()
    sae.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in ck["model_state_dict"].items()})
    H = sae.hidden_dim

    tl = pq.read_table(Path(args.store) / "token_labels.parquet")
    band = np.asarray(tl.column("position_type").to_pylist(), dtype=object)
    bcode = np.select([band == b for b in BANDS], list(range(len(BANDS))), default=-1)
    band_tokens = {b: int((band == b).sum()) for b in BANDS}
    bt = torch.from_numpy(bcode).to(dev)

    fired = torch.zeros(len(BANDS), H, dtype=torch.bool, device=dev)
    row0 = 0
    shards = sorted(glob.glob(str(Path(args.store) / f"layer{args.layer}" / "shard_*.parquet")),
                    key=lambda q: int(Path(q).stem.split("_")[1]))
    with torch.no_grad():
        for sp in shards:
            X = shard_table_to_array(pq.read_table(sp)).astype(np.float32)  # handles dim_ + FixedSizeList 'act'
            n = X.shape[0]
            for s in range(0, n, args.encode_batch):
                e = min(n, s + args.encode_batch)
                c = sae.encode(torch.from_numpy(X[s:e]).to(dev)) > 0
                bb = bt[row0 + s: row0 + e]
                for bi in range(len(BANDS)):
                    m = bb == bi
                    if m.any():
                        fired[bi] |= c[m].any(0)
            row0 += n

    f = fired.cpu().numpy()
    live = f.any(0)
    n_live = int(live.sum())
    out = {
        "sae": args.sae, "layer": args.layer, "n_features": H,
        "band_tokens": band_tokens,
        "n_live_any_band": n_live, "dead_everywhere": int(H - n_live),
        "fire_on": {b: int(f[i].sum()) for i, b in enumerate(BANDS)},
        "frac_features_firing_on": {b: round(float(f[i].sum()) / H, 4) for i, b in enumerate(BANDS)},
        "frac_dead_for_band": {b: round(1 - float(f[i].sum()) / H, 4) for i, b in enumerate(BANDS)},
        "fire_on_bio_protein_or_go": int((f[0] | f[1]).sum()),
        "bio_only_no_text": int(((f[0] | f[1]) & ~f[2]).sum()),
    }
    print(f"[L{args.layer}] features={H} live={n_live} dead-everywhere={H-n_live} ({100*(H-n_live)/H:.1f}%)")
    tok_frac = {b: band_tokens[b] / sum(band_tokens.values()) for b in BANDS}
    for i, b in enumerate(BANDS):
        print(f"   {b:8}: {f[i].sum():6d} feats fire ({100*f[i].sum()/H:5.1f}%)  |  tokens {100*tok_frac[b]:4.1f}% of corpus")
    print(f"   bio(protein|go) total: {int((f[0]|f[1]).sum())} feats  |  text: {int(f[2].sum())} feats")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"   wrote {args.out_json}")


if __name__ == "__main__":
    main()
