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

r"""Auto-interp via an OpenAI-compatible LLM (NVIDIA NIM). For each SAE feature in a role_category,
send its top-activating contexts to the LLM and get a 1-sentence "Fires on..." label.

API-bound (the LLM runs server-side), so throughput scales with --workers (concurrent requests),
not GPUs. Resumable: re-running skips features already in --out. Incremental save every 25.

    NIM_API_KEY=nvapi-... python scripts/autointerp_run.py --pub <dashboard/public> \
        --category reasoning_only --workers 16 --out labels.json
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
from openai import OpenAI

SYS = ("You analyze features of a sparse autoencoder trained on a protein-function reasoning LLM. "
       "Given text snippets where a feature fires most strongly (the token in the brackets is the peak), "
       "state in ONE sentence starting with 'Fires on' the precise common pattern across the snippets. "
       "Be specific; do not restate these instructions.")


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser()
    p.add_argument("--pub", required=True, help="dashboard public/ dir (feature_metadata + feature_examples)")
    p.add_argument("--category", default="reasoning_only", help="role_category to label")
    p.add_argument("--model", default="meta/llama-3.3-70b-instruct")
    p.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    p.add_argument("--api-key-env", default="NIM_API_KEY")
    p.add_argument("--n-examples", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():  # noqa: D103
    a = parse_args()
    client = OpenAI(base_url=a.base_url, api_key=os.environ[a.api_key_env])
    meta = {r["feature_id"]: r for r in pq.read_table(f"{a.pub}/feature_metadata.parquet").to_pylist()}
    ex = {}
    for r in pq.read_table(f"{a.pub}/feature_examples.parquet").to_pylist():
        ex.setdefault(r["feature_id"], []).append(r)
    feats = [fid for fid, m in meta.items() if m.get("role_category") == a.category]
    feats = sorted(feats, key=lambda f: -max((r["max_activation"] for r in ex.get(f, [])), default=0))
    if a.limit:
        feats = feats[:a.limit]
    labels = json.load(open(a.out)) if os.path.exists(a.out) else {}
    todo = [f for f in feats if str(f) not in labels]
    print(f"[{a.category}] {len(feats)} feats, {len(todo)} to do ({len(labels)} cached), {a.workers} workers", flush=True)

    def ctx(fid):
        out = []
        for e in sorted(ex.get(fid, []), key=lambda r: -r["max_activation"])[:a.n_examples]:
            s = list(e["sequence"].split(" "))
            ac = np.array(e["activations"])
            t = int(ac.argmax())
            if t < len(s):
                s[t] = f"[[{s[t]}]]"
            out.append("  - " + " ".join(s[max(0, t - 12): t + 10]))
        return "\n".join(out)

    def one(fid):
        prompt = f"Feature {fid} top activations:\n{ctx(fid)}\n\nOne sentence:"
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=a.model,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
                    temperature=0.2, top_p=0.7, max_tokens=80)
                return fid, r.choices[0].message.content.strip()
            except Exception as e:  # noqa: BLE001  (rate-limit / transient -> retry)
                if attempt == 4:
                    return fid, f"(error:{type(e).__name__})"
                time.sleep(2 * (attempt + 1))

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as exr:
        futs = [exr.submit(one, f) for f in todo]
        for fut in as_completed(futs):
            fid, lab = fut.result()
            labels[str(fid)] = lab
            done += 1
            if done % 25 == 0:
                json.dump(labels, open(a.out, "w"), indent=2)
                print(f"  {done}/{len(todo)}", flush=True)
    json.dump(labels, open(a.out, "w"), indent=2)
    print(f"wrote {len(labels)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
