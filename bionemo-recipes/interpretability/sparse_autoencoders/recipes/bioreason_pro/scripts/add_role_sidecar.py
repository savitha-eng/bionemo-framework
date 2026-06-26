# Adds a `role` column (prompt | response) to a val store's token_labels via re-collate.
# response = teacher-forcing target tokens (labels != -100) = the assistant turn (reasoning + answer);
# prompt = everything the model reads as input (system + protein + go + structured text + GO-GPT preds + question).
import sys, argparse
sys.path.insert(0, "src")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import bioreason_pro_sae.data as brp_data
from bioreason_pro_sae.model_loader import _install_unsloth_stub
_install_unsloth_stub()
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument("--store", required=True); p.add_argument("--split", default="validation")
p.add_argument("--num-proteins", type=int, default=300)
p.add_argument("--ckpt", default="/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
pad = tok.pad_token_id
tr, va, te = brp_data.load_reasoning_splits(max_length_protein=2000)
ds = {"train": tr, "validation": va, "test": te}[a.split]
ds = ds.select(range(min(a.num_proteins, len(ds))))
coll = brp_data.make_collate_fn(tok, 10000, 2000)
role_of = {}  # (protein_id, token_index) -> role
for i, batch in enumerate(DataLoader(ds, batch_size=1, collate_fn=coll)):
    ids = batch["input_ids"][0]; labels = batch["labels"][0]
    keep = (ids != pad).numpy()
    lab = labels.numpy()[keep]
    pid = str(ds[i].get("protein_id", f"row{i}"))
    for j, l in enumerate(lab):  # token_index j = position among kept tokens (matches extract.py renumber)
        role_of[(pid, j)] = "response" if l != -100 else "prompt"
t = pq.read_table(f"{a.store}/token_labels.parquet")
pids = t.column("protein_id").to_pylist(); tidx = t.column("token_index").to_pylist()
roles = [role_of.get((p, int(j)), "prompt") for p, j in zip(pids, tidx)]
out = t.append_column("role", pa.array(roles, pa.string()))
pq.write_table(out, f"{a.store}/token_labels_with_role.parquet")
import collections; c = collections.Counter(roles)
print(f"wrote {a.store}/token_labels_with_role.parquet  rows={out.num_rows:,}")
print(f"  role mix: prompt={c['prompt']:,} ({100*c['prompt']/len(roles):.1f}%)  response={c['response']:,} ({100*c['response']/len(roles):.1f}%)")
# cross with band: how much of each band is prompt vs response?
band = np.array(t.column("position_type").to_pylist(), dtype=object); rl = np.array(roles, dtype=object)
for b in ("protein","go","text"):
    m = band==b; pr=(rl[m]=="prompt").sum(); rs=(rl[m]=="response").sum()
    print(f"  {b:8}: prompt {pr:>8,}  response {rs:>8,}")
