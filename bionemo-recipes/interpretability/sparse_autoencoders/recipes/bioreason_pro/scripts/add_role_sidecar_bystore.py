import sys, argparse; sys.path.insert(0,"src")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import bioreason_pro_sae.data as brp_data
from bioreason_pro_sae.model_loader import _install_unsloth_stub; _install_unsloth_stub()
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
p=argparse.ArgumentParser(); p.add_argument("--store",required=True); p.add_argument("--split",default="train")
p.add_argument("--ckpt",default="/data/savithas/scratch/hf-cache/hub/models--wanglab--bioreason-pro-sft/snapshots/b77bd65fdc3c3e95224dc977cc4d4480fdbf8f2b")
a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.ckpt,trust_remote_code=True); pad=tok.pad_token_id
store_pids=[str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
tr,va,te=brp_data.load_reasoning_splits(max_length_protein=2000); ds={"train":tr,"validation":va,"test":te}[a.split]
allp=[str(x) for x in ds["protein_id"]]; idx={p:i for i,p in enumerate(allp)}
sel=[idx[p] for p in store_pids if p in idx]; print(f"matched {len(sel)}/{len(store_pids)} store proteins in {a.split}")
sub=ds.select(sel); coll=brp_data.make_collate_fn(tok,10000,2000); role_of={}
for k,batch in enumerate(DataLoader(sub,batch_size=1,collate_fn=coll)):
    ids=batch["input_ids"][0]; lab=batch["labels"][0]; keep=(ids!=pad).numpy(); lk=lab.numpy()[keep]
    pid=str(sub[k].get("protein_id")); 
    for j,l in enumerate(lk): role_of[(pid,j)]="response" if l!=-100 else "prompt"
    if k%1000==0: print(f"  {k}/{len(sub)}",flush=True)
t=pq.read_table(f"{a.store}/token_labels.parquet"); pids=t.column("protein_id").to_pylist(); tx=t.column("token_index").to_pylist()
roles=[role_of.get((p,int(j)),"prompt") for p,j in zip(pids,tx)]
pq.write_table(t.append_column("role",pa.array(roles,pa.string())),f"{a.store}/token_labels_with_role.parquet")
from collections import Counter; print("role mix:",dict(Counter(roles)))
