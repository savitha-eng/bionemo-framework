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
# accession token ids: pure-digit tokens + GO/IPR/colon -> lets the reasoning probe drop GO:xxxx / IPRxxxx
# accession strings so a feature can't trivially "read the accession" (leakage control).
import re
acc_ids=set()
for tid in range(len(tok.get_vocab())):
    d=tok.decode([tid]).strip()
    if re.fullmatch(r"\d+", d) or d in ("GO","IPR",":") or d.startswith("IPR"): acc_ids.add(tid)
print(f"accession token ids: {len(acc_ids)}")
store_pids=[str(x) for x in pq.read_table(f"{a.store}/proteins.parquet").column("protein_id").to_pylist()]
tr,va,te=brp_data.load_reasoning_splits(max_length_protein=2000); ds={"train":tr,"validation":va,"test":te}[a.split]
allp=[str(x) for x in ds["protein_id"]]; idx={p:i for i,p in enumerate(allp)}
sel=[idx[p] for p in store_pids if p in idx]; print(f"matched {len(sel)}/{len(store_pids)} store proteins in {a.split}")
sub=ds.select(sel); coll=brp_data.make_collate_fn(tok,10000,2000); role_of={}; acc_of={}
for k,batch in enumerate(DataLoader(sub,batch_size=1,collate_fn=coll)):
    ids=batch["input_ids"][0]; lab=batch["labels"][0]; keep=(ids!=pad).numpy()
    lk=lab.numpy()[keep]; idk=ids.numpy()[keep]
    pid=str(sub[k].get("protein_id"))
    for j,(l,tid) in enumerate(zip(lk,idk)):
        role_of[(pid,j)]="response" if l!=-100 else "prompt"
        if int(tid) in acc_ids: acc_of[(pid,j)]=True
    if k%1000==0: print(f"  {k}/{len(sub)}",flush=True)
t=pq.read_table(f"{a.store}/token_labels.parquet"); pids=t.column("protein_id").to_pylist(); tx=t.column("token_index").to_pylist()
roles=[role_of.get((p,int(j)),"prompt") for p,j in zip(pids,tx)]
isacc=[bool(acc_of.get((p,int(j)),False)) for p,j in zip(pids,tx)]
t=t.append_column("role",pa.array(roles,pa.string())).append_column("is_accession",pa.array(isacc,pa.bool_()))
pq.write_table(t,f"{a.store}/token_labels_with_role.parquet")
from collections import Counter; print("role mix:",dict(Counter(roles)),"| accession tokens:",sum(isacc))
