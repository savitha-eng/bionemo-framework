import json,glob,sys,pyarrow as pa,pyarrow.parquet as pq
pub=sys.argv[1]; lab={}
for f in glob.glob(f"{sys.argv[2]}/autointerp_*.json"):
    if "VALIDATE" in f: continue
    lab.update({int(k):v for k,v in json.load(open(f)).items()})
m=pq.read_table(f"{pub}/feature_metadata.parquet")
cols=[c for c in m.column_names if c!="autointerp_label"]; m=m.select(cols)
m=m.append_column("autointerp_label",pa.array([lab.get(fid,"") for fid in m.column("feature_id").to_pylist()],pa.string()))
pq.write_table(m,f"{pub}/feature_metadata.parquet"); print(f"merged {len(lab)} labels into feature_metadata")
