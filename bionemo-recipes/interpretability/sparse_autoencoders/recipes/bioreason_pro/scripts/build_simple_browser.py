# A dead-simple, self-contained HTML feature browser: one searchable/sortable table grouped by
# role_category, showing the plain-English auto-interp label + GO term + band mix + a top example.
import json, html, numpy as np, pyarrow.parquet as pq, sys
pub=sys.argv[1]; out=sys.argv[2]
meta={r["feature_id"]:r for r in pq.read_table(f"{pub}/feature_metadata.parquet").to_pylist()}
ex={}
for r in pq.read_table(f"{pub}/feature_examples.parquet").to_pylist(): ex.setdefault(r["feature_id"],[]).append(r)
def f(m,k,d=0):
    try: return float(m.get(k) if m.get(k) is not None else d)
    except: return d
def top_ctx(fid):
    e=sorted(ex.get(fid,[]),key=lambda r:-r["max_activation"])
    if not e: return ""
    s=e[0]["sequence"].split(" "); a=np.array(e[0]["activations"]); t=int(a.argmax())
    seg=s[max(0,t-9):t+8]
    if t-max(0,t-9)<len(seg): seg[t-max(0,t-9)]="«"+seg[t-max(0,t-9)]+"»"
    return html.escape(" ".join(seg))[:160]
CAT_DESC={"bio":"fires on protein/GO embeddings (derived biology)",
 "reasoning_only":"fires in the model's OWN reasoning (not the prompt) — genuine",
 "prompt_annotation":"fires on handed-in GO/InterPro text in the prompt — LABEL-READER (leakage)",
 "general_text":"general language (fires in both prompt & reasoning)","dead":"never fires"}
rows=[]
for fid,m in sorted(meta.items()):
    cat=m.get("role_category","?")
    if cat=="dead": continue
    rows.append({"id":fid,"cat":cat,"label":m.get("autointerp_label") or "",
      "go":(m.get("go_label") if m.get("go_label") not in (None,"none") else ""),"auc":round(f(m,"go_auc"),2) or "",
      "bioauc":round(f(m,"go_auc_protein"),2) or "","bioterm":(m.get("go_term_protein") or ""),
      "band":m.get("band_class",""),"p":round(100*f(m,"protein_frac")),"g":round(100*f(m,"go_frac")),"t":round(100*f(m,"text_frac")),
      "act":round(f(m,"max_activation"),1),"ctx":top_ctx(fid)})
data=json.dumps(rows)
COLORS={"bio":"#2a9d8f","reasoning_only":"#e76f51","prompt_annotation":"#e9c46a","general_text":"#8d99ae"}
legend="".join(f'<span style="background:{COLORS[k]};padding:2px 8px;border-radius:4px;margin:3px;color:#000;display:inline-block">{k}</span> {html.escape(v)}<br>' for k,v in CAT_DESC.items() if k!="dead")
H=f"""<!doctype html><html><head><meta charset=utf-8><title>SAE feature browser</title>
<style>body{{font-family:system-ui,Arial;margin:18px;background:#1a1a1a;color:#eee}}
h2{{color:#76B900}} .legend{{font-size:13px;line-height:1.8;margin:8px 0 14px}}
input,select{{padding:6px;margin:4px;background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:4px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}} th,td{{border:1px solid #333;padding:5px 7px;text-align:left;vertical-align:top}}
th{{background:#222;cursor:pointer;position:sticky;top:0}} td.lab{{max-width:340px}} td.ctx{{font-family:monospace;color:#bbb;max-width:380px}}
.pill{{padding:1px 6px;border-radius:4px;color:#000;font-size:11px}} .b{{color:#76B900}}</style></head><body>
<h2>BioReason-Pro L28 SAE — feature browser (8k)</h2>
<div class=legend><b>Categories:</b><br>{legend}
<br><b style="color:#76B900">BIO-AUC</b> = protein-band-only, held-out — the REAL sequence→function signal (use this for biology).
The grayed <b>all-token AUC</b> is leakage-prone (fires on GO terms printed in the prompt) — <b>don't use it for biology.</b> Sorted by BIO-AUC.</div>
<input id=q placeholder="search label / GO / context…" size=40 oninput=render()>
<select id=cat onchange=render()><option value="">all categories</option>
<option>bio</option><option>reasoning_only</option><option>prompt_annotation</option><option>general_text</option></select>
<span id=n></span>
<table><thead><tr>
<th onclick=sortby('id')>feat</th><th onclick=sortby('cat')>category</th><th>auto-interp label</th>
<th onclick=sortby('bioauc')>BIO-AUC<br>(protein-band, real)</th><th>band p/g/t%</th><th onclick=sortby('act')>max</th><th>top example («peak»)</th>
<th onclick=sortby('auc') style="color:#777;font-weight:normal">all-token AUC<br>(⚠ prompt-leakage)</th>
</tr></thead><tbody id=tb></tbody></table>
<script>const D={data},C={json.dumps(COLORS)};let s='bioauc',sd=-1;
function sortby(k){{sd=(s==k)?-sd:1;s=k;render()}}
function render(){{let q=document.getElementById('q').value.toLowerCase(),c=document.getElementById('cat').value;
let r=D.filter(x=>(!c||x.cat==c)&&(!q||(x.label+' '+x.go+' '+x.ctx).toLowerCase().includes(q)));
r.sort((a,b)=>{{let u=a[s],v=b[s];return (u<v?-1:u>v?1:0)*sd}});
document.getElementById('n').textContent=' '+r.length+' features';
document.getElementById('tb').innerHTML=r.slice(0,2000).map(x=>`<tr><td class=b>F${{x.id}}</td>
<td><span class=pill style="background:${{C[x.cat]||'#888'}}">${{x.cat}}</span></td>
<td class=lab>${{x.label||'<i style=color:#666>—</i>'}}</td>
<td style="color:#76B900;font-weight:bold">${{x.bioauc?x.bioauc+'<br><small style=color:#999;font-weight:normal>'+x.bioterm+'</small>':''}}</td>
<td>${{x.p}}/${{x.g}}/${{x.t}}</td><td>${{x.act}}</td><td class=ctx>${{x.ctx}}</td>
<td style="color:#777">${{x.auc?x.auc+(x.go?'<br><small>'+x.go+'</small>':''):''}}</td></tr>`).join('')}}
render();</script></body></html>"""
open(out,"w").write(H); print(f"wrote {out} ({len(rows)} features, {len(H)//1024} KB)")
