"""Rank reasoning SAE features by GOOD-SYNTHESIS quality = long activated phrases scored for mechanism content.

Two stages, grounded (no LLM):
  (1) keep only features that fire on long CONTIGUOUS phrases (single-token firing = lexical/echo);
  (2) score each phrase = weighted mechanism-verbs + beyond-prompt named entities - echo/accession density.

Refinements (from validation):
  - MECHANISM VERBS (catalyzes/recruits/positions/...) weighted 1.0; DISCOURSE connectives (therefore/thus)
    only 0.2 -- the latter fire on answer-formatting boilerplate (e.g. F22404 "therefore the molecular
    function is..."), not synthesis.
  - BEYOND-PROMPT ENTITY BONUS: named molecular players (Vps5, SNX3, CCR4, Gas, PI3P) that do NOT appear in
    that protein's PROMPT band = genuine synthesis (content the model brought in, not echoed). This is the
    strongest synthesis signal; a formatting template names no entities.
  - PER-WINDOW scoring: report the FRACTION of windows that are synthesis, so mixed features (some mechanism,
    some pure IPR/residue echo, e.g. F25291 GPCR) are flagged rather than averaged out.
"""
import numpy as np, pyarrow.parquet as pq, re, json, sys

PUB = ('/data/savithas/phase3-wt/bionemo-recipes/interpretability/sparse_autoencoders/recipes/'
       'bioreason_pro/multimodal_dashboard/public/l30_balanced/feature_examples.parquet')
MECH = re.compile(r'\b(catalyz|recruit|position|potentiat|sculpt|choreograph|mediat|phosphorylat|assembl|'
                  r'stabiliz|integrat|coupl|driv|trigger|activat|inhibit|suppress|promot|initiat|requir|'
                  r'enabl|allow|shap|underli|shuttl|tether|anchor|gat(e|es|ing)|bind|dimeriz|remodel|'
                  r'orchestrat|suppli|form|occup|rearrang|create|nucleat|scaffold)\w*', re.I)
DISC = re.compile(r'\b(therefore|thus|hence|thereby|accordingly|because|so that|consequently)\w*', re.I)
ECHO = re.compile(r'(GO\s*:?\s*[0-9]|IPR\s*[0-9]|[0-9]{3,})', re.I)
ENTITY_TOK = re.compile(r'[A-Z]')            # gene/protein-symbol-like fragment (has a capital), robust to BPE
THR_FRAC, MIN_SPAN, MIN_DENS, NEX = 0.35, 4, 0.4, 12

df = pq.read_table(PUB).to_pandas()
# prompt vocabulary per protein: the GIVEN annotations = tokens in the prompt band. Entities NOT here = novel.
prompt_vocab = {}
for pid, sub in df[df.band == 'prompt'].groupby('protein_id'):
    toks = set()
    for s in sub['sequence']:
        toks.update(t.lower() for t in str(s).split())
    prompt_vocab[pid] = toks

rows = df[df.band == 'reasoning']
out = []
for fid, sub in rows.groupby('feature_id'):
    sub = sub.nlargest(NEX, 'max_activation')
    spans, wq, novelrate = [], [], []
    for _, x in sub.iterrows():
        toks = str(x['sequence']).split()
        act = np.asarray(x['activations'], float)
        m = min(len(toks), len(act))
        if not m:
            continue
        toks, act = toks[:m], act[:m]
        pk = int(act.argmax())
        thr = THR_FRAC * act[pk]
        fire = np.where(act >= thr)[0]
        if not len(fire):
            continue
        s, e = int(fire.min()), int(fire.max())
        slen = e - s + 1
        dens = len(fire) / slen
        if slen < MIN_SPAN or dens < MIN_DENS:
            continue
        span_toks = toks[s:e + 1]
        w = ' '.join(span_toks)
        nt = len(span_toks)
        pv = prompt_vocab.get(x['protein_id'], set())
        ent = [t for t in span_toks if ENTITY_TOK.search(t) and not ECHO.search(t) and len(t) <= 6]
        novel = [t for t in ent if t.lower() not in pv]        # beyond-prompt named entities
        mech = len(MECH.findall(w)) + 0.2 * len(DISC.findall(w))
        echo = len(ECHO.findall(w))
        q = (mech + 0.4 * len(novel) - echo) / nt
        spans.append(slen)
        wq.append(q)
        novelrate.append(len(novel) / nt)
    if len(spans) < 3:
        continue
    span_len = float(np.mean(spans))
    consistency = len(spans) / NEX
    mean_q = float(np.mean(wq))
    frac_synth = float(np.mean([q > 0.05 for q in wq]))        # per-window synthesis fraction
    synth_score = mean_q * min(span_len, 40) / 10 * consistency * (0.5 + frac_synth)
    out.append(dict(feature=int(fid), span_len=round(span_len, 1), n_phrase=len(spans),
                    frac_synth_win=round(frac_synth, 2), novel_entity_rate=round(float(np.mean(novelrate)), 3),
                    quality=round(mean_q, 3), synth_score=round(synth_score, 3)))
out.sort(key=lambda d: -d['synth_score'])
json.dump(out, open('synth_span_ranked.json', 'w'), indent=1)
print(f'scored {len(out)} phrase-firing reasoning features (of {rows.feature_id.nunique()})\n')
print(f'{"rank":>4} {"feat":>7} {"span":>5} {"nphr":>4} {"fracS":>6} {"novEnt":>7} {"qual":>6} {"SCORE":>7}')
for i, d in enumerate(out[:30]):
    print(f'{i+1:>4} F{d["feature"]:6} {d["span_len"]:5} {d["n_phrase"]:4} {d["frac_synth_win"]:6} '
          f'{d["novel_entity_rate"]:7} {d["quality"]:6} {d["synth_score"]:7}')
print(f'\n... {len(out)} total -> synth_span_ranked.json')
