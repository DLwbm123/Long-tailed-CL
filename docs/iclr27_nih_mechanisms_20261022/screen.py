"""Reproducible title/abstract screening of the complete dated public catalog; not full-text reading."""
import json,re,sys
from pathlib import Path
from collections import Counter
patterns={
 'long_tail':r'long[ -]?tail(?:ed)?|class[ -]imbalanc|imbalanced (?:class|data|recognition|learning)|tail class|rare class|minority class',
 'prototype':r'prototyp(?:e|es|ical)|class cent(?:er|re)|class representation|multi[ -]cent(?:er|re)',
 'few_shot':r'few[ -]shot|one[ -]shot|low[ -]shot|n[ -]way.*k[ -]shot|support set|novel class',
 'continual':r'continual learn|class[ -]incremental|incremental learn|catastrophic forget',
 'transfer_geometry':r'neural collapse|feature (?:calibration|distribution)|distribution calibration|optimal transport|subspace.*class|class.*subspace',
}
visual=r'image classific|visual recogni|vision[ -]language|image recogni|fine[ -]grained|medical imag|few[ -]shot classif|long[ -]tailed recogni|semantic segment|image[ -]based'
llm=r'large language model|language model|in[ -]context learning|reasoning|code generat'
rows=[json.loads(x) for x in Path(sys.argv[1]).open() if x.strip()]
assert len({r['id'] for r in rows})==len(rows)
found=[]
for r in rows:
 title=r.get('title',''); abstract=r.get('abstract','');text=title+'\n'+abstract+'\n'+' '.join(r.get('keywords',[]));tags=[k for k,v in patterns.items() if re.search(v,text,re.I)]
 if not tags:continue
 focus=any(k in tags for k in ['long_tail','prototype','few_shot']);vis=bool(re.search(visual,text,re.I));lang=bool(re.search(llm,text,re.I))
 score=sum(4 if re.search(v,title,re.I) else 1 for k,v in patterns.items() if k in tags)+4*vis+3*('long_tail' in tags)+2*('prototype'in tags)+2*('few_shot'in tags)-3*(lang and not vis)
 found.append(dict(id=r['id'],title=title,tags=tags,score=score,visual_signal=vis,language_signal=lang,focus=focus,status=r.get('venueid'),url='https://openreview.net/forum?id='+r['id'],pdf=r.get('pdf'),abstract=abstract,reading_level='metadata-screened only'))
found.sort(key=lambda r:(-r['score'],r['id']))
p=Path(sys.argv[2]);p.mkdir(exist_ok=True)
(p/'SCREENING.json').write_text(json.dumps(dict(snapshot_date='2026-10-18',records=len(rows),unique=len(rows),missing_abstracts=sum(not r.get('abstract') for r in rows),matched=len(found),focus_matched=sum(r['focus'] for r in found),tags=dict(Counter(k for r in found for k in r['tags'])),classification='regex retrieval only, not all abstracts read by a human or full-text reading',papers=found),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(total=len(rows),matched=len(found),tags=dict(Counter(k for r in found for k in r['tags'])))))
for r in found[:100]:print(r['score'],r['id'],','.join(r['tags']),r['title'])
