"""Aggregate saved final predictions; no encoding, fitting, or parameter selection."""
import json,os,time
from pathlib import Path
import torch
start=time.time();torch.set_num_threads(4);root=Path(os.environ['RUN_OUTPUT'])
s=torch.load(root/'stage_5.private.pt',weights_only=True);y=s['labels'];seen=s['seen']
pred={a:torch.tensor(seen)[v.argmax(1)] for a,v in s['scores'].items()}
bins=json.loads((root/'FREQUENCY.json').read_text())['images']['bins']
groups=dict(bins,base=seen[:10],later=seen[10:],old=seen[:18],new=seen[18:]);out={}
for arm in ['uniform_shrink','risk_limited_shrink']:
 out[arm]={}
 for ref in ['cls768_ridge','scale_joint','df_matched_joint']:
  rec={};a=pred[arm]==y;b=pred[ref]==y
  for name,cs in groups.items():
   mask=torch.tensor([int(v) in cs for v in y]);entry=dict(images=int(mask.sum()),candidate_correct=int((a&mask).sum()),reference_correct=int((b&mask).sum()),rescue=int((a&~b&mask).sum()),harm=int((~a&b&mask).sum()))
   assert entry['candidate_correct']-entry['reference_correct']==entry['rescue']-entry['harm'];rec[name]=entry
  out[arm][ref]=rec
 c=pred['cls768_ridge']==y;a=pred[arm]==y;overlaps={}
 for ref in ['scale_joint','df_matched_joint']:
  b=pred[ref]==y;ra=a&~c;rb=b&~c;ha=~a&c;hb=~b&c
  overlaps[ref]=dict(rescue_candidate=int(ra.sum()),rescue_reference=int(rb.sum()),rescue_intersection=int((ra&rb).sum()),rescue_only_candidate=int((ra&~rb).sum()),harm_candidate=int(ha.sum()),harm_reference=int(hb.sum()),harm_intersection=int((ha&hb).sum()),harm_only_candidate=int((ha&~hb).sum()))
 out[arm]['relative_CLS_overlap']=overlaps
out['analysis_seconds']=time.time()-start
(root/'FINAL_GROUP_PAIRS.json').write_text(json.dumps(out,indent=2)+'\n')
