"""Saved class counts and TRAIN curves only. No model/data scoring or fit."""
import json,statistics,os
from pathlib import Path
r=Path(os.environ.get('RESULTS_ROOT',Path(__file__).parent/'results'));aa=json.loads((r/'ALL_CLASS_COUNTS.json').read_text());cc=json.loads((r/'FULL_TRAIN_CURVES.json').read_text());out={}
for ds in ['NIH','ISIC','HK']:
 a=[x for x in aa if x['dataset']==ds and x['arm']=='vision_lora'];last=max(x['stage'] for x in a);n0=next(x['seen'] for x in a if x['stage']==0);f=next(x for x in a if x['stage']==last);tails=[i for i,n in enumerate(f['train_counts']) if n<250];old=[i for i in tails if i<n0];new=[i for i in tails if i>=n0];rows=[]
 for seed in [74002,74003,74004]:
  v=sorted([x for x in a if x['seed']==seed],key=lambda x:x['stage']);birth={i:next(x for x in v if x['seen']>i) for i in tails};fin=v[-1];train=next(x for x in cc if x['dataset']==ds and x['seed']==seed and x['stage']==last)['curves'][-1];row={'seed':seed,'final_TRAIN_macro_BA':100*train['macro_BA'],'final_TRAIN_tail_BA':100*train['tail_BA']}
  for name,idx in [('initial_tail',old),('later_tail',new)]:
   if idx:row[name]={'classes':len(idx),'birth_BA':100*statistics.mean(birth[i]['correct'][i]/birth[i]['counts'][i] for i in idx),'final_BA':100*statistics.mean(fin['correct'][i]/fin['counts'][i] for i in idx)}
  rows.append(row)
 cells=[x for x in cc if x['dataset']==ds];platform=0
 for x in cells:
  t=[z['objective_mean'] for z in x['curves'][-3:]];platform+=int((max(t)-min(t))/max(abs(statistics.mean(t)),1e-12)<.01)
 out[ds]={'last3_objective_relative_range_lt1pct':platform,'stage_cells':len(cells),'seeds':rows}
(r/'CURVE_AND_BIRTH_ANALYSIS.json').write_text(json.dumps(out,indent=2,allow_nan=False))
