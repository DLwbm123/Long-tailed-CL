import json,csv
from pathlib import Path
from common import write

def run(cfg):
 r=Path(cfg['output']);rows=json.loads((r/'STAGE_METRICS.json').read_text());delta={k:[s['arms']['JOINT'][k]-s['arms']['SFT'][k] for s in rows] for k in ['macro','tail','micro']};mean={k:sum(v)/6 for k,v in delta.items()};last={k:v[-1] for k,v in delta.items()};tiny=[]
 for c in [1,2,12]:
  a=rows[-1]['arms'];tiny.append({'class':c,'count':a['SFT']['counts'][c],'SFT_correct':a['SFT']['correct'][c],'JOINT_correct':a['JOINT']['correct'][c]})
 # Per-case harm requires private saved predictions; aggregate correct counts cannot prove no harm.
 import torch
 v=torch.load(r/'VAL_scores_s5.private.pt',weights_only=False,map_location='cpu');y=v['labels'];a=v['scores']['SFT'].argmax(1)==y;b=v['scores']['JOINT'].argmax(1)==y
 for z in tiny:
  mask=y==z['class'];z['harm']=int((mask&a&~b).sum());z['rescue']=int((mask&~a&b).sum())
 both=sum(x>=-1e-8 and y>=-1e-8 for x,y in zip(delta['macro'],delta['tail']));gate=mean['macro']>=.5 and last['macro']>=.5 and last['tail']>=1 and mean['tail']>=-1e-8 and both>=4 and all(z['harm']==0 for z in tiny)
 out={'performance_pilot_gate':gate,'mean_JOINT_minus_internal_SFT_pp':mean,'final_delta_pp':last,'stage_deltas_pp':delta,'stages_macro_tail_nonnegative':both,'tiny':tiny,'attribution_controls_deferred':True,'not_independent_confirmation':True};write(r/'ANALYSIS.json',out)
 with (r/'CLASS_STAGE.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['stage','arm','class','count','correct','recall'])
  for s in rows:
   for arm,m in s['arms'].items():
    for c in range(s['classes']):w.writerow([s['stage'],arm,c,m['counts'][c],m['correct'][c],m['recall'][c]])
