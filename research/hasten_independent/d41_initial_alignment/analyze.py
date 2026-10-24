import json,os
from fractions import Fraction
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);a=json.loads((r/'RESULTS.json').read_text());summary={}
for ds,z in a.items():
 cells=z['cells'];assert len(cells)==3 and z['optimizer_steps']==z['VAL_reads']==z['TEST_reads']==0
 vals={}
 for mode in ['euclidean','hyper']:
  delta=[c['identity'][mode]['macro']-c['original'][mode]['macro'] for c in cells]
  td=[c['identity'][mode]['tail_macro']-c['original'][mode]['tail_macro'] for c in cells if c['original'][mode]['tail_macro'] is not None]
  vals[mode]=dict(mean_delta_pp=sum(delta)/3,mean_tail_delta_pp=None if not td else sum(td)/len(td),paired_both_improved=sum(sum(Fraction(v,q) for v,q in zip(c['identity'][mode]['correct_by_class'],c['identity'][mode]['counts']))>sum(Fraction(v,q) for v,q in zip(c['original'][mode]['correct_by_class'],c['original'][mode]['counts'])) and sum(Fraction(c['identity'][mode]['correct_by_class'][i],c['identity'][mode]['counts'][i]) for i in c['tail_classes'])>sum(Fraction(c['original'][mode]['correct_by_class'][i],c['original'][mode]['counts'][i]) for i in c['tail_classes']) for c in cells) if td else None,original_macro_mean=sum(c['original'][mode]['macro'] for c in cells)/3,identity_macro_mean=sum(c['identity'][mode]['macro'] for c in cells)/3)
  for c in cells:
   assert c['original'][mode]['counts']==c['identity'][mode]['counts']
 summary[ds]=vals
n=summary['NIH']['hyper'];passed=n['mean_delta_pp']>=1 and n['mean_tail_delta_pp'] is not None and n['mean_tail_delta_pp']>=1 and n['paired_both_improved']>=2 and all(summary[d]['hyper']['mean_delta_pp']>=-1 for d in ['ISIC','HK'])
(r/'FEASIBILITY.json').write_text(json.dumps(dict(summary=summary,passed=passed,training_effect_not_evaluated=True),indent=2)+'\n');print(json.dumps(dict(summary=summary,passed=passed),indent=2))
