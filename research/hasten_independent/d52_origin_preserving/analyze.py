import os,json
from fractions import Fraction as Q
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);z=json.loads((r/'RESULTS.json').read_text());summary={};eligible=[];total=0
for ds,v in z.items():
 assert v['passed'] and v['VAL_reads']==v['TEST_reads']==0 and len(v['cells'])==12
 total+=v['optimizer_steps'];ss=[]
 for seed in [74002,74003,74004]:
  gains=[]
  for fold in [0,1]:
   c=next(t for t in v['cells'] if t['seed']==seed and t['fold']==fold and t['arm']=='native');a=next(t for t in v['cells'] if t['seed']==seed and t['fold']==fold and t['arm']=='origin_preserving')
   assert c['optimizer_steps']==a['optimizer_steps'] and c['fit_counts']==a['fit_counts'] and c['holdout_counts']==a['holdout_counts'];assert c['scores']['prototype']==a['scores']['prototype']
   assert a['proof']['bias_removal_no_rng'] and a['proof']['initial_nonbias_core_exact_control'] and a['proof']['origin_exact_zero'] and a['removed_parameters']==1024 and a['bias_absent_step_checks']==a['optimizer_steps']
   def risk(t,cs):
    p=t['scores']['final'];return 100*sum((Q(p['correct'][k],p['counts'][k]) for k in cs),Q(0))/len(cs)
   cs=range(len(a['holdout_counts']));gains.append((risk(a,cs)-risk(c,cs),risk(a,a['tail_classes'])-risk(c,c['tail_classes'])))
  macro=sum((a for a,b in gains),Q(0))/2;tail=sum((b for a,b in gains),Q(0))/2
  ss.append(dict(seed=seed,macro_gain_pp=float(macro),tail_gain_pp=float(tail),both_gain_ge1=macro>=1 and tail>=1))
  if ds=='NIH' and macro>=1 and tail>=1:eligible.append(seed)
 summary[ds]=ss
noharm=all(sum(v[k] for v in summary[d])/3>=-1 for d in ['ISIC','HK'] for k in ['macro_gain_pp','tail_gain_pp'])
f=dict(passed=len(eligible)>=2 and noharm,NIH_eligible_seeds=eligible,other_domains_noharm=noharm,summary=summary,total_optimizer_steps=total,not_full_incremental_accuracy=True)
(r/'FEASIBILITY.json').write_text(json.dumps(f,indent=2)+'\n');print(json.dumps(f,indent=2))
