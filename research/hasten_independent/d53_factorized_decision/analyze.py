import os,json
from fractions import Fraction as Q
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);z=json.loads((r/'RESULTS.json').read_text());summary={};eligible=[];total=0;domain_means={}
for ds,v in z.items():
 assert v['passed'] and v['VAL_reads']==v['TEST_reads']==0 and len(v['cells'])==18
 total+=v['optimizer_steps'];ss=[];agg={ref:[Q(0),Q(0)] for ref in ['native','flat']}
 for seed in [74002,74003,74004]:
  gains={ref:[] for ref in ['native','flat']}
  for fold in [0,1]:
   a=next(t for t in v['cells'] if t['seed']==seed and t['fold']==fold and t['arm']=='factorized')
   assert a['proof']['coarse_init_no_rng'] and a['proof']['initial_core_exact_native'] and a['proof']['coarse_fine_gradient_separated'] and a['max_coarse_gradient']>0
   for ref in ['native','flat']:
    c=next(t for t in v['cells'] if t['seed']==seed and t['fold']==fold and t['arm']==ref);assert c['optimizer_steps']==a['optimizer_steps'] and c['fit_counts']==a['fit_counts'] and c['holdout_counts']==a['holdout_counts'];assert c['scores']['prototype']==a['scores']['prototype']
    def risk(t,cs):
     p=t['scores']['final'];return 100*sum((Q(p['correct'][k],p['counts'][k]) for k in cs),Q(0))/len(cs)
    cs=range(len(a['holdout_counts']));gains[ref].append((risk(a,cs)-risk(c,cs),risk(a,a['tail_classes'])-risk(c,c['tail_classes'])))
  item=dict(seed=seed);joint=True
  for ref,gg in gains.items():
   m=sum((a for a,b in gg),Q(0))/2;t=sum((b for a,b in gg),Q(0))/2;agg[ref][0]+=m/3;agg[ref][1]+=t/3;item[ref]=dict(macro_gain_pp=float(m),tail_gain_pp=float(t),both_gain_ge1=m>=1 and t>=1);joint=joint and m>=1 and t>=1
  item['both_references_gain_ge1']=joint;ss.append(item)
  if ds=='NIH' and joint:eligible.append(seed)
 summary[ds]=ss;domain_means[ds]={ref:dict(macro_gain_pp=float(v[0]),tail_gain_pp=float(v[1]),noharm=v[0]>=-1 and v[1]>=-1) for ref,v in agg.items()}
noharm=all(domain_means[d][ref]['noharm'] for d in ['ISIC','HK'] for ref in ['native','flat'])
f=dict(passed=len(eligible)>=2 and noharm,NIH_eligible_seeds=eligible,other_domains_noharm=noharm,summary=summary,domain_means=domain_means,total_optimizer_steps=total,not_full_incremental_accuracy=True)
(r/'FEASIBILITY.json').write_text(json.dumps(f,indent=2)+'\n');print(json.dumps(f,indent=2))
