"""Exact-count matched comparisons and frozen primary gates; saved outputs only."""
from pathlib import Path
from fractions import Fraction as F
import json,os,statistics
from decision import ARMS
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);assert json.loads((root/'results/CLOSEOUT_VERIFICATION.json').read_text())['passed'];out=root/'results';summaries={};comparisons={};counts=[]
def scores(rows,tail,arm=None):
 result=[]
 for r in sorted(rows,key=lambda v:v['stage']):
  m=r['metrics'][arm] if arm else r;v=[F(c,n) for c,n in zip(m['correct'],m['counts'])];result.append((sum(v)/len(v),sum(v[i] for i in tail if i<len(v))/sum(i<len(v) for i in tail)))
 return [100*sum(v[0] for v in result)/len(result),100*result[-1][0],100*result[-1][1]]
for ds in ['HK']:
 agg=json.loads((root/ds/'aggregate.json').read_text());tail=[i for i,n in enumerate(agg['train_counts']) if n<250];refs={}
 refs['SFT']=[dict(seed=v['seed'],stage=v['stage'],metrics={'sft_reference':v['metrics']['sft_reference']}) for v in agg['results']]
 ref=Path(cfg['reference_run']);p=ref/'assets/references'/('r04_unit_scale' if ds=='NIH' else 'r12_three_datasets')
 if ds!='NIH':p=p/ds
 refs['H0']=[v for v in json.loads((p/'aggregate.json').read_text())['results'] if v['arm']=='H-unit']
 refs['R28']=[v for v in json.loads((ref/'references'/f'R28_LINEAR_{ds}_aggregate.json').read_text())['results'] if v['arm']=='H-unit']
 for label,rr in refs.items():
  for v in agg['results']:
   prior=next(z for z in rr if z['seed']==v['seed'] and z['stage']==v['stage']);m=prior['metrics']['sft_reference' if label=='SFT' else 'final'];assert m['counts']==v['metrics']['fixed_tars_grpo']['counts']
 summaries[ds]={};comparisons[ds]={}
 exact={a:{} for a in ARMS}
 for arm in ARMS:
  for seed in [74002]:exact[arm][seed]=scores([v for v in agg['results'] if v['seed']==seed],tail,arm)
  summaries[ds][arm]=dict(seeds=[dict(seed=s,avg_final_tail=[float(v) for v in exact[arm][s]]) for s in exact[arm]],mean=[float(sum(exact[arm][s][i] for s in exact[arm])/len(exact[arm])) for i in range(3)])
  for v in agg['results']:counts.append(dict(dataset=ds,arm=arm,seed=v['seed'],stage=v['stage'],**v['metrics'][arm]))
 for label in ['H0','R28','SFT','dynamic_tars_grpo','dynamic_tars_exact','fixed_tars_exact']:
  delta=[];joint=0
  for seed in [74002]:
   if label in exact:b=exact[label][seed]
   else:b=scores([v for v in refs[label] if v['seed']==seed],tail,'sft_reference' if label=='SFT' else 'final')
   a=exact['fixed_tars_grpo'][seed];diff=[a[i]-b[i] for i in range(3)];delta.append(diff);joint+=int(diff[1]>0 and diff[2]>0)
  mean=[sum(d[i] for d in delta)/len(delta) for i in range(3)];comparisons[ds][label]=dict(mean_delta_pp=[float(v) for v in mean],exact_delta=[str(v) for v in mean],joint_improved=joint)
gates=dict(full_three_domain_adoption_gate_evaluated=False,single_seed_exploratory=True,scope='HK seed74002; all6stages completed',primary_candidate='fixed_tars_grpo',not_independent_patient_confirmation=True)
for name,value in [('ANALYSIS.json',summaries),('COMPARISONS.json',comparisons),('GATE.json',gates),('ALL_CLASS_COUNTS.json',counts)]: (out/name).write_text(json.dumps(value,indent=2))
print(json.dumps(dict(summary=summaries,comparisons=comparisons,gate=gates),indent=2))
