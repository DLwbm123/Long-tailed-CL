import os,json,runpy
from pathlib import Path
from fractions import Fraction
R=Path(os.environ['RUN_ROOT']);runpy.run_path(str(Path(__file__).with_name('analyze_arm.py')),run_name='__main__');O=R/'results'
now=json.loads((O/'ANALYSIS.json').read_text());h0=json.loads((R/'assets/references/r12_three_datasets/results/ANALYSIS.json').read_text());matched=json.loads((R/'references/R28_LINEAR_ANALYSIS.json').read_text());gates={};audits={};newsteps=0
for ds in now:
 agg=json.loads((R/ds/'aggregate.json').read_text());rr=agg['results'];old=json.loads((R/'references'/f'R28_LINEAR_{ds}_aggregate.json').read_text())['results'];old={(r['seed'],r['stage']):r for r in old};audit=[]
 for r in rr:
  control=old[r['seed'],r['stage']];assert r['audit']['steps']==control['audit']['steps']
  for k in ['correct','counts']:assert r['metrics']['prototype'][k]==control['metrics']['prototype'][k]
  newsteps+=r['audit']['steps'];audit.append(dict(seed=r['seed'],stage=r['stage'],source=r['audit']['hierarchy_leaf_source'],calls=r['audit']['hierarchy_leaf_calls'],rows=r['audit']['hierarchy_leaf_rows'],image_gradient_norm=r['audit']['image_hierarchy_gradient_norm'],text_gradient_norm=r['audit']['text_hierarchy_gradient_norm']))
 audits[ds]=audit
for label,ref in [('historical_H0',h0),('matched_linear64',matched)]:
 gate={}
 for ds in now:
  delta={m:now[ds]['summary']['final'][m]['mean']-ref[ds]['summary']['final'][m]['mean'] for m in ['mean_BA','final_BA','tail_BA']}
  rr=json.loads((R/ds/'aggregate.json').read_text())['results'];last=max(r['stage'] for r in rr);tail=[c for c,n in enumerate(now[ds]['split']['train_counts']) if n<250]
  if label=='historical_H0':
   p=R/'assets/references'/('r04_unit_scale' if ds=='NIH' else 'r12_three_datasets')
   if ds!='NIH':p=p/ds
   old=json.loads((p/'aggregate.json').read_text())['results']
  else:old=json.loads((R/'references'/f'R28_LINEAR_{ds}_aggregate.json').read_text())['results']
  def exact(m):
   z=[Fraction(c,n) for c,n in zip(m['correct'],m['counts'])];return sum(z)/len(z),sum(z[c] for c in tail)/len(tail)
  joint=0
  for seed in [74002,74003,74004]:
   a=next(r for r in rr if r['seed']==seed and r['stage']==last);b=next(r for r in old if r['seed']==seed and r['stage']==last and r['arm']=='H-unit')
   joint+=int(all(x>y for x,y in zip(exact(a['metrics']['final']),exact(b['metrics']['final']))))
  gate[ds]=dict(deltas_pp=delta,paired_both_improved=joint)
 d=gate['NIH']['deltas_pp'];gate['passed']=d['mean_BA']>=1 and d['final_BA']>=1 and d['tail_BA']>=2.857142857 and gate['NIH']['paired_both_improved']>=2 and all(gate[ds]['deltas_pp'][m]>=-1 for ds in ['ISIC','HK'] for m in ['final_BA','tail_BA']);gates[label]=gate
gates['passed']=all(gates[k]['passed'] for k in ['historical_H0','matched_linear64']);assert newsteps==44070
(O/'GATE.json').write_text(json.dumps(gates,indent=2));(O/'HIERARCHY_VERIFICATION.json').write_text(json.dumps(audits,indent=2));v=json.loads((O/'VERIFICATION.json').read_text());v.update(new_records=45,reused_initial_records=0,new_optimizer_steps=newsteps,prototype_counts_exact_control=True);(O/'VERIFICATION.json').write_text(json.dumps(v,indent=2));print(json.dumps(gates,indent=2))
