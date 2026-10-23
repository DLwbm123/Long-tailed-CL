"""All 90 sealed stages; historical and same-precision paired gates."""
import os,json,runpy
from pathlib import Path
from fractions import Fraction
ROOT=Path(os.environ['RUN_ROOT']);OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
for arm in ['linear64','gelu64']:
 os.environ['RUN_ROOT']=str(ROOT/arm);runpy.run_path(str(Path(__file__).with_name('analyze_arm.py')),run_name='__main__')
os.environ['RUN_ROOT']=str(ROOT)
reports={arm:json.loads((ROOT/arm/'results/ANALYSIS.json').read_text()) for arm in ['linear64','gelu64']}
h0=json.loads((ROOT/'assets/references/r12_three_datasets/results/ANALYSIS.json').read_text());gates={};allagg={}
for arm in reports:
 allagg[arm]={}
 for ds in ['NIH','ISIC','HK']:
  agg=json.loads((ROOT/arm/ds/'aggregate.json').read_text());allagg[arm][ds]=agg
  hpath=ROOT/'assets/references'/('r04_unit_scale' if ds=='NIH' else 'r12_three_datasets')
  if ds!='NIH':hpath=hpath/ds
  old=json.loads((hpath/'aggregate.json').read_text())['results'];old={(r['seed'],r['stage']):r for r in old if r['arm']=='H-unit'}
  r24=json.loads((ROOT/'references'/f'R24_{ds}_aggregate.json').read_text())['results'];r24={(r['seed'],r['stage']):r for r in r24}
  for r in agg['results']:
   assert r['audit']['steps']==old[r['seed'],r['stage']]['audit']['steps']
   for k in ['correct','counts']:assert r['metrics']['prototype'][k]==r24[r['seed'],r['stage']]['metrics']['prototype'][k]
 for name in ['ANALYSIS.json','VERIFICATION.json','TAIL_TRAJECTORIES.json','BRANCH_TRANSITIONS.json']:
  (OUT/f'{arm}_{name}').write_text((ROOT/arm/'results'/name).read_text())
 def writeagg():
  for ds in ['NIH','ISIC','HK']:(OUT/f'{arm}_{ds}_aggregate.json').write_text(json.dumps(allagg[arm][ds],indent=2))
 writeagg()
def exact(m,tail):
 rr=[Fraction(c,n) for c,n in zip(m['correct'],m['counts'])];return sum(rr)/len(rr),sum(rr[c] for c in tail)/len(tail)
for label,report,branch in [('historical_H0',h0,'final'),('paired_linear64',reports['linear64'],'final')]:
 gate={}
 for ds in ['NIH','ISIC','HK']:
  delta={m:reports['gelu64'][ds]['summary']['final'][m]['mean']-report[ds]['summary'][branch][m]['mean'] for m in ['mean_BA','final_BA','tail_BA']}
  now=allagg['gelu64'][ds]['results'];last=max(r['stage'] for r in now);tail=[i for i,n in enumerate(reports['gelu64'][ds]['split']['train_counts']) if n<250]
  if label=='historical_H0':
   p=ROOT/'assets/references'/('r04_unit_scale' if ds=='NIH' else 'r12_three_datasets')
   if ds!='NIH':p=p/ds
   old=json.loads((p/'aggregate.json').read_text())['results']
  else:old=allagg['linear64'][ds]['results']
  joint=0
  for seed in [74002,74003,74004]:
   a=next(r for r in now if r['seed']==seed and r['stage']==last and r['arm']=='H-unit');b=next(r for r in old if r['seed']==seed and r['stage']==last and r['arm']=='H-unit')
   joint+=int(all(x>y for x,y in zip(exact(a['metrics']['final'],tail),exact(b['metrics']['final'],tail))))
  gate[ds]=dict(deltas_pp=delta,paired_both_improved=joint)
 d=gate['NIH']['deltas_pp'];gate['passed']=d['mean_BA']>=1 and d['final_BA']>=1 and d['tail_BA']>=2.857142857 and gate['NIH']['paired_both_improved']>=2 and all(gate[ds]['deltas_pp'][m]>=-1 for ds in ['ISIC','HK'] for m in ['final_BA','tail_BA']);gates[label]=gate
gates['passed']=gates['historical_H0']['passed'] and gates['paired_linear64']['passed']
(OUT/'GATE.json').write_text(json.dumps(gates,indent=2));(OUT/'ANALYSIS.json').write_text(json.dumps(reports,indent=2))
checks=sum(json.loads((ROOT/a/'results/VERIFICATION.json').read_text())['prediction_metric_checks'] for a in reports);assert checks==270
(OUT/'VERIFICATION.json').write_text(json.dumps(dict(metric_checks=checks,new_stages=90,new_optimizer_steps=88040,no_test=True,independent_confirmation=False,prototype_counts_exact_R24=True,steps_exact_H0=True,paired_fraction_counts=True),indent=2))
print(json.dumps(gates,indent=2))
