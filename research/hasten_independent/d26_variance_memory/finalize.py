"""Evaluate paired improvement with exact class count fractions, avoiding float ties."""
import json,statistics as st
from fractions import Fraction
from pathlib import Path

def write_gate(out,ref):
 results=json.loads((out/'RESULTS.json').read_text());report=json.loads((out/'ANALYSIS.json').read_text());gates={}
 h0=json.loads((ref/'assets/references/r12_three_datasets/results/ANALYSIS.json').read_text())
 def recalls(m,tail):
  rr=[Fraction(c,n) for c,n in zip(m['correct'],m['counts'])]
  return sum(rr)/len(rr),sum(rr[c] for c in tail)/len(tail)
 for label in ['historical_H0','matched_R24_core']:
  gate={}
  for ds in report:
   old=h0[ds] if label=='historical_H0' else report[ds];branch='final' if label=='historical_H0' else 'core_final'
   delta={m:report[ds]['summary']['final'][m]['mean']-old['summary'][branch][m]['mean'] for m in ['mean_BA','final_BA','tail_BA']}
   rr=results[ds]['records'];last=max(r['stage'] for r in rr);tail=[c for c,n in enumerate(results[ds]['split']['train_counts']) if n<250];joint=0
   if label=='historical_H0':
    p=ref/'assets/references'/('r04_unit_scale' if ds=='NIH' else 'r12_three_datasets')
    if ds!='NIH':p=p/ds
    refs=json.loads((p/'aggregate.json').read_text())['results']
   for seed in [74002,74003,74004]:
    row=next(r for r in rr if r['seed']==seed and r['stage']==last);a=recalls(row['metrics']['final'],tail)
    b=recalls(next(r for r in refs if r['seed']==seed and r['stage']==last and r['arm']=='H-unit')['metrics']['final'],tail) if label=='historical_H0' else recalls(row['metrics']['core_final'],tail)
    joint+=int(all(x>y for x,y in zip(a,b)))
   gate[ds]=dict(deltas_pp=delta,paired_both_improved=joint)
  d=gate['NIH']['deltas_pp'];gate['passed']=d['mean_BA']>=1 and d['final_BA']>=1 and d['tail_BA']>=2.857142857 and gate['NIH']['paired_both_improved']>=2 and all(gate[ds]['deltas_pp'][m]>=-1 for ds in ['ISIC','HK'] for m in ['final_BA','tail_BA']);gates[label]=gate
 gates['passed']=all(gates[k]['passed'] for k in ['historical_H0','matched_R24_core']);gates['paired_comparison']='exact Fraction class counts; equal counts cannot improve due to float roundoff'
 (out/'GATE.json').write_text(json.dumps(gates,indent=2))
if __name__=='__main__':
 import os
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());write_gate(Path(cfg['output']),Path(cfg['reference']))
