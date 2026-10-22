import json,statistics
from pathlib import Path
root=Path(__file__).parent;data=json.loads((root/'results/aggregate.json').read_text());rows=data['results'];arms=['H-unit-ref','B-full','B-no-proj'];seeds=[74002,74003,74004]
summary={}
for a in arms:
 out=[]
 for seed in seeds:
  rr=sorted([r for r in rows if r['arm']==a and r['seed']==seed],key=lambda v:v['stage']);m=[r['metrics']['final']['BA']*100 for r in rr]
  out.append(dict(seed=seed,mean=statistics.mean(m),final=m[-1],stages=m,final_correct=sum(rr[-1]['metrics']['final']['correct']),tail=rr[-1]['metrics']['final']['tail_BA']*100,perclass_correct=rr[-1]['metrics']['final']['correct']))
 summary[a]=dict(seeds=out,mean=statistics.mean(v['mean'] for v in out),final=statistics.mean(v['final'] for v in out),tail=statistics.mean(v['tail'] for v in out))
diffs=[dict(seed=s,mean=summary['B-full']['seeds'][i]['mean']-summary['H-unit-ref']['seeds'][i]['mean'],final=summary['B-full']['seeds'][i]['final']-summary['H-unit-ref']['seeds'][i]['final']) for i,s in enumerate(seeds)]
gate=summary['B-full']['tail']>=summary['H-unit-ref']['tail'] and statistics.mean(v['mean'] for v in diffs)>=1 and statistics.mean(v['final'] for v in diffs)>=1 and sum(v['mean']>0 and v['final']>0 for v in diffs)>=2
(root/'results/ANALYSIS.json').write_text(json.dumps(dict(summary=summary,paired_balanced_minus_reference=diffs,preregistered_development_gate_pass=gate,independent_confirmation=False),indent=2))

print(json.dumps({a:{k:v[k] for k in ['mean','final','tail']} for a,v in summary.items()}))
