"""Saved responses/counts only, after complete campaign; no model rescoring."""
import os,json,statistics
from pathlib import Path
from fractions import Fraction as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);ref=Path(cfg['reference_run']);assert json.loads((root/'STATUS.json').read_text())['phase']=='COMPLETE';out=root/'results';out.mkdir(exist_ok=True);summary={};classcounts=[];steps=0
for ds in ['NIH','ISIC','HK']:
 p=root/ds;x=json.loads((p/'aggregate.json').read_text());train=x['train_counts'];tail=[c for c,n in enumerate(train) if n<250];seal=json.loads((p/'ALL_HEADS_SEALED.json').read_text());assert seal['VAL_reads']==0;steps+=x['optimizer_steps'];rr=x['results'];assert len(rr)==(18 if ds!='ISIC' else 9)
 summary[ds]={}
 for arm in ['support_lora','frozen_support']:
  seeds=[]
  for seed in [74002,74003,74004]:
   rows=sorted([r for r in rr if r['seed']==seed],key=lambda r:r['stage']);per=[]
   for r in rows:
    m=r['metrics'][arm];assert all(n>0 for n in m['counts']);v=[F(c,n) for c,n in zip(m['correct'],m['counts'])];b=sum(v)/len(v);t=sum(v[c] for c in tail if c<len(v))/sum(c<len(v) for c in tail);assert abs(float(b)-m['BA'])<1e-10
    per.append((b,t));classcounts.append(dict(dataset=ds,arm=arm,seed=seed,stage=r['stage'],seen=r['seen'],correct=m['correct'],counts=m['counts'],train_counts=train[:r['seen']]))
   seeds.append(dict(seed=seed,mean_BA=float(100*sum(a for a,b in per)/len(per)),final_BA=float(100*per[-1][0]),tail_BA=float(100*per[-1][1])))
  summary[ds][arm]=dict(seeds=seeds,mean={k:statistics.mean(v[k] for v in seeds) for k in ['mean_BA','final_BA','tail_BA']},sd={k:statistics.stdev(v[k] for v in seeds) for k in ['mean_BA','final_BA','tail_BA']})
assert steps==sum(d['expected_steps'] for d in cfg['domains'].values())==52260
gates={}
for label in ['H0','R28']:
 gates[label]={}
 for ds in summary:
  if label=='R28':p=ref/'references'/f'R28_LINEAR_{ds}_aggregate.json'
  else:
   p=ref/'assets/references'/('r04_unit_scale' if ds=='NIH' else 'r12_three_datasets')
   if ds!='NIH':p=p/ds
   p=p/'aggregate.json'
  old=json.loads(p.read_text())['results'];old=[r for r in old if r['arm']=='H-unit'];train=json.loads((root/ds/'aggregate.json').read_text())['train_counts'];tail=[c for c,n in enumerate(train) if n<250];delta=[];joint=0
  for seed in [74002,74003,74004]:
   new=sorted([r for r in classcounts if r['dataset']==ds and r['arm']=='support_lora' and r['seed']==seed],key=lambda r:r['stage']);prior=sorted([r for r in old if r['seed']==seed],key=lambda r:r['stage']);assert len(new)==len(prior)
   def exact(z):
    v=[F(c,n) for c,n in zip(z['correct'],z['counts'])];return sum(v)/len(v),sum(v[c] for c in tail if c<len(v))/sum(c<len(v) for c in tail)
   a=[exact(r) for r in new];b=[exact(r['metrics']['final']) for r in prior];d=[100*(sum(v[0] for v in a)/len(a)-sum(v[0] for v in b)/len(b)),100*(a[-1][0]-b[-1][0]),100*(a[-1][1]-b[-1][1])];delta.append(d);joint+=int(d[1]>0 and d[2]>0)
  d=[sum(v[i] for v in delta)/3 for i in range(3)];gates[label][ds]=dict(deltas_pp={k:float(v) for k,v in zip(['mean_BA','final_BA','tail_BA'],d)},joint_improved=joint,exact_deltas=[str(v) for v in d])
 d=[F(v) for v in gates[label]['NIH']['exact_deltas']];gates[label]['passed']=d[0]>=1 and d[1]>=1 and d[2]>=F(20,7) and gates[label]['NIH']['joint_improved']>=2 and all(F(gates[label][ds]['exact_deltas'][i])>=-1 for ds in ['ISIC','HK'] for i in [1,2])
gates['passed']=gates['H0']['passed'] and gates['R28']['passed']
for n,x in [('ANALYSIS.json',summary),('GATE.json',gates),('ALL_CLASS_COUNTS.json',classcounts),('VERIFICATION.json',dict(optimizer_steps=steps,trained_stage_updates=45,readout_evaluations=90,all_domains_seeds=True,VAL_after_TRAIN_seal=True,independent_confirmation=False,no_TEST=True))]:(out/n).write_text(json.dumps(x,indent=2))
print(json.dumps(dict(summary=summary,gate=gates),indent=2))
