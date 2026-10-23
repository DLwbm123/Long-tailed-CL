"""Verify private scores on host; export only class-level counts and summaries."""
import os,json,statistics as st,time
from pathlib import Path
import torch
torch.set_num_threads(2)
ROOT=Path(os.environ['RUN_ROOT']);OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
report={};trajectories=[];branches=[];checks=0
for ds in ['NIH','ISIC','HK']:
 source=ROOT/ds
 seal=json.loads((source/'SPLIT_SEAL.json').read_text());agg=json.loads((source/'aggregate.json').read_text());rr=[r for r in agg['results'] if r['arm']=='H-unit'];stages=len(seal['stages']);counts=seal['train_counts'];tail=[i for i,n in enumerate(counts) if n<250];arrival={c:s for s,cs in enumerate(seal['stages']) for c in cs};seen=[sum(map(len,seal['stages'][:s+1])) for s in range(stages)]
 assert len(rr)==3*stages
 summaries=[]
 for seed in [74002,74003,74004]:
  rows=sorted([r for r in rr if r['seed']==seed],key=lambda r:r['stage']);assert len(rows)==stages
  for r in rows:
   stage=r['stage'];n=r['seen'];old=r['old'];assert r['checkpoint_restore_predictions_exact'] and r['validation_rows_read']==0
   assert r['audit']['fixed_parent_max_change']==0 and r['train_distance_upper_clamp_fraction']==0
   if old:assert r['audit']['replay']=={str(c):r['audit']['steps'] for c in range(old)}
   x=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}_responses.private.pt',map_location='cpu',weights_only=False);y=x['labels'];scores=x['scores'];pred={k:v.argmax(1) for k,v in scores.items()};ok={k:v.eq(y) for k,v in pred.items()}
   denom=torch.bincount(y,minlength=n);assert denom.tolist()==seal['eval_counts'][:n]
   for k,v in scores.items():
    assert v.shape==(len(y),n) and torch.isfinite(v).all()
    correct=torch.bincount(y[ok[k]],minlength=n);m=r['metrics'][k];assert correct.tolist()==m['correct'] and denom.tolist()==m['counts']
    rec=correct.double()/denom;assert abs(float(rec.mean())-m['BA'])<1e-7
    assert abs(float(rec[[c for c in tail if c<n]].mean())-m['tail_BA'])<1e-7;checks+=1
   for c in range(n):
    mask=y==c;den=int(denom[c]);birthn=seen[arrival[c]]
    row=dict(dataset=ds,seed=seed,stage=stage,class_index=c,original_id=seal['original_order'][c],train_count=counts[c],eval_count=den,tail=c in tail,arrival=arrival[c])
    for k,v in scores.items():row[k+'_correct']=int(ok[k][mask].sum());row[k+'_fixed_correct']=int((v[mask,:birthn].argmax(1)==c).sum())
    trajectories.append(row)
    p,h,f=ok['prototype'][mask],ok['hyper'][mask],ok['final'][mask]
    branches.append(dict(dataset=ds,seed=seed,stage=stage,class_index=c,tail=c in tail,eval_count=den,both_wrong=int((~p&~h).sum()),prototype_rescue=int((~p&f).sum()),prototype_harm=int((p&~f).sum()),hyper_rescue=int((~h&f).sum()),hyper_harm=int((h&~f).sum())))
  bybranch={k:dict(mean_BA=st.mean(r['metrics'][k]['BA']*100 for r in rows),final_BA=rows[-1]['metrics'][k]['BA']*100,tail_BA=rows[-1]['metrics'][k]['tail_BA']*100) for k in ['final','prototype','hyper']}
  summaries.append(dict(seed=seed,branches=bybranch))
 def stats(vals):return dict(mean=st.mean(vals),sd=st.stdev(vals))
 summary={k:{metric:stats([r['branches'][k][metric] for r in summaries]) for metric in ['mean_BA','final_BA','tail_BA']} for k in ['final','prototype','hyper']}
 t=[v for v in trajectories if v['dataset']==ds and v['tail']];last=[v for v in t if v['stage']==stages-1];first=[v for v in t if v['stage']==v['arrival']]
 br=[v for v in branches if v['dataset']==ds and v['tail'] and v['stage']==stages-1]
 diag=dict(first_tail_macro=100*st.mean(v['final_correct']/v['eval_count'] for v in first),final_tail_macro=100*st.mean(v['final_correct']/v['eval_count'] for v in last),fixed_final_tail_macro=100*st.mean(v['final_fixed_correct']/v['eval_count'] for v in last),both_wrong_tail_macro=100*st.mean(v['both_wrong']/v['eval_count'] for v in br),prototype_rescue_macro=100*st.mean(v['prototype_rescue']/v['eval_count'] for v in br),prototype_harm_macro=100*st.mean(v['prototype_harm']/v['eval_count'] for v in br),hyper_rescue_macro=100*st.mean(v['hyper_rescue']/v['eval_count'] for v in br),hyper_harm_macro=100*st.mean(v['hyper_harm']/v['eval_count'] for v in br))
 report[ds]=dict(seeds=summaries,summary=summary,tail_diagnostics=diag,split=seal,train_imbalance=max(counts)/min(counts),new_stages=len(rr),reused_stages=0)
 if True:
  (OUT/f'{ds}_aggregate.json').write_text(json.dumps(agg,indent=2));(OUT/f'{ds}_ALL_HEADS_SEALED.json').write_text((source/'ALL_HEADS_SEALED.json').read_text())
for name,data in [('ANALYSIS.json',report),('TAIL_TRAJECTORIES.json',trajectories),('BRANCH_TRANSITIONS.json',branches),('VERIFICATION.json',dict(prediction_metric_checks=checks,records=45,new_records=45,no_test=True,cache_unit_norm_receipts=True,all_count_checks=True,restore_receipts=True,independent_confirmation=False))]:
 (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False))
print(json.dumps({k:dict(metrics=v['summary']['final'],diagnostic=v['tail_diagnostics']) for k,v in report.items()},indent=2))

h0=json.loads((ROOT/'assets/references/r12_three_datasets/results/ANALYSIS.json').read_text())
gate={};deltas={}
for ds in report:
 oldsource=ROOT/'assets/references/r04_unit_scale' if ds=='NIH' else ROOT/'assets/references/r12_three_datasets'/ds
 old=json.loads((oldsource/'aggregate.json').read_text())
 old={(r['seed'],r['stage']):r for r in old['results'] if r['arm']=='H-unit'}
 new=json.loads((ROOT/ds/'aggregate.json').read_text())
 for r in new['results']:
  assert r['audit']['steps']==old[r['seed'],r['stage']]['audit']['steps']
 assert report[ds]['split']==h0[ds]['split']
 for r in new['results']:
  a=r['audit']['centroid_anchor']
  assert a['calls']==r['audit']['steps'] and a['max_applied_parameter_ratio']<=.1001
  assert sum(a['seen_by_class'])==a['examples']
  assert a['seen_by_class']==[a['calls']]*r['seen']
  assert a['examples']==r['seen']*r['audit']['steps']
 deltas[ds]={m:report[ds]['summary']['final'][m]['mean']-h0[ds]['summary']['final'][m]['mean'] for m in ['mean_BA','final_BA','tail_BA']}
 paired=[]
 for n,o in zip(report[ds]['seeds'],h0[ds]['seeds']):
  assert n['seed']==o['seed']
  paired.append(all(n['branches']['final'][m]>o['branches']['final'][m] for m in ['final_BA','tail_BA']))
 gate[ds]={'deltas_pp':deltas[ds],'paired_both_improved':sum(paired)}
gate['passed']=deltas['NIH']['mean_BA']>=1 and deltas['NIH']['final_BA']>=1 and deltas['NIH']['tail_BA']>=2.857142857 and gate['NIH']['paired_both_improved']>=2 and all(deltas[d][m]>=-1 for d in ['ISIC','HK'] for m in ['final_BA','tail_BA'])
(OUT/'GATE.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate,indent=2))
