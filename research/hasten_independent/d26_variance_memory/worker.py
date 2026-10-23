"""Frozen TRAIN within-class diagonal metric; reuse sealed R24 core responses."""
import os,json,time,statistics as st,traceback,ast,types
from pathlib import Path
import torch
import torch.nn.functional as F

def run():
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ref=Path(cfg['reference'])
 torch.set_num_threads(4);torch.set_num_interop_threads(2)
 def write(n,x):(out/n).write_text(json.dumps(x,indent=2,allow_nan=False))
 def metric(v,y,n,tail):
  assert torch.isfinite(v).all() and v.shape==(len(y),n)
  den=torch.bincount(y,minlength=n);correct=torch.bincount(y[v.argmax(1)==y],minlength=n);assert (den>0).all()
  rec=correct.double()/den
  return dict(BA=float(rec.mean()*100),tail_BA=float(rec[[c for c in tail if c<n]].mean()*100),counts=den.tolist(),correct=correct.tolist())
 tree=ast.parse((ref/'assets/official/models/hasten.py').read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_ens'];assert len(nodes)==1
 ns=dict(F=F);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<fixed-fusion>','exec'),ns)
 ens=lambda h,p:ns['get_ens'](types.SimpleNamespace(args={'stat':.5}),h,p)
 a=torch.tensor([[1.,2.,3.],[3.,1.,2.]],dtype=torch.double);b=torch.eye(3,dtype=torch.double)
 raw=F.normalize(a,dim=1)@b.T
 assert torch.allclose(F.normalize(a/2,dim=1)@F.normalize(b/2,dim=1).T,raw,atol=1e-14)
 write('SYNTHETIC_ACCEPTANCE.json',dict(isotropic_variance_reduces_to_cosine=True))
 results={};report={};checks=0;maxerr=0.
 for ds in ['NIH','ISIC','HK']:
  write('STATUS.json',dict(phase='PREPARE_TRAIN_STATISTICS',dataset=ds,time=time.time()))
  source=ref/ds;seal=json.loads((source/'SPLIT_SEAL.json').read_text());nst=len(seal['stages']);tail=[i for i,n in enumerate(seal['train_counts']) if n<250]
  assert json.loads((source/'ALL_HEADS_SEALED.json').read_text())['validation_rows_read']==0
  heads={}
  for seed in [74002,74003,74004]:
   for stage in range(nst):
    s=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}.private.pt',map_location='cpu',weights_only=False);n=s['total'];assert n==sum(map(len,seal['stages'][:stage+1])) and len(s['cov'])==n
    mu=s['network']['fc.weight'].double();sigma=s['network']['fc.sigma'].double();v=torch.stack([a.double().diag() for a in s['cov']]).mean(0)
    assert mu.shape==(n,512) and torch.isfinite(mu).all() and torch.isfinite(v).all() and (v>0).all()
    heads[seed,stage]=(mu,sigma,v)
  # No VAL features used until every arrived TRAIN statistic has been validated.
  write(f'{ds}_STATISTICS_SEALED.json',dict(heads=len(heads),validation_feature_reads=0,source='sealed R24 TRAIN means/covariances',extra_regularizer=0))
  vc=torch.load(ref/f'assets/{ds}/val_features.private.pt',map_location='cpu',weights_only=False);x=vc['x'];assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
  idmap={i:j for j,i in enumerate(vc['ids'])};records=[]
  for (seed,stage),(mu,sigma,v) in heads.items():
   n=len(mu);saved=torch.load(source/f'trajectories/{seed}/H-unit/stage{stage}_responses.private.pt',map_location='cpu',weights_only=False);y=saved['labels'];xx=x[[idmap[i] for i in saved['ids']]].double();h=saved['scores']['hyper_core'];p=saved['scores']['prototype'];b=saved['scores']['core_final']
   raw=sigma*(F.normalize(xx,dim=1)@F.normalize(mu,dim=1).T);err=float((raw-p).abs().max());maxerr=max(maxerr,err);assert err<2e-5 and torch.equal(raw.argmax(1),p.argmax(1))
   replay=ens(h,p);assert float((replay-b).abs().max())<1e-7 and torch.equal(replay.argmax(1),b.argmax(1))
   scale=v.rsqrt();changed=sigma*(F.normalize(xx*scale,dim=1)@F.normalize(mu*scale,dim=1).T);mix=ens(h,changed)
   metrics={k:metric(z,y,n,tail) for k,z in dict(final=mix,prototype=changed,core_final=b,raw_prototype=p).items()};checks+=4
   assert metrics['final']['counts']==seal['eval_counts'][:n]
   newok=mix.argmax(1)==y;oldok=b.argmax(1)==y
   rescue=torch.bincount(y[newok&~oldok],minlength=n).tolist();harm=torch.bincount(y[oldok&~newok],minlength=n).tolist()
   assert [c+r-h for c,r,h in zip(metrics['core_final']['correct'],rescue,harm)]==metrics['final']['correct']
   records.append(dict(seed=seed,stage=stage,seen=n,metrics=metrics,rescue=rescue,harm=harm,variance_min=float(v.min()),variance_max=float(v.max()),prototype_replay_maxerr=err))
  results[ds]=dict(records=records,split=seal);summaries=[]
  for seed in [74002,74003,74004]:
   rows=[r for r in records if r['seed']==seed];summaries.append(dict(seed=seed,branches={b:dict(mean_BA=st.mean(r['metrics'][b]['BA'] for r in rows),final_BA=rows[-1]['metrics'][b]['BA'],tail_BA=rows[-1]['metrics'][b]['tail_BA']) for b in ['final','core_final']}))
  report[ds]=dict(seeds=summaries,summary={b:{m:dict(mean=st.mean(s['branches'][b][m] for s in summaries),sd=st.stdev(s['branches'][b][m] for s in summaries)) for m in ['mean_BA','final_BA','tail_BA']} for b in ['final','core_final']})
  write('RESULTS.json',results);write('ANALYSIS.json',report)
 h0=json.loads((ref/'assets/references/r12_three_datasets/results/ANALYSIS.json').read_text());gates={}
 for label in ['historical_H0','matched_R24_core']:
  gate={}
  for ds in report:
   old=h0[ds] if label=='historical_H0' else report[ds];branch='final' if label=='historical_H0' else 'core_final'
   delta={m:report[ds]['summary']['final'][m]['mean']-old['summary'][branch][m]['mean'] for m in ['mean_BA','final_BA','tail_BA']}
   joint=sum(all(a['branches']['final'][m]>b['branches'][branch][m] for m in ['final_BA','tail_BA']) for a,b in zip(report[ds]['seeds'],old['seeds']))
   gate[ds]=dict(deltas_pp=delta,paired_both_improved=joint)
  d=gate['NIH']['deltas_pp'];gate['passed']=d['mean_BA']>=1 and d['final_BA']>=1 and d['tail_BA']>=2.857142857 and gate['NIH']['paired_both_improved']>=2 and all(gate[ds]['deltas_pp'][m]>=-1 for ds in ['ISIC','HK'] for m in ['final_BA','tail_BA']);gates[label]=gate
 gates['passed']=all(gates[k]['passed'] for k in ['historical_H0','matched_R24_core']);write('GATE.json',gates)
 assert checks==180
 write('VERIFICATION.json',dict(metric_checks=checks,stage_seed_records=45,max_raw_prototype_replay_error=maxerr,core_fusion_exact=True,new_optimizer_steps=0,new_encoder_calls=0,no_test=True,independent_confirmation=False,deterministic_reused_heads_not_new_seed_fits=True))
 write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 try:run()
 except BaseException:
  out=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output']);(out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
