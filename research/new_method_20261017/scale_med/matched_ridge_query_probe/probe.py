"""Exactly two matched support-only ridge solves; CPU and sealed features."""
import json,os,time,resource
from pathlib import Path
from collections import Counter
import torch

torch.set_num_threads(4)
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def transform(g,mp):return (((g-mp['gn'][0])/mp['gn'][1])[:,mp['ids']]-mp['on'][0])/mp['on'][1]
def same_mapper(a,b):return torch.equal(a['ids'],b['ids']) and all(torch.equal(x,y) for k in ['gn','on'] for x,y in zip(a[k],b[k]))
def hessian(a):
 h=a.clone();h.diagonal()[:-1].add_(.01);return h

def qweights(y,groups):
 n=Counter(groups);nc={c:len({groups[i] for i in torch.where(y==c)[0]}) for c in range(4)}
 return torch.tensor([1/(4*nc[int(c)]*n[g]) for c,g in zip(y,groups)],dtype=torch.float64)
def points(s,y,groups):
 pred=s.argmax(1);correct=[int((pred[y==c]==c).sum()) for c in range(4)];counts=[int((y==c).sum()) for c in range(4)];rec=[k/n for k,n in zip(correct,counts)]
 ce=torch.nn.functional.cross_entropy(s,y,reduction='none');other=s.clone();other[torch.arange(len(y)),y]=-torch.inf;mar=s[torch.arange(len(y)),y]-other.max(1).values;p=qweights(y,groups)
 cm=torch.bincount(y*4+pred,minlength=16).reshape(4,4)
 return dict(BA=sum(rec)/4,correct=correct,counts=counts,recall=rec,group_balanced_BA=float(p@(pred==y).double()),confusion_matrix=cm.tolist(),classes=[dict(class_index=c,CE=float(ce[y==c].mean()),margin_quantiles=torch.quantile(mar[y==c],torch.tensor([0.,.1,.5,.9,1.],dtype=mar.dtype)).tolist()) for c in range(4)])
def pairs(b,p,y):
 bb,pp=b.argmax(1),p.argmax(1);rows=[]
 for c in range(4):
  ids=y==c;bo,po=bb[ids]==c,pp[ids]==c;r,h=int((~bo & po).sum()),int((bo & ~po).sum());assert r-h==int(po.sum())-int(bo.sum())
  rows.append(dict(class_index=c,rescued=r,harmed=h,net=r-h,both_correct=int((bo & po).sum()),both_wrong=int((~bo & ~po).sum()),harmed_destination=torch.bincount(pp[ids][bo & ~po],minlength=4).tolist()))
 assert abs(25*sum(z['net']/int((y==c).sum()) for c,z in enumerate(rows))-100*sum(float((pp[y==c]==c).double().mean()-(bb[y==c]==c).double().mean()) for c in range(4))/4)<1e-10
 return rows

def gate(pooled,folds,paired):
 if any(n not in pooled for n in ['F','T_CE','T_probe']) or set(folds)!={'0','1'}:return dict(pass_all=False,missing=True,decision='MIXED_MATCHED_RIDGE_DIAGNOSTIC')
 f,p=pooled['F'],pooled['T_probe'];checks=dict(CLASS1_COUNT_RESTORED=p['correct'][1]>=f['correct'][1] and all(v['T_probe']['correct'][1]>=v['F']['correct'][1] for v in folds.values()),CLASS1_F_CORRECT_PRESERVED=paired[1]['harmed']==0,BA_RESTORED=p['BA']>=f['BA']-1e-12 and all(v['T_probe']['BA']>=v['F']['BA']-1e-12 for v in folds.values()),OTHER_CLASS_COST_WITHIN_1PP=all(p['recall'][c]-f['recall'][c]>=-.01-1e-12 for c in [0,2,3]))
 checks={k:bool(v) for k,v in checks.items()};passed=all(checks.values());alllower=p['BA']<f['BA']-1e-12 and all(v['T_probe']['BA']<v['F']['BA']-1e-12 for v in folds.values())
 return dict(**checks,pass_all=passed,decision='MATCHED_RIDGE_POINT_COMPETITIVE' if passed else 'STOP_MATCHED_RIDGE_NONCOMPETITIVE' if alllower else 'MIXED_MATCHED_RIDGE_DIAGNOSTIC')
def synthetic():
 a=torch.eye(8,dtype=torch.float64);w=torch.arange(32,dtype=a.dtype).reshape(8,4)/32;h=hessian(a);b=h@w;assert float((h@w-b).abs().max())==0 and h[-1,-1]==a[-1,-1]
 # The diagonal addition is floating point; its mathematical value uses a tight absolute tolerance.
 assert torch.allclose(h.diagonal()[:-1]-a.diagonal()[:-1],torch.full((7,),.01,dtype=a.dtype),atol=1e-15,rtol=0)
 x=torch.arange(56,dtype=a.dtype).reshape(8,7)/56;shift=torch.ones(7,dtype=a.dtype);assert torch.allclose(x@w[:-1]+w[-1],(x-shift)@w[:-1]+w[-1]+shift@w[:-1],atol=1e-12)
 y=torch.arange(32)%4;groups=['toy'+str(i) for i in range(32)];s=torch.nn.functional.one_hot(y,4).double();tab=points(s,y,groups);pair=pairs(s,s,y);folds={'0':dict(F=tab,T_probe=tab),'1':dict(F=tab,T_probe=tab)};r=gate(dict(F=tab,T_CE=tab,T_probe=tab),folds,pair);assert r['pass_all'];bad=[dict(z) for z in pair];bad[1]['harmed']=1;assert not gate(dict(F=tab,T_CE=tab,T_probe=tab),folds,bad)['pass_all'];assert gate({}, {}, [])['missing']
 return dict(no_synthetic_solve=True,penalty_intercept_and_translation=True,gates_success_failure_missing=True,paired_group_weight_finite_JSON=True)

def main(cfg):
 tests=synthetic();old=Path(cfg['sealed_root']);out=Path(cfg['output'])
 names=[old/'prep/PLAN.private.pt',old/'evaluate/responses.private.pt']
 for k in [0,1]:names.extend([old/'prep'/f'{k}.teacher.private.pt',old/'evaluate'/f'{k}.state.private.pt',old/'evaluate'/f'{k}.g.private.pt'])
 allowed={p.resolve() for p in names};original=torch.load
 def load(p):
  p=Path(p)
  assert p.resolve() in allowed or p.parent==out and p.name in ['0.probe.private.pt','1.probe.private.pt'],'PRIVATE_ASSET_READ_DENIED'
  return original(p,map_location='cpu',weights_only=False)
 torch.load=load
 assert json.loads((old/'train/ENCODER_SEAL.json').read_text())['query_excluded_entire_producer']
 expected=json.loads((old/'evaluate/aggregate.json').read_text());plan=load(old/'prep/PLAN.private.pt');previous=load(old/'evaluate/responses.private.pt');y=plan['y'];groups=plan['groups'];fold=plan['fold'];assert len(y)==2555 and torch.equal(previous['y'],y) and previous['groups']==groups
 scores={n:torch.zeros(2555,4,dtype=torch.float64) for n in ['F','T_CE','T_probe']};coverage=torch.zeros(2555,dtype=torch.long);probe_scores={};fits=[];sizes={};staterr=0.;parity=0.;fres=0.;solves=0
 for k in [0,1]:
  support=fold==k;query=~support;assert not {groups[i] for i in torch.where(support)[0]} & {groups[i] for i in torch.where(query)[0]}
  assert json.loads((old/'evaluate'/f'{k}.FEATURE_SEAL.json').read_text())['rows']==2555
  teacher=load(old/'prep'/f'{k}.teacher.private.pt');art=load(old/'evaluate'/f'{k}.state.private.pt');g=load(old/'evaluate'/f'{k}.g.private.pt');assert same_mapper(teacher['mapper'],art['mapper'])
  mp=teacher['mapper'];d=len(art['W']);ix=torch.triu_indices(d,d);a=torch.zeros(d,d,dtype=torch.float64);a[ix.unbind()]=art['A'];a[ix[1],ix[0]]=art['A'];b=art['B'];assert d==1536 and list(b.shape)==[1536,6]
  assert torch.allclose(a[-1,-1],torch.tensor(1.,dtype=a.dtype),atol=1e-12) and torch.allclose(b[-1,:4],torch.full((4,),.25,dtype=a.dtype),atol=1e-12) and torch.equal(b[:,4:],torch.zeros_like(b[:,4:]))
  ft=hessian(teacher['A'])@teacher['W']-teacher['B'][:,:4];fres=max(fres,float(ft.abs().max()));assert fres<=1e-7
  z=transform(g,mp);x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);sg=[groups[i] for i in torch.where(support)[0]];q=qweights(y[support],sg);cols=torch.tensor([0,17,767,1534,1535]);xs=x[support][:,cols]
  small=xs.T@(xs*q[:,None]);bb=xs.T@(torch.nn.functional.one_hot(y[support],6).double()*q[:,None]);staterr=max(staterr,float((small-a[cols][:,cols]).abs().max()),float((bb-b[cols]).abs().max()));assert staterr<=1e-10
  w=torch.linalg.solve(hessian(a),b);solves+=1;res=float((hessian(a)@w-b).abs().max());assert res<=1e-7 and torch.isfinite(w).all() and torch.equal(w[:,4:],torch.zeros_like(w[:,4:]))
  probe=dict(art,W=w.contiguous(),alpha=.01,head_origin='matched_support_ridge_same_F_mapper',source_commit=cfg['source_commit']);file=out/f'{k}.probe.private.pt';torch.save(probe,file);sizes[str(k)]=file.stat().st_size;assert sizes[str(k)]<=19058800
  back=load(file);assert same_mapper(mp,back['mapper']) and torch.equal(back['W'],w) and torch.equal(back['A'],art['A']) and torch.equal(back['B'],b) and all(torch.equal(back['bias_overrides'][n],v) for n,v in art['bias_overrides'].items())
  testids=torch.where(support)[0][:32];ss=z[testids]@w[:-1,:4]+w[-1,:4];replay=transform(g[testids],back['mapper'])@back['W'][:-1,:4]+back['W'][-1,:4];assert torch.allclose(ss,replay,atol=1e-10,rtol=0)
  probe_scores[k]=dict(query=query,zq=z[query],gf=plan['cached_g'][query],mp=mp,w=w,fw=teacher['W'],cew=art['W']);coverage[query]+=1
  fits.append(dict(fold=k,alpha=.01,residual=res,support_rows=int(support.sum()),query_rows=int(query.sum()),mapper_reestimated=False))
  write(out/f'{k}.HEAD_SEAL.json',dict(alpha=.01,head_origin='matched_support_ridge_same_F_mapper',residual=res,query_excluded_from_fit=True))
 assert solves==2 and (coverage==1).all()
 write(out/'MODEL_SEAL.json',dict(actual_ridge_solves=2,alpha=.01,mapper_reestimated=False,artifacts=sizes,development_used=False,test_used=False,future_used=False))
 for k,v in probe_scores.items():
  mask=v['query'];zq=v['zq'];w=v['w'];fw=v['fw'];cw=v['cew'];ff=transform(v['gf'],v['mp'])@fw[:-1]+fw[-1];ce=zq@cw[:-1,:4]+cw[-1,:4]
  scores['T_probe'][mask]=zq@w[:-1,:4]+w[-1,:4];scores['F'][mask]=ff;scores['T_CE'][mask]=ce
  parity=max(parity,float((ff-previous['score']['F_out'][mask]).abs().max()),float((ce-previous['score']['T_out'][mask]).abs().max()));assert parity<=1e-8
 assert all(torch.isfinite(s).all() for s in scores.values())
 pooled={n:points(s,y,groups) for n,s in scores.items()};folds={str(k):{n:points(s[fold!=k],y[fold!=k],[groups[i] for i in torch.where(fold!=k)[0]]) for n,s in scores.items()} for k in [0,1]}
 for n,ref in [('F','F_out'),('T_CE','T_out')]:assert pooled[n]['correct']==[v['correct'] for v in expected['summary'][ref]['classes']]
 paired={n:pairs(scores['F'],s,y) for n,s in scores.items()};other=pairs(scores['T_CE'],scores['T_probe'],y);gates=gate(pooled,folds,paired['T_probe']);perfoldpairs={str(k):{n:pairs(scores['F'][fold!=k],s[fold!=k],y[fold!=k]) for n,s in scores.items()} for k in [0,1]}
 result=dict(summary=pooled,per_fold=folds,paired_vs_F=paired,probe_vs_CE_paired=other,per_fold_paired_vs_F=perfoldpairs,gate=gates,decision=gates['decision'],fits=fits,actual_ridge_solves=solves,artifact_bytes=sizes,F_residual=fres,support_stat_max_error=staterr,original_score_parity_max_error=parity,exact_original_count_parity=True,synthetic=tests,method_success=False,bootstrap_performed=False,encoder_forwards=0,mapper_fits=0,parameter_updates=0,development_used=False,future_used=False,test_used=False,not_independent_confirmation=True,source_commit=cfg['source_commit'])
 write(out/'aggregate.json',result)

if __name__=='__main__':
 start=time.monotonic();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());code=1
 try:main(cfg);code=0
 except BaseException as e:
  import traceback
  write(Path(cfg['output'])/'FAILURE.json',dict(error=type(e).__name__));(Path(cfg['output'])/'FAILURE.private.txt').write_text(traceback.format_exc())
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(Path(cfg['output'])/'WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized()))
 raise SystemExit(code)
