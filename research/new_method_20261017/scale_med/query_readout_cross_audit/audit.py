"""CPU algebra on sealed group-excluded query features and heads."""
import json,os,time,resource
from pathlib import Path
import torch

torch.set_num_threads(4)
NAMES=['F_on_F','F_on_T','T_on_F','T_on_T']
def transform(g,mp):return (((g-mp['gn'][0])/mp['gn'][1])[:,mp['ids']]-mp['on'][0])/mp['on'][1]
def same_mapper(a,b):return torch.equal(a['ids'],b['ids']) and all(torch.equal(x,y) for k in ['gn','on'] for x,y in zip(a[k],b[k]))
def stats(v):
 if len(v)==0:return dict(count=0,mean=None,quantiles=None)
 assert torch.isfinite(v).all()
 return dict(count=len(v),mean=float(v.mean()),quantiles=torch.quantile(v,torch.tensor([0.,.1,.5,.9,1.],dtype=v.dtype)).tolist())
def points(s,y):
 pred=s.argmax(1);correct=[int((pred[y==c]==c).sum()) for c in range(4)];counts=[int((y==c).sum()) for c in range(4)];rec=[v/n for v,n in zip(correct,counts)]
 other=s.clone();other[torch.arange(len(y)),y]=-torch.inf;m=s[torch.arange(len(y)),y]-other.max(1).values
 return dict(BA=sum(rec)/4,correct=correct,counts=counts,recall=rec,classes=[dict(class_index=c,margin=stats(m[y==c]),CE=stats(torch.nn.functional.cross_entropy(s[y==c],y[y==c],reduction='none'))) for c in range(4)])
def pairs(base,pred,y):
 b,p=base.argmax(1),pred.argmax(1);rows=[]
 for c in range(4):
  ids=y==c;bo,po=b[ids]==c,p[ids]==c;r,h=int((~bo & po).sum()),int((bo & ~po).sum());assert r-h==int(po.sum())-int(bo.sum())
  rows.append(dict(class_index=c,rescued=r,harmed=h,both_correct=int((bo & po).sum()),both_wrong=int((~bo & ~po).sum()),harmed_destination=torch.bincount(p[ids][bo & ~po],minlength=4).tolist()))
 return rows
def algebra(zf,zt,wf,wt,y):
 sf=zf@wf[:-1]+wf[-1];st=zt@wt[:-1]+wt[-1];mid=zt@wf[:-1]+wf[-1];rev=zf@wt[:-1]+wt[-1];err=0.
 for c in range(4):
  for j in range(4):
   if c==j:continue
   a,b,d,e=[s[:,c]-s[:,j] for s in [sf,mid,rev,st]]
   interaction=(zt-zf)@((wt[:-1,c]-wt[:-1,j])-(wf[:-1,c]-wf[:-1,j]))
   err=max(err,float(((e-a)-(b-a)-(e-b)).abs().max()),float(((e-a)-(d-a)-(e-d)).abs().max()),float(((e-b-d+a)-interaction).abs().max()))
 assert err<=1e-8;return err

def synthetic():
 gen=torch.Generator().manual_seed(74293);zf=torch.randn(32,7,dtype=torch.float64,generator=gen);zt=torch.randn(32,7,dtype=zf.dtype,generator=gen);wf=torch.randn(8,4,dtype=zf.dtype,generator=gen);wt=torch.randn(8,4,dtype=zf.dtype,generator=gen);y=torch.arange(32)%4
 err=algebra(zf,zt,wf,wt,y);scores=[z@w[:-1]+w[-1] for z,w in [(zf,wf),(zt,wf),(zf,wt),(zt,wt)]]
 pairs(scores[0],scores[3],y)
 group=torch.arange(32)//2;fold=group%2;assert ((fold==0).int()+(fold==1).int()==1).all() and not set(group[fold==0].tolist()) & set(group[fold==1].tolist())
 json.dumps([points(s,y) for s in scores],allow_nan=False)
 return dict(algebra_error=err,group_query_coverage=True,paired_identity_and_finite_JSON=True)

def main(cfg):
 tests=synthetic();old=Path(cfg['sealed_root']);out=Path(cfg['output'])
 allowed=[old/'prep/PLAN.private.pt',old/'evaluate/responses.private.pt']
 for k in [0,1]:allowed.extend([old/'prep'/f'{k}.teacher.private.pt',old/'evaluate'/f'{k}.state.private.pt',old/'evaluate'/f'{k}.g.private.pt'])
 allowed={p.resolve() for p in allowed};original=torch.load
 def load(p):
  assert Path(p).resolve() in allowed,'PRIVATE_ASSET_READ_DENIED'
  return original(p,map_location='cpu',weights_only=False)
 torch.load=load
 assert json.loads((old/'train/ENCODER_SEAL.json').read_text())['query_excluded_entire_producer']
 assert json.loads((old/'evaluate/MODEL_SEAL.json').read_text())['two_teacher_fits']==2
 expected=json.loads((old/'evaluate/aggregate.json').read_text())
 plan=load(old/'prep/PLAN.private.pt');previous=load(old/'evaluate/responses.private.pt');y=plan['y'];fold=plan['fold'];groups=plan['groups']
 assert len(y)==2555 and torch.equal(y,previous['y']) and groups==previous['groups']
 score={n:torch.zeros(2555,4,dtype=torch.float64) for n in NAMES};cover=torch.zeros(2555,dtype=torch.long);perfold={};decomp={};paths={};parity=0.;err=0.;units=0
 for k in [0,1]:
  mask=fold!=k;ids=torch.where(mask)[0];cover[mask]+=1
  assert not {groups[i] for i in ids} & {groups[i] for i in torch.where(~mask)[0]}
  seal=json.loads((old/'evaluate'/f'{k}.FEATURE_SEAL.json').read_text());assert seal['rows']==2555 and seal['one_pass']
  teacher=load(old/'prep'/f'{k}.teacher.private.pt');artifact=load(old/'evaluate'/f'{k}.state.private.pt');assert same_mapper(teacher['mapper'],artifact['mapper'])
  gf=plan['cached_g'][mask];gt=load(old/'evaluate'/f'{k}.g.private.pt')[mask];mp=teacher['mapper'];zf,zt=transform(gf,mp),transform(gt,mp);wf,wt=teacher['W'][:,:4],artifact['W'][:,:4];yy=y[mask]
  local={n:z@w[:-1]+w[-1] for n,z,w in [('F_on_F',zf,wf),('F_on_T',zt,wf),('T_on_F',zf,wt),('T_on_T',zt,wt)]}
  assert all(torch.isfinite(s).all() for s in local.values())
  for n,s in local.items():score[n][mask]=s
  for n,ref in [('F_on_F','F_out'),('T_on_T','T_out')]:parity=max(parity,float((local[n]-previous['score'][ref][mask]).abs().max()))
  assert parity<=1e-8
  err=max(err,algebra(zf,zt,wf,wt,yy));perfold[str(k)]={n:points(s,yy) for n,s in local.items()};rows=[]
  ff=local['F_on_F'];ft=local['F_on_T'];tf=local['T_on_F'];tt=local['T_on_T'];fok=ff.argmax(1)==yy;mok=ft.argmax(1)==yy;tok=tt.argmax(1)==yy
  paths[str(k)]=[]
  for c in range(4):
   cc=yy==c;orig=cc & fok;harm=orig & ~tok
   categories=dict(reference_wrong_final_wrong=int((orig & ~mok & ~tok).sum()),reference_correct_final_wrong=int((orig & mok & ~tok).sum()),reference_wrong_final_correct=int((orig & ~mok & tok).sum()),reference_correct_final_correct=int((orig & mok & tok).sum()))
   assert sum(categories.values())==int(orig.sum())
   paths[str(k)].append(dict(class_index=c,F_original_correct=int(orig.sum()),T_new_harm=int(harm.sum()),fixed_reference_retained=int((orig & mok).sum()),categories=categories))
   for j in range(4):
    if c==j:continue
    m00,m01,m10,m11=[s[:,c]-s[:,j] for s in [ff,ft,tf,tt]]
    fields=dict(F_margin=m00,fixed_reference_representation=m01-m00,replacement_after_T_feature=m11-m01,head_change_on_F_feature=m10-m00,representation_after_T_head=m11-m10,interaction=m11-m01-m10+m00,T_margin=m11)
    rows.append(dict(class_index=c,competitor=j,subsets={label:{n:stats(v[take]) for n,v in fields.items()} for label,take in [('all',cc),('F_original_correct',orig),('T_new_harm',harm)]}));units+=1
  decomp[str(k)]=rows
 assert (cover==1).all() and units==24
 tab={n:points(s,y) for n,s in score.items()}
 for n,ref in [('F_on_F','F_out'),('T_on_T','T_out')]:assert tab[n]['correct']==[v['correct'] for v in expected['summary'][ref]['classes']]
 fc,mc,tc=[tab[n]['correct'][2] for n in ['F_on_F','F_on_T','T_on_T']]
 flags=dict(REFERENCE_RESTORES_CLASS2=bool(mc>=fc and mc>tc),REFERENCE_RESTORES_BA=bool(tab['F_on_T']['BA']>=tab['F_on_F']['BA'] and tab['F_on_T']['BA']>tab['T_on_T']['BA']),FIXED_REFERENCE_CLASS2_HARM=bool(mc<fc))
 flags['COMPLETE_REFERENCE_RESTORATION_POINT_PATTERN']=flags['REFERENCE_RESTORES_CLASS2'] and flags['REFERENCE_RESTORES_BA']
 result=dict(summary=tab,per_fold=perfold,fixed_rival_margin_cells=units,margin_decomposition=decomp,F_reference_paths=paths,
  paired_vs_F_on_F={n:pairs(score['F_on_F'],s,y) for n,s in score.items()},
  descriptive_flags=flags,parity_score_max_error=parity,algebra_max_error=err,exact_original_endpoint_count_parity=True,
  synthetic=tests,common_mapper_exact=True,query_coverage_exactly_once=True,decision='DESCRIPTIVE_QUERY_READOUT_CROSS_AUDIT_COMPLETE',
  method_success=False,not_path_causal_identification=True,bootstrap_performed=False,encoder_forwards=0,new_fits=0,parameter_updates=0,
  development_used=False,test_used=False,future_used=False,source_commit=cfg['source_commit'])
 (out/'aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
 start=time.monotonic();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());code=1
 try:main(cfg);code=0
 except BaseException as e:
  import traceback
  Path(cfg['output'],'FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__)))
  Path(cfg['output'],'FAILURE.private.txt').write_text(traceback.format_exc())
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);Path(cfg['output'],'WORKER_RESOURCE_RECEIPT.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized())))
 raise SystemExit(code)
