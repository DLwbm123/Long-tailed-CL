"""Frozen initial bias encoder -> original G/P/C incremental sufficient statistics."""
import os,sys,json,time,resource,subprocess,io
from pathlib import Path
import torch,numpy as np
from core_attribution import weights,norm_fit
from core_depth import PERM,WIDTH,mapped,top,project,nbytes,planned,BUDGET
from evaluation import evaluate,metrics
from space import check_space
ARMS=['G','P','C'];NAMES=[a+'_Bias' for a in ARMS]
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else (_ for _ in ()).throw(TypeError(type(z).__name__))));t.replace(p)
def model_for(cfg):
 import timm
 from safetensors.torch import load_file
 bc=json.loads(Path(cfg['benchmark_config']).read_text());torch.manual_seed(74002)
 m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);m.load_state_dict({k:v for k,v in load_file(bc['weight']).items() if not k.startswith('head.')},strict=True);m=m.eval().requires_grad_(False).cuda()
 if cfg['mode']=='FIT':
  state=torch.load(Path(cfg['training_output'])/'encoder.private.pt',weights_only=False);ps=dict(m.named_parameters())
  with torch.no_grad():
   for k,v in state['bias_overrides'].items():assert ps[k].shape==v.shape;ps[k].copy_(v.cuda())
 return m
def block6(m,x):
 x=m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
 for b in m.blocks[:6]:x=b(x)
 return x[:,0].float()
def fit(a,b,k,alpha):
 h=a/k;h=h.clone();h.diagonal()[:-1].add_(alpha);w=torch.linalg.solve(h,b/k);err=float((h@w-b/k).abs().max());assert err<=1e-7 and torch.isfinite(w).all();return w,err
def build(g,h,q):
 gn=norm_fit(g,q);hn=norm_fit(h,q);gg=(g-gn[0])/gn[1];hh=(h-hn[0])/hn[1];ts={}
 for a in ARMS:
  d=WIDTH[a];t=dict(arm=a,d=d,g_norm=gn)
  if a!='G':t['h_norm']=hn
  if a=='P':t['U']=top(hh.T@(hh*q[:,None])/4)
  if a=='C':
   core=gg[:,PERM[:d]];aa=core.T@(core*q[:,None])/4;aa.diagonal().add_(.01);v=torch.linalg.solve(aa.cuda(),(core.T@(hh*q[:,None])/4).cuda()).cpu();e=hh-core@v;t['U']=top(e.T@(e*q[:,None])/4);t['V']=v@t['U']
  t['out_norm']=norm_fit(project(gg,hh,t),q);ts[a]=t
 return ts

def selfcheck(cfg):
 torch.manual_seed(74163);x=torch.cat([torch.randn(24,8,dtype=torch.float64),torch.ones(24,1,dtype=torch.float64)],1);y=torch.arange(24)%4;q=torch.ones(24,dtype=torch.float64)/6;a=x.T@(x*q[:,None]);b=x.T@(torch.nn.functional.one_hot(y,6).double()*q[:,None]);w,e=fit(a,b,4,.01)
 assert torch.allclose(a,x[:12].T@(x[:12]*q[:12,None])+x[12:].T@(x[12:]*q[12:,None]))
 yy=torch.arange(60)%6;gg=['synthetic'+str(i) for i in range(60)];scores={}
 for n in ['G_Matched','P_Matched','C_Matched']+NAMES:
  for st in range(3):scores[n,st]=torch.zeros(60,st+4)
 failed=evaluate(scores,yy,gg,NAMES);assert not failed['gates']['pass_all'];write(Path(cfg['output'])/'synthetic_fail.json',failed)
 for st in range(3):
  scores['G_Bias',st][torch.where(yy<st+4)[0],yy[yy<st+4]]=1
 passed=evaluate(scores,yy,gg,NAMES);assert passed['gates']['pass_all'];write(Path(cfg['output'])/'synthetic_pass.json',passed);assert evaluate({},yy,gg,NAMES)['decision']=='UNRESOLVED_INITIAL_BIAS'
 m=model_for(cfg);proof=torch.load(Path(cfg['preflight_training']).parent/'preflight_encoder.private.pt',weights_only=False);params=dict(m.named_parameters());assert all(k in params and params[k].shape==v.shape for k,v in proof['bias_overrides'].items())
 with torch.no_grad():
  for k,v in proof['bias_overrides'].items():params[k].copy_(v.cuda())
 assert all(torch.equal(params[k].detach().cpu(),v) for k,v in proof['bias_overrides'].items())
 return dict(pass_all=True,three_arm_six_comparison_success_failure_missing_JSON=True,actual_trained_override_binding=True,incremental_statistics=True,system_residual=e)

def main(cfg):
 out=Path(cfg['output']);start=time.monotonic()
 if cfg['mode']=='EVAL_PREFLIGHT':write(out/'aggregate.json',selfcheck(cfg));return
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('SYNTHETIC_FAILURE')
 assert cfg['mode']=='FIT' and json.loads(Path(cfg['preflight_receipt']).read_text())['pass_all']
 seal=json.loads((Path(cfg['training_output'])/'ENCODER_SEAL.json').read_text());assert seal['bias_training_exposures']==25550 and seal['headonly_training_exposures']==25550 and seal['future_frozen'] and seal['source_commit']==cfg['source_commit']
 ref=Path(cfg['reference']).resolve();matched=Path(cfg['matched']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());testroot=Path(bc['output']).resolve();inherited=Path(cfg['inherited_predictions']).resolve();stage=-1;sealed=False
 roots=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  assert not any(p.is_relative_to(r) for r in roots),'learner cannot decode'
  if p.is_relative_to(ref/'inputs'):assert stage<3 and p==ref/'inputs'/f'{stage}.private.pt','retired/future bundle'
  if p.is_relative_to(matched) and p!=inherited:assert p.is_relative_to(matched/f'pixels_{stage}'),'retired/future pixels'
  if p==inherited or p.is_relative_to(testroot):assert sealed,'test before seal'
  for s in range(4):
   if p.is_relative_to(out/('pixels_'+str(s))):assert stage==s,'retired/future pixels'
 sys.addaudithook(guard)
 def reject_probe(path):
  try:guard('open',(str(path),'rb'))
  except AssertionError:return
  raise AssertionError('isolation probe failed')
 reject_probe(inherited);reject_probe(testroot/'ISIC.test.rows.private.json')
 m=model_for(cfg);fits=[];heads={};state={};maps=None;alpha=None;parity={};sizes={};reader_costs=[];counts=[];train_groups=set();val_groups=set();extracted=0
 def checkpoint(phase,**kw):
  check_space(cfg['window_root']);write(out/'status.json',dict(state='RUNNING',phase=phase,stage=stage,elapsed_seconds=time.monotonic()-start,**kw));write(out/'partial.json',dict(fits=fits,counts=counts,artifact_bytes=sizes,parity=parity,reader_costs=reader_costs,feature_rows=extracted,alpha=alpha))
 def extract(expected):
  nonlocal extracted
  checkpoint('original_g_features');feature=out/f'features_{stage}';feature.mkdir();cc=dict(cfg,mode='FEATURE',output=str(feature));cf=out/f'FEATURE{stage}.config.private.json';write(cf,cc);env=os.environ.copy();env.update(RUN_CONFIG=str(cf),FEATURE_STAGE=str(stage));log=(out/f'features{stage}.private.log').open('w');p=subprocess.Popen([cfg['old_python'],cfg['entry']],env=env,stdout=log,stderr=subprocess.STDOUT);write(out/f'FEATURE{stage}_LAUNCH.private.json',dict(pid=p.pid,parent_pid=os.getpid(),argv=[cfg['old_python'],cfg['entry']]))
  try:code=p.wait()
  finally:log.close()
  receipt=json.loads((feature/'EXIT.json').read_text());reader_costs.append(json.loads((feature/'WORKER_RESOURCE_RECEIPT.json').read_text()));assert code==0 and receipt['exit_code']==0
  g=torch.load(feature/'g.private.pt',weights_only=False);assert len(g)==expected
  index=json.loads((matched/f'pixels_{stage}/INDEX.json').read_text());assert index['rows']==expected;values=[]
  for cell in index['chunks']:
   z=torch.load(matched/f'pixels_{stage}'/cell['file'],weights_only=False)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):h=block6(m,z.cuda()).cpu().double()
   assert torch.isfinite(h).all();values.append(h);extracted+=len(z);checkpoint('FP16_block6',processed=cell['first']+len(z),total=expected)
  return g,torch.cat(values)
 for stage in range(3):
  checkpoint('load_stage');obj=torch.load(ref/'inputs'/f'{stage}.private.pt',weights_only=False);y=obj['y'];gg=obj['groups'];tr=~obj['valid'];va=obj['valid'];tg=[gg[i] for i in torch.where(tr)[0].tolist()];vg={gg[i] for i in torch.where(va)[0].tolist()};assert not set(tg)&(vg|val_groups) and not train_groups&vg;train_groups.update(tg);val_groups|=vg;q=weights(y[tr],tg);assert abs(float(q.sum())-(4 if stage==0 else 1))<1e-10
  g,h=extract(len(y));counts.append(dict(stage=stage,rows=len(y),training=int(tr.sum()),validation=int(va.sum())));k=stage+4
  if stage==0:
   maps=build(g[tr],h[tr],q);torch.save(maps,out/'transforms.private.pt');write(out/'TRANSFORM_SEAL.json',dict(training_rows=2555,rank=128,width={a:WIDTH[a] for a in ARMS},test_used=False,source_commit=cfg['source_commit']))
  for arm in ARMS:
   x=mapped(g,h,maps[arm]);aa=x[tr].T@(x[tr]*q[:,None]);aa=(aa+aa.T)/2;bb=x[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]);old=state.get(arm,(torch.zeros_like(aa),torch.zeros_like(bb)));state[arm]=(old[0]+aa,old[1]+bb)
   if stage==0 and arm=='G':
    candidates=[]
    for al in [.0001,.01,1.]:
     w,e=fit(*state[arm],k,al);ba=metrics(x[va]@w[:,:k],y[va],[gg[i] for i in torch.where(va)[0].tolist()],k)['BA'];fits.append(dict(arm=arm,stage=0,alpha=al,residual=e,status='CERTIFIED',initial_validation_BA=ba));candidates.append((ba,al,w))
    _,alpha,w=max(candidates,key=lambda z:(z[0],z[1]));heads[arm,stage]=w;write(out/'SELECTION_SEAL.json',dict(alpha=alpha,source='initial_G_validation_681',grid=[.0001,.01,1.],tie='larger',test_used=False))
   else:
    w,e=fit(*state[arm],k,alpha);heads[arm,stage]=w;fits.append(dict(arm=arm,stage=stage,alpha=alpha,residual=e,status='CERTIFIED'))
   aa,bb=state[arm];assert maps[arm]['d']==WIDTH[arm]
   # Same dimensions and exact dense matrix; packed persistence only avoids serializer overhead.
   ix=torch.triu_indices(len(aa),len(aa));artifact=dict(A=aa[ix.unbind()].contiguous(),B=bb.contiguous(),W=heads[arm,stage].contiguous(),mapper=maps[arm],encoder_bias_overrides=torch.load(Path(cfg['training_output'])/'encoder.private.pt',weights_only=False)['bias_overrides'],bias_representation='absolute_overrides',PERM=PERM,alpha=alpha,classes=k,source_commit=cfg['source_commit']);fn=out/f'{arm}.{stage}.state.private.pt';torch.save(artifact,fn);sizes[f'{arm}.{stage}']=fn.stat().st_size;assert sizes[f'{arm}.{stage}']<=BUDGET
   loaded=torch.load(fn,weights_only=False);recovered=torch.zeros_like(aa);recovered[ix.unbind()]=loaded['A'];recovered[(ix[1],ix[0])]=loaded['A'];assert torch.equal(aa,recovered) and torch.equal(loaded['W'],heads[arm,stage]);rw,re=fit(recovered,loaded['B'],k,alpha);assert float((rw-heads[arm,stage]).abs().max())<=1e-10
  checkpoint('stage_retired');del obj,g,h,x,y,gg;reject_probe(ref/'inputs'/f'{stage-1}.private.pt') if stage else None
 assert len(fits)==11 and len(heads)==9
 write(out/'MODEL_SEAL.json',dict(heads=9,fit_units=11,test_used=False,source_commit=cfg['source_commit'],alpha=alpha,artifact_bytes=sizes,all_transforms_initial_only=True));sealed=True;stage=3
 prior=torch.load(inherited,weights_only=False);y=prior['y'];gg=prior['groups'];assert len(y)==586 and not set(gg)&(train_groups|val_groups)
 rows=json.loads((testroot/'ISIC.test.rows.private.json').read_text());assert torch.equal(y,torch.tensor([r['label'] for r in rows])) and gg==[r['identity_component'] for r in rows]
 g,h=extract(len(y));scores=dict(prior['scores'])
 for arm in ARMS:
  x=mapped(g,h,maps[arm])
  for st in range(3):scores[arm+'_Bias',st]=(x@heads[arm,st])[:,:st+4]
 assert extracted==6112
 torch.save(dict(scores=scores,y=y,groups=gg),out/'predictions.private.pt');r=evaluate(scores,y,gg,NAMES);r.update(source_commit=cfg['source_commit'],fits=fits,counts=counts,artifact_bytes=sizes,parity=parity,feature_rows=extracted,reader_costs=reader_costs,historical_test_exposed=True,patient_isolation='UNKNOWN',engineering_comparison_not_novel_algorithm=True);write(out/'aggregate.json',r);checkpoint('complete');write(out/'status.json',dict(state='COMPLETE',decision=r['decision']))
