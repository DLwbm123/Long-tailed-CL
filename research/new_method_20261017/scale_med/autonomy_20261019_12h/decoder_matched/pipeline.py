"""Matched original pixels -> unchanged modern FP16 block6 -> original G/P/C."""
import os,sys,json,time,resource,subprocess,io
from pathlib import Path
import torch,numpy as np
from core_attribution import weights,norm_fit
from core_depth import PERM,WIDTH,mapped,top,project,nbytes,planned,BUDGET
from evaluation import evaluate,metrics
from space import check_space
ARMS=['G','P','C'];NAMES=[a+'_Matched' for a in ARMS]
def write(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda z:z.item() if isinstance(z,np.generic) else (_ for _ in ()).throw(TypeError(type(z).__name__))));t.replace(p)
def model_for(cfg):
 import timm
 from safetensors.torch import load_file
 bc=json.loads(Path(cfg['benchmark_config']).read_text());torch.manual_seed(74002)
 m=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);m.load_state_dict({k:v for k,v in load_file(bc['weight']).items() if not k.startswith('head.')},strict=True);return m.eval().requires_grad_(False).cuda()
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
 torch.manual_seed(74159);x=torch.randn(32,9,dtype=torch.float64);x=torch.cat([x,torch.ones(32,1,dtype=torch.float64)],1);y=torch.arange(32)%4;q=torch.ones(32,dtype=torch.float64)/8;a=x.T@(x*q[:,None]);b=x.T@(torch.nn.functional.one_hot(y,6).double()*q[:,None]);w,err=fit(a,b,4,.01);assert torch.allclose(a,x[:16].T@(x[:16]*q[:16,None])+x[16:].T@(x[16:]*q[16:,None]))
 # Lossless storage doesn't change coordinates, penalties, solve or predictions.
 ix=torch.triu_indices(len(a),len(a));v=a[ix.unbind()];aa=torch.zeros_like(a);aa[ix.unbind()]=v;aa[(ix[1],ix[0])]=v;assert torch.equal(a,aa);assert torch.allclose(fit(aa,b,4,.01)[0],w)
 yy=torch.arange(60)%6;groups=['synthetic'+str(i) for i in range(60)];s={}
 for n in ['G_ridge','P_ridge','C_ridge']+NAMES:
  for st in range(3):s[n,st]=torch.randn(60,st+4)
 r=evaluate(s,yy,groups,NAMES);assert len(r['comparisons']['C_Matched']['vs_ridge'])==4 and not r['method_success'];json.dumps(r,allow_nan=False)
 assert evaluate({},yy,groups,NAMES)['decision']=='UNRESOLVED_MISSING_CELLS'
 # Deliberately strong synthetic candidate exercises success gate and risks.
 for key in s:
  n,st=key;s[key]=torch.zeros(60,st+4)
  if n=='C_Matched':s[key][torch.arange(60)[yy<st+4],yy[yy<st+4]]=1
 r2=evaluate(s,yy,groups,NAMES);assert r2['gates']['C_Matched']['pass_all'];json.dumps(r2,allow_nan=False)
 m=model_for(cfg);pixels=torch.load(Path(cfg['prior_replay'])/'original/pixels.private.pt',weights_only=False);vals=[];manual=0.
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
  for z in pixels.split(64):
   h=block6(m,z.cuda());capture=[];hook=m.blocks[5].register_forward_hook(lambda mod,args,out:capture.append(out[:,0].float()));m.forward_features(z.cuda());hook.remove();manual=max(manual,float((h-capture[0]).abs().max()));vals.append(h.cpu())
  repeated=block6(m,pixels.cuda()).cpu()
 assert manual<1e-5 and torch.equal(torch.cat(vals),repeated)
 return dict(pass_all=True,original_W_residual=err,incremental_statistics=True,packed_dense_identical=True,three_arm_four_contrast_JSON=True,synthetic_success_failure_missing=True,same_pixels_same_modern_FP16_exact=True,manual_hook_error=manual,encoder_runtime_is_fixed=True)

def main(cfg):
 out=Path(cfg['output']);start=time.monotonic()
 if cfg['mode']=='PREFLIGHT':write(out/'aggregate.json',selfcheck(cfg));return
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('SYNTHETIC_FAILURE')
 assert cfg['mode']=='FORMAL' and json.loads(Path(cfg['preflight_receipt']).read_text())['pass_all']
 ref=Path(cfg['reference']).resolve();cached=Path(cfg['cached_features']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());testroot=Path(bc['output']).resolve();inherited=Path(cfg['inherited_predictions']).resolve();stage=-1;sealed=False
 roots=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  assert not any(p.is_relative_to(r) for r in roots),'learner cannot decode'
  if p.is_relative_to(ref/'inputs'):assert stage<3 and p==ref/'inputs'/f'{stage}.private.pt','retired/future bundle'
  if p.parent==cached and p.name.startswith('stage'):assert p.name==f'stage{stage}.features.private.pt','retired/future mapped inputs'
  if p==inherited or p==cached/'test.features.private.pt' or p.is_relative_to(testroot):assert sealed,'test before seal'
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
  checkpoint('canonical_decode');env=os.environ.copy();env['READ_STAGE']=str(stage);env['CUDA_VISIBLE_DEVICES']='';log=(out/f'reader{stage}.private.log').open('w');p=subprocess.Popen([cfg['reader_python'],cfg['reader_entry']],env=env,stdout=log,stderr=subprocess.STDOUT);write(out/f'READER{stage}_LAUNCH.private.json',dict(pid=p.pid,parent_pid=os.getpid(),argv=[cfg['reader_python'],cfg['reader_entry']]))
  try:code=p.wait()
  finally:log.close()
  root=out/f'pixels_{stage}';receipt=json.loads((root/'EXIT.json').read_text());reader_costs.append(receipt);assert code==0 and receipt['exit_code']==0
  index=json.loads((root/'INDEX.json').read_text());assert index['rows']==expected;values=[]
  for cell in index['chunks']:
   fn=root/cell['file'];z=torch.load(fn,weights_only=False);assert len(z)==cell['rows']
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):h=block6(m,z.cuda()).cpu().double()
   assert torch.isfinite(h).all();values.append(h);extracted+=len(z);checkpoint('FP16_block6',processed=cell['first']+len(z),total=expected)
  return torch.cat(values)
 for stage in range(3):
  checkpoint('load_stage');obj=torch.load(ref/'inputs'/f'{stage}.private.pt',weights_only=False);data=torch.load(cached/f'stage{stage}.features.private.pt',weights_only=False);g=obj['g'];y=obj['y'];gg=obj['groups'];tr=~obj['valid'];va=obj['valid'];tg=[gg[i] for i in torch.where(tr)[0].tolist()];vg={gg[i] for i in torch.where(va)[0].tolist()};assert not set(tg)&(vg|val_groups) and not train_groups&vg;train_groups.update(tg);val_groups|=vg;q=weights(y[tr],tg);assert abs(float(q.sum())-(4 if stage==0 else 1))<1e-10
  h=extract(len(y));counts.append(dict(stage=stage,rows=len(y),training=int(tr.sum()),validation=int(va.sum())));k=stage+4
  if stage==0:
   maps=build(g[tr],h[tr],q);torch.save(maps,out/'transforms.private.pt');write(out/'TRANSFORM_SEAL.json',dict(training_rows=2555,rank=128,width={a:WIDTH[a] for a in ARMS},test_used=False,source_commit=cfg['source_commit']))
  for arm in ARMS:
   x=mapped(g,h,maps[arm]);aa=x[tr].T@(x[tr]*q[:,None]);bb=x[tr].T@(torch.nn.functional.one_hot(y[tr],6).double()*q[:,None]);old=state.get(arm,(torch.zeros_like(aa),torch.zeros_like(bb)));state[arm]=(old[0]+aa,old[1]+bb)
   if arm=='G':
    err=float((x-data['G']).abs().max());assert err<=1e-10;parity[f'G_map.{stage}']=err
   if stage==0 and arm=='G':
    candidates=[]
    for al in [.0001,.01,1.]:
     w,e=fit(*state[arm],k,al);ba=metrics(x[va]@w[:,:k],y[va],[gg[i] for i in torch.where(va)[0].tolist()],k)['BA'];fits.append(dict(arm=arm,stage=0,alpha=al,residual=e,status='CERTIFIED',initial_validation_BA=ba));candidates.append((ba,al,w))
    _,alpha,w=max(candidates,key=lambda z:(z[0],z[1]));heads[arm,stage]=w;write(out/'SELECTION_SEAL.json',dict(alpha=alpha,source='initial_G_validation_681',grid=[.0001,.01,1.],tie='larger',test_used=False));assert alpha==.01,'G replay selected different regularization'
   else:
    w,e=fit(*state[arm],k,alpha);heads[arm,stage]=w;fits.append(dict(arm=arm,stage=stage,alpha=alpha,residual=e,status='CERTIFIED'))
   aa,bb=state[arm];assert nbytes(state[arm])+nbytes(maps[arm])+x.shape[1]*6*8+64==planned(arm,WIDTH[arm])<=BUDGET
   # Same dimensions and exact dense matrix; packed persistence only avoids serializer overhead.
   ix=torch.triu_indices(len(aa),len(aa));artifact=dict(A=aa[ix.unbind()].contiguous(),B=bb.contiguous(),W=heads[arm,stage].contiguous(),mapper=maps[arm],PERM=PERM,alpha=alpha,classes=k,source_commit=cfg['source_commit']);fn=out/f'{arm}.{stage}.state.private.pt';torch.save(artifact,fn);sizes[f'{arm}.{stage}']=fn.stat().st_size;assert sizes[f'{arm}.{stage}']<=BUDGET
  checkpoint('stage_retired');del obj,data,g,h,x,y,gg;reject_probe(ref/'inputs'/f'{stage-1}.private.pt') if stage else None
 assert len(fits)==11 and len(heads)==9
 write(out/'MODEL_SEAL.json',dict(heads=9,fit_units=11,test_used=False,source_commit=cfg['source_commit'],alpha=alpha,artifact_bytes=sizes,all_transforms_initial_only=True));sealed=True;stage=3
 prior=torch.load(inherited,weights_only=False);testx=torch.load(cached/'test.features.private.pt',weights_only=False);test=torch.load(testroot/'ISIC.test.features.private.pt',weights_only=False);g=torch.nn.functional.normalize(test['x'].double(),dim=1);y=prior['y'];gg=prior['groups'];assert len(y)==586 and not set(gg)&(train_groups|val_groups);h=extract(len(y));scores=dict(prior['scores'])
 for arm in ARMS:
  x=mapped(g,h,maps[arm])
  if arm=='G':parity['G_map.test']=float((x-testx['G']).abs().max());assert parity['G_map.test']<=1e-10
  for st in range(3):
   scores[arm+'_Matched',st]=(x@heads[arm,st])[:,:st+4]
   if arm=='G':
    err=float((scores['G_Matched',st]-scores['G_ridge',st]).abs().max());assert err<=1e-8;parity[f'G_score.{st}']=err
 assert extracted==6112
 torch.save(dict(scores=scores,y=y,groups=gg),out/'predictions.private.pt');r=evaluate(scores,y,gg,NAMES);r.update(source_commit=cfg['source_commit'],fits=fits,counts=counts,artifact_bytes=sizes,parity=parity,feature_rows=extracted,reader_costs=reader_costs,historical_test_exposed=True,patient_isolation='UNKNOWN',engineering_comparison_not_novel_algorithm=True);write(out/'aggregate.json',r);checkpoint('complete');write(out/'status.json',dict(state='COMPLETE',decision=r['decision']))
