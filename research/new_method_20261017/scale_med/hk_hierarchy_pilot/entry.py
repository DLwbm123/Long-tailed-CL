"""Frozen CLIP feature pilot: true, flat and shuffled hierarchy, nine sealed models."""
import os,sys,json,time,resource,traceback,csv,math
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from core import NAMES,PARENTS,TYPE,SUB,SEEDS,permutation,weights,metrics,paired,Adapter,loss,selfcheck
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
counts=dict(image_encoder_rows=0,image_encoder_calls=0,text_encoder_rows=0,text_encoder_calls=0,backward=0,optimizer_steps=0,ridge_solves=0)
allowed=set();query_allowed=False

def write(n,v):
 p=out/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def status(**v):write('STATUS.json',v);write('ACTUAL_COUNTS.json',counts)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def rows(role):return [r for r in json.loads((Path(cfg['prior_root'])/f'HK.{role}.rows.private.json').read_text()) if r['label']<13]
def paths(rr):
 root=Path(cfg['image_root']).resolve();pp=[]
 for r in rr:
  p=(root/r['relative_path']).resolve();assert p.is_relative_to(root);pp.append(p)
 return pp

def encode(rr,role):
 global allowed
 if role!='S_old':assert query_allowed and (out/'MODEL_SEAL.json').exists()
 pp=paths(rr);allowed=set(pp);parts=[]
 with torch.no_grad():
  for start in range(0,len(pp),32):
   xx=[]
   for p in pp[start:start+32]:
    with Image.open(p) as im:xx.append(preprocess(im.convert('RGB')))
   z=F.normalize(backbone.encode_image(torch.stack(xx).cuda()).float(),dim=1).cpu()
   assert torch.isfinite(z).all();parts.append(z);counts['image_encoder_rows']+=len(z);counts['image_encoder_calls']+=1
   status(phase='ENCODING',role=role,done=min(start+32,len(pp)),total=len(pp))
 return torch.cat(parts)

def ridge_fit(x,y,gs):
 q=weights(y,gs);x=x.double();center=q@x;scale=((x-center).square()*q[:,None]).sum().sqrt();z=(x-center)/scale
 xx=torch.cat([z,torch.ones(len(x),1,dtype=x.dtype)],1);A=xx.T@(xx*q[:,None]);A=(A+A.T)/2;B=xx.T@(F.one_hot(y,13).double()*q[:,None]);A.diagonal()[:-1].add_(.01)
 W=torch.linalg.solve(A,B);counts['ridge_solves']+=1;err=float((A@W-B).abs().max());assert err<1e-7
 return dict(W=W,center=center,scale=scale,residual=err)
def ridge_score(x,r):return ((x.double()-r['center'])/r['scale'])@r['W'][:-1]+r['W'][-1]
def frozen():assert all(not p.requires_grad and p._version==versions[n] for n,p in backbone.named_parameters())
def oldscore(g,obj):
 m=obj['mapper'];z=(((g-m['gn'][0])/m['gn'][1])[:,m['ids']]-m['on'][0])/m['on'][1]
 return z@obj['W'][:-1]+obj['W'][-1]

def main():
 global backbone,preprocess,versions,query_allowed,allowed
 write('SYNTHETIC.json',selfcheck())
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 train=rows('S_old');assert len(train)==3140
 yy=torch.tensor([r['label'] for r in train]);gg=[r['identity_component'] for r in train]
 index={r['relative_path']:r for r in csv.DictReader((Path(cfg['manifest_root'])/'HK/train.csv').open())}
 names=[set() for _ in range(13)]
 for r in train:names[r['label']].add(index[r['relative_path']]['official_class_name'].replace('-',' '))
 assert [next(iter(n)) for n in names if len(n)==1]==NAMES
 q=weights(yy,gg)
 seenp=set(paths(train));seeng=set(gg);metadata={}
 for role,n in [('E',986),('Qgate',388)]:
  rr=rows(role);pp=set(paths(rr));gs={r['identity_component'] for r in rr}
  assert len(rr)==n and not pp&seenp and not gs&seeng;seenp|=pp;seeng|=gs;metadata[role]=rr
 root=Path(cfg['image_root']).resolve()
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(root):assert p in allowed,'IMAGE_OUTSIDE_CURRENT_AUTHORIZED_SPLIT'
 sys.addaudithook(guard)
 write('PLAN_SEAL.json',dict(source_commit=cfg['source_commit'],class_names=NAMES,parent_names=PARENTS,type=TYPE,sub=SUB,seeds=SEEDS,shuffle={str(s):permutation(s) for s in SEEDS},support_rows=len(train),support_group_counts=[len({gg[i] for i in torch.where(yy==c)[0].tolist()}) for c in range(13)],query_model_selection=False))
 import open_clip
 status(phase='LOADING_BACKBONE')
 backbone,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',precision='fp32',device='cuda',cache_dir=str(out/'assets'))
 backbone.eval().requires_grad_(False);versions={n:p._version for n,p in backbone.named_parameters()}
 tokenizer=open_clip.get_tokenizer('ViT-B-32')
 prompts=['a photo of a '+n+'.' for n in NAMES]+['an endoscopic image of '+n+'.' for n in PARENTS]
 with torch.no_grad():text=F.normalize(backbone.encode_text(tokenizer(prompts).cuda()).float(),dim=1)
 assert text.shape==(20,512) and torch.isfinite(text).all();counts['text_encoder_rows']=20;counts['text_encoder_calls']=1
 write('TEXT_SEAL.json',dict(prompts=prompts,rows=20,normalized=True))
 x=encode(train,'S_old');assert x.shape==(3140,512)
 # Each class prototype averages group means, so duplicated images do not increase its weight.
 prot=[]
 for c in range(13):
  groups=sorted(set(gg[i] for i in torch.where(yy==c)[0].tolist()))
  prot.append(torch.stack([x[torch.tensor([g==key for g in gg])].mean(0) for key in groups]).mean(0))
 proto=F.normalize(torch.stack(prot),dim=1);ridge=ridge_fit(x,yy,gg)
 torch.save(dict(proto=proto,ridge=ridge,text=text.cpu()),out/'BASELINES.private.pt')
 assert same(load(out/'BASELINES.private.pt'),dict(proto=proto,ridge=ridge,text=text.cpu()))
 x=x.cuda();y=yy.cuda();allseals=[]
 for seed in SEEDS:
  orders=torch.multinomial(q,600*64,replacement=True,generator=torch.Generator().manual_seed(seed)).reshape(600,64)
  for arm in ['flat','true','shuffled']:
   model=Adapter().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
   perm=permutation(seed) if arm=='shuffled' else list(range(13));trace=[]
   for step,ids in enumerate(orders):
    opt.param_groups[0]['lr']=.001*.5*(1+math.cos(math.pi*step/600))
    opt.zero_grad(set_to_none=True);val,leaf,h=loss(model,x[ids],y[ids],text,arm,perm)
    assert torch.isfinite(val);val.backward();counts['backward']+=1
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));opt.step();counts['optimizer_steps']+=1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    if step%50==0 or step==599:
     trace.append(dict(step=step+1,loss=float(val.detach()),leaf=float(leaf.detach()),hierarchy=float(h.detach()),gradient_norm=gn));write(f'{seed}.{arm}.TRACE.json',trace)
     status(phase='TRAIN',seed=seed,arm=arm,step=step+1,total=600)
   art=dict(state={n:p.detach().cpu() for n,p in model.state_dict().items()},seed=seed,arm=arm,perm=perm,steps=600,source_commit=cfg['source_commit'])
   path=out/f'{seed}.{arm}.model.private.pt';torch.save(art,path);back=load(path);assert same(art,back)
   restored=Adapter().cuda();restored.load_state_dict(back['state'])
   with torch.no_grad():err=float((model.logits(x[:32],text)-restored.logits(x[:32],text)).abs().max())
   assert err<=1e-6;frozen();seal=dict(seed=seed,arm=arm,steps=600,parameters=524288,artifact_bytes=path.stat().st_size,readback_exact=True,score_readback_maxabs=err,backbone_unchanged=True)
   allseals.append(seal);write('MODEL_RECEIPTS.json',allseals);del model,restored,opt;torch.cuda.empty_cache()
 assert counts['optimizer_steps']==5400
 write('MODEL_SEAL.json',dict(models=9,all_sealed=True,query_pending=True,counts=dict(counts),source_commit=cfg['source_commit']))
 query_allowed=True;results={};saved={};ys=[];gsall=[];pooled={};old=load(Path(cfg['window_root'])/'group_consistency_pilot/HK.FCR.artifact.private.pt')
 for role in ['E','Qgate']:
  rr=metadata[role];qy=torch.tensor([r['label'] for r in rr]);qgs=[r['identity_component'] for r in rr];z=encode(rr,role)
  scores=dict(NCM=10*z@proto.T,Ridge=ridge_score(z,ridge),ZS=10*z@text[:13].cpu().T)
  of=load(Path(cfg['prior_root'])/f'HK.{role}.features.private.pt');keep=of['y']<13;assert torch.equal(of['y'][keep],qy)
  scores['OriginalF']=oldscore(F.normalize(of['x'][keep].double(),dim=1),old)
  assert metrics(scores['OriginalF'],qy,qgs)['correct']==cfg['HK_expected'][role]['correct']
  for seed in SEEDS:
   for arm in ['flat','true','shuffled']:
    model=Adapter().cuda();model.load_state_dict(load(out/f'{seed}.{arm}.model.private.pt')['state']);model.eval()
    with torch.no_grad():logits=model.logits(z.cuda(),text)[:,:13].cpu();scores[f'{seed}.{arm}']=.5*(logits.softmax(1)+scores['NCM'].softmax(1))
    del model
  frozen();tab={k:metrics(s,qy,qgs) for k,s in scores.items()}
  pair={str(seed):{a:paired(scores[f'{seed}.{a}'],scores[f'{seed}.true'],qy) for a in ['flat','shuffled']} for seed in SEEDS}
  results[role]=dict(summary=tab,paired=pair);saved[role]=scores;ys.append(qy);gsall+=qgs
  for k,v in scores.items():pooled.setdefault(k,[]).append(v)
  write(role+'.json',results[role])
 y=torch.cat(ys);ps={k:torch.cat(v) for k,v in pooled.items()};pt={k:metrics(v,y,gsall) for k,v in ps.items()}
 comparisons={}
 for a in ['flat','shuffled']:
  ds=[pt[f'{s}.true']['BA']-pt[f'{s}.{a}']['BA'] for s in SEEDS]
  comparisons[a]=dict(seed_BA_delta=ds,mean_BA_delta=sum(ds)/3,positive_seeds=sum(d>0 for d in ds),cohort_mean_BA_delta={r:sum(results[r]['summary'][f'{s}.true']['BA']-results[r]['summary'][f'{s}.{a}']['BA'] for s in SEEDS)/3 for r in results})
 module_gate=all(v['mean_BA_delta']>=.01 and v['positive_seeds']>=2 and all(d>=-1e-12 for d in v['cohort_mean_BA_delta'].values()) for v in comparisons.values())
 safety=all(results[r]['summary'][f'{seed}.true']['recall'][c]>=results[r]['summary'][base]['recall'][c]-.01-1e-12 for r in results for seed in SEEDS for base in [f'{seed}.flat',f'{seed}.shuffled','OriginalF'] for c in range(13))
 torch.save(saved,out/'responses.private.pt')
 write('aggregate.json',dict(decision='HIERARCHY_PILOT_POINT_SIGNAL' if module_gate and safety else 'NO_SAFE_HIERARCHY_POINT_SIGNAL',module_point_gate=module_gate,allclass_safety_gate=safety,method_success=False,scientific_complete=True,comparisons=comparisons,summary=pt,cohorts=results,counts=counts,source_commit=cfg['source_commit'],historically_exposed=True,patient_independence='UNKNOWN',bootstrap=False,CI=None,hasten_reproduction=False,incremental_forgetting_test=False))
 status(phase='COMPLETE')

start=time.time();code=1
try:main();code=0
except BaseException as e:
 write('FAILURE.json',dict(type=type(e).__name__,scientific_complete=False));(out/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED')
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write('WORKER_EXIT.json',dict(exit_code=code));write('ACTUAL_COUNTS.json',counts)
raise SystemExit(code)
