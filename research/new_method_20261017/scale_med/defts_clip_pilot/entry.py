"""Frozen CLIP, known-group DEFTS pilot; all choices in sealed protocol."""
import os,json,time,resource,traceback,sys,csv
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from PIL import Image

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);torch.set_num_threads(4)
load=torch.load
counts=dict(image_encoder_calls=0,image_encoder_rows=0,text_encoder_calls=0,text_encoder_rows=0,ridge_solves=0,precision_inverses=0,posterior_solves=0,completed_EB_fits=0)

def write(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def phase(n,**kw):write('STATUS.json',dict(phase=n,**kw))
def weights(y,gs,K):
 size=Counter(gs);nc=[len({gs[i] for i in torch.where(y==c)[0]}) for c in range(K)];assert all(nc)
 return torch.tensor([1/(K*nc[int(c)]*size[g]) for c,g in zip(y,gs)],dtype=torch.float64)
def points(s,y,gs,K):
 pred=s.argmax(1);n=torch.bincount(y,minlength=K);assert (n>0).all();ok=[int((pred[y==c]==c).sum()) for c in range(K)];rec=[v/int(t) for v,t in zip(ok,n)]
 return dict(BA=sum(rec)/K,correct=ok,counts=n.tolist(),recall=rec,group_BA=float(weights(y,gs,K)@(pred==y).double()),confusion=torch.bincount(y*K+pred,minlength=K*K).reshape(K,K).tolist())
def rh(a,b,y,K):
 aa=a.argmax(1)==y;bb=b.argmax(1)==y
 return [dict(class_id=c,rescued=int((~aa&bb&(y==c)).sum()),harmed=int((aa&~bb&(y==c)).sum())) for c in range(K)]
def mapold(g,m):return (((g-m['gn'][0])/m['gn'][1])[:,m['ids']]-m['on'][0])/m['on'][1]
def oldscore(g,m):return mapold(g,m['mapper'])@m['W'][:-1]+m['W'][-1]

def encode(rows,root,tag):
 root=Path(root);values=[]
 for k in range(0,len(rows),32):
  ims=[]
  for r in rows[k:k+32]:
   rel=Path(r['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
   with Image.open(root/rel) as im:ims.append(preprocess(im.convert('RGB')))
  with torch.no_grad():z=model.encode_image(torch.stack(ims).cuda()).float();z=F.normalize(z,dim=1)
  assert torch.isfinite(z).all();values.append(z.cpu().double());counts['image_encoder_calls']+=1;counts['image_encoder_rows']+=len(z);write('ACTUAL_COUNTS.json',counts);phase('encoding',dataset=tag,done=min(k+32,len(rows)),total=len(rows))
 v=torch.cat(values);assert v.shape==(len(rows),512)
 torch.save(dict(x=v,y=torch.tensor([r['label'] for r in rows]),groups=[r['identity_component'] for r in rows]),out/(tag+'.features.private.pt'))
 return v

def fit(g,y,gs,text,Q,tag):
 K=len(text);d=g.shape[1];q=weights(y,gs,K);by={}
 for i,key in enumerate(gs):by.setdefault(key,[]).append(i)
 obs=[];labels=[]
 for key,ii in by.items():
  assert len(set(y[ii].tolist()))==1;obs.append(g[ii].mean(0));labels.append(int(y[ii[0]]))
 u=torch.stack(obs);uy=torch.tensor(labels);G=torch.bincount(uy,minlength=K).double();nu=int((G-1).sum());assert nu>0
 mu=torch.stack([u[uy==c].mean(0) for c in range(K)]);delta=u-mu[uy];rawS=delta.T@delta;S=Q.T@rawS@Q;S=(S+S.T)/2;r=Q.shape[1];sigma2=float(S.trace()/(nu*r));tau2=float((mu@Q-text@Q).square().sum()/(K*r)-sigma2*(1/G).mean())
 write(tag+'.EB_ESTIMATE.json',dict(rank=r,group_counts=G.tolist(),degrees_freedom=nu,sigma2=sigma2,tau2=tau2))
 assert sigma2>0 and tau2>0 and torch.isfinite(S).all(),'NONPOSITIVE_EB_VARIANCE'
 den=S+float(S.trace()/nu)*torch.eye(r,dtype=S.dtype);torch.linalg.cholesky(den)
 prec=.1*(nu+2*r+1)*torch.linalg.inv(den);counts['precision_inverses']+=1;I=torch.eye(r,dtype=S.dtype);post=[];maxres=0.
 for c in range(K):
  h=I/tau2+G[c]*prec;b=(text[c]@Q)/tau2+G[c]*prec@(mu[c]@Q);v=torch.linalg.solve(h,b);counts['posterior_solves']+=1;maxres=max(maxres,float((h@v-b).abs().max()));post.append(v)
 theta=torch.stack(post);counts['completed_EB_fits']+=1;assert maxres<=1e-7 and (theta.norm(dim=1)>1e-12).all()
 mean=q@g;scale=((g-mean).square()*q[:,None]).sum().sqrt();assert scale>1e-12
 z=(g-mean)/scale;x=torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1);a=x.T@(x*q[:,None]);a=(a+a.T)/2;b=x.T@(F.one_hot(y,K).double()*q[:,None]);h=a.clone();h.diagonal()[:-1].add_(.01);W=torch.linalg.solve(h,b);counts['ridge_solves']+=1;re=float((h@W-b).abs().max());assert re<=1e-7
 ix=torch.triu_indices(d+1,d+1);jx=torch.triu_indices(d,d)
 obj=dict(A_packed=a[ix.unbind()].contiguous(),B=b,W=W,center=mean,scale=scale,means=mu,group_counts=G,pooled_raw_group_scatter=rawS[jx.unbind()].contiguous(),text=text,Q=Q,posterior=theta,tau2=tau2,sigma2=sigma2,alpha=.01,beta=.1,prompt='a photo of a {class_name}.',backbone='OpenAI CLIP ViT-B/32 frozen shared reference',source_commit=cfg['source_commit'])
 path=out/(tag+'.artifact.private.pt');torch.save(obj,path);assert path.stat().st_size<=19058800
 back=load(path,map_location='cpu',weights_only=False)
 for k,v in obj.items():
  if isinstance(v,torch.Tensor):assert torch.equal(v,back[k])
 dif=max(float((score(g,obj)[k]-score(g,back)[k]).abs().max()) for k in score(g,obj));assert dif<=1e-10
 write(tag+'.HEAD_SEAL.json',dict(bytes=path.stat().st_size,readback_exact=True,support_readback_maxabs=dif,EB_residual=maxres,ridge_residual=re,query_pending=True));return back

def score(g,t):
 q=g@t['Q'];mu=t['means'];pr=mu@t['Q'];th=t['posterior']
 assert (g.norm(dim=1)>1e-12).all() and (mu.norm(dim=1)>1e-12).all() and (pr.norm(dim=1)>1e-12).all() and (q.norm(dim=1)>1e-12).all()
 result=dict(ZS=g@t['text'].T,NCM=F.normalize(g,dim=1)@F.normalize(mu,dim=1).T,Projected=F.normalize(q,dim=1)@F.normalize(pr,dim=1).T,DEFTS=F.normalize(q,dim=1)@F.normalize(th,dim=1).T,Ridge=((g-t['center'])/t['scale'])@t['W'][:-1]+t['W'][-1])
 assert all(torch.isfinite(s).all() for s in result.values());return result

def gates(tab,cells,paired):
 d=tab['DEFTS'];bases=['OriginalF','Ridge'];K=len(d['correct']);others=[c for c in range(K) if c!=1]
 g=dict(BA_GAIN=all(d['BA']>=tab[b]['BA']+.01-1e-12 for b in bases),CLASS1_GAIN=all(d['correct'][1]>=tab[b]['correct'][1]+1 for b in bases),CELL_BA=all(v['DEFTS']['BA']>=v[b]['BA']-1e-12 for v in cells.values() for b in bases),CELL_CLASS1=all(v['DEFTS']['correct'][1]>=v[b]['correct'][1] for v in cells.values() for b in bases),CLASS1_H0=all(paired[b][1]['harmed']==0 for b in bases),ALLCLASS_CELL_COST=all(v['DEFTS']['recall'][c]>=v[b]['recall'][c]-.01-1e-12 for v in cells.values() for b in bases for c in others),VS_PROTOTYPES=d['BA']>tab['Projected']['BA'] and d['BA']>tab['NCM']['BA'])
 return dict(**g,pass_all=all(g.values()))

start=time.time();code=1
try:
 torch.manual_seed(74400);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 phase('admission');assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info(0);assert free>=8*2**30
 write('GPU_START.json',dict(device=0,free_bytes=free,total_bytes=total))
 # Known-group Bayes extension identity, no fitted synthetic model.
 assert abs((1/2+3/4)**-1*(2/2+3*4/4)-3.2)<1e-12
 write('SYNTHETIC.json',dict(scalar_posterior_pass=True))
 import open_clip
 phase('download_or_load_shared_backbone')
 model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',precision='fp32',device='cuda',cache_dir=str(out/'assets'));model.eval().requires_grad_(False);tokenizer=open_clip.get_tokenizer('ViT-B-32')
 phase('metadata_binding');prior=Path(cfg['prior_root']);prod=Path(cfg['producer_root']);man=Path(cfg['manifest_root'])
 plan=load(prod/'prep/PLAN.private.pt',weights_only=False)
 rr=json.loads((prior/'ISIC.S_old.rows.private.json').read_text());rr=[r for r in rr if r['label']<4];assert len(rr)==3236
 ir=[rr[i] for i in plan['row_ids'].tolist()];assert [r['identity_component'] for r in ir]==plan['groups'] and torch.equal(torch.tensor([r['label'] for r in ir]),plan['y'])
 hr=json.loads((prior/'HK.S_old.rows.private.json').read_text());hr=[r for r in hr if r['label']<13]
 semantic=json.loads((man/'ISIC/semantic_map_train_only.json').read_text())['mapping'];names={};allrows={'ISIC':ir,'HK':hr}
 for ds,K in [('ISIC',4),('HK',13)]:
  index={r['relative_path']:r for r in csv.DictReader((man/ds/'train.csv').open())};mapping={c:set() for c in range(K)}
  for r in allrows[ds]:
   a=index[r['relative_path']];name=semantic[a['original_label']]['class_name'] if ds=='ISIC' else a['official_class_name'].replace('-',' ');mapping[r['label']].add(name)
  assert all(len(v)==1 for v in mapping.values());names[ds]=[next(iter(mapping[c])) for c in range(K)]
 write('TEXT_PLAN_SEAL.json',dict(class_names=names,prompt='a photo of a {class_name}.',basis_variance_fraction=.95,selection=False))
 text={};basis={}
 for ds in names:
  with torch.no_grad():v=model.encode_text(tokenizer(['a photo of a '+n+'.' for n in names[ds]]).cuda()).float();v=F.normalize(v,dim=1).cpu().double()
  z=v-v.mean(0);evals,Q=torch.linalg.eigh(z.T@z);ev=evals.flip(0).clamp_min(0);assert ev.sum()>0;r=int(torch.searchsorted(ev.cumsum(0)/ev.sum(),torch.tensor(.95,dtype=ev.dtype)))+1
  counts['text_encoder_calls']+=1;counts['text_encoder_rows']+=len(v);basis[ds]=Q.flip(1)[:,:r];text[ds]=v;write(ds+'.TEXT_SEAL.json',dict(rank=r,variance_retained=float(ev[:r].sum()/ev.sum()),classes=len(v)))
 ig=encode(ir,cfg['image_roots']['ISIC'],'ISIC_initial');hg=encode(hr,cfg['image_roots']['HK'],'HK_support');iy=plan['y'];igs=plan['groups'];hy=torch.tensor([r['label'] for r in hr]);hgs=[r['identity_component'] for r in hr]
 heads=[]
 for k in [0,1]:
  take=plan['fold']==k;gs=[igs[i] for i in torch.where(take)[0]];assert not set(gs)&{igs[i] for i in torch.where(~take)[0]};heads.append(fit(ig[take],iy[take],gs,text['ISIC'],basis['ISIC'],'ISIC'+str(k)))
 hh=fit(hg,hy,hgs,text['HK'],basis['HK'],'HK');write('MODEL_SEAL.json',dict(producers=3,ridge_fits=3,EB_fits=3,backbone_updates=0,all_query_risk_pending=True))
 phase('sealed_query_evaluation');alls={n:torch.zeros(len(iy),4,dtype=torch.float64) for n in ['ZS','NCM','Projected','DEFTS','Ridge']}
 for k,t in enumerate(heads):
  take=plan['fold']!=k
  for n,s in score(ig[take],t).items():alls[n][take]=s
 prev=load(prod/'evaluate/responses.private.pt',weights_only=False);alls['OriginalF']=prev['score']['F_out'];assert torch.equal(prev['y'],iy) and prev['groups']==igs
 tab={n:points(s,iy,igs,4) for n,s in alls.items()};assert tab['OriginalF']['correct']==[237,13,64,1626]
 folds={str(k):{n:points(s[plan['fold']!=k],iy[plan['fold']!=k],[igs[i] for i in torch.where(plan['fold']!=k)[0]],4) for n,s in alls.items()} for k in [0,1]};pa={b:rh(alls[b],alls['DEFTS'],iy,4) for b in ['OriginalF','Ridge','Projected','NCM']};it=dict(summary=tab,folds=folds,paired=pa,gate=gates(tab,folds,pa));write('ISIC.json',it)
 oldh=load(Path(cfg['window_root'])/'group_consistency_pilot/HK.FCR.artifact.private.pt',weights_only=False);paths={r['relative_path'] for r in hr};groups=set(hgs);co={};joined={n:[] for n in alls};ys=[];gsall=[]
 for role in ['E','Qgate']:
  rows=json.loads((prior/f'HK.{role}.rows.private.json').read_text());keep=torch.tensor([r['label']<13 for r in rows]);rows=[r for r in rows if r['label']<13];yy=torch.tensor([r['label'] for r in rows]);gs=[r['identity_component'] for r in rows];pp={r['relative_path'] for r in rows};assert not pp&paths and not set(gs)&groups;paths|=pp;groups|=set(gs)
  gg=encode(rows,cfg['image_roots']['HK'],'HK_'+role);ss=score(gg,hh);old=load(prior/f'HK.{role}.features.private.pt',weights_only=False);assert torch.equal(old['y'][keep],yy);ss['OriginalF']=oldscore(F.normalize(old['x'][keep].double(),dim=1),oldh)
  mt={n:points(s,yy,gs,13) for n,s in ss.items()};assert mt['OriginalF']['correct']==cfg['HK_expected'][role]['correct'];co[role]=dict(summary=mt,paired={b:rh(ss[b],ss['DEFTS'],yy,13) for b in ['OriginalF','Ridge','Projected','NCM']})
  for n in joined:joined[n].append(ss[n])
  ys.append(yy);gsall+=gs
 yy=torch.cat(ys);ss={n:torch.cat(v) for n,v in joined.items()};mt={n:points(s,yy,gsall,13) for n,s in ss.items()};pa={b:rh(ss[b],ss['DEFTS'],yy,13) for b in ['OriginalF','Ridge','Projected','NCM']};ht=dict(cohorts=co,pooled=mt,paired=pa,gate=gates(mt,{k:v['summary'] for k,v in co.items()},pa))
 write('aggregate.json',dict(decision='GROUP_DEFTS_JOINT_POINT_SIGNAL' if it['gate']['pass_all'] and ht['gate']['pass_all'] else 'STOP_GROUP_DEFTS_CLIP_PILOT',method_success=False,ISIC=it,HK=ht,encoder='OpenAI CLIP ViT-B/32 frozen',ridge_fits=3,EB_fits=3,encoder_updates=0,hyperparameter_search=False,bootstrap=None,historically_exposed=True,patient_independence='UNKNOWN',source_commit=cfg['source_commit']))
 phase('COMPLETE');code=0
except BaseException as e:
 write('FAILURE.json',dict(type=type(e).__name__,scientific_complete=False,no_retry=True));(out/'FAILURE.private.txt').write_text(traceback.format_exc());phase('FAILED',error_type=type(e).__name__)
finally:
 write('ACTUAL_COUNTS.json',counts);u=resource.getrusage(resource.RUSAGE_SELF);write('WORKER_EXIT.json',dict(exit_code=code));write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
raise SystemExit(code)
