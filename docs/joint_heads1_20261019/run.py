"""One missing cross-moment pass; fixed joint and duplicate readouts only."""
import csv,json,os,sys,time,traceback,importlib.util
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core import joint_gram,duplicate_head
BASE=Path(os.environ['RUN_CONFIG']).resolve().parents[1]
spec=importlib.util.spec_from_file_location('moment_helpers',BASE/'matched_heads1_20261018/core.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
sys.path.insert(0,str(BASE/'case_hierarchical1_20261018'))
from data import read_rows,Images


def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def metrics(score,labels,seen,old):
 pred=torch.tensor(seen)[score.argmax(1)]
 counts=[int((labels==c).sum()) for c in seen]
 correct=[int(((labels==c)&(pred==c)).sum()) for c in seen]
 recall=[a/b for a,b in zip(correct,counts)]
 return dict(macro=sum(recall)/len(seen),micro=float((pred==labels).double().mean()),
  old_macro=sum(recall[:old])/old if old else None,new_macro=sum(recall[old:])/(len(seen)-old),
  counts=counts,correct=correct,recall=recall,
  confusion=[[int(((labels==c)&(pred==d)).sum()) for d in seen] for c in seen])


def main(cfg):
 torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 root=Path(cfg['output']);root.mkdir(exist_ok=False);write(root/'config.private.json',cfg)
 start=time.time();nt=0;nv=0
 def status(phase,**kw):
  write(root/'status.private.json',dict(state='RUNNING',phase=phase,started=start,updated=time.time(),new_training_images=nt,new_validation_images=nv,**kw))
  assert time.time()-start<cfg['max_wall_seconds']
  assert nt<=cfg['train_count'] and nv<=cfg['val_count']
  if torch.cuda.is_initialized():assert torch.cuda.max_memory_reserved()<=cfg['gpu_reserved_budget_bytes']
 status('data_check')
 train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
 assert len(train)==cfg['train_count'] and len(val)==cfg['val_count']
 assert not ({r['group'] for r in train}&{r['group'] for r in val})
 prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run'])
 for run in [prior,matched]:
  old=json.loads((run/'config.private.json').read_text())
  for key in ['seed','order','task_sizes','manifest','images','weight','author_labels']:assert old[key]==cfg[key]
  assert json.loads((run/'status.private.json').read_text())['state']=='COMPLETE'
 originals=[torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
 patches=[torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
 from safetensors.torch import load_file
 import timm
 encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
 encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
 encoder.eval().requires_grad_(False).cuda()
 def extract(rows):
  for images,idx in DataLoader(Images(rows,cfg),batch_size=64,num_workers=0,shuffle=False):
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=encoder.forward_features(images.cuda())
   assert z.shape[1:]==(197,768)
   yield z[:,0].double(),z[:,1:].float().mean(1).double(),idx
 grams=[];targets=[];parity={}
 for c,old,pm in zip(cfg['order'],originals,patches):
  rows=[r for r in train if r['label']==c]
  assert len({r['group'] for r in rows})==old['groups']
  w=common.weights([r['group'] for r in rows],'cuda')
  cg=torch.zeros(769,769,dtype=torch.float64,device='cuda');pg=torch.zeros_like(cg);cross=torch.zeros(768,768,dtype=torch.float64,device='cuda')
  for cls,patch,idx in extract(rows):
   nt+=len(idx);status('cross_moments',class_id=c)
   mass=w[idx.cuda()];a=common.augmented(cls);b=common.augmented(patch)
   cg+=a.T@(a*mass[:,None]);pg+=b.T@(b*mass[:,None]);cross+=cls.T@(patch*mass[:,None])
  errors=dict(CLS_gram=float((cg.cpu()-old['ridge_xx']).abs().max()),patch_gram=float((pg.cpu()-pm['patch_mean_gram']).abs().max()))
  assert max(errors.values())<cfg['feature_atol'],'Fresh marginal statistics mismatch'
  parity[str(c)]=errors
  torch.save(dict(cross=cross.cpu()),root/f'cross_{c:02d}.private.pt')
  grams.append(joint_gram(old['ridge_xx'].cuda(),pm['patch_mean_gram'].cuda(),cross).cpu())
  targets.append(torch.cat([old['ridge_xy'][:-1],pm['patch_mean_target'][:-1],old['ridge_xy'][-1:]]))
  print(json.dumps(dict(event='class_complete',class_id=c,images=len(rows),errors=errors)),flush=True)
 assert nt==cfg['train_count']
 fitted=[]
 for stage in range(6):
  n=sum(cfg['task_sizes'][:stage+1])
  joint=common.ridge([g.cuda() for g in grams[:n]],[t.cuda() for t in targets[:n]],1.).cpu()
  cg=torch.stack([x['ridge_xx'] for x in originals[:n]]).sum(0).cuda()
  ct=torch.stack([x['ridge_xy'] for x in originals[:n]],1).cuda()
  dup=duplicate_head(cg,ct).cpu();base=torch.linalg.solve(cg+torch.eye(769,device='cuda',dtype=torch.float64),ct).cpu()
  historical=torch.load(matched/f'heads_{stage}.private.pt',weights_only=True)
  assert float((base-historical['cls768_ridge']).abs().max())<cfg['score_atol']
  head=dict(joint=joint,duplicate_raw=dup,cls=base,patch=historical['patchmean768_ridge'])
  fitted.append(head);torch.save(head,root/f'heads_{stage}.private.pt');status('fit_readouts',stage=stage)
 write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,fitted_classes=20,stages=6,new_training_images=nt,marginal_parity=parity))
 query={};vparity={}
 for c in cfg['order']:
  rows=[r for r in val if r['label']==c];blocks=list(extract(rows));nv+=sum(len(idx) for _,_,idx in blocks)
  cls=torch.cat([a.cpu() for a,b,i in blocks]);patch=torch.cat([b.cpu() for a,b,i in blocks])
  old=torch.load(prior/f'val_{c:02d}.private.pt',weights_only=True)[2].double()
  error=float((cls-old).abs().max());assert error<=cfg['feature_atol']
  query[c]=(cls,patch);vparity[str(c)]=error;status('validation_features',class_id=c)
 torch.save(query,root/'query768.private.pt');write(root/'PARITY.json',dict(validation_CLS=vparity))
 results=[]
 for stage,head in enumerate(fitted):
  old=sum(cfg['task_sizes'][:stage]);seen=cfg['order'][:old+cfg['task_sizes'][stage]]
  rows=[r for r in val if r['label'] in seen];labels=torch.tensor([r['label'] for r in rows])
  pos={c:0 for c in seen};cs=[];ps=[]
  for row in rows:
   c=row['label'];i=pos[c];pos[c]+=1;cs.append(query[c][0][i]);ps.append(query[c][1][i])
  cls=torch.stack(cs);patch=torch.stack(ps);a=common.augmented(cls)
  scores=dict(cls768_ridge=a@head['cls'],patchmean768_ridge=common.augmented(patch)@head['patch'],
   joint768_ridge=common.augmented(torch.cat([cls,patch],1))@head['joint'],duplicate_raw=a@head['duplicate_raw'])
  # Exact analytic [CLS/sqrt(2),CLS/sqrt(2),1] prediction; verified on explicit matrices in check.py.
  scores['duplicate_calibrated']=scores['cls768_ridge'].clone()
  previous=torch.load(matched/f'stage_{stage}.private.pt',weights_only=True)
  assert previous['seen']==seen and torch.equal(previous['labels'],labels)
  for arm in ['cls768_ridge','patchmean768_ridge']:
   assert float((scores[arm]-previous['scores'][arm]).abs().max())<cfg['score_atol']
   assert torch.equal(scores[arm].argmax(1),previous['scores'][arm].argmax(1))
  record=dict(stage=stage,seen=seen,metrics={k:metrics(v,labels,seen,old) for k,v in scores.items()},contrasts={})
  pred={k:torch.tensor(seen)[v.argmax(1)] for k,v in scores.items()}
  for cand,ref in [('joint768_ridge','cls768_ridge'),('joint768_ridge','duplicate_raw'),('duplicate_raw','cls768_ridge'),('duplicate_calibrated','cls768_ridge')]:
   p,q=pred[cand],pred[ref]
   record['contrasts'][cand+'-'+ref]=dict(delta_BA=record['metrics'][cand]['macro']-record['metrics'][ref]['macro'],rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
  torch.save(dict(scores=scores,labels=labels,seen=seen),root/f'stage_{stage}.private.pt')
  results.append(record);write(root/'results.json',results);status('stage_complete',stage=stage)
 costs=dict(wall_seconds=time.time()-start,new_training_images=nt,new_validation_images=nv,optimizer_updates=0,attempts=1,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),historical_feature_creation_excluded=True)
 write(root/'costs.json',costs)
 assert sum(p.stat().st_size for p in root.iterdir() if p.is_file())<cfg['output_budget_bytes']
 with (root/'summary.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['stage','arm','BA','old_BA','new_BA'])
  for r in results:
   for arm,m in r['metrics'].items():writer.writerow([r['stage'],arm,m['macro'],m['old_macro'],m['new_macro']])
 write(root/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),costs=costs))
 print('COMPLETE',flush=True)


if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except Exception:
  root=Path(cfg['output']);root.mkdir(exist_ok=True);p=root/'status.private.json'
  state=json.loads(p.read_text()) if p.exists() else {}
  state.update(state='FAILED',updated=time.time(),error=traceback.format_exc());write(p,state);raise
