import csv,json,os,time,traceback
from pathlib import Path
import torch
from core import training_rms,overlap


def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def metrics(score,labels,seen,old):
 pred=torch.tensor(seen)[score.argmax(1)];counts=[int((labels==c).sum()) for c in seen];correct=[int(((labels==c)&(pred==c)).sum()) for c in seen];rec=[a/b for a,b in zip(correct,counts)]
 return dict(macro=sum(rec)/len(seen),micro=float((pred==labels).double().mean()),old_macro=sum(rec[:old])/old if old else None,new_macro=sum(rec[old:])/(len(seen)-old),counts=counts,correct=correct,recall=rec,confusion=[[int(((labels==c)&(pred==d)).sum()) for d in seen] for c in seen])


def main(cfg):
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);start=time.time();root=Path(cfg['output']);root.mkdir(exist_ok=False);write(root/'config.private.json',cfg)
 def status(phase,**kw):
  write(root/'status.private.json',dict(state='RUNNING',phase=phase,started=start,updated=time.time(),new_training_images=0,new_validation_images=0,**kw));assert time.time()-start<cfg['max_wall_seconds']
 status('training_scales')
 prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run']);origin=Path(cfg['origin_run'])
 for p in [prior,matched,joint,origin]:assert json.loads((p/'status.private.json').read_text())['state']=='COMPLETE'
 gc=torch.zeros(769,769,dtype=torch.float64);gm=gc.clone();ends=[sum(cfg['task_sizes'][:i+1]) for i in range(6)];scales=[]
 for i,c in enumerate(cfg['order']):
  gc+=torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True)['ridge_xx'];gm+=torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True)['patch_mean_gram']
  if i+1 not in ends:continue
  stage=ends.index(i+1);heads=torch.load(joint/f'heads_{stage}.private.pt',weights_only=True)
  scales.append(dict(stage=stage,classes=i+1,train_weight_mass=float(gc[-1,-1]),sigma_CLS=training_rms(gc,heads['cls'],i+1),sigma_patch=training_rms(gm,heads['patch'],i+1)))
  status('training_scales',stage=stage)
 write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,head_fits=0,training_scales=scales))
 results=[];overlaps=[]
 for stage,audit in enumerate(scales):
  s=torch.load(origin/f'stage_{stage}.private.pt',weights_only=True);labels=s['labels'];seen=s['seen'];old=sum(cfg['task_sizes'][:stage]);assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]
  scores=dict(s['scores']);c=scores['cls768_ridge'];m=scores['patchmean768_ridge'];scores['score_equal']=(c+m)/2
  scores['score_train_rms']=((c-c.mean(1,keepdim=True))/audit['sigma_CLS']+(m-m.mean(1,keepdim=True))/audit['sigma_patch'])/2
  record=dict(stage=stage,seen=seen,metrics={arm:metrics(v,labels,seen,old) for arm,v in scores.items()},contrasts={})
  pred={arm:torch.tensor(seen)[v.argmax(1)] for arm,v in scores.items()}
  for cand in cfg['new_arms']:
   for ref in ['cls768_ridge','scale_joint','df_matched_joint','score_equal']:
    if cand==ref:continue
    p,q=pred[cand],pred[ref];record['contrasts'][cand+'-'+ref]=dict(delta_BA=record['metrics'][cand]['macro']-record['metrics'][ref]['macro'],rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
  correctness={arm:p==labels for arm,p in pred.items()};base=correctness['cls768_ridge'];patch=correctness['patchmean768_ridge']
  groups=dict(both_correct=base&patch,CLS_only=base&~patch,patch_only=~base&patch,both_wrong=~base&~patch)
  for arm in ['scale_joint','df_matched_joint']:
   a=correctness['cls_same_block'];j=correctness[arm];out=dict(stage=stage,joint_arm=arm,reference='cls768_ridge',rescue={},harm={})
   for kind,ma,mj in [('rescue',a&~base,j&~base),('harm',~a&base,~j&base)]:
    aa=set(ma.nonzero().flatten().tolist());jj=set(mj.nonzero().flatten().tolist());out[kind]['all']=overlap(aa,jj)
    out[kind]['original_groups']={name:overlap(aa&set(mask.nonzero().flatten().tolist()),jj&set(mask.nonzero().flatten().tolist())) for name,mask in groups.items()}
    out[kind]['per_class']={str(c):overlap(aa&set((labels==c).nonzero().flatten().tolist()),jj&set((labels==c).nonzero().flatten().tolist())) for c in seen}
   overlaps.append(out)
  torch.save(dict(scores=scores,labels=labels,seen=seen),root/f'stage_{stage}.private.pt');results.append(record);status('score_and_overlap',stage=stage)
 frequency=json.loads((origin/'FREQUENCY.json').read_text());group_metrics={}
 for criterion,data in frequency['groups'].items():
  group_metrics[criterion]=dict(bins=data['bins'],BA={arm:{name:sum(m['recall'][cfg['order'].index(c)] for c in members)/len(members) for name,members in data['bins'].items()} for arm,m in results[-1]['metrics'].items()})
 write(root/'results.json',results);write(root/'OVERLAP.json',overlaps);write(root/'FREQUENCY.json',group_metrics)
 with (root/'summary.csv').open('w',newline='') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['stage','arm','BA','old_BA','new_BA'])
  for r in results:
   for arm,m in r['metrics'].items():w.writerow([r['stage'],arm,m['macro'],m['old_macro'],m['new_macro']])
 costs=dict(wall_seconds=time.time()-start,new_training_images=0,new_validation_images=0,optimizer_updates=0,head_fits=0,attempts=1,device='CPU',threads=4,historical_creation_excluded=True)
 write(root/'costs.json',costs);assert sum(p.stat().st_size for p in root.iterdir() if p.is_file())<cfg['output_budget_bytes']
 write(root/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),costs=costs));print(json.dumps(dict(costs=costs,final_contrasts=results[-1]['contrasts'],final_overlap=overlaps[-2:])))


if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except Exception:
  root=Path(cfg['output']);root.mkdir(exist_ok=True);p=root/'status.private.json';state=json.loads(p.read_text()) if p.exists() else {};state.update(state='FAILED',updated=time.time(),error=traceback.format_exc());write(p,state);raise
