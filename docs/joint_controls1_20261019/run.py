"""CPU-only scale and df controls using existing sufficient statistics."""
import csv,importlib.util,json,os,time,traceback
from pathlib import Path
import torch
from core import scaled,spectrum,degrees,match_alpha


def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def metrics(scores,labels,seen,old):
 p=torch.tensor(seen)[scores.argmax(1)]
 counts=[int((labels==c).sum()) for c in seen]
 correct=[int(((labels==c)&(p==c)).sum()) for c in seen];recall=[a/b for a,b in zip(correct,counts)]
 return dict(macro=sum(recall)/len(seen),micro=float((p==labels).double().mean()),old_macro=sum(recall[:old])/old if old else None,new_macro=sum(recall[old:])/(len(seen)-old),counts=counts,correct=correct,recall=recall,confusion=[[int(((labels==c)&(p==d)).sum()) for d in seen] for c in seen])


def main(cfg):
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);start=time.time()
 root=Path(cfg['output']);root.mkdir(exist_ok=False);write(root/'config.private.json',cfg)
 def status(phase,**kw):
  write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),started=start,new_training_images=0,new_validation_images=0,**kw))
  assert time.time()-start<cfg['max_wall_seconds']
 status('load_statistics')
 prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run'])
 for p in [prior,matched,joint]:assert json.loads((p/'status.private.json').read_text())['state']=='COMPLETE'
 basepath=joint.parent.parent
 spec=importlib.util.spec_from_file_location('joint_helper',basepath/'joint_heads1_20261019/core.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 originals=[torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
 patches=[torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
 ec=sum(float(x['ridge_xx'][:-1,:-1].trace()) for x in originals[:10])/10
 em=sum(float(x['patch_mean_gram'][:-1,:-1].trace()) for x in patches[:10])/10
 scale=(ec/em)**.5;assert ec>0 and em>0
 g=torch.zeros(1537,1537,dtype=torch.float64);cg=torch.zeros(769,769,dtype=torch.float64);targets=[];fitted=[];audits=[]
 endpoints=[sum(cfg['task_sizes'][:i+1]) for i in range(6)]
 for i,(c,a,b) in enumerate(zip(cfg['order'],originals,patches)):
  cross=torch.load(joint/f'cross_{c:02d}.private.pt',weights_only=True)['cross']
  g+=mod.joint_gram(a['ridge_xx'],b['patch_mean_gram'],cross);cg+=a['ridge_xx']
  targets.append(torch.cat([a['ridge_xy'][:-1],b['patch_mean_target'][:-1],a['ridge_xy'][-1:]]))
  if i+1 not in endpoints:continue
  stage=endpoints.index(i+1);t=torch.stack(targets,1)
  ng,nt,factors=scaled(g,t,scale)
  ev=spectrum(ng);cv=spectrum(cg);targetdf=degrees(cv,1.)
  alpha=match_alpha(ev,targetdf)
  eye=torch.eye(len(g),dtype=torch.float64)
  heads=dict(scale_joint=torch.linalg.solve(ng+eye,nt),df_matched_joint=torch.linalg.solve(ng+alpha*eye,nt),factors=factors)
  fitted.append(heads);torch.save(heads,root/f'heads_{stage}.private.pt')
  audits.append(dict(stage=stage,classes=i+1,CLS_df=targetdf,scale_joint_df=degrees(ev,1.),matched_alpha=alpha,matched_df=degrees(ev,alpha),df_error=degrees(ev,alpha)-targetdf,CLS_energy=float(cg[:-1,:-1].trace()),scaled_joint_energy=float(ng[:-1,:-1].trace())))
  status('fit_control_heads',stage=stage)
 write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,base_CLS_energy=ec,base_patch_energy=em,patch_scale=scale,stages=audits))
 # Validation features and scores are first opened after all six stages are fitted.
 query=torch.load(joint/'query768.private.pt',weights_only=True);results=[]
 for stage,head in enumerate(fitted):
  prior_stage=torch.load(joint/f'stage_{stage}.private.pt',weights_only=True)
  labels=prior_stage['labels'];seen=prior_stage['seen'];old=sum(cfg['task_sizes'][:stage])
  assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]
  pos={c:0 for c in seen};xs=[]
  for c in labels.tolist():
   j=pos[c];pos[c]+=1;xs.append(torch.cat([query[c][0][j],query[c][1][j],torch.ones(1,dtype=torch.float64)]))
  x=torch.stack(xs)*head['factors'];scores=dict(prior_stage['scores'])
  for arm in cfg['new_arms']:scores[arm]=x@head[arm]
  record=dict(stage=stage,seen=seen,metrics={arm:metrics(v,labels,seen,old) for arm,v in scores.items()},contrasts={},class_sensitivity={})
  pred={arm:torch.tensor(seen)[v.argmax(1)] for arm,v in scores.items()}
  for arm in ['joint768_ridge']+cfg['new_arms']:
   for ref in ['cls768_ridge','duplicate_raw','joint768_ridge','scale_joint']:
    if arm==ref or (arm=='joint768_ridge' and ref=='scale_joint'):continue
    p,q=pred[arm],pred[ref];delta=record['metrics'][arm]['macro']-record['metrics'][ref]['macro']
    name=arm+'-'+ref
    record['contrasts'][name]=dict(delta_BA=delta,rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
    if ref=='cls768_ridge':
     diffs=[(a-b)/15 for a,b in zip(record['metrics'][arm]['correct'],record['metrics'][ref]['correct'])]
     loo=[(sum(diffs)-d)/(len(diffs)-1) for d in diffs]
     record['class_sensitivity'][name]=dict(per_class_delta=dict(zip(map(str,seen),diffs)),leave_one_class_out_min=min(loo),leave_one_class_out_max=max(loo),net_positive_classes=sum(d>0 for d in diffs),net_negative_classes=sum(d<0 for d in diffs))
  torch.save(dict(scores=scores,labels=labels,seen=seen),root/f'stage_{stage}.private.pt');results.append(record);status('score_controls',stage=stage)
 write(root/'results.json',results)
 costs=dict(wall_seconds=time.time()-start,new_training_images=0,new_validation_images=0,optimizer_updates=0,attempts=1,device='CPU',threads=4,historical_statistics_and_query_creation_excluded=True)
 write(root/'costs.json',costs)
 assert sum(p.stat().st_size for p in root.iterdir() if p.is_file())<cfg['output_budget_bytes']
 with (root/'summary.csv').open('w',newline='') as f:
  w=csv.writer(f,lineterminator="\n");w.writerow(['stage','arm','BA','old_BA','new_BA'])
  for r in results:
   for arm,m in r['metrics'].items():w.writerow([r['stage'],arm,m['macro'],m['old_macro'],m['new_macro']])
 write(root/'status.private.json',dict(state='COMPLETE',finished=time.time(),started=start,costs=costs))
 print(json.dumps(dict(costs=costs,training_audit=audits,final_contrasts=results[-1]['contrasts'])))


if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except Exception:
  root=Path(cfg['output']);root.mkdir(exist_ok=True);p=root/'status.private.json'
  state=json.loads(p.read_text()) if p.exists() else {};state.update(state='FAILED',updated=time.time(),error=traceback.format_exc());write(p,state);raise
