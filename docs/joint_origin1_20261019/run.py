import importlib.util,json,os,time,traceback,csv
from pathlib import Path
import torch
from core import surrogate_transform,same_block
from diagnose import stage_diagnostics,trajectories


def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def write(p,v):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def metrics(score,labels,seen,old):
 p=torch.tensor(seen)[score.argmax(1)];counts=[int((labels==c).sum()) for c in seen];correct=[int(((labels==c)&(p==c)).sum()) for c in seen];rec=[a/b for a,b in zip(correct,counts)]
 return dict(macro=sum(rec)/len(seen),micro=float((p==labels).double().mean()),old_macro=sum(rec[:old])/old if old else None,new_macro=sum(rec[old:])/(len(seen)-old),counts=counts,correct=correct,recall=rec,confusion=[[int(((labels==c)&(p==d)).sum()) for d in seen] for c in seen])


def energy(grams):
 means=torch.stack([g[:-1,-1] for g in grams]);total=sum(float(g[:-1,:-1].trace()) for g in grams)/len(grams)
 mean=float(means.mean(0).square().sum());classmeans=float(means.square().sum(1).mean());within=total-classmeans;between=classmeans-mean
 assert min(within,between)>=-1e-8
 return dict(total=total,global_mean_energy=mean,between_class_mean_energy=between,within_class_energy=within,global_mean_fraction=mean/total,within_class_fraction=within/total)


def main(cfg):
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);start=time.time();root=Path(cfg['output']);root.mkdir(exist_ok=False);write(root/'config.private.json',cfg)
 def status(phase,**kw):
  write(root/'status.private.json',dict(state='RUNNING',phase=phase,started=start,updated=time.time(),new_training_images=0,new_validation_images=0,**kw));assert time.time()-start<cfg['max_wall_seconds']
 status('training_statistics')
 prior=Path(cfg['prior_run']);matched=Path(cfg['matched_run']);joint=Path(cfg['joint_run']);control=Path(cfg['control_run']);basepath=joint.parent.parent
 helpers=module(basepath/'joint_controls1_20261019/core.py','df_helpers')
 for p in [prior,matched,joint,control]:assert json.loads((p/'status.private.json').read_text())['state']=='COMPLETE'
 originals=[torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
 patches=[torch.load(matched/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
 crosses=[torch.load(joint/f'cross_{c:02d}.private.pt',weights_only=True)['cross'] for c in cfg['order']]
 cg=[x['ridge_xx'] for x in originals];pg=[x['patch_mean_gram'] for x in patches]
 um=[torch.cat([x,y['patch_mean_target'][:-1][None,:]],0) for x,y in zip(crosses,patches)]
 gb=torch.stack(cg[:10]).sum(0);umb=torch.stack(um[:10]).sum(0)
 mapping=torch.linalg.solve(gb+torch.eye(769,dtype=torch.float64)*cfg['surrogate_rho'],umb)
 ec=float(gb[:-1,:-1].trace())/10;eh=float((mapping.T@gb@mapping).trace())/10;shat=(ec/eh)**.5
 transform=surrogate_transform(mapping,shat);torch.save(dict(mapping=mapping,transform=transform),root/'surrogate.private.pt')
 control_seal=json.loads((control/'FIT_SEAL.json').read_text());scale=control_seal['patch_scale'];fits=[];audits=[]
 for stage in range(6):
  n=sum(cfg['task_sizes'][:stage+1]);g=torch.stack(cg[:n]).sum(0);b=torch.stack([x['ridge_xy'] for x in originals[:n]],1);cross=torch.stack(um[:n]).sum(0)
  alpha=control_seal['stages'][stage]['matched_alpha'];head,p=same_block(g,b,cross,alpha)
  sg=transform.T@g@transform;sb=transform.T@b;ev=helpers.spectrum(sg);target=helpers.degrees(helpers.spectrum(g),1.);sa=helpers.match_alpha(ev,target)
  sur=torch.linalg.solve(sg+sa*torch.eye(1537,dtype=torch.float64),sb)
  previous_heads=torch.load(control/f'heads_{stage}.private.pt',weights_only=True)
  wm=previous_heads['df_matched_joint'][768:1536]*scale/2**.5
  fit=dict(cls_same_block=head,predicted_patch_joint=transform@sur,P=p,WM=wm,surrogate_full_head=sur)
  fits.append(fit);torch.save(fit,root/f'heads_{stage}.private.pt')
  audits.append(dict(stage=stage,classes=n,joint_alpha=alpha,CLS_feature_penalty=2*alpha,CLS_bias_penalty=alpha,surrogate_alpha=sa,CLS_df=target,surrogate_df=helpers.degrees(ev,sa),df_error=helpers.degrees(ev,sa)-target,CLS_energy=energy(cg[:n]),patch_energy=energy(pg[:n]),risk_total_weight=n,baseline_average_risk_penalty=1/n))
  status('fit_head_controls',stage=stage)
 write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,surrogate_rho=cfg['surrogate_rho'],surrogate_base_CLS_energy=ec,surrogate_base_patch_energy=eh,surrogate_scale=shat,stages=audits))
 query=torch.load(joint/'query768.private.pt',weights_only=True);results=[];diagnostics=[];saved_previous=None;errors=[]
 for stage,fit in enumerate(fits):
  previous=torch.load(control/f'stage_{stage}.private.pt',weights_only=True);labels=previous['labels'];seen=previous['seen'];old=sum(cfg['task_sizes'][:stage]);assert seen==cfg['order'][:old+cfg['task_sizes'][stage]]
  pos={c:0 for c in seen};cs=[];ms=[]
  for c in labels.tolist():
   j=pos[c];pos[c]+=1;cs.append(query[c][0][j]);ms.append(query[c][1][j])
  c=torch.stack(cs);m=torch.stack(ms);u=torch.cat([c,torch.ones(len(c),1,dtype=torch.float64)],1)
  scores=dict(previous['scores']);scores.update({arm:u@fit[arm] for arm in cfg['new_arms']})
  reconstruction=scores['cls_same_block']+(m-u@fit['P'])@fit['WM'];error=float((reconstruction-scores['df_matched_joint']).abs().max());assert error<cfg['score_atol'];errors.append(error)
  record=dict(stage=stage,seen=seen,metrics={a:metrics(s,labels,seen,old) for a,s in scores.items()},contrasts={})
  pred={a:torch.tensor(seen)[s.argmax(1)] for a,s in scores.items()}
  pairs=[('cls_same_block','cls768_ridge'),('predicted_patch_joint','cls768_ridge'),('df_matched_joint','cls_same_block'),('df_matched_joint','predicted_patch_joint'),('scale_joint','predicted_patch_joint')]
  for a,b in pairs:
   p,q=pred[a],pred[b];record['contrasts'][a+'-'+b]=dict(delta_BA=record['metrics'][a]['macro']-record['metrics'][b]['macro'],rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()),both_wrong=int(((p!=labels)&(q!=labels)).sum()))
  saved=dict(scores=scores,labels=labels,seen=seen);d=stage_diagnostics(saved,saved_previous);d['stage']=stage;diagnostics.append(d);saved_previous=saved
  torch.save(saved,root/f'stage_{stage}.private.pt');results.append(record);status('diagnostics',stage=stage)
 trajectory=trajectories(results,cfg['order']);support=json.loads((prior/'support.json').read_text());frequency={}
 for criterion in ['images','patients']:
  ordered=sorted(cfg['order'],key=lambda c:(support[str(c)][criterion],c));bins=dict(tail=ordered[:7],middle=ordered[7:13],head=ordered[13:]);groupmetrics={}
  for arm,met in results[-1]['metrics'].items():groupmetrics[arm]={name:sum(met['recall'][cfg['order'].index(c)] for c in members)/len(members) for name,members in bins.items()}
  frequency[criterion]=dict(bins=bins,BA=groupmetrics)
 write(root/'results.json',results);write(root/'DIAGNOSTICS.json',diagnostics);write(root/'TRAJECTORIES.json',trajectory);write(root/'FREQUENCY.json',dict(support=support,groups=frequency));write(root/'RECONSTRUCTION.json',dict(max_score_errors=errors,tolerance=cfg['score_atol']))
 with (root/'summary.csv').open('w',newline='') as f:
  writer=csv.writer(f,lineterminator='\n');writer.writerow(['stage','arm','BA','old_BA','new_BA'])
  for r in results:
   for a,m in r['metrics'].items():writer.writerow([r['stage'],a,m['macro'],m['old_macro'],m['new_macro']])
 costs=dict(wall_seconds=time.time()-start,new_training_images=0,new_validation_images=0,optimizer_updates=0,attempts=1,device='CPU',threads=4,historical_creation_excluded=True)
 write(root/'costs.json',costs);assert sum(p.stat().st_size for p in root.iterdir() if p.is_file())<cfg['output_budget_bytes']
 write(root/'status.private.json',dict(state='COMPLETE',started=start,finished=time.time(),costs=costs));print(json.dumps(dict(costs=costs,final_contrasts=results[-1]['contrasts'],reconstruction_errors=errors)))


if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except Exception:
  root=Path(cfg['output']);root.mkdir(exist_ok=True);p=root/'status.private.json';state=json.loads(p.read_text()) if p.exists() else {};state.update(state='FAILED',updated=time.time(),error=traceback.format_exc());write(p,state);raise
