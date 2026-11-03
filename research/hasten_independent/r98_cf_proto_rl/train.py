"""Fixed 6x512 common warmup, then 6x5x256 matched post-training."""
import copy,json,time,csv,torch
from pathlib import Path
from common import write,save,guard
from prototype_policy import Policy
from episodes import episode,noise
from losses import loss_for_arm,ARMS
COUNTS={'formal_optimizer_steps':0,'warmup_steps':0,'post_steps':0,'episodes':0,'joint_rollouts':0,'active_class_actions':0,'prototype_builds':0,'raw_score_rows':0,'counterfactual_utility_calls':0,'utility_calls':0,'image_encoder_rows':0,'cache_feature_rows_accessed':0,'F_rows':0,'VAL_rows':0,'TEST_rows':0,'diagnostic_forward':0,'diagnostic_backward':0,'diagnostic_compute_counts':{}}

def flat_grad(loss,m,retain=False):
 gs=torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True,retain_graph=retain);return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,m.parameters())])

def gradient_diag(m,ref,ep,arm,stage,step):
 loss,ce,aux,stats=loss_for_arm(arm,m,ref,ep,noise(ep,stage,step,0,purpose=13),COUNTS['diagnostic_compute_counts']);a=flat_grad(ce,m,True);b=flat_grad(aux,m);COUNTS['diagnostic_forward']+=1;COUNTS['diagnostic_backward']+=2;return {'stage':stage,'arm':arm,'step':step,'CE_grad_norm':float(a.norm()),'aux_plus_KL_grad_norm':float(b.norm()),'cosine':float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30)),'scope':'raw parameter gradients before clipping, fullS+B, no optimizer update'}

def estimator_diag(m,ref,ep,stage):
 stats={a:[] for a in ['CF','GLOBAL']}
 for j in range(32):
  eps=noise(ep,stage,j,0,purpose=17)
  for arm in stats:
   _,_,aux,_=loss_for_arm(arm,m,ref,ep,eps,COUNTS['diagnostic_compute_counts']);stats[arm].append(flat_grad(aux,m).cpu().double());COUNTS['diagnostic_forward']+=1;COUNTS['diagnostic_backward']+=1
 means={};out=[]
 for arm,v in stats.items():
  g=torch.stack(v);mean=g.mean(0);means[arm]=mean;out.append({'stage':stage,'arm':arm,'independent_G8_replicates':32,'mean_gradient_norm':float(mean.norm()),'sum_coordinate_sample_variance':float(g.var(0,correction=1).sum()),'mean_replicate_gradient_norm':float(g.norm(dim=1).mean()),'raw_before_clipping':True})
 cos=float((means['CF']@means['GLOBAL'])/(means['CF'].norm()*means['GLOBAL'].norm()).clamp_min(1e-30))
 for row in out:row['paired_mean_direction_cosine']=cos
 return out

def checkpoint(c,m,opt,stage,arm,step):
 name=f's{stage}_{arm}_{step}.private.pt';save(Path(c['output'])/name,{'stage':stage,'arm':arm,'step':step,'model':m.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),'counts':dict(COUNTS)});return name

def optimizer(m,lr):return torch.optim.AdamW(m.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)

def fit(c,m,ref,d,stage,arm,steps,lr,diagnostics):
 r=Path(c['output']);opt=optimizer(m,lr);history=[];start=time.time();cpu=time.process_time()
 for step in range(steps+1):
  guard(c)
  if arm!='WARM' and step in [0,128,256]:diagnostics.append(gradient_diag(m,ref,episode(d,stage,step),arm,stage,step))
  if step==steps:break
  opt.zero_grad(set_to_none=True);records=[]
  for half in [False,True]:
   ep=episode(d,stage,step,half);eps=noise(ep,stage,step,int(half));value,_,_,stats=loss_for_arm(arm,m,ref,ep,eps,COUNTS);(value*.5).backward();records.append(stats)
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();assert all(torch.isfinite(p).all() for p in m.parameters());COUNTS['formal_optimizer_steps']+=1;COUNTS['warmup_steps' if arm=='WARM' else 'post_steps']+=1;rec={'stage':stage,'arm':arm,'step':step+1,'grad_norm_before_clip':float(norm),'gradient_clipped':bool(norm>1),'episodes':records,'counts':dict(COUNTS)}
  with (r/'TRAIN_DIAGNOSTICS.private.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
  if (step+1)%16==0:write(r/'ACTIVE_COUNTS.private.json',{'stage':stage,'arm':arm,'step':step+1,'counts':COUNTS})
  if arm!='WARM' and step+1==128:checkpoint(c,m,opt,stage,arm,128)
 name=checkpoint(c,m,opt,stage,'SUP0' if arm=='WARM' else arm,steps);history={'stage':stage,'arm':arm,'steps':steps,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start};write(r/f'COST_s{stage}_{arm}.json',history);return name

def run(c,meta):
 from data_contract import load_stage
 r=Path(c['output']);rows,y,roles=meta;torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);m=Policy().cuda();final=[];diag=[];est=[];start=time.time();cpu=time.process_time()
 for s in range(6):
  d=load_stage(c,s,y,roles);COUNTS['cache_feature_rows_accessed']+=len(d['S'])+len(d['B']);final.append(fit(c,m,None,d,s,'WARM',512,3e-4,diag));write(r/'STATUS.json',{'phase':'WARMUP','stage_complete':s+1,'counts':COUNTS})
 assert COUNTS['formal_optimizer_steps']==3072;write(r/'WARMUP_SEAL.json',{'snapshots':final[:],'count':6,'F_VAL_TEST':0})
 for s in range(6):
  d=load_stage(c,s,y,roles);ref=Policy().cuda();ref.load_state_dict(torch.load(r/f's{s}_SUP0_512.private.pt',weights_only=False,map_location='cpu')['model']);ref.requires_grad_(False);m=copy.deepcopy(ref).requires_grad_(True);est+=estimator_diag(m,ref,episode(d,s,0),s)
  for arm in ARMS:
   m=copy.deepcopy(ref).requires_grad_(True);final.append(fit(c,m,ref,d,s,arm,256,1e-4,diag));write(r/'STATUS.json',{'phase':'POST_TRAIN','stage':s,'arm_complete':arm,'counts':COUNTS})
 assert COUNTS['formal_optimizer_steps']==10752 and len(final)==36;write(r/'TRAIN_COMPLETE.json',{'counts':COUNTS,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start});write(r/'SEAL.json',{'snapshots':final,'final_endpoints':36,'intermediate_endpoints':30,'formal_optimizer_steps':10752,'F_VAL_TEST_before_seal':0,'deploy':'deterministic mean action; stage full arrived TRAIN after fixed F readout','no_endpoint_selection':True});write(r/'GRADIENT_DIAGNOSTICS.json',diag);write(r/'ESTIMATOR_DIAGNOSTICS.json',est)
