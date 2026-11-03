"""Nine paired post-training arms from six borrowed fixed SUP0 references."""
import copy,json,time,torch
from pathlib import Path
from common import write,save,guard
from prototype_policy import Policy
from episodes import episode,noise
from losses import components,ARMS
from calibration import load_reference
COUNTS={'formal_optimizer_steps':0,'episodes':0,'joint_rollouts':0,'active_class_actions':0,'policy_class_prototype_builds':0,'raw_score_rows':0,'replacement_utility_contexts':0,'factual_utility_contexts':0,'utility_contexts':0,'image_encoder_rows':0,'cache_feature_rows_accessed':0,'F_rows':0,'VAL_rows':0,'TEST_rows':0,'diagnostic_forward':0,'diagnostic_backward':0,'diagnostic_compute_counts':{}}

def flat_grad(loss,m,retain=False):
 gs=torch.autograd.grad(loss,tuple(m.parameters()),allow_unused=True,retain_graph=retain);return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(gs,m.parameters())])
def make_episode(d,s,step,half,kappa):
 ep=episode(d,s,step,half);ep['kappa']=float(kappa);ep['tail']=torch.tensor(d['rawcounts'],device=ep['state'].device)<250;return ep

def gradient_diag(m,ref,ep,arm,stage,step):
 value,ce,rl,hold,stats=components(arm,m,ref,ep,noise(ep,stage,step,0,purpose=13),COUNTS['diagnostic_compute_counts']);gc=flat_grad(ce,m,True);gr=flat_grad(rl,m,True);gh=flat_grad(hold,m);COUNTS['diagnostic_forward']+=1;COUNTS['diagnostic_backward']+=3
 def cos(a,b):return float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))
 return {'stage':stage,'arm':arm,'step':step,'CE_grad_norm':float(gc.norm()),'RL_grad_norm':float(gr.norm()),'weighted_hold_grad_norm':float(gh.norm()),'CE_RL_cosine':cos(gc,gr),'CE_hold_cosine':cos(gc,gh),'RL_hold_cosine':cos(gr,gh),**stats,'scope':'S-full+B fixed point, three raw parameter gradients before clipping, no optimizer'}

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
def checkpoint(c,m,opt,s,arm,step):
 name=f's{s}_{arm}_{step}.private.pt';save(Path(c['output'])/name,{'stage':s,'arm':arm,'step':step,'model':m.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),'counts':dict(COUNTS)});return name

def fit(c,m,ref,d,s,arm,kappa,diag):
 r=Path(c['output']);opt=optimizer(m);start=time.time();cpu=time.process_time()
 for step in range(257):
  guard(c)
  if step in [0,128,256]:diag.append(gradient_diag(m,ref,make_episode(d,s,step,False,kappa),arm,s,step))
  if step==256:break
  opt.zero_grad(set_to_none=True);stats=[]
  for half in [False,True]:
   ep=make_episode(d,s,step,half,kappa);val,_,_,_,record=components(arm,m,ref,ep,noise(ep,s,step,int(half)),COUNTS);(.5*val).backward();stats.append(record)
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();assert all(torch.isfinite(p).all() for p in m.parameters());COUNTS['formal_optimizer_steps']+=1
  rec={'stage':s,'arm':arm,'step':step+1,'grad_norm_before_clip':float(norm),'gradient_clipped':bool(norm>1),'episodes':stats}
  with (r/'TRAIN_DIAGNOSTICS.private.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
  if (step+1)%16==0:write(r/'ACTIVE_COUNTS.private.json',{'stage':s,'arm':arm,'step':step+1,'counts':COUNTS})
  if step+1==128:checkpoint(c,m,opt,s,arm,128)
 name=checkpoint(c,m,opt,s,arm,256);write(r/f'COST_s{s}_{arm}.json',{'stage':s,'arm':arm,'steps':256,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start});return name

def run(c,meta,kappas):
 from data_contract import load_stage
 r=Path(c['output']);final=[];diag=[];start=time.time();cpu=time.process_time()
 for s in range(6):
  d=load_stage(c,s,meta[1],meta[2]);COUNTS['cache_feature_rows_accessed']+=len(d['S'])+len(d['B']);ref=load_reference(c,s)
  for arm in ARMS:
   m=copy.deepcopy(ref).requires_grad_(True);final.append(fit(c,m,ref,d,s,arm,kappas[s],diag));write(r/'STATUS.json',{'phase':'POST_TRAIN','stage':s,'arm_complete':arm,'counts':COUNTS})
 assert COUNTS['formal_optimizer_steps']==13824 and len(final)==54;write(r/'TRAIN_COMPLETE.json',{'counts':COUNTS,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'borrowed_warmup_steps_not_new':3072});write(r/'SEAL.json',{'snapshots':final,'final_endpoints':54,'intermediate_endpoints':54,'formal_optimizer_steps':13824,'borrowed_SUP0_references':6,'borrowed_warmup_updates':3072,'calibration_fixed':kappas,'F_VAL_TEST_before_seal':0,'deploy':'deterministic mean action; stage full arrived TRAIN after fixed F readout','no_endpoint_selection':True});write(r/'GRADIENT_DIAGNOSTICS.json',diag)
