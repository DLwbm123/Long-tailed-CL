import time,torch
from pathlib import Path
from core import *
from binding import *
from train_build import table
from common import *

def run(cfg,counter):
 root=Path(cfg['output']);models={};opts={};timing=[]
 for arm in ARMS:
  torch.manual_seed(74002);models[arm]=Actor();opts[arm]=torch.optim.AdamW(models[arm].parameters(),lr=3e-4,weight_decay=0)
 for s in range(6):
  d=load_stage(cfg,s);g=Builder();g.load_state_dict(torch.load(root/f'builder_main_s{s}.private.pt',map_location='cpu',weights_only=False)['model']);g.eval().requires_grad_(False);logs={a:[] for a in ARMS};cpu=time.process_time();wall=time.time()
  for ctx in range(64):
   guard(cfg);t=table(cfg,d,s,ctx,g);mask=wins(t['scores'],t['y']);save(root/'active_context.private.pt',{'stage':s,'context':ctx,**t})
   for arm in ARMS:
    model=models[arm];opt=opts[arm];rng=torch.Generator().manual_seed(74002+10000*s+ctx);oldlog=model(t['state']).detach();oldprob=oldlog.exp();actions=sampled(oldprob,rng);q=None;U=None;baseline=None;local=None;costcalls=0
    if arm in ['M','S','H']:
     q=conditional(t['scores'],t['y'],actions,t['tail'],t['protected'],t['reference']);costcalls+=8*d['C']*3
     if arm=='S':local=sampled(oldprob,rng)
    elif arm=='G':U=utility(t['scores'],t['y'],actions,t['tail'],t['protected'],t['reference'])[0][:,2];costcalls+=8
    losses=[];clip=[]
    for update in range(4):
     guard(cfg);write(root/'ACTIVE_POLICY.private.json',{'stage':s,'context':ctx,'arm':arm,'update':update,'counts':counter});logp=model(t['state']);prob=logp.exp();regular=.04*kl(logp)
     if arm=='C':
      now_actions=sampled(prob.detach(),rng);ceq=conditional(t['scores'],t['y'],now_actions,t['tail'],t['protected'],t['reference'],kind='CE');objective=(prob[None]*ceq).sum((1,2)).mean();costcalls+=8*d['C']*3
     elif arm=='G':objective=group_loss(logp,oldlog,actions,U)
     else:
      b,_=exact_B(prob,mask,t['y'],t['tail']);objective=-b/3
      if arm in ['M','H']:objective+=risk_loss(logp,oldlog,oldprob,q,shuffle=arm=='H')/3
      elif arm=='S':
       objective+=risk_loss(logp,oldlog,oldprob,q,local=local)/3
     loss=objective+regular;assert torch.isfinite(loss);opt.zero_grad();loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();counter['actor_optimizer_steps']+=1;losses.append(float(loss.detach()));clip.append(float(((logp-oldlog).exp().sub(1).abs()>.2).to(DT).mean()))
    logs[arm].append({'context':ctx,'loss':losses,'clip_fraction':clip,'oldprob':oldprob,'actions':actions,'conditional_table':q,'sampled_baseline':baseline,'local_actions':local,'U':U,'reward_calls':costcalls,'CPU_seconds_cumulative':time.process_time()-cpu,'wall_seconds_cumulative':time.time()-wall});counter['joint_reward_calls']+=costcalls
    save(root/'active_policies.private.pt',{'stage':s,'context':ctx,'arm_complete':arm,'models':{a:m.state_dict() for a,m in models.items()},'optimizers':{a:o.state_dict() for a,o in opts.items()},'counts':dict(counter)})
   if ctx in [15,31,63]:
    for arm in ARMS:save(root/f'actor_{arm}_s{s}_u{4*(ctx+1)}.private.pt',{'model':models[arm].state_dict(),'optimizer':opts[arm].state_dict(),'logs':logs[arm],'counts':dict(counter)})
   write(root/'STATUS.json',{'phase':'ACTOR','stage':s,'context_complete':ctx+1,'counts':counter})
  t=table(cfg,d,s,4,g);a=torch.zeros(d['C'],dtype=torch.long);search=[]
  for sweep in range(2):
   for c in range(d['C']):
    trial=a.repeat(3,1);trial[:,c]=torch.arange(3);v=utility(t['scores'],t['y'],trial,t['tail'],t['protected'],t['reference'])[0][:,2];best=int(v.argmax());current=float(v[int(a[c])]);counter['joint_reward_calls']+=3
    if float(v[best])>current:a[c]=best
    search.append({'sweep':sweep,'class':c,'chosen':int(a[c]),'U_candidates':v})
  save(root/f'search_s{s}.private.pt',{'actions':a,'trace':search,'Sids':t['Sids'],'Fids':t['Fids']});timing.append({'stage':s,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall});write(root/'ACTOR_COST.json',timing)
 assert counter['actor_optimizer_steps']==9216
 write(root/'PARAMETERS_AND_RULES_SEALED.json',{'builder_states':12,'actor_states':108,'search_states':6,'VAL_reads':0,'counts':counter})

 def deployment(s,step=256):
  d=load_stage(cfg,s);g=Builder();g.load_state_dict(torch.load(root/f'builder_main_s{s}.private.pt',map_location='cpu',weights_only=False)['model']);g.eval().requires_grad_(False);support=[d['z'][d['y']==c] for c in range(d['C'])];p0=d['head']['prototypes'].to(DT);st,_=state(support,list(range(d['C'])),s,cfg['train_counts'],d['birth'],p0_override=p0)
  with torch.no_grad():proto,r=g(st,p0)
  ast=torch.cat([st,r,r.norm(dim=1,keepdim=True)],1);dist={};models_fixed={}
  for arm in ARMS:
   model=Actor();model.load_state_dict(torch.load(root/f'actor_{arm}_s{s}_u{step}.private.pt',map_location='cpu',weights_only=False)['model']);model.eval().requires_grad_(False)
   with torch.no_grad():prob=model(ast).exp()
   dist[arm]=prob
  C=d['C'];dist.update({'N0':F.one_hot(torch.zeros(C,dtype=torch.long),3).to(DT),'Nhalf':F.one_hot(torch.ones(C,dtype=torch.long),3).to(DT),'Nfull':F.one_hot(torch.full((C,),2,dtype=torch.long),3).to(DT),'U':torch.full((C,3),1/3,dtype=DT),'SEARCH':F.one_hot(torch.load(root/f'search_s{s}.private.pt',map_location='cpu',weights_only=False)['actions'],3).to(DT)})
  return {'stage':s,'proto':proto,'state':ast,'distributions':dist,'S_dep_ids':d['ids'],'p0_exact_SFT':torch.equal(proto[:,0],d['head']['prototypes'].to(DT))}
 for s in range(6):
  for step in [64,128,256]:save(root/f'deploy_s{s}_u{step}.private.pt',deployment(s,step))
 write(root/'ALL_STATES_SEALED.json',{'builder_states':12,'actor_states':108,'deployment_states':18,'final_deployment_states':6,'VAL_reads':0,'TEST_reads':0,'counts':counter})
