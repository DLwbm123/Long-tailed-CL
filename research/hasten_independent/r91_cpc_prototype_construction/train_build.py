import time,torch,itertools
from pathlib import Path
import torch.nn.functional as F
from core import *
from binding import *
from common import *

def init_models():
 models={};opts={}
 for name in ['main','held']:
  torch.manual_seed(74002);g=Builder();models[name]=g;opts[name]=torch.optim.AdamW(g.parameters(),lr=3e-4,weight_decay=1e-4)
 return models,opts

def loss_build(g,st,p0,qz,qy,C):
 cand,_=g(st,p0);logits=20*qz@cand[:,2].T;row=F.cross_entropy(logits,qy,reduction='none');ce=torch.stack([row[qy==c].mean() for c in range(C)]).mean();target=torch.stack([unit(qz[qy==c].mean(0)) for c in range(C)]);return ce+.1*(1-(cand[:,2]*target).sum(1)).mean()

def run(cfg,counter):
 root=Path(cfg['output']);models,opts=init_models();cpu=time.process_time();wall=time.time();records=[]
 for s in range(6):
  d=load_stage(cfg,s)
  for name,g in models.items():
   classes=[c for c in range(d['C']) if name=='main' or c%3!=1];qz,qy,_=query(d,'W',classes);opt=opts[name]
   for update in range(256):
    guard(cfg);opt.zero_grad();losses=[]
    for ep in range(4):
     idx=4*update+ep;st,p0,_=episode(cfg,d,s,idx,classes);loss=loss_build(g,st,p0,qz,qy,len(classes));assert torch.isfinite(loss);(loss/4).backward();losses.append(float(loss.detach()));counter['builder_episode_backward']+=1
    opt.step();counter['builder_optimizer_steps']+=1
    if update%16==15:write(root/'STATUS.json',{'phase':'BUILD','stage':s,'model':name,'update_complete':update+1,'loss':sum(losses)/4,'counts':counter})
   save(root/f'builder_{name}_s{s}.private.pt',{'model':g.state_dict(),'optimizer':opt.state_dict(),'classes_in_fit':classes,'counts':dict(counter)})
  records.append({'stage':s,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall});write(root/'BUILDER_COST.json',records)
  del d
 write(root/'BUILDERS_SEALED.json',{'states':12,'steps':counter['builder_optimizer_steps'],'VAL_reads':0});return records

def table(cfg,d,s,idx,g):
 st,p0,ids=episode(cfg,d,s,idx)
 with torch.no_grad():proto,r=g(st,p0)
 qz,qy,Fids=query(d,'F');bz,by,Bids=query(d,'W');sc=score(qz,proto).detach();sb=score(bz,proto).detach();tail,protected=meta(cfg,d['C'],s);ref=reference(cfg,d,s,Fids,sc[:,:,0]);bref=reference(cfg,d,s,Bids,sb[:,:,0]);return {'state':torch.cat([st,r,r.norm(dim=1,keepdim=True)],1).detach(),'proto':proto.detach(),'Sids':ids,'Fids':Fids,'Bids':Bids,'scores':sc,'B_scores':sb,'y':qy,'B_y':by,'tail':tail,'protected':protected,'reference':ref,'B_reference':bref}

def precheck(cfg,counter):
 root=Path(cfg['output']);rows=[];blockactions=torch.tensor(list(itertools.product(range(3),repeat=4)))
 for s in range(6):
  d=load_stage(cfg,s);C=d['C'];gs={}
  for name in ['main','held']:
   g=Builder();g.load_state_dict(torch.load(root/f'builder_{name}_s{s}.private.pt',map_location='cpu',weights_only=False)['model']);g.eval().requires_grad_(False);gs[name]=g
  for idx in range(5):
   t=table(cfg,d,s,idx,gs['main']);h=table(cfg,d,s,idx,gs['held']);zeros=torch.zeros(C,dtype=torch.long);ones=torch.full((C,),2,dtype=torch.long)
   v,rec,_=utility(t['scores'],t['y'],torch.stack([zeros,ones]),t['tail'],t['protected'],t['reference']);vh,rh,_=utility(h['scores'],h['y'],torch.stack([zeros,ones]),h['tail'],h['protected'],h['reference']);held=torch.tensor([c%3==1 for c in range(C)]);ht=held&t['tail'];allblock=zeros.repeat(81,1);allblock[:,:4]=blockactions
   bv,_,_=utility(t['B_scores'],t['B_y'],allblock,t['tail'],t['protected'],t['B_reference']);best=int(bv[:,2].argmax());fv,fr,_=utility(t['scores'],t['y'],allblock,t['tail'],t['protected'],t['reference']);delta=fr[best]-fr.mean(0)
   rows.append({'stage':s,'support_slot':idx,'main_macro_pp':float((rec[1]-rec[0]).mean()*100),'main_tail_pp':float((rec[1]-rec[0])[t['tail']].mean()*100),'held_macro_pp':float((rh[1]-rh[0])[held].mean()*100),'held_tail_pp':float((rh[1]-rh[0])[ht].mean()*100),'held_tiny1_delta':float(rh[1,1]-rh[0,1]),'block_macro_pp':float(delta.mean()*100),'block_tail_pp':float(delta[t['tail']].mean()*100)})
   counter['precheck_contexts']+=1
  write(root/'STATUS.json',{'phase':'PRECHECK','stage':s,'counts':counter})
 def avg(k,ss=rows):return sum(v[k] for v in ss)/len(ss)
 stage=[{k:avg(k,[v for v in rows if v['stage']==s]) for k in ['main_macro_pp','main_tail_pp','held_macro_pp','held_tail_pp','block_macro_pp','block_tail_pp']} for s in range(6)]
 g1=avg('main_macro_pp')>1e-8 and avg('main_tail_pp')>=-1e-8 and sum(v['main_macro_pp']>=-1e-8 and v['main_tail_pp']>=-1e-8 for v in stage)>=4
 g2=avg('held_macro_pp')>1e-8 and avg('held_tail_pp')>1e-8 and sum(v['held_macro_pp']>=-1e-8 for v in stage)>=4 and all(v['held_tiny1_delta']>=-1e-12 for v in rows)
 g3=avg('block_macro_pp')>1e-8 and avg('block_tail_pp')>1e-8 and sum(v['block_macro_pp']>=-1e-8 and v['block_tail_pp']>=-1e-8 for v in stage)>=4
 result={'passed':bool(g1 and g2 and g3),'main_transfer_gate':bool(g1),'held_class_gate':bool(g2),'block_transfer_gate':bool(g3),'rows':rows,'stage':stage,'VAL_TEST':0,'counts':dict(counter)};write(root/'PRECHECK.json',result);return result
