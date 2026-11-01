import itertools,json,time,resource
from pathlib import Path
import numpy as np
import torch
import counts
from decision import Policy,outcomes,state_tree,exact_group_constants,exact_group_loss,group_loss,kl,sample
from experiment import select_memory
from common import write

def run(cfg):
 start=time.time();cpu=time.process_time();torch.set_num_threads(4);torch.manual_seed(cfg['seed']);m=Policy();states=torch.randn(5,53,dtype=torch.float64)
 with torch.no_grad():old,oldslots,_=m.tree(states);old=old.clone();oldslots=oldslots.clone();m.net[-1].bias.add_(torch.tensor([.23,-.17,.04,-.08]))
 joint,slots,logjoint=m.tree(states);assert abs(float(joint.sum())-1)<1e-12 and (joint>0).all();R=np.linspace(-.7,1.1,16);co=exact_group_constants(old,R,2);exact=exact_group_loss(slots,oldslots,old,co);direct=slots.sum()*0
 for a,b in itertools.product(range(16),repeat=2):direct=direct+old[a]*old[b]*group_loss(slots,oldslots,torch.tensor([a,b]),R)
 assert abs(float((exact-direct).detach()))<1e-11;g1=torch.autograd.grad(exact,tuple(m.parameters()),retain_graph=True);g2=torch.autograd.grad(direct,tuple(m.parameters()),retain_graph=True);err=max(float((x-y).abs().max()) for x,y in zip(g1,g2));assert err<1e-10
 zero=group_loss(slots,oldslots,torch.tensor([0,15]),np.ones(16));assert float(zero.detach())==0
 assert abs(float((group_loss(slots,oldslots,torch.tensor([0,15]),R)-group_loss(slots,oldslots,torch.tensor([0,15]),R,True)).detach()))>1e-5
 draw=sample(oldslots,np.array([[0.,0.],[.999999,.999999]]));assert draw.tolist()==[0,15]
 opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=0);opt.zero_grad();(exact+.04*kl(joint,logjoint)).backward();assert any(z.grad is not None and z.grad.abs().sum()>0 for z in m.parameters());opt.step();counts.C['toy_controller_optimizer_steps']+=1
 scores=np.repeat(np.array([[[.9,.1],[.1,.9]]]),16,axis=0);scores[1,0]=[.1,.9];labels=np.array([0,1]);wr=outcomes(scores,labels,2,1,[1000,3]);assert wr['reward'][0]==1 and wr['reward'][1]==0
 tree=state_tree(scores,labels,wr,2,1,[1000,3],[0,1],np.array([1,1]),np.array([1,1]),1);assert tree.shape==(5,53) and torch.isfinite(tree).all()
 y=torch.tensor([0]*21+[1]*3);roles={'S':list(range(10))+[21],'W':list(range(10,18))+[22],'F':list(range(18,21))+[23]};f=np.random.default_rng(0).normal(size=(24,3));f/=np.linalg.norm(f,axis=1,keepdims=True);memory=select_memory(cfg,list(range(24)),f,y,roles);assert sum(map(len,memory[0].values()))==20 and sum(map(len,memory[1].values()))==3
 for c in memory:
  allrows=sum(memory[c].values(),[]);assert len(set(allrows))==len(allrows)
  for role,ids in memory[c].items():assert set(ids)<=set(roles[role])
 rec={'passed':True,'toy_controller_optimizer_steps':1,'new_real_forward':0,'VAL_TEST_reads':0,'actual_policy_X_loss_error':float((exact-direct).detach().abs()),'actual_policy_X_gradient_error':err,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024};write(Path(cfg['output'])/'SOFTWARE_CHECK.json',rec);return rec
