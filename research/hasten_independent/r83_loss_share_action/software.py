"""CPU toy gradients and real TRAIN metadata coverage; zero visual optimizer/scoring."""
import json,os,time
from pathlib import Path
import torch
import torch.nn.functional as F
from check_decision import check
from diagnostic_stats import check as stats_check
from loss_share import weights
from data import train_rows
from folds import assign_folds
from decision import allocation
x=check();x.update(stats_check());torch.set_num_threads(4);t=time.time()
y=torch.tensor([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]);n=5;torch.manual_seed(74002);z=torch.randn(len(y),n,dtype=torch.float64,requires_grad=True);ce=F.cross_entropy(z,y,reduction='none');base=ce.mean();bg=torch.autograd.grad(base,z,retain_graph=True)[0]
for a in range(n+1):
 loss=(weights(y,n,a,rho=0)*ce).sum();g=torch.autograd.grad(loss,z,retain_graph=True)[0];assert torch.allclose(loss,base,atol=1e-14,rtol=0) and torch.allclose(g,bg,atol=1e-14,rtol=0)
losses=[(weights(y,n,a)*ce).sum() for a in range(n+1)];avg=sum(losses)/(n+1);fixed=(weights(y,n,-1)*ce).sum();assert torch.allclose(avg,fixed,atol=1e-14,rtol=0);assert torch.allclose(torch.autograd.grad(avg,z,retain_graph=True)[0],torch.autograd.grad(fixed,z,retain_graph=True)[0],atol=1e-14,rtol=0);assert weights(y,n,1)[0]>.25
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());dc=cfg['domains']['HK'];rows,labels=train_rows(dc);fold=assign_folds(labels.tolist(),[r['identity_component'] for r in rows]);blocks=0
for stage in range(6):
 ids=[i for classes in dc['stages'][:stage+1] for i,c in enumerate(labels.tolist()) if c in classes];yy=labels[ids];n=sum(map(len,dc['stages'][:stage+1]))
 for b in [0,1]:
  S=[i for i,idx in enumerate(ids) if fold[idx]==1-b]
  for rep in range(3):
   seed=74002+2000000+10000*stage+100*b+10*rep
   for h in [0,1]:
    common=[S[i] for i in torch.randperm(len(S),generator=torch.Generator().manual_seed(seed+h)).tolist()[:192]];ii=allocation(yy,S,common,-1,torch.Generator().manual_seed(seed+800000+h));assert len(ii)==len(set(ii))==256 and set(yy[ii].tolist())==set(range(n))
   blocks+=1
assert blocks==36;x.update(passed=True,rho0_loss_gradient_equivalent=True,uniform_action_mean_loss_gradient_equals_fixed_mean=True,actual_unique_batch_class_coverage_blocks=blocks,CE_share_mass_one=True,visual_optimizer_steps=0,controller_optimizer_steps=0,new_model_scores=0,wall_seconds=time.time()-t);print(json.dumps(x))
