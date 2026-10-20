"""Fixed derivative error decomposition, no optimizer updates or risk scores."""
import torch
from pathlib import Path
from train_base import setup,raw
from train import load_training,schedule,write
from solver import objective

def run(cfg):
 pixels,y,groups,_=load_training(cfg);ids=schedule(y,groups)[0][0];m,params,bkeys,ckeys,heads,pw=setup(cfg)
 for k in bkeys:params[k].requires_grad_(True)
 ps=[params[k] for k in bkeys];initial=[p.detach().clone() for p in ps];xx=pixels[ids].cuda();yy=y[ids].cuda()
 def feature():
  o=raw(m,pw,xx);return torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1)
 g=feature();l=objective(g,yy,'O');gg=torch.autograd.grad(l,g,retain_graph=True)[0];gr=torch.autograd.grad(l,ps);norm=sum(float(z.square().sum()) for z in gr)**.5;dirs=[z/norm for z in gr];cells=[]
 for epsilon in [.001,.0005,.00025]:
  values=[];features=[];achieved=[]
  for sign in [1,-1]:
   with torch.no_grad():
    for p,init,d in zip(ps,initial,dirs):p.copy_(init+sign*epsilon*d)
   # Same grad-enabled execution path as analytical derivative.
   z=feature();values.append(float(objective(z,yy,'O').detach()));features.append(z.detach());achieved.append(sum(float(((p-init)*d).sum()) for p,init,d in zip(ps,initial,dirs)))
  fd=(values[0]-values[1])/(2*epsilon);chain=float((gg.detach()*(features[0]-features[1])/(2*epsilon)).sum());cells.append(dict(epsilon=epsilon,finite_difference=fd,encoder_feature_chain_fd=chain,analytic=norm,achieved_parameter_projection=achieved,loss_positive=values[0],loss_negative=values[1]))
 write(Path(cfg['output'])/'aggregate.json',dict(decision='DERIVATIVE_ERROR_DECOMPOSITION_ONLY',cells=cells,optimizer_updates=0,new_scores=0,training_images=32,not_a_passing_certificate=True))
