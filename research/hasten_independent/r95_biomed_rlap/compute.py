import torch
import torch.nn.functional as F
from model import visual,textual

def text_table(base,m,tokens,births,old_text):
 out=[]
 for b,branch in enumerate(m.branches):
  ix=[i for i,k in enumerate(births) if k==b]
  if not ix:continue
  if b<len(m.branches)-1:
   if b not in old_text:
    with torch.no_grad():old_text[b]=textual(base,tokens[ix],branch).detach()
   out.append(old_text[b])
  else:out.append(textual(base,tokens[ix],branch))
 return torch.cat(out)

def micro_loss(m,vs,text,proto,yy):
 scores,d,balance=m.logits(vs,text,proto,m._births);ce=(-(1+d.gather(1,yy[:,None]).squeeze(1))*scores.log_softmax(1).gather(1,yy[:,None]).squeeze(1)).mean();sim=vs[-1]@text[yy].T/m.log_temp.exp().clamp(.01,1);target=torch.arange(len(yy),device=yy.device);contrast=.5*(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target));return ce+contrast+.01*balance,scores

def fast_group(base,m,batches,bank,by,tokens,births,old_visual,old_text,counter=None):
 m._births=births;n=len(births);den=sum(len(yy) for _,yy,_ in batches);proto,_,rl=m.prototypes(bank,by,n);miss=sum(1 for k in births if k<len(m.branches)-1 and k not in old_text);tt=text_table(base,m,tokens,births,old_text);pp=proto.detach().requires_grad_(True);tx=tt.detach().requires_grad_(True);total=0.;scores=[]
 if counter is not None:counter['text_encoder_rows']+=sum(k==len(m.branches)-1 for k in births)+miss;counter['cached_text_rows']+=sum(k<len(m.branches)-1 for k in births)-miss
 for x,yy,j in batches:
  x=x.cuda();yy=yy.cuda();vs=[]
  for b,branch in enumerate(m.branches):
   if b<len(m.branches)-1:
    values,filled=old_visual[b];assert bool(filled[j].all());a=values[j].cuda();
    if counter is not None:counter['cached_visual_rows']+=len(j)
   else:
    a=visual(base,x,branch)
    if counter is not None:counter['real_image_encoder_rows']+=len(j)
   vs.append(a)
  loss,s=micro_loss(m,vs,tx,pp,yy);scale=len(yy)/den;assert torch.isfinite(loss);(loss*scale).backward();total+=float(loss.detach())*scale;scores.append(s.detach())
 targets=[];grads=[]
 if proto.requires_grad:targets.append(proto);grads.append(pp.grad)
 if tt.requires_grad:targets.append(tt);grads.append(tx.grad)
 if m.arm=='FULL':targets.append(rl);grads.append(rl.new_tensor(.1));total+=.1*float(rl.detach())
 torch.autograd.backward(targets,grads)
 return total,torch.cat(scores)

def slow_group(base,m,batches,bank,by,tokens,births):
 m._births=births;den=sum(len(y) for _,y,_ in batches);total=0.;scores=[]
 for x,yy,j in batches:
  x=x.cuda();yy=yy.cuda();proto,_,rl=m.prototypes(bank,by,len(births));tt=[];vs=[]
  for b,branch in enumerate(m.branches):
   ix=[i for i,k in enumerate(births) if k==b]
   with torch.set_grad_enabled(b==len(m.branches)-1):
    if ix:tt.append(textual(base,tokens[ix],branch))
    vs.append(visual(base,x,branch))
  loss,s=micro_loss(m,vs,torch.cat(tt),proto,yy)
  if m.arm=='FULL':loss=loss+.1*rl
  scale=len(yy)/den;(loss*scale).backward();total+=float(loss.detach())*scale;scores.append(s.detach())
 return total,torch.cat(scores)
