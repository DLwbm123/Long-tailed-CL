"""Shared support/query geometry and instance policy, anchored to strong teacher."""
import copy,math
import torch
from torch import nn
import torch.nn.functional as F
ARMS=('UNIFORM','CE_WEIGHT','RLPO_STYLE','PROTO_GRPO')

def unit(z):return F.normalize(z,dim=-1,eps=1e-8)

class Joint(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.V=nn.Linear(512,16,bias=False);self.U=nn.Linear(16,512,bias=False)
        nn.init.zeros_(self.U.weight)
        self.policy=nn.Sequential(nn.Linear(1032,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.policy[-1].weight);nn.init.zeros_(self.policy[-1].bias)
        self.reference=copy.deepcopy(self.policy).requires_grad_(False)
        if arm=='UNIFORM':self.policy.requires_grad_(False)
    def adapt(self,z):return unit(z+self.U(F.gelu(self.V(z))))
    @torch.no_grad()
    def update_reference(self):
        for ref,p in zip(self.reference.parameters(),self.policy.parameters()):ref.mul_(.9).add_(p,alpha=.1)
    def representation_parameters(self):return list(self.U.parameters())+list(self.V.parameters())
    def policy_parameters(self):return list(self.policy.parameters())

def geometry(h,d):
    parts=[h[ix] for ix in d['ix']];mean=torch.stack([unit(z.mean(0)) for z in parts])
    sim=mean@mean.T;C=len(parts);b=[];rb=[]
    for cl,z in enumerate(parts):
        with torch.no_grad():
            v=sim[cl].clone();v[cl]=-torch.inf
            near=v.topk(min(3,C-1)).values;near=F.pad(near,(0,3-len(near)))
            raw=d['rawcounts'][cl];n=len(z);birth=d['births'][cl]
            scal=z.new_tensor([math.log1p(raw)/math.log(6822),math.log1p(n)/math.log(6822),n>1,(d['stage']-birth)/d.get('stage_denominator',5),birth==d['stage']])
            extra=torch.cat([near,scal])
        features=torch.cat([z,mean[cl].expand(len(z),-1),extra.expand(len(z),-1)],1)
        b.append(features)
    return parts,b

def logits(model,h,d):
    parts,features=geometry(h,d)
    if model.arm=='UNIFORM':return parts,[z.new_zeros(len(z)) for z in parts],[z.new_zeros(len(z)) for z in parts]
    out=[model.policy(v).squeeze(-1) for v in features]
    with torch.no_grad():ref=[model.reference(v.detach()).squeeze(-1) for v in features]
    return parts,out,ref

def prototypes(parts,actions,d):
    # Baseline anchor has the same historical information budget in every arm.
    # Support and query transformation are current, never frozen at old geometry.
    p=[];ws=[]
    for cl,(z,a) in enumerate(zip(parts,actions)):
        w=a.softmax(-1);ws.append(w)
        p.append(unit(d['base'][cl]+w@z-d['raw_mean'][cl]))
    return torch.stack(p,-2),ws

def deterministic(model,d,detach_geometry=False):
    h=model.adapt(d['S']);q=model.adapt(d['B'])
    if detach_geometry:h=h.detach();q=q.detach()
    parts,b,ref=logits(model,h,d);p,w=prototypes(parts,b,d)
    score=20*q@p.T
    return score,parts,b,ref,p,w

def classmean(x,y,C):
    den=torch.bincount(y,minlength=C).double();assert (den>0).all()
    out=torch.zeros(*x.shape[:-1],C,dtype=torch.double,device=x.device)
    out.scatter_add_(-1,y.expand(*x.shape[:-1],-1),x.double())
    return out/den

def behavior_kl(score,reference,y,C):
    a=reference.double().log_softmax(-1).detach();b=score.double().log_softmax(-1)
    return classmean((a.exp()*(a-b)).sum(-1),y,C).mean()

def weight_kl(b,ref):
    return torch.stack([(x.softmax(0)*(x.log_softmax(0)-r.log_softmax(0))).sum() for x,r in zip(b,ref)]).mean()

def reward(scores,d):
    y=d['By'];C=d['C'];ref=d['teacher_scores'].double()
    def values(v):
        pred=v.argmax(-1);correct=(pred==y).double()
        own=v.gather(-1,y.expand(*v.shape[:-2],-1)[...,None]).squeeze(-1)
        wrong=v.masked_fill(F.one_hot(y,C).bool(),-torch.inf).max(-1).values
        return classmean(correct,y,C),classmean(torch.tanh((own-wrong)/2),y,C),correct
    acc,margin,correct=values(scores.double());ra,rm,rc=values(ref)
    harm=classmean(((rc>0)&(correct==0)).double(),y,C)
    h=harm[...,d['protected']].mean(-1) if d['protected'].any() else harm.mean(-1)*0
    r=(acc-ra).mean(-1)+.1*(margin-rm).mean(-1)-h
    return r,{'hard_macro_delta':(acc-ra).mean(-1),'dense_delta':(margin-rm).mean(-1),'harm':h}

def supervised(model,d):
    score,parts,b,ref,p,w=deterministic(model,d)
    ce=F.cross_entropy(score+d['nB'].log()[None],d['By'])
    bhv=behavior_kl(score,d['teacher_scores'],d['By'],d['C'])
    wk=weight_kl(b,ref)
    loss=ce+.1*bhv+.01*wk
    return loss,{'CE':float(ce.detach()),'teacher_behavior_KL':float(bhv.detach()),'weight_KL':float(wk.detach())}

def auxiliary(model,d,eps=None):
    with torch.no_grad():h=model.adapt(d['S']);q=model.adapt(d['B'])
    parts,b,ref=logits(model,h,d);p,w=prototypes(parts,b,d);score=20*q@p.T
    bhv=behavior_kl(score,d['teacher_scores'],d['By'],d['C']);wk=weight_kl(b,ref)
    stats={'teacher_behavior_KL':float(bhv.detach()),'weight_KL':float(wk.detach())}
    if model.arm=='CE_WEIGHT':
        objective=F.cross_entropy(score+d['nB'].log()[None],d['By'])
    elif model.arm=='RLPO_STYLE':
        z=torch.cat(parts);labels=d['Sy'];sim=z@p.T
        own=sim.gather(1,labels[:,None]).squeeze(1)
        wrong=sim.masked_fill(F.one_hot(labels,d['C']).bool(),-torch.inf).max(1).values
        pc=p@p.T;collapse=(pc.sum(1)-pc.diag())/(d['C']-1)
        rw=own-wrong-.1*collapse[labels]
        adv=((rw-rw.mean())/(rw.std(correction=0)+1e-8)).detach()
        # S is class-concatenated, preserving the original sample-level RLPO form.
        objective=-(torch.cat([x.log_softmax(0) for x in b])*adv).mean()
        stats.update(raw_geometric_reward=float(rw.detach().mean()),reward_std=float(rw.detach().std(correction=0)))
    elif model.arm=='PROTO_GRPO':
        assert eps is not None
        actions=[(x[None]+.25*e).detach() for x,e in zip(b,eps)]
        pp,_=prototypes(parts,actions,d)
        with torch.no_grad():
            sg=20*torch.einsum('qd,gcd->gqc',q,pp)
            R,values=reward(sg,d);A=(R-(R.sum()-R)/7).detach()
        terms=[(-.5*((a-x[None])/.25).square()-math.log(.25)-.5*math.log(2*math.pi)).sum(-1) for x,a in zip(b,actions) if len(x)>1]
        logp=torch.stack(terms).sum(0);old=logp.detach();ratio=(logp-old).exp()
        objective=-torch.minimum(ratio*A,ratio.clamp(.8,1.2)*A).mean()
        stats.update(reward_mean=float(R.mean()),reward_std=float(R.std(correction=0)),raw_advantage_std=float(A.std(correction=0)),effective_group=int(float(R.max()-R.min())>1e-12),sample_hard_macro_delta=float(values['hard_macro_delta'].mean()),sample_dense_delta=float(values['dense_delta'].mean()),sample_harm=float(values['harm'].mean()))
    else:raise ValueError(model.arm)
    loss=objective+.1*bhv+.01*wk
    return loss,stats

@torch.no_grad()
def diagnostics(model,d):
    score,parts,b,ref,p,w=deterministic(model,d)
    y=d['By'];pred=score.argmax(1);t=d['teacher_scores'].argmax(1);R,v=reward(score,d)
    recall=classmean((pred==y).double(),y,d['C'])
    return {'B_micro':float((pred==y).double().mean()*100),'B_macro':float(recall.mean()*100),'B_tail':float(recall[d['tail']].mean()*100) if d['tail'].any() else None,'teacher_rescue':int(((pred==y)&(t!=y)).sum()),'teacher_harm':int(((pred!=y)&(t==y)).sum()),'prediction_changes':int((pred!=t).sum()),'deterministic_reward':float(R),'weight_ESS_fraction':float(torch.stack([(1/x.square().sum())/len(x) for x in w]).mean()),'max_weight_multisupport':max(float(x.max()) for x in w if len(x)>1),'single_support_classes':sum(len(x)==1 for x in w),'adapter_U_norm':float(model.U.weight.norm()),'prototype_angle_from_teacher':float(torch.acos((p*d['base']).sum(1).clamp(-1,1)).mean())}
