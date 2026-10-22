"""Stable learned local evidence, patient-mode statistics and all-seen composition."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['GLOBAL','UNSHRUNK','CALIBRATED']


def memory_indices(rows,cap,seed):
    selected=[]
    for c in sorted({r['label'] for r in rows}):
        pp={}
        for i,r in enumerate(rows):
            if r['label']==c:pp.setdefault(r['group'],[]).append(i)
        def key(g,path=''):return hashlib.sha256(f'{seed}:{c}:{g}:{path}'.encode()).digest()
        for g in sorted(pp,key=lambda g:(key(g),g))[:cap]:selected.append(min(pp[g],key=lambda i:(key(g,rows[i]['relative_path']),i)))
    return selected


class EvidenceMap(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg;d=cfg['part_dimension'];m=cfg['parts']
        self.project=nn.Linear(768,d);self.queries=nn.Parameter(torch.randn(m,d)*.02)
        self.decoder=nn.Linear(d,768)
    def forward(self,tokens,reconstruct=False):
        raw=F.normalize(tokens,dim=-1);h=F.normalize(self.project(raw),dim=-1)
        assignment=F.softmax(h[:,1:]@F.normalize(self.queries,dim=1).T/self.cfg['assignment_temperature'],dim=-1)
        attention=assignment/assignment.sum(1,keepdim=True).clamp_min(1e-12)
        local=F.normalize(attention.transpose(1,2)@h[:,1:],dim=-1)
        features=torch.cat([h[:,0],local.flatten(1)],1)
        if not reconstruct:return features
        return features,assignment@self.decoder(local),raw[:,1:]


def primitive_objective(model,head,tokens,labels,cfg):
    clean=F.normalize(tokens,dim=-1)
    masked=clean.clone()
    masked[:,1:]*=(torch.rand((len(masked),masked.shape[1]-1,1),device=masked.device)>=cfg['patch_dropout'])/(1-cfg['patch_dropout'])
    features,reconstruction,_=model(masked,reconstruct=True)
    ce=F.cross_entropy(cfg['logit_scale']*F.normalize(features,dim=1)@F.normalize(head,dim=1).T,labels)
    reconstruction_loss=(reconstruction-clean[:,1:]).square().mean()*768
    q=F.normalize(model.queries,dim=1);off=q@q.T-torch.eye(len(q),device=q.device,dtype=q.dtype)
    diversity=off.square().sum()/(len(q)*(len(q)-1))
    return ce+cfg['reconstruction_weight']*reconstruction_loss+cfg['dictionary_diversity_weight']*diversity,ce,reconstruction_loss


def patient_features(features,rows,c):
    pp={}
    for i,r in enumerate(rows):
        if r['label']==c:pp.setdefault(r['group'],[]).append(i)
    return np.stack([features[ix].double().mean(0).numpy() for ix in pp.values()])


def base_prior(features,rows,classes,cfg):
    d=cfg['part_dimension'];m=cfg['parts'];v=[]
    for c in classes:
        x=patient_features(features,rows,c)[:,d:].reshape(-1,m,d)
        v.append(np.var(x,axis=0,ddof=1) if len(x)>1 else np.zeros((m,d)))
    return np.maximum(np.mean(v,axis=0),cfg['variance_floor'])


def fit_classes(features,rows,classes,prior,cfg):
    """Deterministic two-mode patient clustering; no old images/features or optimization."""
    result=[];d=cfg['part_dimension'];m=cfg['parts'];cost=dict(patient_rows=0,kmeans_distance_pairs=0)
    for c in classes:
        x=patient_features(features,rows,c);n=len(x);cost['patient_rows']+=n
        k=min(cfg['modes'],n);centers=[x[0]]
        while len(centers)<k:
            dist=np.stack([((x-t)**2).sum(1) for t in centers],1).min(1)
            centers.append(x[int(dist.argmax())])
        centers=np.stack(centers);assignment=np.zeros(n,dtype=int)
        for _ in range(cfg['kmeans_iterations']):
            distances=((x[:,None]-centers[None])**2).sum(2);assignment=distances.argmin(1)
            cost['kmeans_distance_pairs']+=n*k
            for j in range(k):
                if np.any(assignment==j):centers[j]=x[assignment==j].mean(0)
        # Recompute membership at the retained centers, then retain its exact empirical moments.
        assignment=((x[:,None]-centers[None])**2).sum(2).argmin(1);cost['kmeans_distance_pairs']+=n*k
        mu=[];variance=[];calibrated=[];weights=[];counts=[]
        for j in range(cfg['modes']):
            member=x[assignment==j] if j<k else x[:0]
            count=len(member);parts=member[:,d:].reshape(-1,m,d)
            mean=parts.mean(0) if count else x[:,d:].reshape(-1,m,d).mean(0)
            var=parts.var(0,ddof=1) if count>1 else np.zeros((m,d))
            rho=count/(count+cfg['prior_patient_strength'])
            mu.append(mean);variance.append(np.maximum(var,cfg['variance_floor']))
            calibrated.append(np.maximum(rho*var+(1-rho)*prior,cfg['variance_floor']))
            weights.append(count/n);counts.append(count)
        result.append(dict(label=c,mean=np.stack(mu),variance=np.stack(variance),calibrated=np.stack(calibrated),weights=np.array(weights),counts=np.array(counts),patients=n,global_mean=x.mean(0)))
    return result,cost


def stack_stats(records,device='cpu',dtype=torch.float32):
    return {key:torch.tensor(np.stack([r[key] for r in records]),device=device,dtype=dtype) for key in ['mean','variance','calibrated','weights','global_mean']}


def local_evidence(features,stats,calibrated,cfg):
    d=cfg['part_dimension'];z=features[:,d:].reshape(-1,cfg['parts'],d)
    mean=stats['mean'];variance=stats['calibrated' if calibrated else 'variance']
    # [batch,class,mode,part]; Gaussian volume correction prevents broad modes winning by radius.
    gaussian=-.5*(((z[:,None,None]-mean[None]).square()/variance[None])+torch.log(2*torch.pi*variance)[None]).sum(-1)
    class_mean=(mean*stats['weights'][:,:,None,None]).sum(1)
    match=(F.normalize(z,dim=-1)[:,None]*F.normalize(class_mean,dim=-1)[None]).sum(-1)
    attention=F.softmax(match/cfg['class_attention_temperature'],dim=-1)
    per_mode=(gaussian*attention[:,:,None]).sum(-1)
    return cfg['logit_scale']/d*torch.logsumexp(per_mode+torch.log(stats['weights'].clamp_min(1e-30))[None],dim=-1)


class Classifier(nn.Module):
    def __init__(self,means):
        super().__init__();self.head=nn.Parameter(F.normalize(means.clone(),dim=1));self.mixing=nn.Parameter(torch.tensor([0.,-2.]))
    def expand(self,means):self.head=nn.Parameter(torch.cat([self.head.detach(),F.normalize(means,dim=1)]))
    def forward(self,features,stats,arm,cfg):
        global_score=cfg['logit_scale']*F.normalize(features,dim=1)@F.normalize(self.head,dim=1).T
        if arm=='GLOBAL':return global_score+0*self.mixing.sum()
        local=local_evidence(features,stats,arm=='CALIBRATED',cfg)
        weights=F.softmax(self.mixing,dim=0)
        return weights[0]*global_score+weights[1]*local


def objective(model,x,y,stats,arm,cfg):
    score=model(x,stats,arm,cfg);ce=F.cross_entropy(score,y)
    head=F.normalize(model.head,dim=1);centers=F.normalize(stats['global_mean'],dim=1)
    align=(1-(head*centers).sum(1)).mean()
    k=len(head);gram=head@head.T
    mask=~torch.eye(k,device=x.device,dtype=torch.bool)
    geometry=(gram[mask]+1/(k-1)).square().mean()
    return ce+cfg['center_alignment_weight']*align+cfg['class_geometry_weight']*geometry,ce
