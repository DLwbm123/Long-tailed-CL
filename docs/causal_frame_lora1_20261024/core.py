"""Native last-four Q/V LoRA and arrival-only patient-oriented fixed simplex targets.
Core implementation only. No execution or qualification evidence is implied.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=['NATIVE_RIDGE','LEARNED_CE','FRAME_CE','FRAME_DR']


def symmetric_power(x,power,record):
    values,vectors=torch.linalg.eigh(x)
    record(dict(check='frame_symmetric_power',power=power,eigen_min=float(values.min()),eigen_max=float(values.max())))
    if power<0:assert float(values.min())>1e-12
    assert float(values.min())>=-1e-12
    return (vectors*values.clamp_min(0).pow(power))@vectors.T


class ArrivalFrame:
    """Only arrived class rows exist; total class count is declared protocol metadata."""
    def __init__(self,record,total_classes=20):
        assert total_classes>1
        self.total=total_classes;self.labels=[];self.rows=None;self.record=record

    def append(self,means,labels):
        assert len(means)==len(labels)==len(set(labels)) and not set(labels)&set(self.labels)
        assert len(labels)+len(self.labels)<=self.total
        mu=F.normalize(means.detach().cpu().double(),dim=1);rho=-1/(self.total-1)
        if self.rows is None:
            n=len(labels);g=(1-rho)*torch.eye(n,dtype=torch.float64)+rho*torch.ones(n,n,dtype=torch.float64)
            # Fixed patient orientation for this legal base set; no future row initialized.
            self.rows=symmetric_power(g,.5,self.record)@symmetric_power(mu@mu.T,-.5,self.record)@mu
        else:
            for mean in mu:
                gram=self.rows@self.rows.T
                coeff=torch.linalg.solve(gram,torch.full((len(gram),),rho,dtype=torch.float64))
                parallel=coeff@self.rows;radius=1-float(parallel@parallel)
                self.record(dict(check='arrival_frame_radius',old_rows=len(self.rows),radius=radius))
                assert radius>=-1e-10
                if len(self.rows)==self.total-1:new=parallel
                else:
                    residual=mean-torch.linalg.solve(gram,self.rows@mean)@self.rows
                    self.record(dict(check='arrival_frame_residual',old_rows=len(self.rows),norm=float(torch.linalg.vector_norm(residual))))
                    assert float(torch.linalg.vector_norm(residual))>1e-10
                    new=parallel+max(radius,0.)**.5*F.normalize(residual,dim=0)
                self.rows=torch.cat([self.rows,new[None]])
        self.labels.extend(labels)
        return self.rows

    def receipt(self):
        rho=-1/(self.total-1);n=len(self.labels)
        target=(1-rho)*torch.eye(n,dtype=torch.float64)+rho*torch.ones(n,n,dtype=torch.float64)
        return dict(arrived_labels=list(self.labels),rows=n,dimension=self.rows.shape[1],Gram_error=float(abs(self.rows@self.rows.T-target).max()))


class QVLoRA(nn.Module):
    def __init__(self,qkv,rank=8):
        super().__init__();self.base=copy.deepcopy(qkv).requires_grad_(False)
        d=qkv.in_features;assert qkv.out_features==3*d
        self.qA=nn.Parameter(torch.empty(rank,d,dtype=qkv.weight.dtype,device=qkv.weight.device))
        self.vA=nn.Parameter(torch.empty_like(self.qA))
        self.qB=nn.Parameter(torch.zeros(d,rank,dtype=qkv.weight.dtype,device=qkv.weight.device))
        self.vB=nn.Parameter(torch.zeros_like(self.qB))
        nn.init.kaiming_uniform_(self.qA,a=5**.5);nn.init.kaiming_uniform_(self.vA,a=5**.5)
        self.scale=1. # alpha=rank, rank8; no dropout or bias adapter.

    def forward(self,x):
        q,k,v=self.base(x).chunk(3,dim=-1)
        return torch.cat([q+self.scale*F.linear(F.linear(x,self.qA),self.qB),k,v+self.scale*F.linear(F.linear(x,self.vA),self.vB)],dim=-1)


class Model(nn.Module):
    def __init__(self,backbone,arm,rank=8):
        super().__init__();assert arm in ARMS[1:] and len(backbone.blocks)==12
        self.arm=arm;self.blocks=nn.Sequential(*copy.deepcopy(list(backbone.blocks.children())[8:]))
        self.norm=copy.deepcopy(backbone.norm);self.requires_grad_(False)
        for block in self.blocks:
            block.attn.fused_attn=False
            assert isinstance(block.attn.q_norm,nn.Identity) and isinstance(block.attn.k_norm,nn.Identity)
            assert isinstance(block.ls1,nn.Identity) and isinstance(block.ls2,nn.Identity)
            assert isinstance(block.mlp.norm,nn.Identity) and block.mlp.act.approximate=='none'
            block.attn.qkv=QVLoRA(block.attn.qkv,rank)
        self.register_parameter('head',None);self.labels=[];self.eval()

    def arrive(self,means,frame,labels):
        assert not set(labels)&set(self.labels) and len(means)==len(labels)
        reference=next(self.blocks.parameters());means=means.to(reference);frame=frame.to(reference)
        if self.arm=='LEARNED_CE':
            new=F.normalize(means.detach().clone(),dim=1)
            head=new if self.head is None else torch.cat([self.head.detach(),new.to(self.head)])
            self.head=nn.Parameter(head)
        else:
            assert len(frame)==len(self.labels)+len(labels)
            if self.head is not None:assert torch.equal(self.head.detach().cpu(),frame[:len(self.labels)].to(self.head).cpu())
            self.head=nn.Parameter(frame.detach().clone(),requires_grad=False)
        self.labels.extend(labels)

    def features(self,tokens):
        x=self.norm(self.blocks(tokens));return F.normalize(torch.cat([x[:,0],x[:,1:].mean(1)],dim=1),dim=1)

    def forward(self,tokens):return 25*self.features(tokens)@F.normalize(self.head,dim=1).T

    def loss(self,tokens,labels):
        z=self.features(tokens)
        if self.arm=='FRAME_DR':return .5*((z*F.normalize(self.head,dim=1)[labels]).sum(1)-1).square()
        return F.cross_entropy(25*z@F.normalize(self.head,dim=1).T,labels,reduction='none')

    def adapter_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items() if k=='head' or k.endswith(('.qA','.qB','.vA','.vB'))}


def merged_body(native_body,adapter):
    """NumPy reference receives original fixed blocks8..11 and actual low-rank state."""
    out={k:np.asarray(v,dtype=np.float64).copy() for k,v in native_body.items()}
    for i in range(4):
        p=f'blocks.{i}.attn.qkv.';w=out[p+'weight'];d=w.shape[1]
        w[:d]+=np.asarray(adapter[p+'qB'],dtype=np.float64)@np.asarray(adapter[p+'qA'],dtype=np.float64)
        w[2*d:]+=np.asarray(adapter[p+'vB'],dtype=np.float64)@np.asarray(adapter[p+'vA'],dtype=np.float64)
    return out
