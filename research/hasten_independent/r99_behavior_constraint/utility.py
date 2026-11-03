"""Float64 class utility and exact single-column counterfactuals."""
import torch
import torch.nn.functional as F

def class_values(scores,y):
 scores=scores.double();C=scores.shape[-1];own=scores.gather(-1,y[None,:,None].expand(scores.shape[0],-1,1)).squeeze(-1);wrong=scores.masked_fill(F.one_hot(y,C).bool()[None],-torch.inf).max(-1).values;v=torch.sigmoid(own-wrong);den=torch.bincount(y,minlength=C).double();assert (den>0).all();out=torch.zeros(scores.shape[0],C,device=scores.device,dtype=torch.double);return out.scatter_add(1,y[None].expand(len(scores),-1),v)/den

def from_values(v,ref,protected):return v.mean(-1)-(torch.relu(ref-v)[...,protected].mean(-1) if protected.any() else 0)
def utility(scores,y,ref,protected):return from_values(class_values(scores,y),ref,protected)

@torch.no_grad()
def counterfactual_credit(scores,ref_scores,y,ref_values,protected,active):
 scores=scores.double();ref_scores=ref_scores.double();G,Q,C=scores.shape;cols=torch.where(active)[0];own=scores.gather(-1,y[None,:,None].expand(G,-1,1)).squeeze(-1);wrong=scores.masked_fill(F.one_hot(y,C).bool()[None],-torch.inf);top,ind=wrong.topk(2,dim=-1);refcols=ref_scores[:,cols].T[None].expand(G,-1,-1)
 # Each replacement changes own score for its class and the competitor maximum for others.
 cw=torch.where(ind[:,:,0][:,None,:]==cols[None,:,None],top[:,:,1][:,None,:],top[:,:,0][:,None,:]);cw=torch.maximum(cw,refcols);cw=torch.where(y[None,None,:]==cols[None,:,None],top[:,:,0][:,None,:],cw);co=torch.where(y[None,None,:]==cols[None,:,None],refcols,own[:,None,:]);val=torch.sigmoid(co-cw);agg=torch.zeros(G,len(cols),C,device=scores.device,dtype=torch.double);agg.scatter_add_(2,y[None,None].expand(G,len(cols),Q),val);agg/=torch.bincount(y,minlength=C)[None,None];u=utility(scores,y,ref_values,protected);minus=from_values(agg,ref_values,protected);D=torch.zeros(G,C,device=scores.device,dtype=torch.double);D[:,cols]=u[:,None]-minus;return u,D
