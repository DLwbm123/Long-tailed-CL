"""Coordinate Gaussian KL versus class-balanced raw prediction reference KL."""
import torch
from prototype_policy import gaussian_KL

def behavior_KL(scores,reference,y):
 lp=scores.double().log_softmax(-1);lr=reference.detach().double().log_softmax(-1);row=(lr.exp()*(lr-lp)).sum(-1);n=scores.shape[-1];den=torch.bincount(y,minlength=n).double();assert (den>0).all();return (row/den[y]).sum()/n

def preservation(kind,mu,ref_mu,scores,ref_scores,y,active,kappa):
 if kind=='ACT':return .04*gaussian_KL(mu,ref_mu.detach(),active)
 if kind=='BHV':return .04*kappa*behavior_KL(scores,ref_scores,y)
 assert kind=='NONE';return scores.sum()*0
