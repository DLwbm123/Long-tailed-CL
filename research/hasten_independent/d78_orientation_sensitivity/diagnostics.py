"""Paired sufficient statistics; no individual predictions are written."""
import torch

def paired_stats(a,b,y,nclass):
 assert a.shape==b.shape and a.shape==(len(y),nclass)
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 pa,pb=a.argmax(1),b.argmax(1);ca,cb=pa==y,pb==y
 ap=a.softmax(1)[torch.arange(len(y)),y];bp=b.softmax(1)[torch.arange(len(y)),y]
 aa=a.clone();bb=b.clone();aa[torch.arange(len(y)),y]=-torch.inf;bb[torch.arange(len(y)),y]=-torch.inf
 am=a[torch.arange(len(y)),y]-aa.max(1).values;bm=b[torch.arange(len(y)),y]-bb.max(1).values
 out=[]
 for c in range(nclass):
  k=y==c;n=int(k.sum());assert n>0
  out.append(dict(class_index=c,count=n,original_correct=int(ca[k].sum()),flipped_correct=int(cb[k].sum()),both_correct=int((ca[k]&cb[k]).sum()),original_only_correct=int((ca[k]&~cb[k]).sum()),flipped_only_correct=int((~ca[k]&cb[k]).sum()),neither_correct=int((~ca[k]&~cb[k]).sum()),prediction_disagreements=int((pa[k]!=pb[k]).sum()),original_true_probability_mean=float(ap[k].double().mean()),flipped_true_probability_mean=float(bp[k].double().mean()),original_true_margin_mean=float(am[k].double().mean()),flipped_true_margin_mean=float(bm[k].double().mean())))
 return out
