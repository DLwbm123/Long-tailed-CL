"""TRAIN-only endpoint geometry and policy summaries. No gradient or optimizer."""
import math
import torch
import torch.nn.functional as F

def h_snapshot(p):
 h=torch.zeros_like(p,dtype=torch.double);p=p.double()
 for k in range(1,8):
  q=k/8;v=q*(1-q);h+=math.comb(8,k)*p.pow(k)*(1-p).pow(7-k)*v/(math.sqrt(v)+1e-8)
 return h

def summarize(z,y,fold,nclass):
 z=z.cpu().double();y=y.cpu();fold=fold.cpu();assert len(z)==len(y)==len(fold) and torch.isfinite(z).all();assert torch.allclose(z.norm(dim=1),torch.ones(len(z),dtype=torch.double),atol=2e-6)
 counts=torch.bincount(y,minlength=nclass);assert bool((counts>1).all())
 sums=torch.zeros(nclass,z.shape[1],dtype=torch.double);sums.index_add_(0,y,z)
 sq=torch.zeros(nclass,dtype=torch.double);sq.index_add_(0,y,z.square().sum(1));means=sums/counts[:,None];proto=F.normalize(means,dim=1);assert bool((means.norm(dim=1)>1e-8).all())
 fs=[];fc=[]
 for b in [0,1]:
  m=fold==b;v=torch.zeros_like(sums);v.index_add_(0,y[m],z[m]);c=torch.bincount(y[m],minlength=nclass);assert bool((c>0).all());fs.append(v);fc.append(c)
 records=[];allp={k:torch.empty(len(y),dtype=torch.double) for k in ['raw','natural']};margin=torch.empty(len(y),dtype=torch.double);correct={k:torch.empty(len(y),dtype=torch.bool) for k in allp}
 for b in [0,1]:
  mask=fold==b;yy=y[mask];support=F.normalize(fs[1-b]/fc[1-b][:,None],dim=1)
  raw=20*z[mask]@support.T;own=raw.gather(1,yy[:,None]).squeeze(1);wrong=raw.clone();wrong.scatter_(1,yy[:,None],float('-inf'));margin[mask]=own-wrong.max(1).values
  for mode,logits in [('raw',raw),('natural',raw+counts.double().log()[None,:])]:
   allp[mode][mask]=logits.softmax(1).gather(1,yy[:,None]).squeeze(1);correct[mode][mask]=logits.argmax(1)==yy
 cos=proto@proto.T;cos.fill_diagonal_(float('-inf'))
 for c in range(nclass):
  m=y==c;nn=int(counts[c]);var=(sq[c]-sums[c].square().sum()/nn)/(nn-1);assert var>=-1e-5
  rec=dict(class_index=c,count=nn,fold_counts=[int(v[c]) for v in fc],mean_embedding_norm=float(means[c].norm()),unbiased_within_trace=float(var.clamp_min(0)),mean_distinct_pair_cosine=float((sums[c].square().sum()-sq[c])/(nn*(nn-1))),nearest_other_centroid_cosine=float(cos[c].max()),nearest_other_class=int(cos[c].argmax()),crossfold_raw_margin_mean=float(margin[m].mean()),crossfold_raw_margin_quantiles=torch.quantile(margin[m],torch.tensor([0.,.25,.5,.75,1.],dtype=torch.double)).tolist())
  for mode in allp:
   p=allp[mode][m];rec[mode]=dict(correct=int(correct[mode][m].sum()),correct_probability_mean=float(p.mean()),correct_probability_quantiles=torch.quantile(p,torch.tensor([0.,.25,.5,.75,1.],dtype=torch.double)).tolist(),endpoint_hypothetical_snapshot_h_mean=float(h_snapshot(p).mean()))
  records.append(rec)
 moments=dict(counts=counts,sums=sums,squared_norm_sums=sq,fold_counts=torch.stack(fc),fold_sums=torch.stack(fs),prototypes=proto)
 return records,moments
