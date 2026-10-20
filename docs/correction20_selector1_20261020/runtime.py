"""Read-only training cache operations, with explicit fit row lineage."""
import json,zlib
from pathlib import Path
import numpy as np
import torch
from core import weights,spectrum,matched_alpha


def write(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class Data:
 def __init__(self,cfg):
  self.cfg=cfg;source=Path(cfg['cache_run']);idx=np.load(source/'train_index.private.npz');self.y=idx['labels'].astype(int);self.g=idx['groups'];self.fold=idx['folds'];self.cache=np.load(source/'train768.private.npy',mmap_mode='r');self.ci=torch.tensor(list(range(768))+[1536])
  assert self.cache.shape==(68058,1536) and np.array_equal(np.unique(self.y),np.arange(20))
  order=np.argsort(self.g,kind='stable');assert np.all((self.g[order][1:]!=self.g[order][:-1])|(self.fold[order][1:]==self.fold[order][:-1]))
  identity=json.loads((source/'config.private.json').read_text());assert identity['seed']==cfg['seed'] and identity['order']==cfg['order']
  assert json.loads((source/'status.private.json').read_text())['state']=='COMPLETE'
 def x(self,ids):
  return torch.cat([torch.from_numpy(np.array(self.cache[ids],copy=True)).double(),torch.ones(len(ids),1,dtype=torch.float64)],1)
 def moment(self,ids):
  yy=self.y[ids];ww=weights(yy,self.g[ids]);raw=torch.zeros(1537,1537,dtype=torch.float64);base=raw.clone();b=torch.zeros(1537,20,dtype=torch.float64)
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];x=self.x(rows);w=torch.from_numpy(ww[off:off+len(rows)]);raw+=x.T@(w[:,None]*x);mask=self.y[rows]<10;xx=x[mask];base+=xx.T@(w[mask,None]*xx)
   for c in np.unique(self.y[rows]):b[:,c]+=(x[self.y[rows]==c]*w[self.y[rows]==c,None]).sum(0)
  fac=torch.ones(1537,dtype=torch.float64);fac[:768]=2**-.5;fac[768:1536]=float(base[:768,:768].trace()/base[768:1536,768:1536].trace())**.5*2**-.5
  proto=(b[:1536]*fac[:1536,None]).T;norm=torch.linalg.vector_norm(proto,dim=1);present=np.bincount(yy,minlength=20)>0;proto=proto/norm.clamp_min(1e-300)[:,None]
  return raw,b,fac,proto,present
 def scores(self,ids,head,key='joint'):
  result=[]
  for off in range(0,len(ids),2048):
   x=self.x(ids[off:off+2048]);result.append((x[:,self.ci]@head['CLS'] if key=='CLS' else (x*head['factors'])@head[key]).numpy())
  return np.concatenate(result)
 def cosine(self,ids,head):
  result=[]
  for off in range(0,len(ids),2048):
   z=self.x(ids[off:off+2048])[:,:1536]*head['factors'][:1536];z=z/torch.linalg.vector_norm(z,dim=1).clamp_min(1e-300)[:,None];c=(z@head['prototypes'].T).numpy();c[:,~head['present']]=0.;result.append(c)
  return np.concatenate(result)
 def subfold(self,ids):
  return np.array([zlib.crc32(('utility-probe-temp-v1:'+str(v)).encode('utf8'))%2 for v in self.g[ids]],dtype=int)
 def ridge(self,a,b):
  h=torch.linalg.solve(a,b);r=float((a@h-b).abs().max()/max(1.,float(b.abs().max())));assert r<1e-9;return h,r
 def df(self,raw,b,fac):
  ev=spectrum(raw[self.ci][:,self.ci]);target=float((ev/(ev+1)).sum());ng=raw*fac[:,None]*fac[None,:];alpha=matched_alpha(spectrum(ng),target);h,r=self.ridge(ng+alpha*torch.eye(1537,dtype=torch.float64),b*fac[:,None]);return h,alpha,r


def rms(scores,y,g):
 w=weights(y,g);s=scores-scores.mean(1,keepdims=True);t=float(np.sqrt(np.sum(w[:,None]*s*s)/(20*w.sum())));assert t>1e-12 and np.isfinite(t);return t


def summary(scores,y,g,fold,cfg):
 from core import metrics
 results=[]
 for outer in [None,0,1,2]:
  mask=np.ones(len(y),dtype=bool) if outer is None else fold==outer;ss={k:v[mask] for k,v in scores.items()};yy=y[mask];gg=g[mask];ok={k:v.argmax(1)==yy for k,v in ss.items()};iw=1/(20*np.bincount(yy,minlength=20)[yy]);pw=weights(yy,gg);pw/=pw.sum();contrasts={}
  for a in ss:
   for ref in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction']:
    if a==ref:continue
    rescue=ok[a]&~ok[ref];harm=~ok[a]&ok[ref];contrasts[a+'-'+ref]=dict(rescue=int(rescue.sum()),harm=int(harm.sum()),BA_rescue=float(iw@rescue),BA_harm=float(iw@harm),delta_BA=float(iw@(rescue.astype(float)-harm)),PP_rescue=float(pw@rescue),PP_harm=float(pw@harm))
  masks={'both_correct':ok['cls768_ridge']&ok['scale_joint'],'CLS_only':ok['cls768_ridge']&~ok['scale_joint'],'joint_only':~ok['cls768_ridge']&ok['scale_joint'],'both_wrong':~ok['cls768_ridge']&~ok['scale_joint']}
  groups={k:dict(images=int(m.sum()),BA_mass=float(iw@m),PP_mass=float(pw@m),correct_BA_mass={a:float(iw@(m&v)) for a,v in ok.items()},correct={a:int((m&v).sum()) for a,v in ok.items()}) for k,m in masks.items()}
  results.append(dict(scope='pooled' if outer is None else f'outer_{outer}',metrics={a:metrics(v,yy,gg,cfg['order'],cfg['frequency_bins']) for a,v in ss.items()},contrasts=contrasts,groups=groups))
 return results
