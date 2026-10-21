"""Two exact moment calculations; no sample or query loader."""
import json,time
from pathlib import Path
import numpy as np
import torch


def write(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)


def assembled(a,b,c):
    d=len(a)-1;g=np.zeros((2*d+1,2*d+1));g[:d,:d]=a[:-1,:-1];g[d:2*d,d:2*d]=b[:-1,:-1];g[:d,d:2*d]=c;g[d:2*d,:d]=c.T
    g[:d,-1]=a[:-1,-1];g[-1,:d]=a[-1,:-1];g[d:2*d,-1]=b[:-1,-1];g[-1,d:2*d]=b[-1,:-1];g[-1,-1]=a[-1,-1]
    return g


def scalar_record(f,mean,j,mass):
    k=len(mean);r=np.full(k,-1/(k-1));r[j]=1
    gap=float(mean@r);variance=float(r@f@r-gap*gap/mass)
    assert variance>=-1e-8
    return dict(half_squared_risk=float((np.trace(f)-2*mean[j]+mass*(1-1/k))/2),score_second_moment=float(np.trace(f)),true_mean_score=float(mean[j]),mean_gap_to_average_wrong=gap,variance_gap_to_average_wrong=variance)


def block_record(a,b,c,w,j):
    d=len(a)-1;u,v,z=w[:d],w[d:-1],w[-1];mu=a[:-1,-1]@u+b[:-1,-1]@v;mass=float(a[-1,-1])
    q=u.T@c@v;f=u.T@a[:-1,:-1]@u+v.T@b[:-1,:-1]@v+q+q.T+np.outer(mu,z)+np.outer(z,mu)+mass*np.outer(z,z)
    return scalar_record(f,mu+mass*z,j,mass)


def self_check():
    rng=np.random.default_rng(74002);x=rng.normal(size=(9,4));x=np.column_stack([x,np.ones(9)]);m=np.arange(1,10,dtype=float);m/=m.sum();w=rng.normal(size=(5,3));w-=w.mean(1,keepdims=True)
    g=x.T@(m[:,None]*x);a=g[np.ix_([0,1,4],[0,1,4])];b=g[np.ix_([2,3,4],[2,3,4])];c=g[:2,2:4]
    assert np.max(np.abs(assembled(a,b,c)-g))<1e-12
    rec=block_record(a,b,c,w,1);scores=x@w;t=np.eye(3)[1]-1/3
    direct=float((m[:,None]*(scores-t)**2).sum()/2)
    assert abs(rec['half_squared_risk']-direct)<1e-12


def inputs(cfg):
    ctl=Path(cfg['control']);seal=json.loads((ctl/'FIT_SEAL.json').read_text());old=torch.load(ctl/'heads_5.private.pt',weights_only=True)
    w=old['df_matched_joint'].numpy();w=w-w.mean(1,keepdims=True)
    ce=np.load(Path(cfg['ce'])/'head.private.npy');assert ce.shape==w.shape==(1537,20) and np.max(np.abs(ce.mean(1)))<1e-12
    assert json.loads((Path(cfg['ce'])/'VERIFICATION.json').read_text())['state']=='PASS'
    return dict(ridge=w,offline_CE=ce),old['factors'].numpy(),seal['stages'][-1]['matched_alpha']


def moments(cfg,c):
    a=torch.load(Path(cfg['prior'])/f'class_{c:02d}.private.pt',weights_only=True)['ridge_xx'].numpy()
    b=torch.load(Path(cfg['matched'])/f'class_{c:02d}.private.pt',weights_only=True)['patch_mean_gram'].numpy()
    cross=torch.load(Path(cfg['joint'])/f'cross_{c:02d}.private.pt',weights_only=True)['cross'].numpy()
    assert abs(a[-1,-1]-1)<1e-12 and abs(b[-1,-1]-1)<1e-12
    return a,b,cross
