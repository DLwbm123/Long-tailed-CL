"""Fixed-cap transforms; residualization is a change of coordinates."""
import torch
from core_attribution import weights,norm_fit,stats,solve,ba,ALPHAS
BUDGET=19058800
ARMS=['G','R','P','C','S']
PERM=torch.randperm(1536,generator=torch.Generator().manual_seed(74010))
def nbytes(v):
    if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
    if isinstance(v,dict):return sum(nbytes(x) for x in v.values())
    if isinstance(v,(tuple,list)):return sum(nbytes(x) for x in v)
    return 0
def planned(a,d):
    z=d if a in ['G','S'] else d+128
    return 8*((z+1)**2+2*(z+1)*6+1537+(769 if a!='G' else 0)+(z+1)+(768*128 if a in ['R','P','C'] else 0)+(d*128 if a=='C' else 0)+(z if a=='S' else 0))+64
WIDTH={a:max(d for d in range(1,2305 if a=='S' else 1537) if planned(a,d)<=BUDGET) for a in ARMS}
def aug(x):return torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)
def project(g,h,t):
    if t['arm']=='G':return g[:,PERM[:t['d']]]
    if t['arm']=='S':return torch.cat([g,h],1)[:,t['indices']]
    core=g[:,PERM[:t['d']]];extra=h@t['U']
    if t['arm']=='C':extra-=core@t['V']
    return torch.cat([core,extra],1)
def mapped(g,h,t):
    g=(g-t['g_norm'][0])/t['g_norm'][1]
    if t['arm']!='G':h=(h-t['h_norm'][0])/t['h_norm'][1]
    q=project(g,h,t);return aug((q-t['out_norm'][0])/t['out_norm'][1])
def top(c):
    _,u=torch.linalg.eigh(c.cuda());u=u[:,-128:].cpu()
    idx=u.abs().argmax(0);sign=u[idx,torch.arange(128)].sign();return u*sign
def supervised(x,y,w,d):
    x=aug(x).cuda();y=y.cuda();w=w.cuda()/4;hot=torch.nn.functional.one_hot(y,4)
    L=float(torch.linalg.eigvalsh(x.T@(x*w[:,None]))[-1]);step=1/max(L,1e-12)
    beta=torch.zeros(x.shape[1],4,dtype=x.dtype,device=x.device)
    initial=x.T@((torch.full_like(hot,.25,dtype=x.dtype)-hot)*w[:,None]);pen=.01*float(initial[:-1].norm(dim=1).max())
    for _ in range(500):
        grad=x.T@((torch.softmax(x@beta,1)-hot)*w[:,None]);nextb=beta-step*grad
        norm=nextb[:-1].norm(dim=1).clamp_min(1e-30);nextb[:-1]*=(1-step*pen/norm).clamp_min(0)[:,None]
        residual=float((nextb-beta).norm()/max(step,1e-30));beta=nextb
    importance=beta[:-1].norm(dim=1).cpu();indices=torch.argsort(importance,descending=True,stable=True)[:d]
    return indices,dict(iterations=500,step=step,group_penalty=pen,proximal_residual=residual)
def transforms(g,h,y,groups):
    w=weights(y,groups);gn=norm_fit(g,w);hn=norm_fit(h,w);gg=(g-gn[0])/gn[1];hh=(h-hn[0])/hn[1]
    out={};selector={}
    for arm in ARMS:
        d=WIDTH[arm];t=dict(arm=arm,d=d,g_norm=gn)
        if arm!='G':t['h_norm']=hn
        if arm=='R':t['U']=torch.linalg.qr(torch.randn(768,128,generator=torch.Generator().manual_seed(74011),dtype=torch.float64),mode='reduced')[0]
        if arm=='P':t['U']=top(hh.T@(hh*w[:,None])/4)
        if arm=='C':
            core=gg[:,PERM[:d]];a=core.T@(core*w[:,None])/4;a.diagonal().add_(.01)
            A=torch.linalg.solve(a.cuda(),(core.T@(hh*w[:,None])/4).cuda()).cpu();e=hh-core@A
            t['U']=top(e.T@(e*w[:,None])/4);t['V']=A@t['U']
        if arm=='S':t['indices'],selector=supervised(torch.cat([gg,hh],1),y,w,d)
        q=project(gg,hh,t);t['out_norm']=norm_fit(q,w);out[arm]=t
    return out,selector
def coordinate_check(g,h,t,head):
    # Direct coordinates use W_direct=T W_residual; its penalty is T^-T D T^-1.
    g=(g-t['g_norm'][0])/t['g_norm'][1];h=(h-t['h_norm'][0])/t['h_norm'][1];core=g[:,PERM[:t['d']]]
    direct=aug(torch.cat([core,h@t['U']],1));d=t['d'];raw=head[:-1]/t['out_norm'][1]
    wd=raw.clone();wd[:d]-=t['V']@raw[d:];bias=head[-1]-t['out_norm'][0]@raw
    score=direct@torch.cat([wd,bias[None]],0);reference=mapped(g*t['g_norm'][1]+t['g_norm'][0],h*t['h_norm'][1]+t['h_norm'][0],t)@head
    error=float((score-reference).abs().max());assert error<1e-7 and torch.equal(score.argmax(1),reference.argmax(1));return error
def selfcheck():
    torch.manual_seed(74010);x=torch.randn(30,7,dtype=torch.float64);x=aug(x);y=torch.tensor([0]*10+[1]*10+[2]*10);groups=[str(i) for i in range(30)]
    a,b=stats(x,y,groups,3);a0,b0=stats(x[:20],y[:20],groups[:20],3);a1,b1=stats(x[20:],y[20:],groups[20:],3)
    w=solve(a,b,.01,3,'cpu');assert torch.allclose(w,solve(a0+a1,b0+b1,.01,3,'cpu'),atol=1e-10)
    T=torch.eye(8,dtype=x.dtype);T[:3,3:7]=torch.randn(3,4,dtype=x.dtype)*.1;T[-1,:7]=torch.randn(7,dtype=x.dtype)*.1
    z=x@T;az,bz=stats(z,y,groups,3);wz=solve(az,bz,.01,3,'cpu');inv=torch.linalg.inv(T);D=torch.eye(8,dtype=x.dtype);D[-1,-1]=0
    wd=torch.linalg.solve(a/3+.01*inv.T@D@inv,b/3)
    assert torch.allclose(x@wd,z@wz,atol=1e-9)
    assert WIDTH==dict(G=1535,R=1375,P=1375,C=1318,S=1534)
