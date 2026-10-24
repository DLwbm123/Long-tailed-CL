"""One shared SSF trajectory; independent same-gradient momentum-SGD audit."""
import time,math
import numpy as np
import torch

def optimizer(m,cfg):return torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=cfg['learning_rate'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'],foreach=False)

def update(m,opt,raw,labels,cfg,count,record,audit=False,step=0):
    tick=time.perf_counter();m.eval();opt.zero_grad(set_to_none=True);lr=cfg['learning_rate']*.5*(1+math.cos(math.pi*step/cfg['supervised_updates']))
    for group in opt.param_groups:group['lr']=lr
    value=0.
    for i in range(len(raw)):
        loss=m.objective(raw[i:i+1].cuda().double(),labels[i:i+1].cuda())/len(raw);finite=bool(torch.isfinite(loss));record(dict(check='finite_objective',value=float(loss.detach()),PASS=finite));assert finite;count['forward_rows']+=1;loss.backward();count['backward_rows']+=1;value+=float(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],cfg['gradient_clip']);record(dict(check='finite_gradient',value=float(norm),PASS=bool(torch.isfinite(norm))));assert bool(torch.isfinite(norm));expected=[];audit_seconds=0.
    if audit:
        t=time.perf_counter()
        for group in opt.param_groups:
            for w in group['params']:
                assert w.grad is not None
                old=w.detach().cpu().numpy().copy();g=w.grad.detach().cpu().numpy().copy()+group['weight_decay']*old;st=opt.state[w];buf=st.get('momentum_buffer');v=g if buf is None else group['momentum']*buf.detach().cpu().numpy()+g;expected.append((w,old-group['lr']*v,v.copy()))
        audit_seconds+=time.perf_counter()-t
    opt.step();count['updates']+=1
    if audit:
        t=time.perf_counter();error=scaled=buferror=0.
        for w,e,v in expected:
            d=abs(w.detach().cpu().numpy()-e);error=max(error,float(d.max()));scaled=max(scaled,float((d/(1e-10+1e-10*abs(e))).max()));buferror=max(buferror,float(abs(opt.state[w]['momentum_buffer'].detach().cpu().numpy()-v).max()))
        record(dict(check='independent_same_gradient_momentum_SGD',step=step,max_error=error,normalized_error=scaled,momentum_error=buferror,PASS=scaled<=1 and buferror<=1e-9));assert scaled<=1 and buferror<=1e-9;audit_seconds+=time.perf_counter()-t
    opt.zero_grad(set_to_none=True);torch.cuda.synchronize();return dict(objective=value,norm=float(norm),production_seconds=time.perf_counter()-tick-audit_seconds,audit_seconds=audit_seconds,learning_rate=lr)
