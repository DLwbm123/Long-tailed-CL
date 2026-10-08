"""Small CPU check for the readout identity, matched input, and mismatch gate."""
import time
import numpy as np
import torch
import prototype_coherent as method
from run_readout_cross import rebuild,decompose,diagonal_guard,paired,metrics


def check():
    started=time.process_time();torch.set_num_threads(2);torch.manual_seed(109)
    def unit(n):return torch.nn.functional.normalize(torch.randn(n,6,dtype=torch.float64),dim=1)
    x=unit(12);y=torch.tensor([0]*6+[1]*6)
    bank=method.append(method.empty(6),x,y,y,torch.full((12,),1/6,dtype=torch.float64))
    before=unit(16);after=torch.nn.functional.normalize(before+.1*unit(16),dim=1)
    labels=torch.tensor([2]*10+[3]*6);seeds=method.seed_components(before,labels)
    s,Ws,Gs,Hs,_=rebuild(bank,before,after,labels,'SHIFT',seeds)
    a,Wa,Ga,Ha,_=rebuild(bank,before,after,labels,'AFFINE',seeds)
    dm,dq,error=decompose(Gs,Ga,Hs,Ha,Ws,Wa)
    assert error<1e-10 and dm[:,2:].abs().max()<1e-12
    assert torch.allclose(s['mu'][2:],a['mu'][2:]) and torch.allclose(s['Q'][2:],a['Q'][2:])
    for kind in ('SHIFT','AFFINE'):
        neutral,W,_,_,_=rebuild(bank,before,before,labels,kind,seeds)
        assert torch.allclose(neutral['mu'][:2],bank['mu'],atol=1e-12)
        assert torch.allclose(neutral['Q'][:2],bank['Q'],atol=1e-12)
    z=unit(9);scores=(z@Ws).numpy();c=dict(score_atol=1e-5,score_rtol=1e-4,head_atol=1e-5,head_rtol=1e-4)
    assert diagonal_guard(scores,scores,Ws,Ws,s,s,c)['passed']
    changed=scores.copy();changed[0,0]+=10
    assert not diagonal_guard(scores,changed,Ws,Ws,s,s,c)['passed']
    truth=np.array([0,1,2,3]);u=np.eye(4);v=u.copy();v[2,0]=2
    p=paired(u,v,truth)
    assert p['shift_only_correct']==1 and p['within_new_correct_cross_group_failure']['newly_failed']==1
    old_labels=torch.arange(6).repeat_interleave(6);old_x=unit(36)
    old=method.append(method.empty(6),old_x,old_labels,old_labels,torch.full((36,),1/6,dtype=torch.float64))
    late_labels=labels+4;late_seeds=method.seed_components(before,late_labels)
    s,Ws,Gs,Hs,_=rebuild(old,before,after,late_labels,'SHIFT',late_seeds)
    a,Wa,Ga,Ha,_=rebuild(old,before,after,late_labels,'AFFINE',late_seeds)
    dm,_,late_error=decompose(Gs,Ga,Hs,Ha,Ws,Wa)
    assert dm[:,6:].abs().max()<1e-12 and torch.allclose(s['Q'][6:],a['Q'][6:])
    u=np.eye(8);v=u.copy();v[6,4]=2
    assert paired(u,v,np.arange(8))['within_new_correct_cross_group_failure']['newly_failed']==1
    v=u.copy();v[2,6]=2
    late=metrics(v,np.arange(8),list(range(8)))
    assert late['old_BA']==5/6 and late['new_BA']==1
    return dict(status='PASS',optimizer_updates=0,gpu_seconds=0,cpu_core_seconds=time.process_time()-started,
        head_identity_relative_error=error,T4_head_identity_relative_error=late_error,
        checks=['matched new moments','identity mapping','head difference identity',
            'diagonal mismatch rejected','paired cross-group failure','six-old two-new boundary and reconstruction'])


if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
