"""Independent dense quadratic, identity, common mode and scope fixtures."""
import json,time,sys
from pathlib import Path
import torch,numpy as np
import prototype_coherent as native
import core1_competition as competition
import concm_mechanism as method
from concm_mechanism_worker import barrier
from concm_mechanism_metrics import diagnose

def check(path):
    start=time.monotonic();p=json.loads(Path(path).read_text());method.validate_parameters(p)
    torch.set_num_threads(2);torch.manual_seed(74002);d,k=12,4
    x=torch.randn(35,d,dtype=torch.float64);y=torch.repeat_interleave(torch.arange(k),torch.tensor([3,6,10,16]));x=torch.nn.functional.normalize(x,dim=1)
    weights=x.new_tensor([1/int((y==v).sum()) for v in y]);bank=native.append(native.empty(d),x,y,y,weights);R=native.metric(bank);W,_=native.head(bank);pairs=competition.competition(bank,W);W,_=competition.bank_head(bank,pairs,regularizer=R)
    gamma=.1*float(torch.trace(bank['Q'].mean(0)+.001*R)/d);H=torch.kron((bank['Q'].mean(0)+.001*R).contiguous(),torch.eye(k,dtype=x.dtype));rhs=bank['mu'].T/k
    for c in range(k):
        for j in range(k):
            if c==j:continue
            e=x.new_zeros(k);e[c]=1;e[j]=-1
            H+=.5*pairs[c,j]*torch.kron((bank['Q'][c]/k).contiguous(),torch.outer(e,e).contiguous());rhs+=.5*pairs[c,j]*(bank['mu'][c]/k)[:,None]*e[None,:]
    heads={};residual=0.
    for v in method.VARIANTS:
        w,a=method.fit(bank,pairs,R,W,v);heads[v]=w
        if v not in ('MPC','MPC_DSM'):
            target,_=method.native.matching_anchors(bank['mu'],W)
            if v=='REG_ZERO':target=torch.zeros_like(W)
            if v=='IDENTITY':target=W
            if v=='ETF_DISC':target*=((W-W.mean(1,keepdim=True)).norm()/W.norm())
            direct=torch.linalg.solve(H+gamma*torch.eye(d*k,dtype=x.dtype),(rhs+gamma*target).flatten()).reshape(d,k)
            torch.testing.assert_close(w,direct,atol=1e-8,rtol=1e-7)
        assert a['norms']['orthogonal_energy_error']<1e-12;residual=max(residual,a['solver']['relative_residual'])
    rows=method.donors(bank,list(range(k)));assert all(abs(sum(a['weight'] for a in rows if a['label']==c)-1)<1e-12 for c in range(k))
    # A common column offset changes full norm, not prediction or WC.
    common=W+torch.randn(d,1,dtype=x.dtype);torch.testing.assert_close(W-W.mean(1,keepdim=True),common-common.mean(1,keepdim=True));assert torch.equal((x@W).argmax(1),(x@common).argmax(1))
    data=dict(order=list(range(k)),task_sizes=[2,2],train_counts={str(i):10 for i in range(k)},val_counts={str(i):6 for i in range(k)})
    labels=np.repeat(np.arange(k),6);xx=np.eye(k)[labels];head=np.eye(k);off=xx@head;hh={v:head.copy() for v in method.VARIANTS};hh['MPC'][:,[2,3]]=hh['MPC'][:,[3,2]];scores={v:xx@h for v,h in hh.items()};reports=diagnose(off,scores,xx,head,hh,labels,data,2)
    assert reports['MPC']['paired']['new']['harmed']==12 and reports['IDENTITY']['paired']['all']['changed']==0
    bad=dict(p,test_accessed=True)
    try:method.validate_parameters(bad)
    except ValueError:pass
    else:raise AssertionError('Test admitted')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='PREPARE',variant='MECHANISM')))
        try:barrier(root,p,'fixture','evaluate','MECHANISM')
        except ValueError:pass
        else:raise AssertionError('Early val admitted')
    return dict(status='PASS',dense_objective_oracle=True,identity_target_optimum=True,common_mode_prediction_invariant=True,norm_orthogonal_decomposition=True,donor_weights_sum_one=True,paired_new_harm_fixture=True,early_val_refused=True,test_refused=True,medical_images_read=0,GPU_processes=0,optimizer_updates=0,max_solve_residual=residual,pipeline_seconds=time.monotonic()-start)
if __name__=='__main__':print(json.dumps(check(sys.argv[1]),indent=2))
