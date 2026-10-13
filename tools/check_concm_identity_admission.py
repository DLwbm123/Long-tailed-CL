"""Zero-information identity and malformed-support trust-boundary fixture."""
import os,json,copy,time
from pathlib import Path
import torch
import concm_identity as method
import prototype_coherent as native
import core1_competition as core

def run():
    start=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_available()
    method.validate_design(json.loads(Path(os.environ['Q128_PROTOCOL']).read_text()))
    mu=torch.eye(4,dtype=torch.float64)[:3];bank=dict(mu=mu,Q=mu[:,:,None]*mu[:,None,:],n=[1,1,1],components=[dict(label=c,center=mu[c],mass=mu.new_tensor(1.),radius=mu.new_tensor(0.),count=1) for c in range(3)])
    R=native.metric(bank);init,_=native.head(bank);pairs=core.competition(bank,init);W,_=core.bank_head(bank,pairs,regularizer=R)
    for v in ('MPC','TAIL','MPC_TAIL'):
        delta,a=method.fit(bank,pairs,R,W,2,v);assert not delta.any() and a['exact_identity']
    d0,_=method.fit(bank,pairs,R,W,2,'TAIL_REG');d1,a=method.fit(bank,pairs,R,W,2,'DSM');assert a['ETF_direction_force_norm']==0.;torch.testing.assert_close(d0,d1,atol=0.,rtol=0.)
    bad=copy.deepcopy(bank);bad['n'][0]=2
    try:method.borrow(bad)
    except ValueError:pass
    else:raise AssertionError('False component support accepted')
    bad=copy.deepcopy(bank);bad['Q'][0,0,1]=.1
    try:method.borrow(bad)
    except ValueError:pass
    else:raise AssertionError('Asymmetric moment accepted')
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,formal_optimizer_updates=0,zero_information_exact_identity=True,
        no_tail_exact_ETF_off_identity=True,component_count_mismatch_refused=True,asymmetric_moment_refused=True,pipeline_seconds=time.monotonic()-start)
if __name__=='__main__':print(json.dumps(run(),indent=2))
