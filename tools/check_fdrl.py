"""Independent small checks for moment rewards and on-policy temporal credit."""
import time
import torch
import fdrl_control as ctl
import prototype_coherent as method


def run():
    started=time.process_time();torch.set_num_threads(2);torch.manual_seed(218)
    x=torch.randn(24,5,dtype=torch.float64);y=torch.arange(4).repeat_interleave(6);w=torch.randn(5,4,dtype=x.dtype)
    old=ctl.meta_append(method.empty(5),x[:12],y[:12])
    actual=ctl.risks(old,w,x[12:],y[12:])
    error=.5*(x@w-torch.nn.functional.one_hot(y,4)).square().sum(1)
    oracle=torch.stack([error[y==c].mean() for c in range(4)])
    assert torch.allclose(actual,oracle,atol=1e-10) and not old['components']
    delta=torch.randn(5,dtype=x.dtype)/10
    actual=ctl.risks(method.translate(old,delta),w,x[12:]+delta,y[12:])
    error=.5*((x+delta)@w-torch.nn.functional.one_hot(y,4)).square().sum(1)
    assert torch.allclose(actual,torch.stack([error[y==c].mean() for c in range(4)]),atol=1e-10)
    a=torch.tensor([.4,.3,.2,.1],dtype=x.dtype);b=torch.tensor([.3,.35,.1,.11],dtype=x.dtype)
    assert abs(ctl.reward(a,b,2,[1,3])-(100*((.1-.05+.1-.01)/4-.05/2-.01/2-(.05+.01)/2)))<1e-10
    assert torch.allclose(ctl.discounted([1.,2.,3.]),torch.tensor([1+.95*2+.95**2*3,2+.95*3,3.],dtype=x.dtype))
    grad=dict(weighted_fd_norm=1.,task_norm=2.,cosine=-.5)
    state=ctl.features(a,a,2,[1,3],grad,.2,old,[.5,.5,0.]);assert state.shape==(16,)
    policy=ctl.Policy(32);trajectory=[]
    for i in range(8):
        s=state.clone();s[0]=i/8;action,p=policy.sample(s)
        trajectory.append(dict(policy_state=s,action=action,behavior=p,reward=float(action==2)))
    before=[p.detach().clone() for p in policy.parameters]
    audit=policy.update(trajectory)
    assert any(not torch.equal(p,q) for p,q in zip(before,policy.parameters)) and policy.updates==1
    assert len(set(t['action'] for t in trajectory))>1
    try:policy.update(trajectory)
    except ValueError as e:assert 'Off-policy' in str(e)
    else:raise AssertionError('Stale behavior accepted')
    # One-step episodes must still produce an actor gradient: no within-episode standardization.
    policy=ctl.Policy(12);action,p=policy.sample(state)
    policy.update([dict(policy_state=state,action=action,behavior=p,reward=1.)])
    assert policy.forward(state)[0][action] > p[action]
    return dict(status='PASS',checks=['empirical class-balanced risk identity','translated moment oracle',
        'reward protection arithmetic','discounted temporal returns','state bounds','on-policy behavior guard',
        'categorical main-path sampling','one-transition actor credit'],toy_policy_updates=2,
        scientific_optimizer_updates=0,cpu_core_seconds=time.process_time()-started)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
