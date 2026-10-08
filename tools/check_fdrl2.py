"""Small independent oracles for cross-epoch returns and reward admission."""
import copy
import time
import torch
import fdrl_control as ctl
from audit_fdrl_reward import empirical
from summarize_fdrl_calibration import summarize


def run():
    start=time.process_time();torch.set_num_threads(2)
    x=torch.tensor([[3.,1.],[1.,4.],[2.,2.]],dtype=torch.float64)
    y=torch.tensor([0,1,0]);w=torch.eye(2,dtype=torch.float64)
    values=empirical(x,y,w)
    assert values['recall']==[1.,1.] and values['margin']==[1.,3.]
    assert values['loss']==[2.5,5.]
    policy=ctl.Policy(19);trajectory=[]
    for i in range(4):
        state=torch.zeros(16,dtype=torch.float64);state[0]=i/4
        action,p=policy.sample(state)
        trajectory.append(dict(policy_state=state,action=action,behavior=p,reward=float(i==3)))
    audit=policy.update(trajectory)
    assert torch.allclose(torch.tensor(audit['returns']),torch.tensor([.95**3,.95**2,.95,1.]))
    assert audit['advantage'][0]>0 and policy.updates==1
    base=dict(status='COMPLETE',actual_updates=48,expected_updates=48,real_old_images_accessed=False,
        future_images_accessed=False,validation_images_accessed=False,test_accessed=False)
    arms=[]
    for i in range(3):
        arms.append(dict(proxy_reward=i,oracle_reward=i,proxy_before=[.3,.3],proxy_after=[.3,.3],
            oracle_before=dict(recall=[.5,.5]),oracle_after=dict(recall=[.5+i*.02,.5+i*.02])))
    records=[dict(base,arms=copy.deepcopy(arms)) for _ in range(20)]
    assert summarize(records)['passed']
    for r in records:
        for a in r['arms']:a['oracle_reward']=-a['proxy_reward']
    assert not summarize(records)['passed'] and summarize(records)['pairwise_agreement']==0
    assert not summarize(records[:19])['passed']
    return dict(status='PASS',checks=['empirical loss, recall and competitor margin oracle',
        'task-end reward reaches first-epoch action with unchanged behavior',
        'reward admission accepts aligned and rejects reversed or incomplete evidence'],
        toy_policy_updates=1,scientific_optimizer_updates=0,cpu_core_seconds=time.process_time()-start)


if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
