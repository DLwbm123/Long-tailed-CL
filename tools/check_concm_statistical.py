"""Independent sample/moment oracle and full-report synthetic fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import concm_statistical as method
import prototype_coherent as native
import core1_competition as competition
from concm_statistical_metrics import stage,aggregate
from collect_concm_statistical import report,validate
from lca_pair_worker import JOBS
from concm_statistical_worker import barrier


def check(protocol):
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    p=json.loads(Path(protocol).read_text());method.validate_parameters(p)
    d,k=12,4;x=torch.randn(35,d,dtype=torch.float64);y=torch.repeat_interleave(torch.arange(k),torch.tensor([3,6,10,16]))
    x=torch.nn.functional.normalize(x,dim=1);group=y.clone();weights=x.new_tensor([1/int((y==v).sum()) for v in y])
    bank=native.append(native.empty(d),x,y,group,weights);R=native.metric(bank)
    reference,_=native.head(bank);pairs=competition.competition(bank,reference)
    reference,_=competition.bank_head(bank,pairs,regularizer=R)
    saved=copy.deepcopy(bank);completed,ca=method.complete(bank)
    shifted=x+(completed['mu']-bank['mu'])[y]
    for c in range(k):
        z=shifted[y==c]
        torch.testing.assert_close(z.mean(0),completed['mu'][c],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(z.T@z/len(z),completed['Q'][c],rtol=1e-12,atol=1e-12)
    assert ca['eta'][0]>ca['eta'][-1]
    maximum=0.;actions={}
    for variant in method.VARIANTS:
        W,a=method.fit(bank,pairs,R,reference,variant)
        revised=bank if variant=='DSM' else completed
        gamma=a['structure_coefficient'];target=None
        if gamma:target,_=method.matching_anchors(revised['mu'],reference)
        # Explicit samples independently produce the full Hessian and linear term.
        sample=x if variant=='DSM' else shifted
        base=sample.T@(weights[:,None]*sample)/k+.001*R+gamma*torch.eye(d,dtype=x.dtype)
        H=torch.kron(base.contiguous(),torch.eye(k,dtype=x.dtype))
        rhs=sample.T@(weights[:,None]*torch.nn.functional.one_hot(y,k).double())/k
        for c in range(k):
            z=sample[y==c];q=z.T@z/(len(z)*k)
            for j in range(k):
                if c==j:continue
                e=x.new_zeros(k);e[c]=1;e[j]=-1
                H+=.5*pairs[c,j]*torch.kron(q.contiguous(),torch.outer(e,e).contiguous())
                rhs+=.5*pairs[c,j]*(z.mean(0)/k)[:,None]*e[None,:]
        if gamma:rhs+=gamma*target
        direct=torch.linalg.solve(H,rhs.flatten()).reshape(d,k)
        torch.testing.assert_close(W,direct,rtol=1e-7,atol=1e-8)
        maximum=max(maximum,a['solver']['relative_residual']);actions[variant]=a['head_delta_norm']
        assert actions[variant]>0
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'head.pt';torch.save(dict(head=W.float(),variant=variant),path)
            assert torch.equal(torch.load(path,weights_only=False)['head'],W.float())
    assert torch.equal(bank['mu'],saved['mu']) and torch.equal(bank['Q'],saved['Q'])
    invalid=copy.deepcopy(p);invalid['test_accessed']=True
    try:method.validate_parameters(invalid)
    except ValueError:pass
    else:raise AssertionError('Test scope accepted')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='PREPARE',variant='MPC')))
        try:barrier(root,p,'fixture','evaluate','MPC')
        except ValueError:pass
        else:raise AssertionError('Early evaluation accepted')
    fixture=copy.deepcopy(p);fixture['control_references']={};fixture['AIR_references']={}
    runs={v:{} for v in method.VARIANTS};preflights={v:{} for v in method.VARIANTS};prepared={}
    for name in JOBS:
        ds=name.split('_')[0];data=fixture['data'][ds];stages=[]
        for task in range(1,len(data['task_sizes'])+1):
            seen=data['order'][:sum(data['task_sizes'][:task])]
            labels=np.concatenate([np.full(data['val_counts'][str(c)],c) for c in seen])
            local=np.array([seen.index(int(c)) for c in labels]);off=np.zeros((len(labels),len(seen)),dtype=np.float32)
            off[np.arange(len(labels)),local]=1;on=off.copy();on[0]=0;on[0,(local[0]+1)%len(seen)]=2
            s=stage(off,on,on,labels,data,task);s.update(published_control_exact=True,encoder_state_unchanged=True,mechanical=dict(solver=dict(relative_residual=0.)))
            assert s['paired']['all']['harmed']==1 and s['paired']['all']['rescued']==0
            stages.append(s)
        metrics={mode:aggregate([s['metrics'][mode] for s in stages],data) for mode in ('OFF','ON')}
        fixture['control_references'][name]=metrics['OFF'];fixture['AIR_references'][ds]=metrics['OFF']
        prepared[name]=dict(status='PREPARED',source_commit='fixture',stages=[{} for _ in stages])
        for v in method.VARIANTS:
            runs[v][name]=dict(dataset=ds,variant=v,version=method.VERSION,source_commit='fixture',metrics=metrics,stages=stages,
                status=dict(status='COMPLETE'),optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False)
            preflights[v][name]=dict(status='PASS',parameters_unchanged=True,CPU_GPU_scores_close=True,classification_metrics_read=False,
                source_commit='fixture',optimizer_updates=0,train_accessed=False,test_accessed=False)
    costs=[dict(variant=v,job=n,operation=op,gpu=i%2,returncode=0,residence_seconds=1.) for v in method.VARIANTS for i,n in enumerate(JOBS) for op in ('preflight','evaluate')]
    s=dict(lock=fixture,source_commit='fixture',round_name='fixture',program=dict(status='COMPLETE',active={},failures=[],costs=costs,cpu_costs=[dict(returncode=0,gpu=None,residence_seconds=1.) for _ in JOBS]),
        prepared=prepared,runs=runs,preflights=preflights,closed_round_cost_ledger={},cpu_checks={},engineering_events=[])
    assert validate(s)
    bad=copy.deepcopy(s);bad['runs']['MPC']['HK_base']['stages'][-1]['metrics']['ON']['per_class_n']['5']=2
    try:validate(bad)
    except ValueError:pass
    else:raise AssertionError('Wrong denominator accepted')
    with tempfile.TemporaryDirectory() as tmp:
        a=report(s,Path(tmp)/'complete');assert a['status']=='PASS'
        assert len(json.loads((Path(tmp)/'complete/RESULTS.json').read_text())['rows'])==18
        partial=copy.deepcopy(s);partial['program']['status']='INCOMPLETE';partial['runs']['MPC'].pop('HK_base')
        a=report(partial,Path(tmp)/'partial');assert a['status']=='INCOMPLETE'
        assert json.loads((Path(tmp)/'partial/PARTIAL_RESULTS.json').read_text())['metrics']['HK']['MPC_B'] is None
    return dict(status='PASS',GPU_processes=0,medical_images_read=0,formal_optimizer_updates=0,
        exact_sample_moment_translation=True,explicit_sample_quadratic_oracle=True,all_variants_active_in_fixture=True,
        source_statistics_unchanged=True,ETF_geometry=True,checkpoint_roundtrip=True,illegal_test_refused=True,
        early_evaluation_refused=True,
        complete_18_row_report=True,wrong_denominator_refused=True,missing_not_zero=True,
        max_solve_residual=maximum,synthetic_head_changes=actions,pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':
    import sys
    print(json.dumps(check(sys.argv[1]),indent=2))
