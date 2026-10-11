"""CPU-only frozen-scope, historical-head compatibility and contribution report checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import module_additions as modules
import prototype_coherent as native
from run_prototype_single import manifests
from run_prototype_coherent import fit_split
from run_residual_drift_cycle import validate
from summarize_residual_drift import FIELDS, MODULES, report


def run(config, previous_modules, control_runs):
    started=time.monotonic();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized();validate(config)
    for key,value in [('gpus',[0,2])]:
        bad=copy.deepcopy(config);bad[key]=value
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed resource scope accepted')
    for key,value in [('residual_drift_weight',.2),('expected_steps',469),('order',[0]),('evaluation_split','official_test')]:
        bad=copy.deepcopy(config);bad['jobs'][0][key]=value
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed frozen scientific scope accepted')
    stages={};counts={}
    for ds in ('ISIC','HK'):
        j=next(j for j in config['jobs'] if j['name']==ds)
        train,val=manifests(j);steps=[];pos=0
        for size in j['task_sizes']:
            classes=j['order'][pos:pos+size];pos+=size
            rows=[r for r in train if r['label'] in classes];fit,meta=fit_split(rows,j['split_seed'])
            steps.append(2*((len(fit)+63)//64))
        assert sum(steps)==j['expected_steps']
        counts[ds]=dict(train_n=len(train),validation_n=len(val),train_class_n={str(c):sum(r['label']==c for r in train) for c in j['order']},validation_class_n={str(c):sum(r['label']==c for r in val) for c in j['order']})
        for arm in MODULES[:2]:
            st=control_runs[ds+'_'+arm]['metrics']['stages']
            seen=[];pos=0
            for size,t in zip(j['task_sizes'],st):
                seen+=j['order'][pos:pos+size];pos+=size
                assert t['seen']==seen and t['per_class_n']=={str(c):counts[ds]['validation_class_n'][str(c)] for c in seen}
        stages[ds]=steps
    torch.manual_seed(74002);x=torch.randn(24,6,dtype=torch.float64);y=torch.arange(3).repeat_interleave(8)
    bank=native.append(native.empty(6),x,y,torch.zeros(24,dtype=torch.long),x.new_full((24,),1/8))
    exact=[]
    for path,arm in previous_modules:
        spec=importlib.util.spec_from_file_location('old_'+arm,path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        parts=None if arm=='base' else torch.randn(3,4,6,dtype=x.dtype)
        a=old.head(bank,arm,'ISIC',[0,1,2],parts);b=modules.head(bank,arm,'ISIC',[0,1,2],parts,drift_weight=0)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2]==b[2]
        if parts is not None:
            tokens=[torch.randn(3,5,3,dtype=x.dtype) for _ in range(2)]
            assert torch.equal(old.local_parts(tokens),modules.local_parts(tokens))
            shift=torch.randn(4,6,dtype=x.dtype);labels=torch.tensor([0,1,2]);cur=torch.randn(3,4,6,dtype=x.dtype)
            assert torch.equal(old.local_memory(parts,shift,cur,labels),modules.local_memory(parts,shift,cur,labels))
        exact.append(arm)
    # Synthetic complete four-cell report; never new classification evaluation.
    s=dict(program=dict(status='COMPLETE',active={},failures=[],costs=[dict(job=str(i),operation='fixture',gpu=i%2,returncode=0,residence_seconds=0) for i in range(12)],elapsed_seconds=0,whole_round_wall_seconds=0,resource_wait_seconds=0),runs=copy.deepcopy(control_runs),preflights={},closed_round_cost_ledger={},source_commit='SYNTHETIC_ONLY',control_matching={'fixture':True},cpu_check_seconds=0)
    for ds in ('ISIC','HK'):
        for a,b in [('residual_drift','base'),('local_transport_drift','local_transport')]:
            v=copy.deepcopy(control_runs[ds+'_'+b]);v['metrics'].update(residual_drift_weight=.1,evaluation_split='development_validation')
            v['status'].update(status='COMPLETE',steps={'ISIC':468,'HK':176}[ds],policy_updates=0,max_solve_residual=1e-10)
            diag=[]
            for t in v['metrics']['stages']:
                for epoch in (1,2):
                    d=dict(task=t['task'],epoch=epoch,steps=v['status']['steps'],max_batch_solve_residual=1e-10,activation={'extra_columns':4*len(t['seen']),'prior_delta':1},residual_drift=dict(weight=.1,detached=True,fit_only=True,historical_samples=0,residual_trace=1,residual_frobenius_norm=1))
                    if epoch==2:d.update(boundary_residual_drift=dict(weight=.1,current_train_including_meta=True,historical_samples=0,residual_trace=1,residual_frobenius_norm=1),boundary_solve=dict(relative_residual=1e-10))
                    diag.append(d)
            v['scalar_diagnostics']=diag;s['runs'][ds+'_'+a]=v
            s['preflights'][ds+'_'+a]=dict(status=dict(status='COMPLETE',steps=0,policy_updates=0,max_solve_residual=1e-10),scalars=dict(status='PASS',adapter_updates=0,policy_updates=0,gradient_norm=1,residual_drift=dict(weight=.1,nonzero_activation_not_proven=True)))
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'out';report(s,dest)
        contrasts=json.loads((dest/'CONTRIBUTIONS.json').read_text());assert len(contrasts)==10 and all(set(c['delta_pp'])==set(FIELDS) for c in contrasts)
        assert (dest/'PER_CLASS_RECALL.csv').is_file()
        for c in contrasts:
            if c['contrast']=='interaction':assert all(v==0 for v in c['delta_pp'].values())
        bad=copy.deepcopy(s);bad['runs']['ISIC_residual_drift']['status']['steps']+=1
        try:report(bad,Path(tmp)/'invalid')
        except ValueError:pass
        else:raise AssertionError('Incomplete budget accepted')
        bad=copy.deepcopy(s);bad['runs']['HK_local_transport_drift']['metrics']['stages'][-1]['per_class_n']['3']+=1
        try:report(bad,Path(tmp)/'invalid_n')
        except ValueError:pass
        else:raise AssertionError('Denominator mismatch accepted')
    assert not torch.cuda.is_initialized()
    return dict(status='PASS',gpu_work=False,formal_updates=0,test_accessed=False,images_read=False,
                matched_zero_weight_heads_exact=exact,local_pool_and_regional_transport_exact=True,
                stage_updates=stages,class_counts=counts,scope_and_budget_rejection=True,
                all_ten_contrasts_and_interaction_fixture=True,invalid_budget_and_denominator_rejected=True,
                cpu_pipeline_seconds=time.monotonic()-started)


if __name__=='__main__':
    p=json.loads(Path(os.environ['Q151_CHECK']).read_text())
    print(json.dumps(run(p['config'],p['previous_modules'],p['control_runs']),allow_nan=False))
