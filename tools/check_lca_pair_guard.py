"""Production CPU self-check, synthetic only, with early-evaluation rejection."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import core1_competition as competition
import module_additions as modules
import lca_old_pair_repair as repair
from check_lca_old_pair_repair import fixture,rejected
from lca_pair_metrics import stage,aggregate
from lca_pair_worker import JOBS,validate,val_barrier,metadata,admit


def run(c,p,tmp):
    start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(74002)
    assert not torch.cuda.is_available();validate(c,p)
    for key,value in [('test_accessed',True),('train_images_allowed',True),('cells',['A0','AL','B','P','P_raw']),
                      ('native_energy_radius',.2),('batch_size',32),('primary_control','base')]:
        bad=copy.deepcopy(p);bad[key]=value;rejected(lambda:validate(c,bad))
    checks=[];maximum=0.
    for arm in ('base','local_transport'):
        x,y,bank=fixture();local=torch.randn(5,4,7,dtype=x.dtype) if arm=='local_transport' else None
        _,R,_=modules.head(bank,arm,'HK',list(range(5)),local)
        pairs=competition.competition(bank,torch.zeros(7,5,dtype=x.dtype))
        W,a=competition.bank_head(bank,pairs,regularizer=R)
        delta,b=repair.fit(bank,pairs,R,W,3);maximum=max(maximum,a['relative_residual'],b['relative_residual'])
        j=c['jobs'][0 if arm=='base' else 1];m=metadata(c,j,p,1)
        # Exercise the actual production provenance/sidecar API at the fixture's dimensions.
        m.update(order=list(range(5)),seen=list(range(5)))
        state=repair.checkpoint(delta,3,m);assert torch.equal(repair.restore(state,m,3,(7,5)),delta.float())
        wrong=dict(m,source='wrong');rejected(lambda:repair.restore(state,wrong,3,(7,5)))
        off=(x.float()@W.float()).numpy();change=(x.float()@delta.float()).numpy()
        on=repair.guard(torch.from_numpy(off),torch.from_numpy(change),3).numpy()
        d=dict(order=list(range(5)),task_sizes=[3,2],train_counts={str(i):bank['n'][i] for i in range(5)},
            val_counts={str(i):int((y==i).sum()) for i in range(5)})
        r=stage(off,on,off+change,y.numpy(),d,2)
        assert r['effect']['new_margin_changed']==r['paired']['new']['rescued']==r['paired']['new']['harmed']==0
        mask=y.numpy()<3;s1=stage(off[mask,:3],off[mask,:3],off[mask,:3],y.numpy()[mask],d,1)
        assert s1['paired']['old']['n']==0 and s1['paired']['all']['changed']==0
        result=aggregate([s1['metrics']['ON'],r['metrics']['ON']],d)
        assert np.isfinite(result['forgetting']) and result['final_macro_F1']>=0
        bad=copy.deepcopy(d);bad['val_counts']['0']+=1;rejected(lambda:stage(off,on,off+change,y.numpy(),bad,2))
        assert r['margins']['0']['on']['mean'] is None
        checks.append(arm)
    with tempfile.TemporaryDirectory(dir=tmp) as path:
        root=Path(path)
        (root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='PREPARE')))
        rejected(lambda:val_barrier(root,p,'preflight'))
        (root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='EVALUATE',sidecars_sealed_at=1)))
        for name in JOBS:
            folder=root/'prepared'/name;folder.mkdir(parents=True)
            (folder/'PREPARED.json').write_text(json.dumps(dict(status='PREPARED',stages=[{}]*len(p['data'][name.split('_')[0]]['task_sizes']))))
        rejected(lambda:val_barrier(root,p,'evaluate'))
        (root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='EVALUATE',sidecars_sealed_at=1,preflights_sealed_at=2)))
        for name in JOBS:
            folder=root/'preflight'/name;folder.mkdir(parents=True)
            (folder/'RESULTS.json').write_text(json.dumps(dict(status='PASS',optimizer_updates=0,parameters_unchanged=True)))
        val_barrier(root,p,'evaluate')
    from collect_lca_pair_guard import report,validate as validate_report
    p=copy.deepcopy(p);runs={};prepared={};preflights={};rng=np.random.default_rng(74002)
    for job in c['jobs']:
        name=job['job_name'];d=p['data'][job['name']];reports=[]
        for task in range(1,len(d['task_sizes'])+1):
            seen=d['order'][:sum(d['task_sizes'][:task])];old=sum(d['task_sizes'][:task-1])
            labels=np.repeat(seen,[d['val_counts'][str(v)] for v in seen]);off=rng.normal(size=(len(labels),len(seen))).astype('float32')
            change=rng.normal(size=off.shape).astype('float32')*.2;change[:,old:]=0
            on=repair.guard(torch.from_numpy(off),torch.from_numpy(change),old).numpy()
            result=stage(off,on,off+change,labels,d,task)
            result.update(published_control_exact=True,encoder_state_unchanged=True,mechanical=dict(native_residual=0.,exact_identity=not bool(old)))
            reports.append(result)
        p['control_references'][name]=aggregate([t['metrics']['OFF'] for t in reports],d)
        runs[name]=dict(dataset=job['name'],control=job['control'],version=p['version'],source_commit=c['source_commit'],
            control_source=p['controls'][job['control']]['source'],status=dict(status='COMPLETE'),optimizer_updates=0,policy_updates=0,
            train_accessed=False,test_accessed=False,stages=reports,metrics={mode:aggregate([t['metrics'][mode] for t in reports],d) for mode in ('OFF','ON')})
        prepared[name]=dict(status='PREPARED',stages=[{}]*len(reports))
        preflights[name]=dict(status='PASS',source_commit=c['source_commit'],parameters_unchanged=True,CPU_GPU_guard_exact=True,optimizer_updates=0,test_accessed=False)
        fake=dict(seen=d['order'],steps=d['cumulative_steps'][-1],module=job['control'],head=torch.zeros(1536,len(d['order'])),
            bank=dict(n=[d['train_counts'][str(v)] for v in d['order']]),model={'synthetic':torch.ones(1)},module_memory=None)
        admit(fake,d,job['control'],len(d['task_sizes']))
        bad=copy.deepcopy(fake);bad['steps']+=1;rejected(lambda:admit(bad,d,job['control'],len(d['task_sizes'])))
    costs=[dict(job=name,operation=op,gpu=i%2,returncode=0,residence_seconds=0.) for i,name in enumerate(JOBS) for op in ('preflight','evaluate')]
    cpu=[dict(job=name,operation='prepare',gpu=None,returncode=0,residence_seconds=0.) for name in JOBS]
    snapshot=dict(program=dict(status='COMPLETE',active={},failures=[],costs=costs,cpu_costs=cpu,sidecars_sealed_at=1,preflights_sealed_at=2),
        runs=runs,prepared=prepared,preflights=preflights,lock=p,source_commit=c['source_commit'],round_name='SYNTHETIC_NOT_SCIENCE',
        closed_round_cost_ledger={},cpu_checks=dict(synthetic=True),engineering_events=[])
    with tempfile.TemporaryDirectory(dir=tmp) as path:
        result=report(snapshot,Path(path)/'synthetic_report');assert result['status']=='PASS'
        assert len(list(__import__('csv').DictReader((Path(path)/'synthetic_report/STAGE_CLASS_RECALLS.csv').open())))==630
    bad=copy.deepcopy(snapshot);bad['runs'][JOBS[0]]['stages'][0]['metrics']['ON']['per_class_n']['4']+=1
    rejected(lambda:validate_report(bad))
    return dict(status='PASS',controls=checks,early_val_and_scope_rejected=True,stage_one_identity_and_empty_old=True,
        sidecar_roundtrip_and_provenance=True,all_signed_pair_counts_and_small_n_privacy=True,
        exact_new_raw_margin_and_prediction_preservation=True,full_stage_aggregate=True,full_report_and_bad_denominator=True,
        max_solve_residual=maximum,
        GPU_processes=0,medical_images_read=0,original_checkpoints_read=0,optimizer_updates=0,
        pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':
    c=json.loads(Path(os.environ['Q167_CPU_CONFIG']).read_text())
    result=run(c['campaign'],json.loads(Path(c['lock']).read_text()),c['temporary_root'])
    Path(c['receipt']).write_text(json.dumps(result,indent=2)+'\n')
