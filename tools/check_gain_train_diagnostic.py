"""Production CPU fixtures: sealed inference, fixed readouts and anonymous reports."""
import copy,csv,json,os,tempfile,time
from pathlib import Path
import numpy as np
import torch
import gain_train_diagnostic as worker
import collect_gain_train_diagnostic as collector
from gain_joint_production import CANDIDATES
from gain_joint_worker import pack,restore
from lca_pair_worker import JOBS
from concm_statistical_metrics import stage


def reject(fn):
    try:fn()
    except (ValueError,AssertionError):return
    raise AssertionError('Illegal scope accepted')


def run(protocol):
    began=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());c=dict(gpus=[0,1],jobs=[dict(job_name=n) for n in JOBS],source_commit='fixture');worker.validate(c,p)
    for key,value in [('gpus',[0,2]),('old_image_training_replay',True),('val_accessed',True),('test_accessed',True),('optimizer_updates',1),('independent_calibration',True),('selection_enabled',True),('steps',[.1,.25,.5,1.])]:
        bad=copy.deepcopy(p);bad[key]=value;reject(lambda:worker.validate(c,bad))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'train.csv').write_text('split,relative_path,identity_component,original_label\ntrain,x.png,g1,0\ntrain,y.png,g2,1\n')
        (root/'val.csv').write_text('DO_NOT_OPEN')
        rows=worker.read_train(dict(manifest=str(root)),dict(order=[0,1],train_counts={'0':1,'1':1}));assert len(rows)==2
        (root/'train.csv').write_text('split,relative_path,identity_component,original_label\ntest,x.png,g1,0\n')
        reject(lambda:worker.read_train(dict(manifest=str(root)),dict(order=[0],train_counts={'0':1})))
        (root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='DIAGNOSE')))
        reject(lambda:worker.barrier(root,c,p,'diagnose'))
    d=1536;k=23;old=21;n=2*k;rng=np.random.default_rng(74002);data=copy.deepcopy(p['data']['HK']);data['train_counts']={str(c):2 for c in data['order']}
    labels=np.repeat(data['order'],2);x=rng.normal(0,.02,(n,d)).astype(np.float32);before=x+rng.normal(0,.005,x.shape).astype(np.float32)
    head=rng.normal(0,.01,(d,k)).astype(np.float32);ray=np.zeros_like(head);ray[:,:old]=rng.normal(0,.01,(d,old));ray[:,:old]-=ray[:,:old].mean(1,keepdims=True)
    deltas={v:(step*ray).astype(np.float32) for v,step in zip(CANDIDATES,(.125,.25,.5,1.))}
    mu=torch.stack([torch.from_numpy(x[labels==label].astype(float).mean(0)) for label in data['order']]);bank=dict(mu=mu,Q=torch.stack([torch.outer(m,m) for m in mu]))
    previous=dict(mu=mu[:old].clone());snapshot=bank['mu'].clone();mechanical={}
    for v in CANDIDATES:
        md=dict(dataset='HK',control='base',control_source=p['controls']['base']['source'],order=data['order'],seen=data['order'],seed=74002,task=6,source=worker.SOURCE,variant=v,version='gain_joint_predictive_v2',control_steps=176,optimizer_updates=0)
        side=pack(torch.from_numpy(deltas[v]).double(),old,md);assert torch.equal(restore(side,old,md,(d,k)),torch.from_numpy(deltas[v]))
        m=dict(exact_identity=False,scenarios=[dict(scenario=j,classes=[dict(label=i,known_rescue=.1,known_harm=.05,unknown_mass=0.) for i in range(k)]) for j in range(3)])
        mechanical[v]=dict(point=m,predictive=m)
    a,private=worker.report_stage(x,before,labels,np.tile([0,1],k),head,deltas,bank,previous,mechanical,data,6)
    assert len(a['drift'])==23 and len(private)==21 and torch.equal(bank['mu'],snapshot)
    assert all(z['second'].shape==(46,46) and np.isfinite(z['second']).all() for z in private.values())
    assert all(h['guarded']['paired']['new']['harmed']==0 for h in a['heads'].values())
    assert all(len(h['guarded']['estimates'])==23*6 and len(h['guarded']['fold_counts'])==46 for h in a['heads'].values())
    assert all(z['independent_calibration'] is False for z in a['drift'])
    # Complete report coverage uses synthetic scores with the exact frozen train denominators.
    s=dict(lock=p,source_commit='fixture',CPU_check=dict(process_residence_seconds=1.),runs={},preflights={},program=dict(status='COMPLETE',active={},costs=[],cpu_costs=[],failures=[],whole_round_wall_seconds=2.))
    for name in JOBS:
        ds=name.split('_')[0];dd=p['data'][ds];stages=[];exposures=0
        for task in range(1,len(dd['task_sizes'])+1):
            seen=dd['order'][:sum(dd['task_sizes'][:task])];lab=np.concatenate([np.full(dd['train_counts'][str(v)],v) for v in seen]);y=np.array([seen.index(int(v)) for v in lab]);off=np.eye(len(seen))[y];dat=dict(dd,val_counts=dd['train_counts']);report=stage(off,off,off,lab,dat,task)
            for mm in report['metrics'].values():mm['training_n']=mm.pop('validation_n')
            report.update(estimates=[dict(label=v,estimator=kind,scenario=scenario,n=dd['train_counts'][str(v)],predicted_known_rescue=0. if task==1 else .1,predicted_known_harm=0. if task==1 else .05,unknown_mass=0.,actual_rescue=0.,actual_harm=0.,known_rescue_minus_actual=0.,known_harm_minus_actual=0.,independent_calibration=False) for v in seen for kind in ('point','predictive') for scenario in range(3)],fold_counts=[dict(label=v,fold=f,n=dd['train_counts'][str(v)]//2,missing=False,rescue_rate=0.,harm_rate=0.) for v in seen for f in (0,1)],class_confusion=[])
            stages.append(dict(task=task,heads={v:dict(guarded=copy.deepcopy(report),raw=copy.deepcopy(report)) for v in CANDIDATES},drift=[dict(label=v,n=dd['train_counts'][str(v)]) for v in seen],boundaries=[],independent_calibration=False))
            exposures+=len(lab)*(1 if task==1 else 2)
        s['runs'][name]=dict(status='COMPLETE',source_commit='fixture',version=worker.VERSION,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False,independent_calibration=False,stages=stages,image_exposures=exposures)
        s['preflights'][name]=dict(status='PASS',source_commit='fixture',version=worker.VERSION,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False,independent_calibration=False,parameters_unchanged=True,CPU_GPU_scores_close=True,classification_metrics_read=False,real_training_images=64)
        s['program']['cpu_costs'].append(dict(job=name,operation='prepare',gpu=None,returncode=0,process_started=True,residence_seconds=1.))
        for operation in ('preflight','diagnose'):s['program']['costs'].append(dict(job=name,operation=operation,gpu=0 if name.endswith('_base') else 1,returncode=0,process_started=True,residence_seconds=1.))
    assert collector.validate(s);bad=copy.deepcopy(s);bad['runs']['HK_base']['stages'][-1]['heads']['C1']['guarded']['metrics']['ON']['per_class_n']['5']=99;reject(lambda:collector.validate(bad))
    with tempfile.TemporaryDirectory() as tmp:
        out=collector.report(s,Path(tmp));assert out['main_rows']==72 and out['class_rows']==1008 and out['estimator_macro_rows']==432 and out['ranking_rows']==648
    incomplete=copy.deepcopy(s);incomplete['program']['status']='INCOMPLETE';incomplete['program']['failures']=[dict(operation='diagnose',returncode=1)]
    with tempfile.TemporaryDirectory() as tmp:
        out=collector.report(incomplete,Path(tmp));assert out['status']=='INCOMPLETE' and (Path(tmp)/'results/PARTIAL_RESULTS.json').exists()
    return dict(status='PASS',full_high_dimensional_cpu_PASS=True,synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,medical_val_reads=0,optimizer_updates=0,policy_updates=0,dimensions=[1536,23],private_drift_dimension=46,unchanged_native_head_and_direction=True,sidecars_roundtrip=True,original_denominators_and_exposure_checked=True,train_only_reader_checked=True,illegal_scope_and_early_diagnosis_refused=True,independent_calibration_claim_refused=True,all_four_candidate_reports=True,main_rows=72,class_rows=1008,pipeline_seconds=time.monotonic()-began)


if __name__=='__main__':print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
