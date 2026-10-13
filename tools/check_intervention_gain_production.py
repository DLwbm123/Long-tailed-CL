"""Synthetic production checks: cached math, barriers, shared readouts, reports."""
import copy,json,tempfile,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import prototype_coherent as native
import module_additions as modules
import core1_competition as core
import intervention_gain as method
from check_intervention_gain import run as oracle,reject
import intervention_gain_worker as worker
import collect_intervention_gain as collector
from concm_statistical_metrics import aggregate
from run_module_cycle import save as atomic_save

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);atomic_save(path,value)

def run(protocol):
    begun=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());c=dict(gpus=[0,1],jobs=[dict(job_name=n) for n in worker.JOBS]);worker.validate(c,p)
    for key,value in [('test_accessed',True),('optimizer_updates',1),('gpus',[0,2]),('main_head','GAIN_RESCUE_ONLY'),('heads',list(worker.HEADS)+['OTHER']),('batch_size',32),('validation_retuning',True),('automatic_retry',True),('seed',1)]:
        bad=copy.deepcopy(p);bad[key]=value;reject(lambda:worker.validate(c,bad))
    with tempfile.TemporaryDirectory() as temp:
        f=Path(temp)/'design.json';save(f,p['design_parameters']);mathcheck=oracle(f)
        root=Path(temp)/'barrier';root.mkdir();save(root/'PROGRAM_STATE.json',dict(phase='PREFLIGHT',variant='GAIN',sidecars_sealed_at=None));reject(lambda:worker.barrier(root,p,'fixture','preflight'))
        s=dict(lock=copy.deepcopy(p),source_commit='fixture',reporting_source_commit='fixture',prepared={},runs={},preflights={},ledger={},cpu_checks=dict(process_residence_seconds=1.,pipeline_seconds=.5,all_CPU_attempt_residence_seconds=1.),engineering_events=[],program=dict(status='COMPLETE',active={},costs=[],cpu_costs=[],failures=[],whole_round_wall_seconds=2.))
        for n in worker.JOBS:
            ds=n.split('_')[0];arm=n[len(ds)+1:];data=p['data'][ds];stages=[];prep=[];refs=[];exposures=0
            for task in range(1,len(data['task_sizes'])+1):
                seen=data['order'][:sum(data['task_sizes'][:task])];old=sum(data['task_sizes'][:task-1]);labels=np.concatenate([np.full(data['val_counts'][str(c)],c) for c in seen]);features=np.random.default_rng(74002+task).normal(size=(len(labels),8)).astype(np.float32);head=np.random.default_rng(task).normal(size=(8,len(seen))).astype(np.float32);off=features@head;delta=np.zeros_like(head)
                if old:
                    delta[:,:old]=np.random.default_rng(task*997).normal(0,.05,size=(8,old));delta[:,:old]-=delta[:,:old].mean(1,keepdims=True)
                on,raw,change=worker.score(off,features,delta,old);a,b=worker.diagnosis(off,on,raw,features,head,delta,labels,data,task);a.update(published_control_exact=True,encoder_state_unchanged=True,mechanical=dict(relative_residual=0.,exact_identity=not bool(delta.any())),risk=worker.gain_report({},off,on,labels,seen,old));refs.append(a['metrics']['OFF']);exposures+=len(labels)
                md=dict(dataset=ds,control=arm,control_source=p['controls'][arm]['source'],order=data['order'],seen=seen,seed=74002,task=task,source='fixture',version=method.VERSION,control_steps=data['cumulative_steps'][task-1],optimizer_updates=0)
                payload=worker.pack(torch.from_numpy(delta).double(),old,dict(md,variant='GAIN'))
                assert torch.equal(worker.restore(payload,old,dict(md,variant='GAIN'),delta.shape),torch.from_numpy(delta))
                bad=copy.deepcopy(payload);bad['production_metadata']['control_steps']=-1;reject(lambda:worker.restore(bad,old,dict(md,variant='GAIN'),delta.shape))
                variants={v:dict(sidecar_bytes=100,metadata=dict(md,variant=v),mechanical=dict(relative_residual=0.,exact_identity=not bool(delta.any()))) for v in worker.HEADS}
                prep.append(dict(task=task,variants=variants,source_file=dict(bytes=100,mtime_ns=1),native_solve=dict(relative_residual=0.)))
                stages.append(dict(task=task,heads={v:copy.deepcopy(a) for v in worker.HEADS},raw_heads={v:copy.deepcopy(b) for v in worker.HEADS},comparisons={on+'−'+before:copy.deepcopy(a) for on,before in worker.CONTRASTS}))
            s['lock']['control_references'][n]=dict(stages=refs)
            s['prepared'][n]=dict(status='PREPARED',job=n,source_commit='fixture',optimizer_updates=0,medical_images_read=0,stages=prep)
            save(root/'prepared'/n/'PREPARED.json',s['prepared'][n])
            metrics={v:{mode:aggregate([z['raw_heads' if mode=='raw' else 'heads'][v]['metrics']['ON'] for z in stages],data) for mode in ('guarded','raw')} for v in worker.HEADS}
            common=dict(version=method.VERSION,source_commit='fixture',optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False)
            s['runs'][n]=dict(common,job=n,dataset=ds,control=arm,status=dict(status='COMPLETE'),stages=stages,metrics=metrics,control_metrics=aggregate(refs,data),validation_image_exposures=exposures)
            s['preflights'][n]=dict(common,status='PASS',parameters_unchanged=True,CPU_GPU_scores_close=True,guard_checked=True,classification_metrics_read=False,real_validation_images=64)
            for op in ('preflight','evaluate'):s['program']['costs'].append(dict(job=n,operation=op,gpu=0 if arm=='base' else 1,returncode=0,process_started=True,residence_seconds=1.))
            s['program']['cpu_costs'].append(dict(job=n,operation='prepare',gpu=None,returncode=0,process_started=True,residence_seconds=1.))
        save(root/'PROGRAM_STATE.json',dict(phase='PREFLIGHT',variant='GAIN',sidecars_sealed_at=1));worker.barrier(root,p,'fixture','preflight')
        save(root/'PROGRAM_STATE.json',dict(phase='EVALUATE',variant='GAIN',sidecars_sealed_at=1,preflights_sealed_at={}));reject(lambda:worker.barrier(root,p,'fixture','evaluate'))
        for n,a in s['preflights'].items():save(root/'GAIN'/'preflight'/n/'RESULTS.json',a)
        save(root/'PROGRAM_STATE.json',dict(phase='EVALUATE',variant='GAIN',sidecars_sealed_at=1,preflights_sealed_at={'GAIN':1}));worker.barrier(root,p,'fixture','evaluate')
        assert collector.validate(s)
        bad=copy.deepcopy(s);bad['runs']['HK_base']['stages'][-1]['heads']['GAIN']['metrics']['ON']['per_class_n']['5']=2;reject(lambda:collector.validate(bad))
        bad=copy.deepcopy(s);bad['prepared']['ISIC_base']['stages'][0]['variants']['GAIN']['metadata']['control_steps']=0;reject(lambda:collector.validate(bad))
        result=Path(temp)/'fixture';result.mkdir();collector.report(s,result);report=json.loads((result/'results/RESULTS.json').read_text());assert len(report['rows'])==22 and len(report['raw_rows'])==16
        contributions=json.loads((result/'results/CONTRIBUTIONS.json').read_text());assert len(contributions['contrasts'])==40 and len(contributions['interactions'])==2
        # Synthetic signed counts exercise rescue/harm and the guard's old false positives.
        assert any(z['heads']['GAIN']['paired']['all']['changed'] for a in s['runs'].values() for z in a['stages'])
        assert any(z['raw_heads']['GAIN']['paired']['new']['harmed'] for a in s['runs'].values() for z in a['stages'])
    return dict(status='PASS',production_integrated=True,synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,formal_optimizer_updates=0,design_dense_oracle=mathcheck,checkpoint_roundtrip=True,early_validation_and_illegal_scope_refused=True,four_real_preflight_barrier=True,bad_denominator_and_metadata_refused=True,full_report_rows=22,raw_report_rows=16,contrasts=40,interactions=2,guarded_and_raw_signed_counts=True,pipeline_seconds=time.monotonic()-begun)
if __name__=='__main__':
    import os
    print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
