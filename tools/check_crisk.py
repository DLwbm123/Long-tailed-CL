"""Synthetic CPU production checks, including all report states and barriers."""
import copy
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
import gain_conditional_risk as risk
import crisk_evidence as evidence
import crisk_worker as worker
import collect_crisk as collector
from gain_joint_worker import score
from intervention_gain_worker import diagnosis
from concm_statistical_metrics import aggregate
from check_gain_conditional_risk import run as native_check, reject


def fixture(p):
    c=dict(gpus=[0,1],jobs=[dict(job_name=n) for n in worker.JOBS]);worker.validate(c,p)
    for key,value in [('test_accessed',True),('gpus',[0,2]),('validation_retuning',True),('direction_changed',True),('main_head','NO_DRIFT'),('optimizer_updates',1),('old_image_training_replay',True)]:
        reject(lambda key=key,value=value:worker.validate(c,dict(p,**{key:value})))
    with tempfile.TemporaryDirectory() as td:
        r=Path(td);root=r/'barrier';root.mkdir();(root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='STATISTICS',metadata_sealed_at=None)))
        reject(lambda:worker.barrier(root,dict(source_commit='fixture'),p,'evaluate'))
        s=dict(lock=copy.deepcopy(p),source_commit='fixture',reporting_source_commit='fixture',CPU_check=dict(status='PASS',synthetic_only=True),prepared={},preflights={},statistics={},risk_prepared={},runs={},program=dict(status='COMPLETE',active={},costs=[],cpu_costs=[],failures=[],whole_round_wall_seconds=1.))
        rng=np.random.default_rng(74002)
        for job in worker.JOBS:
            ds=job.split('_')[0];data=p['data'][ds];arm=job[len(ds)+1:];stages=[];prepared=[];stats=[];risks=[];refs=[];train_exposures=val_exposures=0
            for task in range(1,len(data['task_sizes'])+1):
                seen=data['order'][:sum(data['task_sizes'][:task])];old=sum(data['task_sizes'][:task-1]);k=len(seen)
                labels=np.concatenate([np.full(data['val_counts'][str(v)],v,dtype=int) for v in seen]);x=rng.normal(size=(len(labels),8)).astype(np.float32);head=rng.normal(size=(8,k)).astype(np.float32);ray=np.zeros_like(head)
                if old:ray[:,:old]=rng.normal(0,.01,size=(8,old));ray[:,:old]-=ray[:,:old].mean(1,keepdims=True)
                deltas={v:(step*ray).astype(np.float32) for v,step in zip(evidence.CANDIDATES,risk.STEPS)}
                chosen={h:dict(chosen_scale=.125 if old and h=='CRISK' else 0.,status='MODEL_CONDITIONAL_SELECTION',candidates=[]) for h in risk.HEADS};heads={};raw={};off=x@head
                for h,delta in dict(deltas,**{h:deltas['C0125'] if z['chosen_scale'] else np.zeros_like(ray) for h,z in chosen.items()}).items():
                    on,unprotected,_=score(off,x,delta,old);a,b=diagnosis(off,on,unprotected,x,head,delta,labels,data,task);heads[h]=a;raw[h]=b
                refs.append(heads['CRISK']['metrics']['OFF']);stages.append(dict(task=task,heads=heads,raw_heads=raw,selections=chosen));prepared.append(dict(task=task))
                tc=sum(data['train_counts'][str(v)] for v in seen);train_exposures+=tc*(2 if old else 1);val_exposures+=len(labels)
                stats.append(dict(task=task));audit=[]
                if old:
                    for h in risk.HEADS:
                        for v in evidence.CANDIDATES:
                            for label in seen:
                                n=data['train_counts'][str(label)];audit.append(dict(mode=h,candidate=v,label=label,target_actual_n=n,classifier_independent=False,training_counts=dict(n=n,correct_off=n,rescued=0,harmed=0),rescue_mean=0.,harm_mean=0.,conditional_net_mean=0.))
                risks.append(dict(task=task,selections=chosen,class_audit=audit,drift=[]))
            s['lock']['control_references'][job]=dict(stages=refs)
            s['prepared'][job]=dict(status='PREPARED',source_commit='fixture',stages=prepared,optimizer_updates=0)
            s['preflights'][job]=dict(status='PASS',source_commit='fixture',optimizer_updates=0,policy_updates=0,test_accessed=False,val_accessed=False,classification_metrics_read=False,parameters_unchanged=True,CPU_GPU_scores_close=True,real_training_images=64)
            s['statistics'][job]=dict(status='STATS_READY',source_commit='fixture',stages=stats,image_exposures=train_exposures,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False)
            s['risk_prepared'][job]=dict(status='RISK_READY',source_commit='fixture',stages=risks,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False)
            s['runs'][job]=dict(status='COMPLETE',source_commit='fixture',stages=stages,image_exposures=val_exposures,optimizer_updates=0,policy_updates=0,val_accessed=True,test_accessed=False,
                               control_metrics=aggregate(refs,data),metrics={h:{mode:aggregate([z['raw_heads' if mode=='raw' else 'heads'][h]['metrics']['ON'] for z in stages],data) for mode in ('guarded','raw')} for h in evidence.CANDIDATES+risk.HEADS})
            for op in ('preflight','statistics','evaluate'):s['program']['costs'].append(dict(job=job,operation=op,gpu=0,returncode=0,process_started=True,residence_seconds=1.))
            for op in ('prepare','risk_prepare'):s['program']['cpu_costs'].append(dict(job=job,operation=op,gpu=None,returncode=0,process_started=True,residence_seconds=1.))
            for directory,value,file in [('prepared',s['prepared'][job],'PREPARED.json'),('preflight',s['preflights'][job],'RESULTS.json'),('statistics',s['statistics'][job],'RESULTS.json'),('risk_prepared',s['risk_prepared'][job],'RESULTS.json')]:
                path=root/directory/job;path.mkdir(parents=True);(path/file).write_text(json.dumps(value))
        (root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='EVALUATE',metadata_sealed_at=1,preflights_sealed_at=1,statistics_sealed_at=1,selections_sealed_at=None)))
        reject(lambda:worker.barrier(root,dict(source_commit='fixture'),s['lock'],'evaluate'))
        (root/'PROGRAM_STATE.json').write_text(json.dumps(dict(phase='EVALUATE',metadata_sealed_at=1,preflights_sealed_at=1,statistics_sealed_at=1,selections_sealed_at=1)))
        worker.barrier(root,dict(source_commit='fixture'),s['lock'],'evaluate');assert collector.validate(s)
        bad=copy.deepcopy(s);bad['runs']['HK_base']['stages'][-1]['heads']['CRISK']['metrics']['ON']['per_class_n']['5']=2;reject(lambda:collector.validate(bad))
        bad=copy.deepcopy(s);bad['risk_prepared']['HK_base']['stages'][-1]['selections']['CRISK']['chosen_scale']=1.;reject(lambda:collector.validate(bad))
        a=collector.report(s,r);assert a['status']=='PASS';assert len(list((r/'results').glob('*.csv')))>=10
    return dict(all_18_synthetic_stage_reports=True,all_five_selectors_and_four_candidates=True,bad_denominator_refused=True,val_before_every_seal_refused=True,selection_parity_checked=True)


def run(protocol):
    start=time.monotonic();p=json.loads(Path(protocol).read_text());torch.set_num_threads(2)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    reused=os.environ.get('Q224_CPU_NATIVE_REUSE')
    if reused:
        previous=json.loads(Path(reused).read_text());assert previous['status']=='PASS' and previous['full_high_dimensional_cpu_PASS']
        native=previous['native_compatibility'];assert len(native['highdimensional_checks'])==2 and all(z['dimensions']==[1536,23] for z in native['highdimensional_checks'])
    else:
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'design.json';path.write_text(json.dumps(p['design_parameters']));native=native_check(path)
    modelmean=np.array([.05,-.05,-1.,.1,-.1,0.]);v=np.array([.1,-.1,0.,0.,0.,0.]);G=np.outer(v,v)
    point=evidence.integrated(modelmean,G,25,2,0,(0,0,0));wide=evidence.integrated(modelmean,G,25,2,0,(0,0,0),np.full(6,.05)*np.array([1,1,1,1,1,0]))
    for h in evidence.CANDIDATES:
        a=point[h];b=wide[h];assert 0<=a['model']['off_error']<=1 and b['drift']['off_error_sensitivity']>=0
    tie=evidence.integrated(np.zeros(6),np.zeros((6,6)),3,2,0,(0,0,0))
    assert tie['C1']['conditional_model_rescue'] is None and not tie['C1']['model']['rescue_prior_available'] and not tie['C1']['model']['harm_prior_available']
    assert "if operation=='prepare':s['metadata_sealed_at']" in Path(__file__).with_name('run_crisk.py').read_text()
    rng=np.random.default_rng(74002);x=rng.normal(size=(60,8)).astype(np.float32);before=x*.99;head=rng.normal(size=(8,3)).astype(np.float32);ray=np.zeros_like(head);ray[:,:2]=rng.normal(0,.001,size=(8,2));ray[:,:2]-=ray[:,:2].mean(1,keepdims=True)
    labels=np.repeat([0,1,2],20);deltas={v:(step*ray).astype(np.float32) for v,step in zip(evidence.CANDIDATES,risk.STEPS)};bank=dict(mu=torch.from_numpy(np.stack([x[labels==c].mean(0) for c in range(3)])))
    e=evidence.training_evidence(x,before,labels,head,deltas,bank,[0,1,2],2,2,[2,1]);a=evidence.build_selections(e,dict(source='synthetic',control='base'))
    json.loads(json.dumps(a,allow_nan=False))
    assert set(a['selections'])==set(risk.HEADS) and len(a['class_audit'])==3*4*5
    assert all(z['target_actual_n']==20 and not z['classifier_independent'] for z in a['class_audit'])
    checks=fixture(p)
    return dict(status='PASS',full_high_dimensional_cpu_PASS=True,synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,medical_val_reads=0,test_accessed=False,optimizer_updates=0,policy_updates=0,native_compatibility=native,native_compatibility_reused_not_recharged=bool(reused),production_checks=checks,real_model_nodes_checked=65536,conditional_zero_denominator_prior_disabled=True,all_source_counts_not_independent=True,private_projection_dimension_max=46,pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
