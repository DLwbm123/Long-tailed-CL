"""Frozen original-checkpoint admission, CPU sidecars, zero-update GPU inference."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import core1_competition as competition
import lca_old_pair_repair as repair
from lca_pair_metrics import stage,aggregate
from lca_switch_diagnostic import read_val
from run_module_cycle import save

JOBS=[ds+'_'+arm for ds in ('ISIC','HK') for arm in ('base','local_transport')]


def validate(c,p):
    if (p['version']!=repair.VERSION or p['status']!='FROZEN_EVALUATION_ONLY'
            or p['cells']!=['A0','AL','B','P'] or p['gpus']!=[0,1] or c['gpus']!=[0,1]
            or p['seed']!=74002 or p['optimizer_updates'] or p['training_feedback']
            or p['train_images_allowed'] or p['test_accessed'] or p['validation_retuning']
            or p['relative_pair_curvature']!=1. or p['native_energy_radius']!=.1 or p['pair_target']!=1.
            or p['normalized'] is not True or p['guard'] is not True
            or p['primary_control']!='local_transport' or p['batch_size']!=64 or p['workers']!=0
            or p['automatic_retry'] or [j['job_name'] for j in c['jobs']]!=JOBS):
        raise ValueError('Unauthorized production scope')


def val_barrier(root,p,operation):
    state=json.loads((root/'PROGRAM_STATE.json').read_text())
    if state['phase']!=operation.upper() or not state.get('sidecars_sealed_at'):
        raise ValueError('Validation opened before all sidecars sealed')
    for name in JOBS:
        a=json.loads((root/'prepared'/name/'PREPARED.json').read_text())
        if a['status']!='PREPARED' or len(a['stages'])!=len(p['data'][name.split('_')[0]]['task_sizes']):
            raise ValueError('Missing complete sidecar trajectory')
    if operation=='evaluate':
        if not state.get('preflights_sealed_at'):raise ValueError('Preflight barrier missing')
        for name in JOBS:
            a=json.loads((root/'preflight'/name/'RESULTS.json').read_text())
            if a['status']!='PASS' or a['optimizer_updates'] or not a['parameters_unchanged']:
                raise ValueError('All four genuine zero-update preflights required')


def metadata(c,j,p,task):
    d=p['data'][j['name']]
    return dict(dataset=j['name'],control=j['control'],control_source=p['controls'][j['control']]['source'],
        order=d['order'],seen=d['order'][:sum(d['task_sizes'][:task])],seed=74002,task=task,source=c['source_commit'])


def original(j,p):
    parent=Path(j['parent_root']);control=p['controls'][j['control']]
    if parent.name!=control['round']:raise ValueError('Original round differs')
    pc=json.loads((parent/'CONFIG.private.json').read_text())
    receipt=json.loads((parent/'DELIVERY.private.json').read_text())
    old=next(x for x in pc['jobs'] if x['job_name']==j['job_name'])
    d=p['data'][j['name']]
    if (pc['source_commit']!=control['source'] or old['source_commit']!=control['source']
            or receipt['commit']!=control['delivery'] or not receipt['nas_saved']
            or old['module']!=j['control'] or old['name']!=j['name'] or old['method']!='static_pc'
            or old['order']!=d['order'] or old['task_sizes']!=d['task_sizes']
            or old['seed']!=74002 or old['split_seed']!=74002 or old['epochs']!=2
            or old['batch_size']!=64 or old['lr']!=.0003
            or old['expected_steps']!=d['cumulative_steps'][-1]
            or old['evaluation_split']!='development_validation' or old.get('test_accessed',False)
            or Path(old['output']).resolve()!=parent.resolve()/'runs'/j['job_name']):
        raise ValueError('Delivered original configuration/provenance differs')
    return old


def admit(st,d,arm,task):
    seen=d['order'][:sum(d['task_sizes'][:task])]
    if (st['seen']!=seen or st['steps']!=d['cumulative_steps'][task-1] or st['module']!=arm
            or st['head'].shape!=(1536,len(seen)) or st['head'].dtype!=torch.float32
            or st['bank']['n']!=[d['train_counts'][str(v)] for v in seen]
            or not torch.isfinite(st['head']).all() or not st['model']
            or (arm=='base' and st['module_memory'] is not None)):
        raise ValueError('Original checkpoint stage, count or budget differs')


def fingerprint(path):
    st=path.stat();return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def run(c,index,operation):
    from run_prototype_single import Images,extract
    from run_multilabel import ApartFeatures
    from run_medical_v2 import transform
    from torch.utils.data import DataLoader
    import module_additions as modules
    start=time.monotonic();root=Path(c['root']);p=json.loads(Path(c['protocol']).read_text());validate(c,p)
    j=c['jobs'][index];d=p['data'][j['name']];old=original(j,p)
    folder=root/('prepared' if operation=='prepare' else 'preflight' if operation=='preflight' else 'runs')/j['job_name']
    folder.mkdir(parents=True,exist_ok=False)
    def state(status,**kw):
        save(folder/'STATUS.json',dict(status=status,operation=operation,optimizer_updates=0,policy_updates=0,
            train_accessed=False,test_accessed=False,**kw))
    def budget():
        if time.time()>=c['deadline']:raise TimeoutError('Frozen two-hour limit')
    torch.set_num_threads(2);torch.manual_seed(74002)
    state('RUNNING',phase='ADMISSION')
    try:
        if operation=='prepare':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('CPU preparation only')
            stages=[]
            for task in range(1,len(d['task_sizes'])+1):
                budget();state('RUNNING',task=task,phase='CPU_ANALYTIC_SIDECAR')
                path=Path(old['output'])/f'stage_{task}.pt';fp=fingerprint(path)
                st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task)
                bank=st['bank'];local=st['module_memory']
                R,_=modules.metric(bank,j['control'],j['name'],st['seen'],local)
                W,a=competition.bank_head(bank,st['boundary_pair_weights'].double(),regularizer=R)
                error=float((W-st['head'].double()).norm()/W.norm().clamp_min(1e-30))
                if (error>p['saved_head_relative_tolerance'] or not torch.allclose(W.float(),st['head'],
                        atol=p['saved_head_element_atol'],rtol=p['saved_head_element_rtol'])):
                    raise ValueError('Saved native float32 head reconstruction differs')
                old_count=sum(d['task_sizes'][:task-1])
                delta,audit=repair.fit(bank,st['boundary_pair_weights'].double(),R,W,old_count)
                side=repair.checkpoint(delta,old_count,metadata(c,j,p,task))
                restored=repair.restore(side,metadata(c,j,p,task),old_count,st['head'].shape)
                assert torch.equal(restored,delta.float())
                if fingerprint(path)!=fp:raise ValueError('Source checkpoint changed during read')
                torch.save(side,folder/f'stage_{task}.pt')
                stages.append(dict(task=task,source_file=fp,sidecar_bytes=(folder/f'stage_{task}.pt').stat().st_size,
                    reconstruction_relative_error=error,native_solve=a,mechanical=audit,metadata=side['metadata']))
                del st,bank,local,R,W,delta,side,restored
            save(folder/'PREPARED.json',dict(status='PREPARED',job=j['job_name'],source_commit=c['source_commit'],
                stages=stages,optimizer_updates=0,medical_images_read=0,cpu_pipeline_seconds=time.monotonic()-start))
            state('PREPARED',stages=len(stages));return
        if operation not in ('preflight','evaluate'):raise ValueError('No training entry')
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():raise ValueError('GPU01 required')
        val_barrier(root,p,operation);budget()
        prepared=json.loads((root/'prepared'/j['job_name']/'PREPARED.json').read_text())
        if prepared['source_commit']!=c['source_commit']:raise ValueError('Sidecar science source differs')
        rows=read_val(old,d);reports=[];exposures=0;side_bytes=0;check_bytes=0
        tasks=[len(d['task_sizes'])] if operation=='preflight' else range(1,len(d['task_sizes'])+1)
        for task in tasks:
            state('RUNNING',task=task,phase='READ_ONLY_FEATURES');budget()
            path=Path(old['output'])/f'stage_{task}.pt';a=prepared['stages'][task-1]
            if fingerprint(path)!=a['source_file'] or a['metadata']!=metadata(c,j,p,task):raise ValueError('Frozen checkpoint changed')
            st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task)
            side=torch.load(root/'prepared'/j['job_name']/f'stage_{task}.pt',map_location='cpu',weights_only=False)
            old_count=sum(d['task_sizes'][:task-1]);delta=repair.restore(side,metadata(c,j,p,task),old_count,st['head'].shape)
            seen=st['seen'];head=st['head'].numpy();model=st['model'];del st
            encoder=ApartFeatures(old['legacy_repo'],old['weight'],len(d['order']),'cuda:0',74002)
            encoder.load_state_dict(model,strict=True);encoder.requires_grad_(False).eval()
            selected=[r for r in rows if r['label'] in seen]
            if operation=='preflight':selected=selected[:64]
            loader=DataLoader(Images(selected,old['images'],transform(False),74002+task*100003),
                batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            with torch.no_grad():x,labels=extract(encoder,loader,budget)
            if not all(torch.equal(v.detach().cpu(),model[key]) for key,v in encoder.state_dict().items()):
                raise ValueError('Inference changed encoder state')
            off=x@head;change=x@delta.numpy();raw=off+change
            on=repair.guard(torch.from_numpy(off),torch.from_numpy(change),old_count).numpy()
            if operation=='preflight':
                gpu_on=repair.guard(torch.from_numpy(off).cuda(),torch.from_numpy(change).cuda(),old_count).cpu().numpy()
                if not np.array_equal(gpu_on,on):raise ValueError('CPU/GPU guard differs')
                if not np.isfinite(on).all() or not np.isfinite(raw).all():raise ValueError('Nonfinite inference')
                result=dict(status='PASS',job=j['job_name'],version=repair.VERSION,source_commit=c['source_commit'],
                    real_validation_images=len(selected),optimizer_updates=0,policy_updates=0,parameters_unchanged=True,
                    CPU_GPU_guard_exact=True,classification_metrics_read=False,train_accessed=False,test_accessed=False,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-start)
                save(folder/'RESULTS.json',result);state('PASS');return
            r=stage(off,on,raw,labels,d,task);reference=p['control_references'][j['job_name']]['stages'][task-1]
            for key in ('seen','per_class_n','per_class_recall','accuracy'):
                if r['metrics']['OFF'][key]!=reference[key]:raise ValueError('OFF parity with published control differs')
            r.update(published_control_exact=True,encoder_state_unchanged=True,mechanical=a['mechanical'],
                reconstruction_relative_error=a['reconstruction_relative_error'])
            reports.append(r);exposures+=len(selected);check_bytes+=a['source_file']['bytes'];side_bytes+=a['sidecar_bytes']
            del encoder,model,side,delta,x,off,on,raw,change
        result=dict(job=j['job_name'],dataset=j['name'],control=j['control'],version=repair.VERSION,
            source_commit=c['source_commit'],control_source=p['controls'][j['control']]['source'],
            stages=reports,metrics={mode:aggregate([s['metrics'][mode] for s in reports],d) for mode in ('OFF','ON')},
            optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,
            validation_image_exposures=exposures,source_checkpoint_bytes=check_bytes,sidecar_bytes=side_bytes,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-start)
        save(folder/'RESULTS.json',result);state('COMPLETE',stages=len(reports))
    except BaseException as exc:
        state('INCOMPLETE',error_type=type(exc).__name__);raise


if __name__=='__main__':
    run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()),int(os.environ['Q167_JOB']),os.environ['Q167_OPERATION'])
