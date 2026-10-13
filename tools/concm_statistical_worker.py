"""Original checkpoint statistics only; no medical train or test loader."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import concm_statistical as method
import core1_competition as competition
from concm_statistical_metrics import stage,aggregate
from lca_pair_worker import original,admit,fingerprint,metadata,JOBS
from lca_switch_diagnostic import read_val
from run_module_cycle import save


def barrier(root,p,source,operation,variant):
    program=json.loads((root/'PROGRAM_STATE.json').read_text())
    if program['phase']!=operation.upper() or program['variant']!=variant or not program.get('sidecars_sealed_at'):
        raise ValueError('Validation opened before statistics seal')
    for name in JOBS:
        prepared=json.loads((root/'prepared'/name/'PREPARED.json').read_text())
        ds=name.split('_')[0]
        if prepared['status']!='PREPARED' or prepared['source_commit']!=source or len(prepared['stages'])!=len(p['data'][ds]['task_sizes']):
            raise ValueError('Four complete preparations required')
    if operation=='evaluate':
        if not program.get('preflights_sealed_at',{}).get(variant):raise ValueError('Preflight seal missing')
        for name in JOBS:
            f=json.loads((root/variant/'preflight'/name/'RESULTS.json').read_text())
            if (f['status']!='PASS' or f['optimizer_updates'] or f['test_accessed'] or f['source_commit']!=source
                    or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close']):
                raise ValueError('Four genuine preflights required')


def run(c,index,operation,variant):
    start=time.monotonic();root=Path(c['root']);p=json.loads(Path(c['protocol']).read_text())
    method.validate_parameters(p)
    if (operation not in ('prepare','preflight','evaluate') or c['gpus']!=[0,1]
            or [j['job_name'] for j in c['jobs']]!=JOBS
            or (operation!='prepare' and variant not in method.VARIANTS)):
        raise ValueError('Invalid production scope')
    j=c['jobs'][index];d=p['data'][j['name']];old=original(j,p)
    folder=root/'prepared'/j['job_name'] if operation=='prepare' else root/variant/('preflight' if operation=='preflight' else 'runs')/j['job_name']
    folder.mkdir(parents=True,exist_ok=False)
    def state(status,**kw):
        save(folder/'STATUS.json',dict(status=status,operation=operation,variant=variant,
            optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,**kw))
    def budget():
        if time.time()>=c['deadline']:raise TimeoutError('Frozen four-hour deadline')
    def meta(task,v):return dict(metadata(c,j,p,task),variant=v,version=method.VERSION)
    torch.set_num_threads(2);torch.manual_seed(74002)
    state('RUNNING',phase='ADMISSION')
    try:
        if operation=='prepare':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('CPU only preparation')
            import module_additions as modules
            stages=[]
            for task in range(1,len(d['task_sizes'])+1):
                budget();state('RUNNING',task=task,phase='CPU_ANALYTIC_SIDECAR')
                path=Path(old['output'])/f'stage_{task}.pt';fp=fingerprint(path)
                st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task)
                R,_=modules.metric(st['bank'],j['control'],j['name'],st['seen'],st['module_memory'])
                W,a=competition.bank_head(st['bank'],st['boundary_pair_weights'].double(),regularizer=R)
                error=float((W-st['head'].double()).norm()/W.norm().clamp_min(1e-30))
                if error>p['saved_head_relative_tolerance'] or not torch.allclose(W.float(),st['head'],atol=p['saved_head_element_atol'],rtol=p['saved_head_element_rtol']):
                    raise ValueError('Published original head reconstruction mismatch')
                variants={}
                for v in method.VARIANTS:
                    head,audit=method.fit(st['bank'],st['boundary_pair_weights'].double(),R,W,v)
                    side=dict(head=head.float(),metadata=meta(task,v))
                    sidepath=folder/f'{v}_{task}.pt';torch.save(side,sidepath)
                    back=torch.load(sidepath,map_location='cpu',weights_only=False)
                    if back['metadata']!=meta(task,v) or not torch.equal(back['head'],head.float()):raise ValueError('Sidecar roundtrip mismatch')
                    variants[v]=dict(sidecar_bytes=sidepath.stat().st_size,metadata=side['metadata'],mechanical=audit)
                if fingerprint(path)!=fp:raise ValueError('Historical checkpoint changed')
                stages.append(dict(task=task,source_file=fp,reconstruction_relative_error=error,native_solve=a,variants=variants))
                del st,R,W,head,side,back
            save(folder/'PREPARED.json',dict(status='PREPARED',job=j['job_name'],source_commit=c['source_commit'],stages=stages,
                optimizer_updates=0,medical_images_read=0,cpu_pipeline_seconds=time.monotonic()-start))
            state('PREPARED',stages=len(stages));return
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():raise ValueError('GPU01 only')
        barrier(root,p,c['source_commit'],operation,variant)
        from run_multilabel import ApartFeatures
        from run_medical_v2 import transform
        from run_prototype_single import Images,extract
        from torch.utils.data import DataLoader
        torch.backends.cuda.matmul.allow_tf32=False
        prepared=json.loads((root/'prepared'/j['job_name']/'PREPARED.json').read_text())
        rows=read_val(old,d);reports=[];exposures=0;side_bytes=0;check_bytes=0
        tasks=[len(d['task_sizes'])] if operation=='preflight' else range(1,len(d['task_sizes'])+1)
        for task in tasks:
            budget();state('RUNNING',task=task,phase='READ_ONLY_VALIDATION')
            path=Path(old['output'])/f'stage_{task}.pt';a=prepared['stages'][task-1]
            if fingerprint(path)!=a['source_file']:raise ValueError('Source checkpoint changed')
            st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task)
            side=torch.load(root/'prepared'/j['job_name']/f'{variant}_{task}.pt',map_location='cpu',weights_only=False)
            if side['metadata']!=meta(task,variant) or side['head'].shape!=st['head'].shape or not torch.isfinite(side['head']).all():
                raise ValueError('Sidecar provenance or head mismatch')
            seen=st['seen'];head=st['head'].numpy();newhead=side['head'].numpy();model=st['model'];del st
            encoder=ApartFeatures(old['legacy_repo'],old['weight'],len(d['order']),'cuda:0',74002)
            encoder.load_state_dict(model,strict=True);encoder.requires_grad_(False).eval()
            selected=[r for r in rows if r['label'] in seen]
            if operation=='preflight':selected=selected[:64]
            loader=DataLoader(Images(selected,old['images'],transform(False),74002+task*100003),batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            with torch.no_grad():x,labels=extract(encoder,loader,budget)
            if not all(torch.equal(v.detach().cpu(),model[key]) for key,v in encoder.state_dict().items()):raise ValueError('Encoder changed during inference')
            off=x@head;on=x@newhead
            if operation=='preflight':
                gpu=(torch.from_numpy(x).cuda()@torch.from_numpy(newhead).cuda()).cpu().numpy()
                if not np.allclose(gpu,on,atol=2e-6,rtol=2e-5) or not np.isfinite(on).all():raise ValueError('CPU GPU score compatibility failed')
                result=dict(status='PASS',job=j['job_name'],variant=variant,version=method.VERSION,source_commit=c['source_commit'],real_validation_images=len(selected),
                    optimizer_updates=0,policy_updates=0,parameters_unchanged=True,CPU_GPU_scores_close=True,classification_metrics_read=False,
                    train_accessed=False,test_accessed=False,peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-start)
                save(folder/'RESULTS.json',result);state('PASS');return
            report=stage(off,on,on,labels,d,task);ref=p['control_references'][j['job_name']]['stages'][task-1]
            for key in ('seen','per_class_n','per_class_recall','accuracy'):
                if report['metrics']['OFF'][key]!=ref[key]:raise ValueError('Published control parity failed')
            report.update(published_control_exact=True,encoder_state_unchanged=True,mechanical=a['variants'][variant]['mechanical'],reconstruction_relative_error=a['reconstruction_relative_error'])
            reports.append(report);exposures+=len(selected);side_bytes+=a['variants'][variant]['sidecar_bytes'];check_bytes+=a['source_file']['bytes']
            del encoder,model,side,x,off,on
        result=dict(job=j['job_name'],dataset=j['name'],control=j['control'],variant=variant,version=method.VERSION,source_commit=c['source_commit'],
            stages=reports,metrics={mode:aggregate([s['metrics'][mode] for s in reports],d) for mode in ('OFF','ON')},
            optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,validation_image_exposures=exposures,
            source_checkpoint_bytes=check_bytes,sidecar_bytes=side_bytes,peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-start)
        save(folder/'RESULTS.json',result);state('COMPLETE',stages=len(reports))
    except BaseException as exc:
        state('INCOMPLETE',error_type=type(exc).__name__);raise


if __name__=='__main__':
    run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()),int(os.environ['Q169_JOB']),os.environ['Q169_OPERATION'],os.environ.get('Q169_VARIANT'))
