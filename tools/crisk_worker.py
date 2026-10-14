"""Frozen training-side paired inference. No optimizer, val reader or selector."""
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from gain_joint_production import CANDIDATES
from gain_joint_worker import restore, score
from gain_calibration_plan import diagnostic_folds, paired_score_moments
from lca_pair_worker import original, admit, fingerprint, JOBS
from intervention_gain_worker import diagnosis
from concm_statistical_metrics import stage
from run_module_cycle import save

VERSION = 'crisk_training_evidence_evaluation_v4'
SOURCE = '533562177c4fa39f18dcbe981e2c0d0f596a09ac'


def validate(c,p):
    import gain_conditional_risk as risk
    risk.validate_design(p['design_parameters'])
    exact=dict(version=VERSION,status='FROZEN_CRISK_EVALUATION',gpus=[0,1],seed=74002,
        optimizer_updates=0,policy_updates=0,train_images_allowed=True,old_train_inference_only=True,
        old_image_training_replay=False,test_accessed=False,validation_retuning=False,automatic_retry=False,
        direction_changed=False,steps=[.125,.25,.5,1.],batch_size=64,workers=0,
        checkpoint_stages=18,sidecars_reused=72,previous_sidecar_source=SOURCE,
        primary_control='local_transport',main_head='CRISK',heads=list(risk.HEADS),
        production_integrated=True,GPU_launch_authorized=True,independent_calibration=False,
        deadline_seconds=86400,model_nodes=65536,model_batch=4096,
        model_distribution='current_train_class_projected_shared_scale_student_t',
        model_df='target_train_n_minus_1_even_for_recentered_donors',RMS_sensitivity_multiplier=1.,
        borrow_shape_temperature=.1)
    if any(p.get(k)!=v for k,v in exact.items()) or c['gpus']!=[0,1] or [j['job_name'] for j in c['jobs']]!=JOBS:
        raise ValueError('Frozen CRISK production scope required')


def read_train(orig,d):
    # Deliberately never call manifests(), read_val() or a train-mode transform.
    with (Path(orig['manifest'])/'train.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        rel=Path(r['relative_path']);r['label']=int(r['original_label'])
        if r['split']!='train' or rel.is_absolute() or '..' in rel.parts or not r['identity_component'] or r['label'] not in d['order']:
            raise ValueError('Invalid training-only manifest')
    if len({r['relative_path'] for r in rows})!=len(rows) or {str(v):sum(r['label']==v for r in rows) for v in d['order']}!=d['train_counts']:
        raise ValueError('Original training counts differ')
    groups={}
    for r in rows:groups.setdefault(r['identity_component'],set()).add(r['label'])
    if any(len(v)!=1 for v in groups.values()):raise ValueError('Group spans classes')
    return rows


def barrier(root,c,p,op):
    s=json.loads((root/'PROGRAM_STATE.json').read_text())
    if s['phase']!=op.upper() or not s.get('metadata_sealed_at'):raise ValueError('Wrong operation or metadata seal')
    for name in JOBS:
        a=json.loads((root/'prepared'/name/'PREPARED.json').read_text())
        if a['status']!='PREPARED' or a['source_commit']!=c['source_commit'] or len(a['stages'])!=len(p['data'][name.split('_')[0]]['task_sizes']):raise ValueError('All metadata preparations required')
    if op in ('statistics','risk_prepare','evaluate'):
        if not s.get('preflights_sealed_at'):raise ValueError('Four genuine zero checks required')
        for name in JOBS:
            f=json.loads((root/'preflight'/name/'RESULTS.json').read_text())
            if f['status']!='PASS' or f['source_commit']!=c['source_commit'] or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or f['optimizer_updates'] or f['policy_updates'] or f['val_accessed'] or f['test_accessed'] or f['classification_metrics_read']:raise ValueError('Invalid genuine preflight')
    if op in ('risk_prepare','evaluate'):
        if not s.get('statistics_sealed_at'):raise ValueError('All training evidence required')
        for name in JOBS:
            z=json.loads((root/'statistics'/name/'RESULTS.json').read_text())
            if z['status']!='STATS_READY' or z['source_commit']!=c['source_commit'] or z['val_accessed'] or z['test_accessed']:raise ValueError('Incomplete training evidence')
    if op=='evaluate':
        if not s.get('selections_sealed_at'):raise ValueError('All selections before val required')
        import gain_conditional_risk as risk
        for name in JOBS:
            z=json.loads((root/'risk_prepared'/name/'RESULTS.json').read_text())
            if z['status']!='RISK_READY' or z['source_commit']!=c['source_commit'] or set(z['stages'][0]['selections'])!=set(risk.HEADS) or z['val_accessed'] or z['test_accessed']:raise ValueError('Incomplete selections')
            if any(x['status']=='NOT_READY' for st in z['stages'] for x in st['selections'].values()):raise ValueError('Missing evidence cannot evaluate')


def run(c,index,op):
    begun=time.monotonic();root=Path(c['root']);p=json.loads(Path(c['protocol']).read_text());validate(c,p)
    if op not in ('prepare','preflight','statistics','risk_prepare','evaluate'):raise ValueError('No training/evaluation tuning entry')
    j=c['jobs'][index];d=p['data'][j['name']];orig=original(j,p)
    folder=root/dict(prepare='prepared',preflight='preflight',statistics='statistics',risk_prepare='risk_prepared',evaluate='runs')[op]/j['job_name'];folder.mkdir(parents=True,exist_ok=False)
    def state(status,**kw):save(folder/'STATUS.json',dict(status=status,operation=op,optimizer_updates=0,policy_updates=0,train_inference_only=op!='evaluate',val_accessed=op=='evaluate',test_accessed=False,**kw))
    def budget():
        if time.time()>=c['deadline']:raise TimeoutError('Frozen diagnostic deadline')
    torch.set_num_threads(2);torch.manual_seed(74002);state('RUNNING',phase='ADMISSION')
    try:
        prior_root=Path(c['sidecar_root']);prior=json.loads((prior_root/'prepared'/j['job_name']/'PREPARED.json').read_text())
        if prior['status']!='PREPARED' or prior['source_commit']!=SOURCE:raise ValueError('Fixed direction source differs')
        if op=='prepare':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('CPU preparation only')
            stages=[]
            for task,a in enumerate(prior['stages'],1):
                budget();state('RUNNING',task=task,phase='READ_ONLY_METADATA')
                path=Path(orig['output'])/f'stage_{task}.pt';fp=fingerprint(path)
                if fp!=a['source_file']:raise ValueError('Delivered control checkpoint changed')
                st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task);variants={}
                for v in CANDIDATES:
                    md=a['variants'][v]['metadata'];sidepath=prior_root/'prepared'/j['job_name']/f'{v}_{task}.pt'
                    expected=dict(dataset=j['name'],control=j['control'],control_source=p['controls'][j['control']]['source'],order=d['order'],seen=st['seen'],seed=74002,task=task,source=SOURCE,variant=v,version='gain_joint_predictive_v2',control_steps=d['cumulative_steps'][task-1],optimizer_updates=0)
                    if md!=expected:raise ValueError('Fixed sidecar provenance differs')
                    side=torch.load(sidepath,map_location='cpu',weights_only=False);delta=restore(side,sum(d['task_sizes'][:task-1]),md,st['head'].shape)
                    if task==1 and delta.any():raise ValueError('First stage nonidentity')
                    variants[v]=dict(sidecar_file=fingerprint(sidepath),metadata=md,mechanical=a['variants'][v]['mechanical'])
                stages.append(dict(task=task,source_file=fp,variants=variants));del st,side,delta
            save(folder/'PREPARED.json',dict(status='PREPARED',job=j['job_name'],source_commit=c['source_commit'],stages=stages,medical_images_read=0,optimizer_updates=0));state('PREPARED',stages=len(stages));return
        if op=='risk_prepare':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('CPU risk preparation only')
            barrier(root,c,p,op)
            import crisk_evidence
            import gain_conditional_risk as risk
            stages=[]
            for task in range(1,len(d['task_sizes'])+1):
                budget();e=torch.load(root/'statistics'/j['job_name']/f'EVIDENCE_{task}.private.pt',map_location='cpu',weights_only=False)
                expected=dict(source_commit=c['source_commit'],job=j['job_name'],task=task,sidecar_source=SOURCE)
                if e['metadata']!=expected:raise ValueError('Training evidence source differs')
                meta=dict(dataset=j['name'],control=j['control'],source=c['source_commit'],sidecar_source=SOURCE,seed=74002)
                a=crisk_evidence.build_selections(e,meta,budget,lambda **kw:state('RUNNING',**kw));a['task']=task
                if any(x['status']=='NOT_READY' for x in a['selections'].values()):raise ValueError('Incomplete production evidence')
                stages.append(a);save(folder/'PARTIAL_RESULTS.json',dict(stages=stages))
            save(folder/'RESULTS.json',dict(status='RISK_READY',version=VERSION,source_commit=c['source_commit'],job=j['job_name'],stages=stages,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False,independent_calibration=False));state('RISK_READY',stages=len(stages));return
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():raise ValueError('GPU01 only')
        barrier(root,c,p,op)
        if op=='evaluate':
            from lca_switch_diagnostic import read_val
            rows=read_val(orig,d)
            selected_risks=json.loads((root/'risk_prepared'/j['job_name']/'RESULTS.json').read_text())
        else:rows=read_train(orig,d)
        prepared=json.loads((root/'prepared'/j['job_name']/'PREPARED.json').read_text())
        from run_multilabel import ApartFeatures
        from run_medical_v2 import transform
        from run_prototype_single import Images,extract
        from torch.utils.data import DataLoader
        torch.backends.cuda.matmul.allow_tf32=False
        encoder=ApartFeatures(orig['legacy_repo'],orig['weight'],len(d['order']),'cuda:0',74002).requires_grad_(False).eval()
        def features(st,selected,task):
            encoder.load_state_dict(st['model'],strict=True);encoder.requires_grad_(False).eval()
            loader=DataLoader(Images(selected,orig['images'],transform(False),74002+task*100003),batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            with torch.no_grad():x,y=extract(encoder,loader,budget)
            if not np.array_equal(y,np.array([r['label'] for r in selected])) or not all(torch.equal(value.detach().cpu(),st['model'][key]) for key,value in encoder.state_dict().items()):raise ValueError('Canonical pairing or frozen model changed')
            return x,y
        reports=[];exposures=0
        for task in ([len(d['task_sizes'])] if op=='preflight' else range(1,len(d['task_sizes'])+1)):
            budget();state('RUNNING',task=task,phase='VALIDATION_INFERENCE' if op=='evaluate' else 'CURRENT_TRAIN_INFERENCE')
            a=prepared['stages'][task-1];a_source_file=a['source_file'];path=Path(orig['output'])/f'stage_{task}.pt'
            if fingerprint(path)!=a['source_file']:raise ValueError('Frozen checkpoint changed')
            st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task);seen=st['seen'];old=sum(d['task_sizes'][:task-1]);selected=[r for r in rows if r['label'] in seen]
            if op=='preflight':selected=[r for r in selected if r['label'] in seen[:old]][:32]+[r for r in selected if r['label'] in seen[old:]][:32]
            x,labels=features(st,selected,task);exposures+=len(selected);head=st['head'].numpy();deltas={}
            for v in CANDIDATES:
                sidepath=prior_root/'prepared'/j['job_name']/f'{v}_{task}.pt'
                if fingerprint(sidepath)!=a['variants'][v]['sidecar_file']:raise ValueError('Sidecar changed')
                side=torch.load(sidepath,map_location='cpu',weights_only=False);deltas[v]=restore(side,old,a['variants'][v]['metadata'],head.shape).numpy()
            before=previous=None
            if old and op!='evaluate':
                state('RUNNING',task=task,phase='PAIRED_PREVIOUS_ENCODER')
                prevpath=Path(orig['output'])/f'stage_{task-1}.pt'
                if fingerprint(prevpath)!=prepared['stages'][task-2]['source_file']:raise ValueError('Previous checkpoint changed')
                previous=torch.load(prevpath,map_location='cpu',weights_only=False);admit(previous,d,j['control'],task-1);before,_=features(previous,selected,task);exposures+=len(selected)
            if op=='preflight':
                off=x@head;gx=torch.from_numpy(x).cuda();goff=gx@torch.from_numpy(head).cuda()
                if not np.allclose(goff.cpu().numpy(),off,atol=2e-6,rtol=2e-5):raise ValueError('CPU GPU OFF mismatch')
                for v,delta in deltas.items():
                    on,raw,change=score(off,x,delta,old);gd=gx@torch.from_numpy(delta).cuda()
                    from intervention_gain import guard
                    gon=guard(goff,gd,old).cpu().numpy()
                    if not np.allclose(gon,on,atol=2e-6,rtol=2e-5) or not np.allclose(gd.cpu().numpy(),change,atol=2e-6,rtol=2e-5) or not np.array_equal(on[:,old:],off[:,old:]):raise ValueError('Frozen CPU GPU guard failed')
                if before is None or not np.isfinite(x-before).all() or any(q.requires_grad for q in encoder.parameters()):raise ValueError('Paired zero-update check failed')
                save(folder/'RESULTS.json',dict(status='PASS',version=VERSION,source_commit=c['source_commit'],job=j['job_name'],real_training_images=len(selected),image_exposures=exposures,parameters_unchanged=True,CPU_GPU_scores_close=True,classification_metrics_read=False,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False,independent_calibration=False,peak_gpu_bytes=torch.cuda.max_memory_allocated()));state('PASS');return
            if op=='statistics':
                from crisk_evidence import training_evidence
                state('RUNNING',task=task,phase='TRAINING_EVIDENCE_AGGREGATION')
                e=training_evidence(x,before,labels,head,deltas,st['bank'],seen,old,task,d['task_sizes'])
                e['metadata']=dict(source_commit=c['source_commit'],job=j['job_name'],task=task,sidecar_source=SOURCE)
                torch.save(e,folder/f'EVIDENCE_{task}.private.pt')
                reports.append(dict(task=task,training_image_exposures=len(selected),previous_encoder_image_exposures=0 if not old else len(selected),seen=seen,drift=e['drift']))
            else:
                import gain_conditional_risk as risk
                from intervention_gain_worker import diagnosis
                chosen=selected_risks['stages'][task-1]['selections'];heads={};rawheads={};scored=dict(deltas)
                for mode,choice in chosen.items():
                    if choice['status']=='NOT_READY' or choice['chosen_scale'] not in (0.,)+risk.STEPS:raise ValueError('Val-time selector changed')
                    scored[mode]=np.zeros_like(head) if not choice['chosen_scale'] else deltas[CANDIDATES[risk.STEPS.index(choice['chosen_scale'])]]
                off=x@head;ref=p['control_references'][j['job_name']]['stages'][task-1]
                for v,delta in scored.items():
                    a,b=diagnosis(off,*score(off,x,delta,old)[:2],x,head,delta,labels,d,task)
                    for key in ('seen','per_class_n','per_class_recall','accuracy'):
                        if a['metrics']['OFF'][key]!=ref[key]:raise ValueError('Published OFF parity differs')
                    if a['paired']['new']['harmed'] or task==1 and a['paired']['all']['changed']:raise ValueError('New guard/first-stage identity failed')
                    heads[v]=a;rawheads[v]=b
                reports.append(dict(task=task,heads=heads,raw_heads=rawheads,selections=chosen))
            save(folder/'PARTIAL_RESULTS.json',dict(version=VERSION,source_commit=c['source_commit'],job=j['job_name'],stages=reports,image_exposures=exposures,optimizer_updates=0,policy_updates=0,val_accessed=op=='evaluate',test_accessed=False))
            if fingerprint(path)!=a_source_file:raise ValueError('Checkpoint changed after inference')
            del st,previous,x,before,side,deltas
        result=dict(status='STATS_READY' if op=='statistics' else 'COMPLETE',version=VERSION,source_commit=c['source_commit'],job=j['job_name'],dataset=j['name'],control=j['control'],stages=reports,image_exposures=exposures,optimizer_updates=0,policy_updates=0,val_accessed=op=='evaluate',test_accessed=False,independent_calibration=False,peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-begun)
        if op=='evaluate':
            from concm_statistical_metrics import aggregate
            result['metrics']={v:{mode:aggregate([z['raw_heads' if mode=='raw' else 'heads'][v]['metrics']['ON'] for z in reports],d) for mode in ('guarded','raw')} for v in list(CANDIDATES)+list(risk.HEADS)}
            result['control_metrics']=aggregate([z['heads']['CRISK']['metrics']['OFF'] for z in reports],d)
        save(folder/'RESULTS.json',result);state(result['status'],stages=len(reports))
    except BaseException as exc:state('INCOMPLETE',error_type=type(exc).__name__);raise


if __name__=='__main__':run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()),int(os.environ['Q224_JOB']),os.environ['Q224_OPERATION'])
