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

VERSION = 'gain_training_paired_diagnostic_v1'
SOURCE = '533562177c4fa39f18dcbe981e2c0d0f596a09ac'


def validate(c,p):
    exact=dict(version=VERSION,status='FROZEN_TRAIN_DIAGNOSTIC',primary_control='local_transport',
               gpus=[0,1],seed=74002,optimizer_updates=0,policy_updates=0,
               train_images_allowed=True,old_train_inference_only=True,old_image_training_replay=False,
               val_accessed=False,test_accessed=False,validation_retuning=False,automatic_retry=False,
               direction_changed=False,steps=[.125,.25,.5,1.],heads=list(CANDIDATES),
               batch_size=64,workers=0,checkpoint_stages=18,sidecars_reused=72,
               previous_sidecar_source=SOURCE,independent_calibration=False,
               classification_inference_only=True,selection_enabled=False)
    if any(p.get(k)!=v for k,v in exact.items()) or c['gpus']!=[0,1] or [j['job_name'] for j in c['jobs']]!=JOBS:
        raise ValueError('Frozen training-side inference scope required')


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
    state=json.loads((root/'PROGRAM_STATE.json').read_text())
    if state['phase']!=op.upper() or not state.get('sidecars_sealed_at'):
        raise ValueError('Train images before four metadata seals')
    for name in JOBS:
        a=json.loads((root/'prepared'/name/'PREPARED.json').read_text())
        if a['status']!='PREPARED' or a['source_commit']!=c['source_commit'] or len(a['stages'])!=len(p['data'][name.split('_')[0]]['task_sizes']):
            raise ValueError('Missing preparation')
    if op=='diagnose':
        if not state.get('preflights_sealed_at'):raise ValueError('Real preflight seal missing')
        for name in JOBS:
            f=json.loads((root/'preflight'/name/'RESULTS.json').read_text())
            if f['status']!='PASS' or f['source_commit']!=c['source_commit'] or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or f['optimizer_updates'] or f['policy_updates'] or f['val_accessed'] or f['test_accessed'] or f['classification_metrics_read']:
                raise ValueError('Four genuine zero-update checks required')


def estimates(mechanical,paired,seen):
    rows=[]
    for kind in ('point','predictive'):
        for z in mechanical[kind]['scenarios']:
            predicted={v['label']:v for v in z['classes']}
            for i,label in enumerate(seen):
                a=paired[str(label)];b=predicted.get(i);identity=mechanical[kind]['exact_identity']
                r=0. if identity else b['known_rescue'];h=0. if identity else b['known_harm'];unknown=0. if identity else b['unknown_mass']
                rows.append(dict(label=label,estimator=kind,scenario=z['scenario'],n=a['n'],
                                 predicted_known_rescue=r,predicted_known_harm=h,unknown_mass=unknown,
                                 actual_rescue=a['rescued']/a['n'],actual_harm=a['harmed']/a['n'],
                                 known_rescue_minus_actual=r-a['rescued']/a['n'],
                                 known_harm_minus_actual=h-a['harmed']/a['n'],
                                 partial_comparison=bool(unknown),independent_calibration=False))
    return rows


def fold_counts(off,on,labels,fold_ids,seen):
    y=np.array([seen.index(int(v)) for v in labels]);po=off.argmax(1);pn=on.argmax(1);rows=[]
    for f in (0,1):
        for i,label in enumerate(seen):
            m=(labels==label)&(fold_ids==f);n=int(m.sum())
            rows.append(dict(fold=f,label=label,n=n,rescued=int((m&(po!=i)&(pn==i)).sum()),
                             harmed=int((m&(po==i)&(pn!=i)).sum()),missing=n==0,
                             rescue_rate=None if not n else float(((po!=i)&(pn==i))[m].mean()),
                             harm_rate=None if not n else float(((po==i)&(pn!=i))[m].mean()),
                             classifier_independent=False))
    return rows


def report_stage(x,before,labels,fold_ids,head,deltas,bank,previous_bank,mechanical,d,task):
    seen=d['order'][:sum(d['task_sizes'][:task])];old=sum(d['task_sizes'][:task-1]);data=dict(d,val_counts=d['train_counts'])
    if x.shape!=(len(labels),1536) or head.shape!=(1536,len(seen)) or set(deltas)!=set(CANDIDATES):raise ValueError('Frozen feature/readout dimensions differ')
    off=x@head;heads={};drift=[];boundaries=[];private={};shift=None
    if old:
        if before.shape!=x.shape:raise ValueError('Paired canonical rows differ')
        shift=np.stack([(x-before)[labels==label].astype(float).mean(0) for label in seen[old:]]).mean(0)
    for v,delta in deltas.items():
        on,raw,_=score(off,x,delta,old);a,b=diagnosis(off,on,raw,x,head,delta,labels,data,task)
        for report in (a,b):
            for m in report['metrics'].values():m['training_n']=m.pop('validation_n')
        a.update(estimates=estimates(mechanical[v],a['paired'],seen),fold_counts=fold_counts(off,on,labels,fold_ids,seen),independent_calibration=False)
        heads[v]=dict(guarded=a,raw=b)
    for i,label in enumerate(seen):
        m=labels==label;mean=x[m].astype(float).mean(0);proxy=bank['mu'][i].numpy();P=np.concatenate((head,deltas['C1']),1).astype(float)
        G=paired_score_moments(before[m],x[m],P,shift) if old and i<old else None
        if G is not None:private[str(label)]=G
        S=bank['Q'][i].numpy()-np.outer(proxy,proxy);projected=P.T@S@P
        actual_scores=x[m].astype(float)@P;actual_cov=np.cov(actual_scores,rowvar=False,bias=True) if m.sum()>1 else np.zeros((2*len(seen),)*2)
        entry=dict(label=label,n=int(m.sum()),old_class=i<old,transport_age=task-next(t for t in range(1,task+1) if label in d['order'][:sum(d['task_sizes'][:t])]),
                   mean_residual_norm=float(np.linalg.norm(mean-proxy)),score_mean_residual_norm=float(np.linalg.norm((mean-proxy)@P)),
                   projected_model_scatter_trace=float(np.trace(projected)),projected_actual_scatter_trace=float(np.trace(actual_cov)),
                   projected_covariance_error_norm=float(np.linalg.norm(actual_cov-projected)),
                   paired_residual_score_mean_norm=None if G is None else float(np.linalg.norm(G['mean'])),
                   paired_residual_second_trace=None if G is None else float(np.trace(G['second'])),
                   paired_residual_second_norm=None if G is None else float(np.linalg.norm(G['second'])),
                   common_shift_norm=None if shift is None else float(np.linalg.norm(shift)),
                   paired_source='adjacent_frozen_encoders_same_train_images' if G is not None else None,
                   old_class_distribution_directly_observed=i<old,independent_calibration=False)
        if i<old:
            transported=previous_bank['mu'][i].numpy()+shift
            entry['reconstructed_transport_error_norm']=float(np.linalg.norm(proxy-transported))
        drift.append(entry)
        for j,competitor in enumerate(seen):
            if i==j:continue
            gap=x[m].astype(float)@(head[:,i]-head[:,j]).astype(float);pgap=float(proxy@(head[:,i]-head[:,j]))
            boundaries.append(dict(label=label,competitor=competitor,n=int(m.sum()),proxy_gap=pgap,actual_gap_mean=float(gap.mean()),
                                   proxy_gap_error=pgap-float(gap.mean()),actual_negative_gap_count=int((gap<0).sum())))
    return dict(task=task,heads=heads,drift=drift,boundaries=boundaries,training_image_exposures=len(labels),
                previous_encoder_image_exposures=0 if not old else len(labels),independent_calibration=False),private


def run(c,index,op):
    begun=time.monotonic();root=Path(c['root']);p=json.loads(Path(c['protocol']).read_text());validate(c,p)
    if op not in ('prepare','preflight','diagnose'):raise ValueError('No training/evaluation tuning entry')
    j=c['jobs'][index];d=p['data'][j['name']];orig=original(j,p)
    folder=root/('prepared' if op=='prepare' else 'preflight' if op=='preflight' else 'runs')/j['job_name'];folder.mkdir(parents=True,exist_ok=False)
    def state(status,**kw):save(folder/'STATUS.json',dict(status=status,operation=op,optimizer_updates=0,policy_updates=0,train_inference_only=True,val_accessed=False,test_accessed=False,**kw))
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
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():raise ValueError('GPU01 only')
        barrier(root,c,p,op);rows=read_train(orig,d);foldmap=diagnostic_folds([r['identity_component'] for r in rows],74002)
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
            budget();state('RUNNING',task=task,phase='CURRENT_TRAIN_INFERENCE')
            a=prepared['stages'][task-1];path=Path(orig['output'])/f'stage_{task}.pt'
            if fingerprint(path)!=a['source_file']:raise ValueError('Frozen checkpoint changed')
            st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task);seen=st['seen'];old=sum(d['task_sizes'][:task-1]);selected=[r for r in rows if r['label'] in seen]
            if op=='preflight':selected=[r for r in selected if r['label'] in seen[:old]][:32]+[r for r in selected if r['label'] in seen[old:]][:32]
            x,labels=features(st,selected,task);exposures+=len(selected);head=st['head'].numpy();deltas={}
            for v in CANDIDATES:
                sidepath=prior_root/'prepared'/j['job_name']/f'{v}_{task}.pt'
                if fingerprint(sidepath)!=a['variants'][v]['sidecar_file']:raise ValueError('Sidecar changed')
                side=torch.load(sidepath,map_location='cpu',weights_only=False);deltas[v]=restore(side,old,a['variants'][v]['metadata'],head.shape).numpy()
            before=previous=None
            if old:
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
            folds=np.array([foldmap[r['identity_component']] for r in selected]);mechanical={v:a['variants'][v]['mechanical'] for v in CANDIDATES}
            state('RUNNING',task=task,phase='AGGREGATE_DIAGNOSTICS')
            report,private=report_stage(x,before,labels,folds,head,deltas,st['bank'],None if previous is None else previous['bank'],mechanical,d,task)
            torch.save(private,folder/f'DRIFT_{task}.private.pt');reports.append(report)
            save(folder/'PARTIAL_RESULTS.json',dict(version=VERSION,source_commit=c['source_commit'],job=j['job_name'],stages=reports,image_exposures=exposures,independent_calibration=False))
            if fingerprint(path)!=a['source_file']:raise ValueError('Checkpoint changed after inference')
            del st,previous,x,before,side,deltas,private
        save(folder/'RESULTS.json',dict(status='COMPLETE',version=VERSION,source_commit=c['source_commit'],job=j['job_name'],dataset=j['name'],control=j['control'],stages=reports,image_exposures=exposures,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False,independent_calibration=False,peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-begun));state('COMPLETE',stages=len(reports))
    except BaseException as exc:state('INCOMPLETE',error_type=type(exc).__name__);raise


if __name__=='__main__':run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()),int(os.environ['Q220_JOB']),os.environ['Q220_OPERATION'])
