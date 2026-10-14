"""Frozen boundary-only four-head inference, with no train or test loader."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
import intervention_gain as method
from gain_joint_production import fit_all,expanded,CANDIDATES,HEADS
import gain_joint as joint
from intervention_gain_worker import diagnosis
import core1_competition as competition
from concm_statistical_metrics import stage,aggregate,summary
from lca_pair_worker import original,admit,fingerprint,metadata,JOBS
from lca_switch_diagnostic import read_val
from run_module_cycle import save

CONTRASTS=[(on,before) for on,before in [('JOINT_POINT_SELECTED','LEGACY_SELECTED'),('JOINT_PRED_SELECTED','LEGACY_SELECTED'),('JOINT_PRED_SELECTED','JOINT_POINT_SELECTED')]]

def pack(delta,old,meta):
    base={k:meta[k] for k in ('dataset','control','control_source','order','seen','seed','task','source')}
    side=method.checkpoint(delta,old,base);side.update(version=joint.VERSION,production_metadata=dict(meta));return side

def restore(side,old,meta,shape):
    if side.get('production_metadata')!=meta or side.get('version')!=joint.VERSION or meta['version']!=joint.VERSION or meta['variant'] not in CANDIDATES or meta['optimizer_updates']!=0 or side['delta'].shape!=shape or side['old_count']!=old:raise ValueError('Production provenance differs')
    expected=pack(side['delta'],old,meta)
    if expected['metadata']!=side.get('metadata'):raise ValueError('Native metadata differs')
    return side['delta'].clone()

def validate(c,p):
    joint.validate_design(p['design_parameters'])
    exact=dict(version=joint.VERSION,status='FROZEN_EVALUATION_ONLY',primary_control='local_transport',heads=list(HEADS),cells=['A0','AL','B','P'],gpus=[0,1],seed=74002,optimizer_updates=0,policy_updates=0,training_feedback=False,train_images_allowed=False,test_accessed=False,validation_retuning=False,automatic_retry=False,batch_size=64,workers=0,sidecars=72,source_stage_checkpoints=18,shared_validation_image_exposures=15966,preflight_image_exposures=256,normal_GPU_processes=8,deadline_seconds=14400,main_head='JOINT_PRED_SELECTED',guard='old_max_nonincrease_new_logits_fixed',raw_readouts_reported=True,performance_head_selection=False,production_integrated=True,GPU_launch_authorized=True,medical_data_access_authorized=True,checkpoint_access_authorized=True)
    if any(p.get(k)!=v for k,v in exact.items()) or c['gpus']!=[0,1] or [j['job_name'] for j in c['jobs']]!=JOBS:raise ValueError('Frozen production scope differs')

def barrier(root,p,source,operation):
    s=json.loads((root/'PROGRAM_STATE.json').read_text())
    if s['phase']!=operation.upper() or s.get('variant')!='JOINT' or not s.get('sidecars_sealed_at'):raise ValueError('Val before complete statistics seal')
    for n in JOBS:
        a=json.loads((root/'prepared'/n/'PREPARED.json').read_text())
        if a['status']!='PREPARED' or a['source_commit']!=source or len(a['stages'])!=len(p['data'][n.split('_')[0]]['task_sizes']) or any(set(z['variants'])!=set(CANDIDATES) for z in a['stages']):raise ValueError('All four preparations required')
    if operation=='evaluate':
        if not s.get('preflights_sealed_at',{}).get('JOINT'):raise ValueError('Real preflight seal missing')
        for n in JOBS:
            f=json.loads((root/'JOINT'/'preflight'/n/'RESULTS.json').read_text())
            if f['status']!='PASS' or f['source_commit']!=source or f['version']!=joint.VERSION or f['optimizer_updates'] or f['policy_updates'] or f['test_accessed'] or f['classification_metrics_read'] or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or not f['guard_checked']:raise ValueError('Four real zero-update PASS required')

def score(off,x,delta,old):
    change=x@delta
    if not np.isfinite(change).all() or np.any(change[:,old:]):raise ValueError('Finite old-only correction required')
    on=method.guard(torch.from_numpy(off),torch.from_numpy(change),old).numpy()
    raw=off+change
    if not np.array_equal(on[:,old:],off[:,old:]):raise ValueError('New logits changed')
    return on,raw,change

def run(c,index,operation,variant):
    start=time.monotonic();root=Path(c['root']);p=json.loads(Path(c['protocol']).read_text());validate(c,p)
    if operation not in ('prepare','preflight','evaluate') or operation!='prepare' and variant!='JOINT':raise ValueError('Invalid operation')
    j=c['jobs'][index];d=p['data'][j['name']];orig=original(j,p)
    folder=root/'prepared'/j['job_name'] if operation=='prepare' else root/'JOINT'/('preflight' if operation=='preflight' else 'runs')/j['job_name'];folder.mkdir(parents=True,exist_ok=False)
    def state(status,**kw):save(folder/'STATUS.json',dict(status=status,operation=operation,optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,**kw))
    def budget():
        if time.time()>=c['deadline']:raise TimeoutError('Frozen four-hour deadline')
    def meta(task,v):return dict(metadata(c,j,p,task),variant=v,version=joint.VERSION,control_steps=d['cumulative_steps'][task-1],optimizer_updates=0)
    torch.set_num_threads(2);torch.manual_seed(74002);state('RUNNING',phase='ADMISSION')
    try:
        if operation=='prepare':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_available():raise ValueError('CPU only')
            import module_additions as modules
            stages=[]
            for task in range(1,len(d['task_sizes'])+1):
                budget();state('RUNNING',task=task,phase='NATIVE_HEAD')
                path=Path(orig['output'])/f'stage_{task}.pt';fp=fingerprint(path);st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task)
                bank=st['bank'];pairs=st['boundary_pair_weights'].double();R,_=modules.metric(bank,j['control'],j['name'],st['seen'],st['module_memory']);W,qa=competition.bank_head(bank,pairs,regularizer=R)
                error=float((W-st['head'].double()).norm()/W.norm().clamp_min(1e-30))
                if error>p['saved_head_relative_tolerance'] or not torch.allclose(W.float(),st['head'],atol=p['saved_head_element_atol'],rtol=p['saved_head_element_rtol']):raise ValueError('Saved control head reconstruction failed')
                old=sum(d['task_sizes'][:task-1]);state('RUNNING',task=task,phase='COVARIANCE_GEOMETRY')
                legacy=p['legacy_references'][j['job_name']][task-1]
                fitted,mechanical=fit_all(bank,pairs,R,W,old,st['head'],legacy)
                variants={}
                for v in CANDIDATES:
                    budget();state('RUNNING',task=task,head=v,phase='ANALYTIC_SIDECAR')
                    delta=fitted[v];a=mechanical['candidates'][v];side=pack(delta,old,meta(task,v));sidepath=folder/f'{v}_{task}.pt';torch.save(side,sidepath);back=torch.load(sidepath,map_location='cpu',weights_only=False)
                    if not torch.equal(restore(back,old,meta(task,v),st['head'].shape),delta.float()):raise ValueError('Roundtrip mismatch')
                    h=W+delta;common=h.mean(1,keepdim=True);disc=h-common
                    a['norms']=dict(full=float(h.norm()),discriminative=float(disc.norm()),common=float(common.norm()*len(st['seen'])**.5),delta=float(delta.norm()),reference_full=float(W.norm()),reference_discriminative=float((W-W.mean(1,keepdim=True)).norm()))
                    if old==0 and delta.any():raise ValueError('Stage one not identity')
                    variants[v]=dict(sidecar_bytes=sidepath.stat().st_size,metadata=side['production_metadata'],mechanical=a)
                if fingerprint(path)!=fp:raise ValueError('Original checkpoint changed')
                stages.append(dict(task=task,source_file=fp,native_solve=qa,reconstruction_relative_error=error,variants=variants,selections=mechanical['selections'],credibility=mechanical['credibility'],direction=mechanical['direction']))
                del st,bank,pairs,R,W,fitted,delta,side,back,h,common,disc
            save(folder/'PREPARED.json',dict(status='PREPARED',job=j['job_name'],source_commit=c['source_commit'],stages=stages,optimizer_updates=0,medical_images_read=0,cpu_pipeline_seconds=time.monotonic()-start));state('PREPARED',stages=len(stages));return
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():raise ValueError('GPU01 only')
        barrier(root,p,c['source_commit'],operation)
        from run_multilabel import ApartFeatures
        from run_medical_v2 import transform
        from run_prototype_single import Images,extract
        from torch.utils.data import DataLoader
        torch.backends.cuda.matmul.allow_tf32=False;prepared=json.loads((root/'prepared'/j['job_name']/'PREPARED.json').read_text());rows=read_val(orig,d);reports=[];exposures=0
        tasks=[len(d['task_sizes'])] if operation=='preflight' else range(1,len(d['task_sizes'])+1)
        for task in tasks:
            budget();state('RUNNING',task=task,phase='READ_ONLY_VALIDATION');a=prepared['stages'][task-1];path=Path(orig['output'])/f'stage_{task}.pt'
            if fingerprint(path)!=a['source_file']:raise ValueError('Source changed')
            st=torch.load(path,map_location='cpu',weights_only=False);admit(st,d,j['control'],task);old=sum(d['task_sizes'][:task-1]);deltas={}
            for v in CANDIDATES:
                side=torch.load(root/'prepared'/j['job_name']/f'{v}_{task}.pt',map_location='cpu',weights_only=False);deltas[v]=restore(side,old,meta(task,v),st['head'].shape).numpy()
            deltas=expanded(deltas,a['selections'])
            seen=st['seen'];head=st['head'].numpy();model=st['model'];del st,side
            encoder=ApartFeatures(orig['legacy_repo'],orig['weight'],len(d['order']),'cuda:0',74002);encoder.load_state_dict(model,strict=True);encoder.requires_grad_(False).eval()
            selected=[r for r in rows if r['label'] in seen]
            if operation=='preflight':selected=selected[:64]
            loader=DataLoader(Images(selected,orig['images'],transform(False),74002+task*100003),batch_size=64,num_workers=0,shuffle=False,drop_last=False)
            with torch.no_grad():x,labels=extract(encoder,loader,budget)
            if not all(torch.equal(value.detach().cpu(),model[key]) for key,value in encoder.state_dict().items()):raise ValueError('Encoder changed')
            off=x@head;scores={v:score(off,x,delta,old) for v,delta in deltas.items()}
            if operation=='preflight':
                gx=torch.from_numpy(x).cuda();goff=gx@torch.from_numpy(head).cuda()
                if not np.allclose(goff.cpu().numpy(),off,atol=2e-6,rtol=2e-5):raise ValueError('OFF CPU GPU mismatch')
                for v,delta in deltas.items():
                    gd=gx@torch.from_numpy(delta).cuda();gon=method.guard(goff,gd,old).cpu().numpy()
                    if not np.allclose(gon,scores[v][0],atol=2e-6,rtol=2e-5) or not np.allclose(gd.cpu().numpy(),scores[v][2],atol=2e-6,rtol=2e-5):raise ValueError('CPU GPU guarded score failed')
                result=dict(status='PASS',job=j['job_name'],version=joint.VERSION,source_commit=c['source_commit'],heads=list(HEADS),real_validation_images=len(selected),optimizer_updates=0,policy_updates=0,parameters_unchanged=True,CPU_GPU_scores_close=True,guard_checked=True,classification_metrics_read=False,train_accessed=False,test_accessed=False,peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-start)
                save(folder/'RESULTS.json',result);state('PASS');return
            stage_reports={};raw_reports={};ref=p['control_references'][j['job_name']]['stages'][task-1]
            for v,(on,raw,change) in scores.items():
                checked,unchecked=diagnosis(off,on,raw,x,head,deltas[v],labels,d,task)
                for key in ('seen','per_class_n','per_class_recall','accuracy'):
                    if checked['metrics']['OFF'][key]!=ref[key]:raise ValueError('Published control parity failed')
                if task==1 and not np.array_equal(on,off):raise ValueError('First-stage identity failed')
                checked.update(published_control_exact=True,encoder_state_unchanged=True,mechanical=a['variants'][v]['mechanical'] if v in CANDIDATES else a['selections'][v])
                stage_reports[v]=checked;raw_reports[v]=unchecked
            comparisons={on+'−'+before:stage(scores[before][0],scores[on][0],scores[on][0],labels,d,task) for on,before in CONTRASTS}
            reports.append(dict(task=task,heads=stage_reports,raw_heads=raw_reports,comparisons=comparisons));exposures+=len(selected)
            save(folder/'PARTIAL_STAGE_RESULTS.json',dict(job=j['job_name'],dataset=j['name'],control=j['control'],source_commit=c['source_commit'],version=joint.VERSION,stages=reports,test_accessed=False,optimizer_updates=0))
            if fingerprint(path)!=a['source_file']:raise ValueError('Source changed')
            del encoder,model,x,off,scores,deltas,loader
        result=dict(job=j['job_name'],dataset=j['name'],control=j['control'],version=joint.VERSION,source_commit=c['source_commit'],stages=reports,metrics={v:{mode:aggregate([z[mode+'_heads' if mode=='raw' else 'heads'][v]['metrics']['ON'] for z in reports],d) for mode in ('guarded','raw')} for v in HEADS},control_metrics=aggregate([z['heads']['JOINT_PRED_SELECTED']['metrics']['OFF'] for z in reports],d),optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,validation_image_exposures=exposures,source_checkpoint_bytes=sum(z['source_file']['bytes'] for z in prepared['stages']),sidecar_bytes=sum(v['sidecar_bytes'] for z in prepared['stages'] for v in z['variants'].values()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),worker_wall_seconds=time.monotonic()-start)
        save(folder/'RESULTS.json',result);state('COMPLETE',stages=len(reports))
    except BaseException as exc:state('INCOMPLETE',error_type=type(exc).__name__);raise

if __name__=='__main__':run(json.loads(Path(os.environ['Q128_CONFIG']).read_text()),int(os.environ['Q185_JOB']),os.environ['Q185_OPERATION'],os.environ.get('Q185_VARIANT'))
