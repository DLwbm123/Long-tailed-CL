"""Frozen medical VLM matrix. Paths and worker assignments use environment only."""
from __future__ import annotations
import fcntl, gc, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from ct13_runtime.factories import build_apart, build_clip
from ct13_runtime.medical_vlm import build_medical
from route_a.run_ct13_real import _extract_rows, _load_bank, _save_bank, _stage_metrics, _write_csv, _open_rgb, _stack
from route_a.semantic_prior import text_prompts, unit_columns
from shared.dual_moment_bank import DualMomentBank
from tools.run_nb2_vlm_r1 import (_read, _write, _sha, _manifest, _label, _names, _task_order,
    _tail, _parent_lock, _clip_lock, _guard, _allow, _append_access, SIZES)
from tools.medvlm_math import CORE, EXTRA, fit, score, reliability_details

SEEDS=(1993,1994,1995)
SOURCE_FILES=('tools/run_medvlm.py','tools/medvlm_math.py','tools/finalize_medvlm.py',
    'ct13_runtime/medical_vlm.py','ct13_runtime/factories.py','tools/run_nb2_vlm_r1.py',
    'tools/finalize_nb2_vlm_r1.py','route_a/run_ct13_real.py','route_a/semantic_prior.py',
    'route_a/spectral_prior_ridge.py','shared/dual_moment_bank.py','shared/frozen_dual_features.py',
    'third_party/APART/utils/inc_net.py','third_party/APART/utils/medical_v2.py',
    'third_party/APART/backbone/vision_transformer_adapter_pool_a.py')

def heartbeat(out,phase,**fields):
    _write(out/('HEARTBEAT_'+os.environ.get('MED_WORKER','main')+'.json'),
           {'phase':phase,'time':time.time(),'pid':os.getpid(),**fields})

def events(out,phase,model,ds,seed,task=0,apart=True):
    def record(n):
        _append_access(out/('access_'+os.environ.get('MED_WORKER','main')+'.jsonl'),
            {'phase':phase,'model':model,'dataset':ds,'seed':seed,'task':task,
             'image_reads':n,'apart_batches':int(apart),'vlm_batches':1,'time':time.time()})
    return record

def model_for(tag,cfg):
    return build_clip(lock=_clip_lock(cfg)) if tag=='G' else build_medical(Path(cfg['medical_assets'])/tag)

def text_bundle(model,ds,names):
    domain='isic' if ds=='ISIC' else 'hyperkvasir'
    raw=model['encode_text'](model['tokenizer'](text_prompts(domain,names,0)+text_prompts(domain,names,1))).numpy().astype(np.float64)
    if raw.shape!=(2*len(names),512):raise ValueError('TEXT_SHAPE')
    t0,t1=raw[:len(names)].T,raw[len(names):].T
    return unit_columns((t0+t1)/2),unit_columns(t0),unit_columns(t1)

def old_stage(cfg,ds,seed,task):return Path(cfg['generic_run'])/'stages'/ds/str(seed)/f'task_{task:02d}'
def stage_path(out,tag,ds,seed,task):return out/'stages'/tag/ds/str(seed)/f'task_{task:02d}'

def check_stage(path,source,protocol):
    lock=_read(path/'STATE_LOCK.json')
    if lock['source']!=source or lock['protocol']!=protocol:raise ValueError('STAGE_SOURCE_CHANGED')
    for name,digest in lock['files'].items():
        if _sha(path/name)!=digest:raise ValueError('STAGE_FILE_CHANGED:'+name)
    return lock

def load_weights(path):
    with np.load(path/'CORE.npz',allow_pickle=False) as z:w={k:z[k] for k in z.files}
    if (path/'EXTRA.npz').exists():
        with np.load(path/'EXTRA.npz',allow_pickle=False) as z:w.update({k:z[k] for k in z.files})
    return w

def generic_audit(cfg,out):
    rows=[];summaries=[]
    for ds in SIZES:
        with np.load(Path(cfg['generic_run'])/'private'/f'{ds}_VAL_PREDICTIONS.npz',allow_pickle=False) as z:
            keys=[json.loads(str(x)) for x in z['index']]
            pred={(k['seed'],k['task'],k['method']):v for k,v in zip(keys,z['predictions'])}
            labels=z['labels']
        for seed in SEEDS:
            for task in range(1,len(SIZES[ds])+1):
                p=old_stage(cfg,ds,seed,task);lock=_read(p/'STATE_W_LOCK.json')
                for name in ('BANK.npz','READOUTS.npz'):
                    if _sha(p/name)!=lock['file_sha256'][name]:raise ValueError('GENERIC_INPUT_CHANGED')
                bank=_load_bank(p/'BANK.npz');meta=_read(p/'READOUTS.json')
                with np.load(p/'READOUTS.npz',allow_pickle=False) as z:
                    delta=z['A6']-z['A2'];norm=float(np.linalg.norm(delta));rel=norm/max(float(np.linalg.norm(z['A2'])),1e-30)
                    per=np.linalg.norm(delta,axis=0)
                a,b=pred[seed,task,'A6'],pred[seed,task,'A2'];valid=b>=0
                changed=int(np.sum((a!=b)&valid));corr=int(np.sum((a==labels)&(b!=labels)&valid));dest=int(np.sum((a!=labels)&(b==labels)&valid))
                for j,r in enumerate(reliability_details(bank,np.asarray(meta['text_u']),meta['a_scale'])):
                    if abs(r['gamma']-lock['gamma'][j])>1e-12:raise ValueError('GENERIC_GAMMA_RECOMPUTE')
                    rows.append({'dataset':ds,'seed':seed,'task':task,**r,'class_W_delta_norm':float(per[j])})
                category=('A_GATE_OFF' if not any(lock['gamma']) else 'B_NUMERICALLY_TINY' if rel<1e-8 else 'C_ARGMAX_UNCHANGED' if changed==0 else 'D_PREDICTIONS_CHANGED')
                summaries.append({'dataset':ds,'seed':seed,'task':task,'category':category,
                    'W_delta_norm':norm,'W_delta_relative':rel,'numerically_tiny_threshold':1e-8,
                    'prediction_changes':changed,'corrections':corr,'destructions':dest})
    _write_csv(out/'GENERIC_PRIOR_AUDIT.csv',rows)
    _write(out/'GENERIC_PRIOR_AUDIT.json',{'stages':summaries,'new_image_forwards':0,'gamma_changed':False})

def save_core(out,tag,ds,seed,task,bank,w,source,protocol,bank_source=None):
    p=stage_path(out,tag,ds,seed,task);tmp=p.with_name(p.name+'.part')
    if tmp.exists():raise ValueError('INCOMPLETE_STAGE_REQUIRES_REVIEW:'+str(tmp))
    tmp.mkdir(parents=True)
    files=['CORE.npz']
    np.savez_compressed(tmp/'CORE.npz',**w)
    if bank_source is None:
        _save_bank(bank,tmp/'BANK.npz');files.append('BANK.npz')
    _write(tmp/'STATE_LOCK.json',{'model':tag,'dataset':ds,'seed':seed,'task':task,
        'class_order':bank.class_order,'source':source,'protocol':protocol,
        'bank_source':str(bank_source) if bank_source else None,
        'files':{name:_sha(tmp/name) for name in files},'new_neural_epochs':0,'optimizer_steps':0})
    tmp.rename(p)

def fit_generic(cfg,out,task_lock,names,source,protocol):
    model=model_for('G',cfg)
    for ds in SIZES:
        for seed in SEEDS:
            for task in range(1,len(SIZES[ds])+1):
                p=stage_path(out,'G',ds,seed,task)
                if p.exists():check_stage(p,source,protocol);continue
                old=old_stage(cfg,ds,seed,task);bank=_load_bank(old/'BANK.npz')
                text=text_bundle(model,ds,[names[ds][c] for c in bank.class_order])
                w=fit(bank,*text)
                save_core(out,'G',ds,seed,task,bank,w,source,protocol,old/'BANK.npz')
                heartbeat(out,'fit_G',dataset=ds,seed=seed,task=task)
    if model['state_hash']()!=model['initial_state_hash']:raise ValueError('G_STATE_DRIFT')
    del model;gc.collect();torch.cuda.empty_cache()

def qualify(cfg,out,task_lock,train,names,allowed):
    model=model_for('B',cfg);receipts=[]
    for ds in SIZES:
        for seed in SEEDS:
            order=_task_order(task_lock,ds,seed)
            rows=[r for c in order[:2] for r in [v for v in train[ds] if _label(v)==c][:2]]
            parent,lock=_parent_lock(cfg,task_lock,ds,seed);apart=build_apart(checkpoint=parent,lock=lock)
            root=Path(cfg['datasets'][ds]['image_root']);_allow(allowed,root,rows)
            start=time.time()
            a,u,h=_extract_rows(rows,root,apart,model,batch_size=48,on_batch=events(out,'qualification','B',ds,seed))
            if not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5) or not np.allclose(np.linalg.norm(u,axis=1),1,atol=1e-5):raise ValueError('FEATURE_NORM')
            text_bundle(model,ds,[names[ds][c] for c in order[:2]])
            if apart['state_hash']()!=apart['restore']['network_sha256']:raise ValueError('PARENT_STATE_DRIFT')
            receipts.append({'dataset':ds,'seed':seed,'samples':len(rows),'seconds':time.time()-start,
                'parent_sha256':lock['parent_sha256'],'shape':list(h.shape),'strict_restore':'PASS'})
            del apart;gc.collect();torch.cuda.empty_cache()
    if model['state_hash']()!=model['initial_state_hash']:raise ValueError('B_STATE_DRIFT')
    _write(out/'QUALIFICATION.json',{'status':'PASS','receipts':receipts,
        'preprocess':str(model['preprocess']),'tokenizer_class':type(model['tokenizer']).__name__,
        'context_length':256,'val_access':0})
    del model;gc.collect();torch.cuda.empty_cache()

def fit_medical(cfg,out,task_lock,train,names,allowed,source,protocol,jobs,tag='B'):
    model=model_for(tag,cfg)
    for ds,seed in jobs:
        order=_task_order(task_lock,ds,seed);parent,lock=_parent_lock(cfg,task_lock,ds,seed)
        apart=build_apart(checkpoint=parent,lock=lock);bank=DualMomentBank();position=0
        for task,size in enumerate(SIZES[ds],1):
            current=order[position:position+size];position+=size;p=stage_path(out,tag,ds,seed,task)
            if p.exists():
                check_stage(p,source,protocol);bank=_load_bank(p/'BANK.npz');continue
            for cid in current:
                rows=[r for r in train[ds] if _label(r)==cid]
                root=Path(cfg['datasets'][ds]['image_root']);_allow(allowed,root,rows)
                _,_,h=_extract_rows(rows,root,apart,model,batch_size=48,on_batch=events(out,'fit',tag,ds,seed,task))
                bank.add_class(cid,h,task=task,component_ids=[r['identity_component'] for r in rows]);del h
            if bank.class_order!=order[:position]:raise ValueError('PREFIX_ORDER')
            text=text_bundle(model,ds,[names[ds][c] for c in bank.class_order]);w=fit(bank,*text)
            save_core(out,tag,ds,seed,task,bank,w,source,protocol)
            heartbeat(out,'fit_'+tag,dataset=ds,seed=seed,task=task)
            print('SEALED',tag,ds,seed,task,flush=True)
        if apart['state_hash']()!=apart['restore']['network_sha256']:raise ValueError('PARENT_STATE_DRIFT')
        del apart,bank;gc.collect();torch.cuda.empty_cache()
    if model['state_hash']()!=model['initial_state_hash']:raise ValueError('MEDICAL_STATE_DRIFT')

def extensions(out,source,protocol):
    # Called only after every G/B core prefix is sealed. No image access.
    for p in sorted((out/'stages').glob('*/*/*/task_*')):
        lock=check_stage(p,source,protocol)
        if (p/'EXTRA_LOCK.json').exists():
            if _sha(p/'EXTRA.npz')!=_read(p/'EXTRA_LOCK.json')['sha256']:raise ValueError('EXTRA_CHANGED')
            continue
        bank=_load_bank(Path(lock['bank_source']) if lock['bank_source'] else p/'BANK.npz')
        w=load_weights(p);ex=fit(bank,w['text'],w['text0'],w['text1'],extensions=True)
        with (p/'EXTRA.npz.part').open('wb') as f:np.savez_compressed(f,**ex)
        (p/'EXTRA.npz.part').rename(p/'EXTRA.npz')
        _write(p/'EXTRA_LOCK.json',{'sha256':_sha(p/'EXTRA.npz'),'source':source,'protocol':protocol})
        heartbeat(out,'extensions',model=lock['model'],dataset=lock['dataset'],seed=lock['seed'],task=lock['task'])

def vlm_only(model,rows,root,callback):
    parts=[]
    for i in range(0,len(rows),48):
        sub=rows[i:i+48];images=[_open_rgb(root/r['relative_path']) for r in sub]
        v=model['forward'](_stack([model['preprocess'](im) for im in images])).numpy().astype(np.float64)
        v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12);parts.append(v);callback(len(sub))
    return np.concatenate(parts)

def evaluate(cfg,out,task_lock,allowed,source,protocol):
    if not (out/'FIT_COMPLETE.json').exists():raise ValueError('PREMATURE_VAL')
    private=out/'private';private.mkdir(exist_ok=True);(private/'scores').mkdir(exist_ok=True)
    stage_rows=[];class_rows=[];bundles={};parity=[]
    for tag in cfg['models']:
        model=model_for(tag,cfg)
        for ds in (('ISIC',) if tag=='D' else SIZES):
            root=Path(cfg['datasets'][ds]['image_root'])
            if _sha(Path(cfg['datasets'][ds]['val_manifest']))!=task_lock[ds]['manifest_sha256']['val']:raise ValueError('VAL_MANIFEST_CHANGED')
            val=_manifest(Path(cfg['datasets'][ds]['val_manifest']),'val',root)
            labels=np.asarray([_label(r) for r in val]);_allow(allowed,root,val)
            bundles.setdefault(ds,{'labels':labels,'components':np.asarray([r['identity_component'] for r in val]),'pred':[],'index':[]})
            u_cached=None
            for seed in SEEDS:
                if tag=='G':
                    parent,lock=_parent_lock(cfg,task_lock,ds,seed);apart=build_apart(checkpoint=parent,lock=lock)
                    a,u,_,p0=_extract_rows(val,root,apart,model,batch_size=48,with_p=True,on_batch=events(out,'val','G',ds,seed))
                    np.savez_compressed(private/f'APART_{ds}_{seed}.npz',a=a,p=p0)
                    if apart['state_hash']()!=apart['restore']['network_sha256']:raise ValueError('VAL_APART_DRIFT')
                    del apart;gc.collect();torch.cuda.empty_cache()
                else:
                    if u_cached is None:u_cached=vlm_only(model,val,root,events(out,'val',tag,ds,seed,apart=False))
                    u=u_cached
                    with np.load(private/f'APART_{ds}_{seed}.npz',allow_pickle=False) as z:a=z['a'];p0=z['p']
                order=_task_order(task_lock,ds,seed);position=0
                for task,size in enumerate(SIZES[ds],1):
                    position+=size;seen=order[:position];current=seen[-size:]
                    path=stage_path(out,tag,ds,seed,task);check_stage(path,source,protocol)
                    if _sha(path/'EXTRA.npz')!=_read(path/'EXTRA_LOCK.json')['sha256']:raise ValueError('EXTRA_CHANGED')
                    w=load_weights(path);stored={}
                    methods=list(CORE+EXTRA)+(['P','F','F2'] if tag=='G' else [])
                    for method in methods:
                        full=method if method in ('P','F','F2') else tag+'.'+method
                        scores=p0@np.load(old_stage(cfg,ds,seed,task)/'P_W.npy',allow_pickle=False) if method=='P' else score(w,method,a,u)
                        metric,cr,pred=_stage_metrics(scores,labels,seen,current,_tail(task_lock,cfg,ds))
                        valid=pred>=0;old=np.isin(labels,seen[:-size]);new=np.isin(labels,current)
                        metric['old_to_current']=int(np.sum(valid&old&np.isin(pred,current)))
                        metric['current_to_old']=int(np.sum(valid&new&np.isin(pred,seen[:-size])))
                        common={'dataset':ds,'seed':seed,'task':task,'method':full}
                        stage_rows.append({**common,'seen_classes':len(seen),**metric});class_rows.extend([{**common,**r} for r in cr])
                        bundles[ds]['pred'].append(pred.astype(np.int16));bundles[ds]['index'].append(common);stored[method]=scores
                    np.savez_compressed(private/'scores'/f'{tag}_{ds}_{seed}_{task}.npz',**stored)
                    heartbeat(out,'evaluate',model=tag,dataset=ds,seed=seed,task=task)
        if model['state_hash']()!=model['initial_state_hash']:raise ValueError('VAL_VLM_DRIFT')
        del model;gc.collect();torch.cuda.empty_cache()
    for ds,b in bundles.items():
        np.savez_compressed(private/f'{ds}_PREDICTIONS.npz',labels=b['labels'],components=b['components'],predictions=np.stack(b['pred']),index=np.asarray([json.dumps(x,sort_keys=True) for x in b['index']]))
    _write_csv(out/'stage_metrics.csv',stage_rows);_write_csv(out/'class_metrics.csv',class_rows)
    _write(out/'PREDICTIONS_LOCK.json',{'status':'LOCKED','files':{ds:_sha(private/f'{ds}_PREDICTIONS.npz') for ds in bundles}})

def setup(cfg,out):
    root=Path(__file__).resolve().parents[1]
    protocol=_read(root/'docs/medvlm_protocol.json')
    protocol['models_executed']=cfg['models']
    source={'code_commit':cfg['code_commit'],'code_sha256':{p:_sha(root/p) for p in SOURCE_FILES},
            'task_lock_sha256':_sha(Path(cfg['task_lock'])),'generic_run_id':Path(cfg['generic_run']).name,
            'model_asset_locks':{tag:_read(Path(cfg['medical_assets'])/tag/'ASSET_LOCK.json') for tag in cfg['models'] if tag!='G'}}
    for name,value in [('protocol_lock.json',protocol),('source_lock.json',source)]:
        p=out/name
        if p.exists() and _read(p)!=value:raise ValueError('LOCK_ALREADY_EXISTS_WITH_DIFFERENT_CONTENT:'+name)
        if not p.exists():_write(p,value)
    return _sha(out/'source_lock.json'),_sha(out/'protocol_lock.json')

def main():
    cfg=_read(Path(os.environ['MED_CONFIG']));out=Path(cfg['run_root']);out.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Fail closed even if an upstream library ignores its offline flags.
    sys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('RUNTIME_NETWORK_DISABLED')) if event=='socket.connect' else None)
    allowed=_guard();task_lock=_read(Path(cfg['task_lock']));train={};names={}
    for ds in SIZES:
        if _sha(Path(cfg['datasets'][ds]['train_manifest']))!=task_lock[ds]['manifest_sha256']['train']:raise ValueError('TRAIN_MANIFEST_CHANGED')
        train[ds]=_manifest(Path(cfg['datasets'][ds]['train_manifest']),'train',Path(cfg['datasets'][ds]['image_root']))
        names[ds]=_names(ds,cfg,train[ds])
    source,protocol=setup(cfg,out)
    if os.environ.get('MED_ROLE')=='fit':
        fit_medical(cfg,out,task_lock,train,names,allowed,source,protocol,json.loads(os.environ['MED_JOBS']))
        return
    lock=(out/'.process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.time();heartbeat(out,'started')
    if not (out/'GENERIC_PRIOR_AUDIT.json').exists():generic_audit(cfg,out)
    if not (out/'QUALIFICATION.json').exists():qualify(cfg,out,task_lock,train,names,allowed)
    if cfg.get('phase')=='qualify':return
    jobs=[(ds,seed) for ds in SIZES for seed in SEEDS];children=[]
    for i,gpu in enumerate(cfg['gpus']):
        env=os.environ.copy();env.update(MED_ROLE='fit',MED_WORKER=str(i),MED_JOBS=json.dumps(jobs[i::len(cfg['gpus'])]),CUDA_VISIBLE_DEVICES=str(gpu))
        log=(out/f'worker_{i}.log').open('ab')
        children.append(subprocess.Popen([sys.executable,os.environ['MED_ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT));log.close()
    fit_generic(cfg,out,task_lock,names,source,protocol)
    codes=[p.wait() for p in children]
    if any(codes):raise RuntimeError('FIT_WORKER_FAILURE:'+str(codes))
    if 'D' in cfg['models']:fit_medical(cfg,out,task_lock,train,names,allowed,source,protocol,[('ISIC',s) for s in SEEDS],tag='D')
    extensions(out,source,protocol)
    expected=90+(12 if 'D' in cfg['models'] else 0)
    stages=list((out/'stages').glob('*/*/*/task_*/STATE_LOCK.json'))
    if len(stages)!=expected:raise ValueError('FIT_COVERAGE')
    _write(out/'FIT_COMPLETE.json',{'status':'ALL_MODELS_LOCKED','stages':len(stages),'val_access':0})
    evaluate(cfg,out,task_lock,allowed,source,protocol)
    from tools.finalize_medvlm import finalize
    report=finalize(out,cfg)
    _write(out/'resource_report.json',{'wall_seconds':time.time()-start,'gpu_indices':cfg['gpus'],
        'new_neural_epochs':0,'optimizer_steps':0,'private_val_cache_reused':True,'generic_fit_image_forwards':0})
    _write(out/'completion_receipts.json',{'status':'PRIMARY_MATRIX_COMPLETE',
        'stage_rows':report['stage_rows'],'class_rows':report['class_rows'],
        'fit_stages':expected,'dermlip_status':report['dermlip_status'],
        'source':source,'protocol':protocol,'new_neural_epochs':0,'optimizer_steps':0})
    heartbeat(out,'COMPLETE')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if os.environ.get('MED_CONFIG'):
            cfg=_read(Path(os.environ['MED_CONFIG']))
            _write(Path(cfg['run_root'])/('FAILED_'+os.environ.get('MED_WORKER','main')+'.json'),
                {'error':repr(e),'traceback':traceback.format_exc(),'time':time.time()})
        raise
