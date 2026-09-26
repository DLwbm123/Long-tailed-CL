"""One GPU, sequential logical streams, atomic stages and isolated evaluation."""
from __future__ import annotations
import csv,fcntl,gc,hashlib,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from tools.run_nb2_vlm_r1 import _read,_write,_sha,_manifest,_label,_names,_task_order,_tail,_parent_lock,_allow,_append_access,SIZES
from tools.run_medvlm import model_for,text_bundle,SOURCE_FILES as OLD_SOURCE
from ct13_runtime.factories import build_apart
from route_a.run_ct13_real import _open_rgb,_stack,_stage_metrics,_write_csv
from shared.frozen_dual_features import _unwrap_pre_logits,normalize_rows
from tools.rfvila_reference import projections,group_folds,cse,predict
from tools.rfvila_math import tensor,transform,solve,select_cv,append
SEEDS=(1993,1994,1995)
ROOT=Path(__file__).resolve().parents[1]
FILES=tuple(dict.fromkeys(OLD_SOURCE+('tools/run_rfvila.py','tools/rfvila_math.py','tools/rfvila_reference.py','tools/finalize_rfvila.py','tools/rfvila_control.py','tools/rfvila_delivery.py','tools/rfvila_io.py','tools/qualify_rfvila.py','tests/test_rfvila_production.py','tests/test_rfvila_reference.py','tests/test_rfvila_report.py','docs/rfvila_plan/03_PROTOCOL_PROPOSAL.json')))

class Budget:
    def __init__(self,out):self.lock=_read(out/'CLOCK_LOCK.json')
    def elapsed(self):return max(time.time()-self.lock['t0_epoch'],time.monotonic()-self.lock['t0_monotonic'])
    def check(self,kind,estimate=0):
        limit={'tier':7200,'fit':30600,'forward':37800,'compute':39600,'finish':43200}[kind]
        if self.elapsed()+1.5*estimate>=limit:raise TimeoutError('INCOMPLETE_BUDGET:'+kind)

def guard(cfg,role):
    allowed=set();out=Path(cfg['run_root']).resolve();images=[Path(d['image_root']).resolve() for d in cfg['datasets'].values()]
    def audit(event,args):
        if event=='socket.connect':raise PermissionError('OFFLINE_MODEL_PROCESS')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();s=str(p).lower()
        mode=args[1] if len(args)>1 else 'r';reading=not isinstance(mode,str) or not any(x in mode for x in ['w','a','x'])
        if not reading:return
        if p.name.lower().startswith(('test.','reserved.')) or '/reserved/' in s or '/test/' in s:raise PermissionError('TEST_RESERVED_DENIED')
        if role in ('fit','qualify'):
            if p.name=='val.csv' or 'predictions' in p.name.lower() or '/scores/' in s or '/private/val' in s:raise PermissionError('FIT_VAL_DENIED')
            if '/runs/' in s and not p.is_relative_to(out):raise PermissionError('HISTORICAL_RUN_DENIED_TO_FIT')
        if any(p.is_relative_to(root) for root in images) and str(p) not in allowed:raise PermissionError('OLD_FUTURE_IMAGE_DENIED')
    sys.addaudithook(audit);return allowed

def event(out,kind,**kw):
    _append_access(out/'ACCESS_EVENTS.jsonl',{'time':time.time(),'kind':kind,**kw})

def check_models(apart,models):
    if apart and apart['state_hash']()!=apart['restore']['network_sha256']:raise ValueError('PARENT_DRIFT')
    for m in models.values():
        if m['state_hash']()!=m['initial_state_hash']:raise ValueError('VLM_DRIFT')

def extract(rows,root,apart,models,out,budget,phase,ds,seed,task):
    result={k:[] for k in (['a'] if apart else [])+list(models)}
    for start in range(0,len(rows),48):
        budget.check('forward',15);batch=rows[start:start+48];images=[_open_rgb(root/r['relative_path']) for r in batch]
        if apart:
            v=apart['forward'](_stack([apart['preprocess'](im) for im in images]))
            result['a'].append(normalize_rows(_unwrap_pre_logits(v,apart=True)).numpy().astype(np.float64))
        for tag,m in models.items():
            v=m['forward'](_stack([m['preprocess'](im) for im in images]))
            result[tag].append(normalize_rows(v).numpy().astype(np.float64))
        event(out,'image_forward',phase=phase,dataset=ds,seed=seed,task=task,n=len(batch),apart=int(apart is not None),vlm=list(models))
    result={k:np.concatenate(v) for k,v in result.items()}
    for k,v in result.items():
        if v.shape!=(len(rows),1536 if k=='a' else 512) or not np.isfinite(v).all():raise ValueError('FEATURE_SHAPE')
    return result

def backup(out,token,paths,budget):
    ack=out/'backup_acks'/f'{token}.json'
    if ack.exists():
        if _read(ack)['status']!='VERIFIED':raise ValueError('BACKUP_ACK_BAD')
        return
    _write(out/'BACKUP_REQUEST.json',{'token':token,'paths':paths,'time':time.time()})
    while not ack.exists():
        budget.check('finish',60)
        if (out/'BACKUP_FAILURE.json').exists():raise RuntimeError('BLOCKED_BACKUP')
        time.sleep(2)
    if _read(ack)['status']!='VERIFIED':raise ValueError('BACKUP_ACK_BAD')

def methods(protocol):return [m for m in protocol['methods'] if m['tier'] in protocol['admitted_tiers']]
def bases(protocol):return [m for m in methods(protocol) if m['lambda_policy']=='task1_group_cv']
def key(m):return m['method']
def mapped(m,raw,maps,device):
    a=tensor(raw['a'],device);u=None if m['branch']=='F' else tensor(raw[m['branch']],device)
    ra,ru=maps[m['rp_seed']] if m['rp_seed'] else (None,None)
    return transform(a,u,ra,ru,m['feature'])

def load_maps(out,device):
    lock=_read(out/'RANDOM_MAP_LOCK.json');result={}
    for seed in (67101,67102):
        arr=[]
        for branch in ('a','u'):
            name=f'R_{seed}_{branch}.npy';p=out/'private/maps'/name
            if _sha(p)!=lock['files'][name]:raise ValueError('RANDOM_MAP_CHANGED')
            arr.append(tensor(np.load(p,allow_pickle=False),device))
        result[seed]=arr
    return result

def fingerprints(out):return {'source':_sha(out/'SOURCE_LOCK.json'),'protocol':_sha(out/'PROTOCOL_LOCK.json'),'random_map':_sha(out/'RANDOM_MAP_LOCK.json')}
def check_lock(out,p):
    d=_read(p/'STATE_LOCK.json')
    if d['locks']!=fingerprints(out):raise ValueError('STATE_LOCK_DRIFT')
    if _sha(p/'W.npz')!=d['W_sha256']:raise ValueError('W_DRIFT')
    return d

def fit(cfg,out,allowed,budget):
    protocol=_read(out/'PROTOCOL_LOCK.json');base=bases(protocol);device=protocol['solver']['backend'];maps=load_maps(out,device)
    tasklock=_read(Path(cfg['task_lock']));models={tag:model_for(tag,cfg) for tag in ('G','B')}
    for ds in SIZES:
        data=cfg['datasets'][ds];root=Path(data['image_root'])
        if _sha(Path(data['train_manifest']))!=tasklock[ds]['manifest_sha256']['train']:raise ValueError('TRAIN_CHANGED')
        train=_manifest(Path(data['train_manifest']),'train',root);names=_names(ds,cfg,train)
        for seed in SEEDS:
            order=_task_order(tasklock,ds,seed);parent,pl=_parent_lock(cfg,tasklock,ds,seed);apart=build_apart(checkpoint=parent,lock=pl)
            banks={};lambdas={};cv={};position=0;previous=None
            for task,size in enumerate(SIZES[ds],1):
                current=order[position:position+size];position+=size
                sp=out/'stages'/ds/str(seed)/f'task_{task:02d}';state=out/'states'/ds/str(seed)/f'task_{task:02d}.npz'
                token=f'{ds}_{seed}_{task:02d}'
                if sp.exists():
                    d=check_lock(out,sp)
                    backup(out,token,[str(sp.relative_to(out)),str(state.relative_to(out))],budget)
                    previous=(sp,state)
                    # Earlier states may be discarded after their successor was backed up.
                    if task<len(SIZES[ds]) and (sp.parent/f'task_{task+1:02d}').exists():continue
                    if _sha(state)!=d['state_sha256']:raise ValueError('RESUME_STATE_DRIFT')
                    with np.load(state,allow_pickle=False) as z:
                        for m in base:
                            k=key(m);banks[k]=[tensor(z[k+'_S'],device),tensor(z[k+'_Q'],device),d['seen'].copy(),d['counts'].copy()]
                    lambdas=d['lambdas'];cv=d['cv'];continue
                budget.check('fit',cfg['stage_estimates'][f'{ds}_{seed}_{task}'])
                rawparts=[];taskrows=[]
                for cid in current:
                    rows=[r for r in train if _label(r)==cid];_allow(allowed,root,rows)
                    rawparts.append(extract(rows,root,apart,models,out,budget,'fit',ds,seed,task));taskrows+=rows
                raw={k:np.concatenate([v[k] for v in rawparts]) for k in rawparts[0]};del rawparts
                y=np.array([_label(r) for r in taskrows]);comp=np.array([r['identity_component'] for r in taskrows]);fold=group_folds(y,comp) if task==1 else None
                weights={};residuals={};traces={};arrays={}
                for m in base:
                    budget.check('fit',cfg['solve_estimate_seconds']);k=key(m);x=mapped(m,raw,maps,device)
                    if task==1:
                        lam,detail=select_cv(x,y,comp,current,fold=fold);lambdas[k]=lam;cv[k]=detail
                        banks[k]=[torch.zeros((x.shape[1],x.shape[1]),dtype=torch.float64,device=device),torch.empty((x.shape[1],0),dtype=torch.float64,device=device),[],[]]
                        event(out,'cv',dataset=ds,seed=seed,method=k,**detail)
                    S,Q,seen,counts=banks[k]
                    for cid in current:S,Q=append(S,Q,seen,counts,cid,x[tensor(y==cid,device).bool()])
                    if seen!=order[:position]:raise ValueError('CLASS_PREFIX')
                    banks[k]=[S,Q,seen,counts]
                    w,res=solve(S,Q,lambdas[k]);weights[k]=w.cpu().numpy();residuals[k]=res;traces[k]=float(S.trace())
                    arrays[k+'_S']=S.cpu().numpy();arrays[k+'_Q']=Q.cpu().numpy()
                    # Historical fixed-lambda references use these exact current-stream linear statistics.
                    if m['feature']=='linear':
                        for lm in methods(protocol):
                            if lm['branch']==m['branch'] and lm['lambda_policy']=='fixed':
                                if lm['fixed_lambda']==lambdas[k]:lw,lr=w,res
                                else:lw,lr=solve(S,Q,lm['fixed_lambda'])
                                weights[key(lm)]=lw.cpu().numpy();residuals[key(lm)]=lr
                    del x
                del raw,taskrows,y,comp
                for tag,model in models.items():weights['text_'+tag]=text_bundle(model,ds,[names[c] for c in order[:position]])[0]
                check_models(apart,models)
                tmp=sp.with_name(sp.name+'.part');tmp.mkdir(parents=True,exist_ok=False);state.parent.mkdir(parents=True,exist_ok=True)
                with state.with_suffix('.part').open('wb') as f:np.savez(f,**arrays)
                state.with_suffix('.part').replace(state)
                np.savez(tmp/'W.npz',**weights)
                d={'dataset':ds,'seed':seed,'task':task,'seen':order[:position],'current':current,'counts':banks[key(base[0])][3],
                   'locks':fingerprints(out),'state_sha256':_sha(state),'W_sha256':_sha(tmp/'W.npz'),'lambdas':lambdas,'cv':cv,
                   'residuals':residuals,'traces':traces,'parent_sha256':pl['parent_sha256'],'stage_time':time.time(),'randomness':'fixed saved maps; no stochastic operation after map generation','neural_epochs':0,'optimizer_steps':0}
                _write(tmp/'STATE_LOCK.json',d);tmp.rename(sp)
                event(out,'stage_sealed',dataset=ds,seed=seed,task=task,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                backup(out,token,[str(sp.relative_to(out)),str(state.relative_to(out))],budget)
                # Keep latest and one rollback state; only this run's now-backed-up temporary state is pruned.
                stale=state.with_name(f'task_{task-2:02d}.npz')
                if task>2 and stale.exists():stale.unlink()
                previous=(sp,state);del arrays,weights
            del banks,apart;gc.collect();torch.cuda.empty_cache()
    states=list((out/'stages').glob('*/*/task_*/STATE_LOCK.json'))
    if len(states)!=45:raise ValueError('FIT_COVERAGE')
    _write(out/'FIT_COMPLETE.json',{'status':'ALL_ADMITTED_STATES_LOCKED','stages':45,'methods':len(methods(protocol)),'val_access':0,'locks':fingerprints(out)})

def evaluate(cfg,out,allowed,budget):
    complete=_read(out/'FIT_COMPLETE.json')
    if complete['locks']!=fingerprints(out):raise ValueError('FIT_LOCK_CHANGED')
    protocol=_read(out/'PROTOCOL_LOCK.json');base=bases(protocol);maps=load_maps(out,protocol['solver']['backend']);tasklock=_read(Path(cfg['task_lock']))
    models={tag:model_for(tag,cfg) for tag in ('G','B')};stage_rows=[];class_rows=[];coverage=[];parity=[]
    private=out/'private';(private/'scores').mkdir(exist_ok=True)
    # Old APART cache has no standalone identity header: require the original source, manifest, row ordering, dtype, and run inventory.
    oldroot=Path(cfg['medical_run']);inv=_read(oldroot/'private/RUN_SHA256_INVENTORY.json')
    inventory={r['path']:r['sha256'] for r in inv['files']}
    oldsource=_read(oldroot/'source_lock.json')
    for name in ('ct13_runtime/factories.py','route_a/run_ct13_real.py','shared/frozen_dual_features.py','tools/run_medvlm.py','tools/run_nb2_vlm_r1.py'):
        if _sha(ROOT/name)!=oldsource['code_sha256'][name]:raise ValueError('CACHE_TRANSFORM_SOURCE_CHANGED')
    for ds in SIZES:
        data=cfg['datasets'][ds];root=Path(data['image_root'])
        if _sha(Path(data['val_manifest']))!=tasklock[ds]['manifest_sha256']['val']:raise ValueError('VAL_CHANGED')
        val=_manifest(Path(data['val_manifest']),'val',root);labels=np.array([_label(r) for r in val]);components=np.array([r['identity_component'] for r in val]);ids=np.array([r['sample_id'] for r in val])
        _allow(allowed,root,val);raw=extract(val,root,None,models,out,budget,'evaluation',ds,0,0);bundle=[];index=[]
        for seed in SEEDS:
            cp=oldroot/'private'/f'APART_{ds}_{seed}.npz'
            if _sha(cp)!=inventory[str(cp.relative_to(oldroot))]:raise ValueError('CACHE_HASH_CHANGED')
            with np.load(cp,allow_pickle=False) as z:raw['a']=z['a']
            if raw['a'].dtype!=np.float64 or raw['a'].shape!=(len(val),1536):raise ValueError('CACHE_LAYOUT')
            parent,pl=_parent_lock(cfg,tasklock,ds,seed)
            # Provenance of each old G stage explicitly points to the locked original parent.
            orig=_read(Path(cfg['generic_run'])/'stages'/ds/str(seed)/'task_01/STATE_W_LOCK.json')
            if orig['parent_sha256']!=pl['parent_sha256']:raise ValueError('CACHE_PARENT_MISMATCH')
            feature={key(m):mapped(m,raw,maps,protocol['solver']['backend']) for m in base}
            for task,size in enumerate(SIZES[ds],1):
                budget.check('compute',20);sp=out/'stages'/ds/str(seed)/f'task_{task:02d}';d=check_lock(out,sp);seen=d['seen'];current=d['current']
                with np.load(sp/'W.npz',allow_pickle=False) as z:w={k:z[k] for k in z.files}
                scores={};preds={}
                for m in methods(protocol):
                    k=key(m)
                    if m['base_method']:
                        parent_scores=scores[m['base_method']];cos=raw[m['branch']]@w['text_'+m['branch']]
                        s=cse(parent_scores,cos,seen,alpha=m['alpha'],permute=m['feature']=='cse_permuted')
                    else:
                        fm=m['branch']+'.LIN' if m['lambda_policy']=='fixed' else k
                        s=(feature[fm]@tensor(w[k],protocol['solver']['backend'])).cpu().numpy()
                    metric,cr,pred=_stage_metrics(s,labels,seen,current,_tail(tasklock,cfg,ds));scores[k]=s;preds[k]=pred
                    metric={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metric.items()}
                    valid=np.isin(labels,seen);old=np.isin(labels,[c for c in seen if c not in current]);new=np.isin(labels,current)
                    metric.update(old_to_current=int(np.sum(valid&old&np.isin(pred,current))),current_to_old=int(np.sum(valid&new&np.isin(pred,[c for c in seen if c not in current]))))
                    common={'dataset':ds,'seed':seed,'task':task,'method':k}
                    stage_rows.append({**common,'seen_classes':len(seen),**metric})
                    class_rows.extend([{**common,**r,'components':len(np.unique(components[labels==r['class_id']]))} for r in cr])
                    bundle.append(pred.astype(np.int16));index.append(common)
                    if m['base_method']:
                        base_s=scores[m['base_method']];bp=preds[m['base_method']];keep=np.stack([np.lexsort((np.array(seen),-r))[:min(5,len(seen))] for r in base_s]);top=np.array(seen)[keep]
                        margin=np.sort(base_s,axis=1)[:,-1]-np.sort(base_s,axis=1)[:,-2]
                        shift=np.max(abs(s-base_s),axis=1)
                        groups=[('class',str(c),labels==c) for c in seen]+[('old','all',old),('current','all',new),('tail','all',np.isin(labels,list(_tail(tasklock,cfg,ds))))]
                        for group,cid,mask in groups:
                            mask=mask&valid;n=int(mask.sum())
                            coverage.append({**common,'group':group,'class_id':cid,'n':n,'true_class_topk':int(np.sum(mask&np.any(top==labels[:,None],axis=1))),
                              'corrections':int(np.sum(mask&(pred==labels)&(bp!=labels))),'destructions':int(np.sum(mask&(pred!=labels)&(bp==labels))),
                              'wrong_to_wrong':int(np.sum(mask&(pred!=labels)&(bp!=labels)&(pred!=bp))),'outside_topk_winner':int(np.sum(mask&~np.any(top==pred[:,None],axis=1))),
                              'mean_margin':float(margin[mask].mean()) if n else None,'mean_semantic_shift':float(shift[mask].mean()) if n else None,
                              'shift_over_margin':float(np.mean(shift[mask]>margin[mask])) if n else None})
                np.savez_compressed(private/'scores'/f'{ds}_{seed}_{task}.npz',**scores,seen_ids=np.array(seen),sample_ids=ids,labels=labels)
                # Compare actual scores, not rounded historical aggregates.
                for new,tag,old in [('F.LOCK1e3','G','F'),('G.LOCK5e4','G','J'),('G.LOCK1e3','G','J1'),('B.LOCK5e4','B','J'),('B.LOCK1e3','B','J1')]:
                    op=oldroot/'private/scores'/f'{tag}_{ds}_{seed}_{task}.npz'
                    if _sha(op)!=inventory[str(op.relative_to(oldroot))]:raise ValueError('HISTORICAL_SCORE_CHANGED')
                    with np.load(op,allow_pickle=False) as z:oldscore=z[old]
                    parity.append({'dataset':ds,'seed':seed,'task':task,'method':new,'max_absolute_score_delta':float(np.max(abs(oldscore-scores[new]))),'prediction_differences':int(np.sum(predict(oldscore,seen)!=predict(scores[new],seen)))})
            del feature;gc.collect();torch.cuda.empty_cache()
        np.savez_compressed(private/f'{ds}_PREDICTIONS.npz',labels=labels,components=components,sample_ids=ids,predictions=np.stack(bundle),index=np.array([json.dumps(v) for v in index]))
    check_models(None,models)
    _write_csv(out/'stage_metrics.csv',stage_rows);_write_csv(out/'class_metrics.csv',class_rows);_write_csv(out/'candidate_coverage.csv',coverage)
    _write(out/'SOLVER_PARITY.json',{'historical_score_comparisons':parity,'synthetic':_read(out/'PRODUCTION_TESTS.json'),'cache_provenance':'original source/transform/parent/manifest/order/dtype and saved file SHA verified'})
    _write(out/'EVALUATION_COMPLETE.json',{'stage_rows':len(stage_rows),'class_rows':len(class_rows),'time':time.time()})

def main():
    cfg=_read(Path(os.environ['RF_CONFIG']));out=Path(cfg['run_root']);budget=Budget(out);role=os.environ['RF_ROLE']
    torch.set_grad_enabled(False);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.device_count()!=1:raise ValueError('EXACTLY_ONE_VISIBLE_GPU_REQUIRED')
    allowed=guard(cfg,role)
    if role=='fit':fit(cfg,out,allowed,budget)
    elif role=='eval':evaluate(cfg,out,allowed,budget)
    else:raise ValueError('UNKNOWN_ROLE')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        out=Path(_read(Path(os.environ['RF_CONFIG']))['run_root'])
        _write(out/f'FAILED_{os.environ.get("RF_ROLE","unknown")}_{int(time.time())}.json',{'error':repr(e),'time':time.time(),'traceback':traceback.format_exc()})
        raise
