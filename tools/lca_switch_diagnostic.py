"""Final-checkpoint, paired ON/OFF analytic-head diagnosis; no training path."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import core1_competition as competition
import lca_supported as lca

VERSION = 'lca_final_state_switch_v1'
JOBS = [ds+'_'+arm for ds in ('ISIC', 'HK')
        for arm in ('lca_supported', 'local_transport_lca_supported')]


def validate_scope(c, lock, old):
    name = c['job_name']; ds = old['name']; data = lock['data'].get(ds)
    if (c['version'] != VERSION or lock['version'] != VERSION or name not in JOBS
            or c['modes'] != ['OFF', 'ON'] or c['train_allowed'] or c['test_accessed']
            or c['batch_size'] != 64 or c['workers'] != 0 or not data
            or old['job_name'] != name or old['order'] != data['order']
            or old['task_sizes'] != data['task_sizes'] or old['seed'] != 74002
            or old['split_seed'] != 74002 or old['method'] != 'static_pc'
            or old['source_commit'] != lock['checkpoint_source_commit']
            or old['lca_version'] != 'supported_old_margin_v1'
            or old['lca_map_ridge'] != .001 or old['lca_curvature_cap'] != .1
            or old['expected_steps'] != data['training_updates']
            or old['module'] != ('local_transport' if 'local_transport' in name else 'base')
            or old['evaluation_split'] != 'development_validation'
            or c['task'] != len(data['task_sizes']) or old.get('test_accessed', False)):
        raise ValueError('Diagnostic scope differs from frozen final-state switch')
    parent = Path(c['parent_root']).resolve()
    if Path(old['output']).resolve() != parent/'runs'/name:
        raise ValueError('Checkpoint is outside the delivered parent round')
    if Path(c['output']).resolve() != Path(c['root']).resolve()/'runs'/name:
        raise ValueError('Independent diagnostic output required')
    return data


def read_val(old, data):
    # This entry deliberately never calls manifests(), which also opens train.csv.
    with (Path(old['manifest'])/'val.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        p = Path(r['relative_path']); r['label'] = int(r['original_label'])
        if (r['split'] != 'val' or p.is_absolute() or '..' in p.parts
                or not r['identity_component'] or r['label'] not in data['order']):
            raise ValueError('Validation manifest scope differs')
    counts = {str(c):sum(r['label'] == c for r in rows) for c in data['order']}
    if counts != data['val_counts'] or len({r['relative_path'] for r in rows}) != len(rows):
        raise ValueError('Validation support differs')
    return rows


@torch.no_grad()
def heads(bank, pairs, R, S, M, saved_double=None, saved_float=None):
    on, on_audit = lca.bank_head(bank, pairs, R, S, M)
    off, off_audit = lca.bank_head(bank, pairs, R, torch.zeros_like(S), M)
    if saved_double is not None:
        rebuild = float((on-saved_double).norm()/saved_double.norm().clamp_min(1e-30))
        if rebuild > 1e-8 or not torch.equal(saved_double.float(), saved_float):
            raise ValueError('Saved head/rebuild provenance mismatch')
    else:
        rebuild = 0.
    k = len(bank['n']); q = bank['Q']/k; mu = bank['mu']/k
    A = q.sum(0)+.001*R
    lap, targets = competition.laplacians(pairs)
    rhs = mu.T+.5*mu.T@targets
    def native(w):
        return A@w+.5*torch.einsum('cdk,ckl->dl', q@w, lap)
    def energy(w):
        return float(.5*(w*native(w)).sum()-(w*rhs).sum())
    off_e, on_e = energy(off), energy(on)
    extra_off, extra_on = float(lca.loss(off,S,M)), float(lca.loss(on,S,M))
    tol = 1e-8*max(1., abs(off_e), abs(on_e), extra_off, extra_on)
    if on_e < off_e-tol or on_e+extra_on > off_e+extra_off+tol:
        raise ValueError('Counterfactual objective optimality check failed')
    delta = on-off; old_count = int((M.abs().sum(0)>0).sum())
    force_off, force_on = S@off@M, S@on@M
    audit = dict(on_solve=on_audit,off_solve=off_audit,rebuild_relative_error=rebuild,
        force_at_off_norm=float(force_off.norm()),force_at_on_norm=float(force_on.norm()),
        force_at_off_relative_rhs=float(force_off.norm()/rhs.norm().clamp_min(1e-30)),
        force_at_on_relative_rhs=float(force_on.norm()/rhs.norm().clamp_min(1e-30)),
        delta_head_norm=float(delta.norm()),delta_head_relative=float(delta.norm()/off.norm().clamp_min(1e-30)),
        delta_old_head_norm=float(delta[:,:old_count].norm()),
        delta_new_head_norm=float(delta[:,old_count:].norm()),
        new_direct_force_norm=float(force_on[:,old_count:].norm()),
        native_energy_off=off_e,native_energy_on=on_e,native_energy_change=on_e-off_e,
        extra_energy_off=extra_off,extra_energy_on=extra_on,
        on_objective_change=on_e+extra_on-off_e-extra_off,
        penalty_trace=float(S.trace()),margin_rank=old_count-1,
        solver_relative_residual_max=max(on_audit['relative_residual'],off_audit['relative_residual']))
    return off, on, audit


def describe(x):
    x = np.asarray(x,dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty/nonfinite diagnostic scalar group')
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),minimum=float(x.min()),
        maximum=float(x.max()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),
        p95=float(np.quantile(x,.95)),positive=int((x>0).sum()),zero=int((x==0).sum()),negative=int((x<0).sum()))


def summarize_scores(off, on, labels, data):
    order = data['order']; old_count = len(order)-data['task_sizes'][-1]
    y = np.array([order.index(int(v)) for v in labels]); n = len(y); k = len(order)
    if (off.shape != (n,k) or on.shape != off.shape or not np.isfinite(off).all()
            or not np.isfinite(on).all() or {str(c):int((labels==c).sum()) for c in order} != data['val_counts']):
        raise ValueError('Scores or paired denominators differ')
    po, pn = off.argmax(1), on.argmax(1)
    counts = {}; flow = {}; margins = {}; metrics = {}
    groups = [('all',np.ones(n,dtype=bool)),('old',y<old_count),('new',y>=old_count)]
    groups += [(str(c),labels==c) for c in order]
    def true_margin(z):
        other=z.copy();other[np.arange(n),y]=-np.inf
        return z[np.arange(n),y]-other.max(1)
    def group_margin(z):
        old=z[:,:old_count].max(1);new=z[:,old_count:].max(1)
        return np.where(y<old_count,old-new,new-old)
    tm_off,tm_on=true_margin(off),true_margin(on)
    gm_off,gm_on=group_margin(off),group_margin(on)
    for name,mask in groups:
        co,cn=po[mask]==y[mask],pn[mask]==y[mask]
        changed=po[mask]!=pn[mask]
        counts[name]=dict(n=int(mask.sum()),changed=int(changed.sum()),rescued=int((~co&cn).sum()),
            harmed=int((co&~cn).sum()),both_correct=int((co&cn).sum()),both_wrong=int((~co&~cn).sum()),
            wrong_to_different_wrong=int((~co&~cn&changed).sum()),
            correct_off=int(co.sum()),correct_on=int(cn.sum()),net_correct=int(cn.sum()-co.sum()))
        a=counts[name]
        assert a['changed']==a['rescued']+a['harmed']+a['wrong_to_different_wrong']
        assert a['net_correct']==a['rescued']-a['harmed']
        margins[name]=dict(true_class_delta=describe((tm_on-tm_off)[mask]),
            old_new_group_delta=describe((gm_on-gm_off)[mask]),
            true_class_off=describe(tm_off[mask]),true_class_on=describe(tm_on[mask]))
    for mode,pred,z in [('OFF',po,off),('ON',pn,on)]:
        true_old=y<old_count;pred_old=pred<old_count;correct=pred==y
        flow[mode]=dict(old_correct=int((true_old&correct).sum()),
            old_to_new=int((true_old&~pred_old).sum()),old_to_old_wrong=int((true_old&pred_old&~correct).sum()),
            new_correct=int((~true_old&correct).sum()),new_to_old=int((~true_old&pred_old).sum()),
            new_to_new_wrong=int((~true_old&~pred_old&~correct).sum()))
        assert sum(flow[mode].values())==n
        recalls={str(c):float((pred[labels==c]==order.index(c)).mean()) for c in order}
        f1=[]
        for ix in range(k):
            tp=int(((pred==ix)&(y==ix)).sum());fp=int(((pred==ix)&(y!=ix)).sum());fn=int(((pred!=ix)&(y==ix)).sum())
            f1.append(2*tp/(2*tp+fp+fn))
        ranked=sorted(order,key=lambda c:(-data['train_counts'][str(c)],c));tail=ranked[k//2:]
        metrics[mode]=dict(accuracy=float(correct.mean()),balanced_accuracy=float(np.mean(list(recalls.values()))),
            old_recall=float(np.mean([recalls[str(c)] for c in order[:old_count]])),
            new_recall=float(np.mean([recalls[str(c)] for c in order[old_count:]])),
            tail_recall=float(np.mean([recalls[str(c)] for c in tail])),macro_f1=float(np.mean(f1)),
            per_class_recall=recalls,per_class_n=data['val_counts'],validation_n=n)
    delta=on.astype(np.float64)-off.astype(np.float64)
    rms=lambda a:float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))
    score_change=dict(rms=rms(delta),maximum_abs=float(np.abs(delta).max()),
        relative_rms=rms(delta)/max(rms(off),1e-30),
        centered_rms=rms(delta-delta.mean(1,keepdims=True)),
        centered_relative_rms=rms(delta-delta.mean(1,keepdims=True))/max(rms(off-off.mean(1,keepdims=True)),1e-30),
        float32_head_readout=True)
    return dict(metrics=metrics,paired=counts,confusion=flow,margin_changes=margins,score_change=score_change)


def run(c):
    from run_prototype_single import Images,extract,save
    from run_multilabel import ApartFeatures
    from run_medical_v2 import transform
    from run_lca_supported_cycle import checkpoint_metadata
    from torch.utils.data import DataLoader
    import module_additions as modules
    started=time.monotonic();out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    def state(status,**kw):
        save(out/'STATUS.json',dict(status=status,optimizer_updates=0,policy_updates=0,
            train_accessed=False,test_accessed=False,**kw))
    def budget():
        if time.time()>=c['deadline']:raise TimeoutError('Frozen diagnostic deadline')
    state('RUNNING',phase='CHECKPOINT_ADMISSION')
    try:
        lock=json.loads(Path(c['diagnostic_lock']).read_text());parent=Path(c['parent_root'])
        pc=json.loads((parent/'CONFIG.private.json').read_text())
        old=next(j for j in pc['jobs'] if j['job_name']==c['job_name'])
        data=validate_scope(c,lock,old)
        p=json.loads((parent/'PROGRAM_STATE.json').read_text());d=json.loads((parent/'DELIVERY.private.json').read_text())
        if p['status']!='COMPLETE' or p['active'] or d['commit']!=lock['parent_delivery_commit'] or not d['nas_saved']:
            raise ValueError('Original round must be completely delivered')
        if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():
            raise ValueError('GPU01 only')
        torch.set_num_threads(4);torch.manual_seed(74002);torch.cuda.manual_seed_all(74002)
        checkpoint=Path(old['output'])/f"stage_{c['task']}.pt"
        st=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if (st['lca_metadata']!=checkpoint_metadata(old,c['task'],old['expected_steps'])
                or st['seen']!=old['order'] or st['steps']!=old['expected_steps'] or st['module']!=old['module']
                or st['lca_old_count']!=len(old['order'])-old['task_sizes'][-1]):
            raise ValueError('Final checkpoint metadata differs')
        bank={key:value.cuda() if torch.is_tensor(value) else value for key,value in st['bank'].items()}
        bank['components']=[{key:value.cuda() if torch.is_tensor(value) else value for key,value in v.items()}
                            for v in st['bank']['components']]
        local=None if st['module_memory'] is None else st['module_memory'].double().cuda()
        R,_=modules.metric(bank,old['module'],old['name'],old['order'],local)
        saved=st['lca_head_double'].cuda();S,M=lca.restore(st['lca_state'],len(saved),saved.shape[1],saved)
        if not torch.allclose(M,lca.old_margin_metric(saved.shape[1],st['lca_old_count'],saved),atol=1e-10,rtol=1e-10):
            raise ValueError('Saved old/new mask differs')
        state('RUNNING',phase='FIXED_STATE_SOLVE')
        off,on,mechanical=heads(bank,st['boundary_pair_weights'].double().cuda(),R,S,M,saved,st['head'].cuda())
        off_np=off.float().cpu().numpy();on_np=st['head'].numpy()
        del bank,local,R,S,M,saved,off,on
        budget();state('RUNNING',phase='VALIDATION_FEATURES')
        encoder=ApartFeatures(old['legacy_repo'],old['weight'],len(old['order']),'cuda:0',74002)
        encoder.load_state_dict(st['model'],strict=True);encoder.requires_grad_(False).eval()
        rows=read_val(old,data)
        loader=DataLoader(Images(rows,old['images'],transform(False),74002+c['task']*100003),
            batch_size=64,num_workers=0,shuffle=False,drop_last=False)
        x,labels=extract(encoder,loader,budget)
        if not all(torch.equal(v.detach().cpu(),st['model'][key]) for key,v in encoder.state_dict().items()):
            raise ValueError('Read-only diagnosis changed encoder state')
        result=summarize_scores(x@off_np,x@on_np,labels,data)
        reference=lock['on_references'][c['job_name']]
        if (result['metrics']['ON']['per_class_recall']!=reference['per_class_recall']
                or result['metrics']['ON']['per_class_n']!=reference['per_class_n']
                or result['metrics']['ON']['accuracy']!=reference['accuracy']):
            raise ValueError('ON parity with delivered final validation differs')
        result.update(job_name=c['job_name'],dataset=old['name'],task=c['task'],mechanical=mechanical,
            version=VERSION,checkpoint_source_commit=old['source_commit'],diagnostic_source_commit=c['source_commit'],
            checkpoint_metadata_verified=True,on_published_parity=True,encoder_state_unchanged=True,
            optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,
            validation_image_exposures=len(rows),head_readout_exposures=2*len(rows),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_bytes=checkpoint.stat().st_size,
            worker_wall_seconds=time.monotonic()-started)
        save(out/'RESULTS.json',result);state('COMPLETE',phase='ANONYMOUS_AGGREGATES_SAVED')
    except BaseException as exc:
        state('INCOMPLETE',error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    job=campaign['jobs'][int(os.environ['Q165_JOB'])]
    run(dict(job,root=campaign['root'],parent_root=campaign['parent_root'],
        source_commit=campaign['source_commit'],deadline=campaign['deadline']))
