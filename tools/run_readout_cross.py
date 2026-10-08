"""Controlled T2/T4 readout reconstruction; private configuration arrives on stdin."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import affine_moments
import prototype_coherent as method
from next1_support import fixed_split, summarize
from run_multilabel import ApartFeatures
from run_pcrl import common_shift
from run_prototype_single import Images, extract, manifests, save


def features_job(c):
    out = Path(c['root'])/c['job']; out.mkdir(exist_ok=False)
    task=c.get('task',2);old_count=2*(task-1);prefix=c['job'].startswith('PREFIX')
    fit_n,meta_n,val_n,previous_steps,final_steps={2:(1079,254,149,276,310),4:(5020,1078,295,318,476)}[task]
    start = time.monotonic()
    def status(state, **extra):
        save(out/'STATUS.json', dict(status=state, elapsed_seconds=time.monotonic()-start,
             optimizer_updates=0, test_accessed=False, **extra))
    def budget():
        if time.monotonic()-start > c['feature_cap_seconds'] or time.time() > c['deadline']:
            raise TimeoutError('Frozen feature extraction limit')
    try:
        status('RUNNING', phase='load_frozen_encoder')
        torch.set_num_threads(4); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        encoder = ApartFeatures(c['legacy_repo'], c['weight'], 8, 'cuda:0', c['seed'])
        state = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
        expected = c['order'][:old_count if prefix else old_count+2]
        if state['seen'] != expected or not state.get('fit_only') or state['steps'] != (previous_steps if prefix else final_steps):
            raise ValueError('Unexpected frozen checkpoint')
        encoder.load_state_dict(state['model'], strict=True); encoder.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in encoder.parameters())
        torch.save({k:state[k] for k in ('bank','head','seen','steps','fit_only')}, out/'STATE.private.pt')
        del state
        train, val = manifests(c)
        rows = [r for r in train if r['label'] in c['order'][old_count:old_count+2]]
        fi, mi = fixed_split(rows, c)
        assert len(fi)==fit_n and len(mi)==meta_n
        assert not ({rows[i]['identity_component'] for i in fi} & {rows[i]['identity_component'] for i in mi})
        from run_medical_v2 import transform
        def get(rows):
            data = DataLoader(Images(rows,c['images'],transform(False),c['seed']+task*100003),
                batch_size=64, num_workers=4, multiprocessing_context='spawn',
                generator=torch.Generator().manual_seed(c['seed']+task*2003))
            return extract(encoder,data,budget)
        status('RUNNING', phase='current_features')
        x,y = get(rows)
        # Match original canonical batches; only permanent fit rows enter the map/head.
        values = dict(fit=x[fi], labels=y[fi], fit_positions=np.asarray(fi))
        if not prefix:
            status('RUNNING', phase='development_features')
            values['val'],values['val_labels'] = get([r for r in val if r['label'] in c['order'][:old_count+2]])
            assert len(values['val'])==val_n
        np.savez_compressed(out/'FEATURES.private.npz',**values)
        status('COMPLETE',phase='FEATURES_READY',fit_n=len(fi),meta_used_for_fit=0,
               peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status('INCOMPLETE',error=str(exc)); raise


def rebuild(bank, before, after, labels, drift, seeds):
    if drift=='SHIFT':
        shift = common_shift(before,after,labels)
        old = method.translate(bank,shift); audit = dict(kind='translation')
    elif drift in ('AFFINE','AFFINE_MEAN'):
        mapping = affine_moments.fit(before,after,labels)
        old = affine_moments.transport(bank,mapping); audit = mapping['diagnostics']
        if drift=='AFFINE_MEAN':
            mu, moved = bank['mu'], old['mu']
            # Class-specific translation preserves covariance and component offsets.
            old['Q'] = (bank['Q']-mu[:,:,None]*mu[:,None,:]
                        +moved[:,:,None]*moved[:,None,:])
            old['components'] = [dict(c,center=c['center']+moved[c['label']]-mu[c['label']])
                                 for c in bank['components']]
            audit = dict(audit,kind='affine_mean_preserved_covariance')
    else:
        raise ValueError('Unfrozen readout')
    group,difficulty = method.memberships(after,labels,seeds)
    weights = method.sample_weights(labels,group,difficulty,after.new_zeros(len(seeds)))
    full = method.append(old,after,labels,group,weights)
    head,_ = method.head(full,0.)
    G = full['Q'].mean(0)+.001*torch.eye(before.shape[1],dtype=before.dtype)
    H = full['mu'].T/len(full['n'])
    return full,head,G,H,audit


def metrics(scores, labels, order):
    detail = summarize(scores,labels,order)
    recalls = [detail['per_class'][str(c)]['recall'] for c in order]
    return dict(BA=float(np.mean(recalls)),old_BA=float(np.mean(recalls[:-2])),
                new_BA=float(np.mean(recalls[-2:])),details=detail)


def paired(a,b,truth):
    old_count=a.shape[1]-2
    pa,pb = a.argmax(1),b.argmax(1);ca,cb = pa==truth,pb==truth
    result = dict(n=len(truth),both_correct=int((ca&cb).sum()),shift_only_correct=int((ca&~cb).sum()),
                  affine_only_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),
                  prediction_changed=int((pa!=pb).sum()))
    new = truth>=old_count
    fa = (a[:,old_count:].argmax(1)+old_count==truth)&(pa<old_count)&new
    fb = (b[:,old_count:].argmax(1)+old_count==truth)&(pb<old_count)&new
    result['within_new_correct_cross_group_failure'] = dict(shift=int(fa.sum()),affine=int(fb.sum()),
        newly_failed=int((~fa&fb).sum()),recovered=int((fa&~fb).sum()),both_failed=int((fa&fb).sum()))
    return result


def decompose(Gs,Ga,Hs,Ha,Ws,Wa):
    mean = torch.linalg.solve(Ga,Ha-Hs)
    second = -torch.linalg.solve(Ga,(Ga-Gs)@Ws)
    error = float((Wa-Ws-mean-second).norm()/(Wa-Ws).norm().clamp_min(1e-15))
    if error > 1e-7: raise ValueError('Head difference identity failed')
    return mean,second,error


def diagonal_guard(scores,reference,head,saved_head,bank,saved_bank,c):
    checks = dict(scores_close=bool(np.allclose(scores,reference,atol=c['score_atol'],rtol=c['score_rtol'])),
        predictions_equal=bool(np.array_equal(scores.argmax(1),reference.argmax(1))),
        head_close=bool(torch.allclose(head,saved_head.double(),atol=c['head_atol'],rtol=c['head_rtol'])),
        means_close=bool(torch.allclose(bank['mu'],saved_bank['mu'],atol=1e-6,rtol=1e-5)),
        moments_close=bool(torch.allclose(bank['Q'],saved_bank['Q'],atol=1e-6,rtol=1e-5)))
    return dict(passed=all(checks.values()),checks=checks,max_score_error=float(np.max(np.abs(scores-reference))),
                max_head_error=float((head-saved_head).abs().max()))


def analyze(c):
    root=Path(c['root']); public=root/'public'; started=time.monotonic(); cpu=time.process_time()
    task=c.get('task',2);old_count=2*(task-1);order=c['order'][:2*task]
    prefix_jobs=c.get('prefix_jobs',dict(SHIFT='PREFIX',AFFINE='PREFIX'))
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    def budget():
        if time.monotonic()-started>c['analysis_cap_seconds'] or time.time()>c['deadline']:
            raise TimeoutError('Readout audit CPU limit')
    cells={}; diagonals={}; geometry={}; row_scores={}; row_matrices={}
    for row in ('SHIFT','AFFINE'):
        prefix_path=root/prefix_jobs[row]
        prefix=torch.load(prefix_path/'STATE.private.pt',map_location='cpu',weights_only=False)
        p=np.load(prefix_path/'FEATURES.private.npz');before=torch.from_numpy(p['fit']).double()
        labels=torch.tensor([order.index(int(y)) for y in p['labels']])
        assert len(prefix['bank']['n'])==old_count
        assert [int((labels==k).sum()) for k in (old_count,old_count+1)]=={2:[1057,22],4:[2310,2710]}[task]
        seeds=method.seed_components(before,labels)
        budget(); x=np.load(root/row/'FEATURES.private.npz')
        assert np.array_equal(p['labels'],x['labels']) and np.array_equal(p['fit_positions'],x['fit_positions'])
        current=torch.from_numpy(x['fit']).double(); observed=np.load(c['reference_scores'][row])
        assert np.array_equal(x['val_labels'],observed['labels'])
        state=torch.load(root/row/'STATE.private.pt',map_location='cpu',weights_only=False)
        row_scores[row]={};row_matrices[row]={}
        for drift in ('SHIFT','AFFINE'):
            budget(); key=row+'_'+drift
            bank,W,G,H,audit=rebuild(prefix['bank'],before,current,labels,drift,seeds)
            scores=x['val']@W.float().numpy()
            cells[key]=metrics(scores,x['val_labels'],order)
            cells[key]['map']=audit
            if row==drift:
                diagonals[row]=diagonal_guard(scores,observed['scores'],W,state['head'],bank,state['bank'],c)
            class_geometry=[]
            for k,class_id in enumerate(order):
                budget(); Q,mu=bank['Q'][k],bank['mu'][k]
                covariance=Q-torch.outer(mu,mu)
                spectrum=torch.linalg.eigvalsh((covariance+covariance.T)/2)
                class_geometry.append(dict(class_id=class_id,trace_second=float(Q.trace()),mean_norm=float(mu.norm()),
                    covariance_min_eigenvalue=float(spectrum[0]),covariance_max_eigenvalue=float(spectrum[-1])))
            eig=torch.linalg.eigvalsh(G)
            geometry[key]=dict(classes=class_geometry,G_min_eigenvalue=float(eig[0]),
                G_max_eigenvalue=float(eig[-1]),G_condition=float(eig[-1]/eig[0]))
            row_scores[row][drift]=scores;row_matrices[row][drift]=(W,G,H)
            np.savez_compressed(root/(key+'.private.npz'),scores=scores,labels=x['val_labels'],head=W.numpy())
        x.close();observed.close();p.close()
    save(public/'DIAGONAL_CHECK.json',diagonals)
    save(public/'MOMENT_DIAGNOSTICS.json',geometry)
    if not all(v['passed'] for v in diagonals.values()):
        save(public/'RESULTS.json',dict(status='STOPPED_DIAGONAL_MISMATCH',diagonal=diagonals,
             non_diagonal_interpretation=False,optimizer_updates=0,test_accessed=False))
        raise ValueError('Diagonal reconstruction failed; no counterfactual interpretation')
    rows={};margin={}
    for row in ('SHIFT','AFFINE'):
        budget(); x=np.load(root/row/'FEATURES.private.npz')
        truth=np.asarray([order.index(int(y)) for y in x['val_labels']]);new=truth>=old_count
        a,b=row_scores[row]['SHIFT'],row_scores[row]['AFFINE']
        rows[row]=dict(deltas={k:cells[row+'_AFFINE'][k]-cells[row+'_SHIFT'][k] for k in ('BA','old_BA','new_BA')},
                       paired=paired(a,b,truth))
        Ws,Gs,Hs=row_matrices[row]['SHIFT'];Wa,Ga,Ha=row_matrices[row]['AFFINE']
        dm,dq,error=decompose(Gs,Ga,Hs,Ha,Ws,Wa)
        assert float(dm[:,old_count:].abs().max())<1e-12
        z=torch.from_numpy(x['val']).double();base=z@Ws;final=z@Wa
        ids=torch.where(torch.from_numpy(new))[0];target=torch.from_numpy(truth[new]);competitor=base[ids,:old_count].argmax(1)
        def margins(s):return s[ids,target]-s[ids,competitor]
        ms,ma=margins(base),margins(final);mm,mq=margins(z@dm),margins(z@dq)
        if not torch.allclose(ma-ms,mm+mq,atol=1e-9,rtol=1e-7):raise ValueError('Margin identity failed')
        margin[row]=dict(head_identity_relative_error=error,reference='same strongest old competitor under SHIFT readout',
             new_n=len(ids),shift_margin_mean=float(ms.mean()),affine_margin_mean=float(ma.mean()),
             delta_mean=float((ma-ms).mean()),mean_rhs_contribution=float(mm.mean()),
             second_moment_contribution=float(mq.mean()),G_relative_change=float((Ga-Gs).norm()/Gs.norm()),
             new_head_relative_change=float((Wa[:,old_count:]-Ws[:,old_count:]).norm()/Ws[:,old_count:].norm()),
             algebraic_not_independent_causal_effects=True)
        x.close()
    interaction={k:rows['AFFINE']['deltas'][k]-rows['SHIFT']['deltas'][k] for k in ('BA','old_BA','new_BA')}
    result=dict(status='COMPLETE_DIAGNOSTIC',task=task,cells=cells,rows=rows,
        frozen_encoders=len(c['checkpoints']),readout_cells=4,optimizer_updates=0,policy_updates=0,test_accessed=False,
        meta_used_for_fit=0,independent_confirmation=False,full_trajectory_attribution=False,
        shared_history_across_rows=task==2,
        analysis_cpu_core_seconds=time.process_time()-cpu,analysis_wall_seconds=time.monotonic()-started)
    result['interaction' if task==2 else 'row_effect_difference']=interaction
    save(public/'MARGIN_DECOMPOSITION.json',margin);save(public/'RESULTS.json',result)
    return result


def run(c):
    root=Path(c['root']);public=root/'public';public.mkdir(exist_ok=True)
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'PROGRAM_STATE.json').exists():raise ValueError('No automatic restart')
    names=c.get('job_names',['PREFIX','SHIFT','AFFINE'])
    jobs=[];active={};started=time.time();phase='FEATURES';outcome='RUNNING'
    def record(**extra):
        now=time.time();total=sum(j.get('elapsed_seconds',now-j['started']) for j in jobs)
        reserved=sum(max(0,c['feature_cap_seconds']-(now-j['started'])) for j in jobs if j['id'] in active)
        value=dict(status=outcome,phase=phase,started=started,deadline=c['deadline'],jobs=jobs,
            gpu_seconds=total,gpu_reserved_seconds=reserved,gpu_seconds_limit=c['gpu_seconds_limit'],
            prior_gpu_seconds=c['prior_gpu_seconds'],cumulative_gpu_seconds=c['prior_gpu_seconds']+total,
            optimizer_updates=0,test_accessed=False,publication_verified=False,**extra)
        save(root/'PROGRAM_STATE.json',value);save(public/'BUDGET_LEDGER.json',value)
    def reap():
        for name,p in list(active.items()):
            code=p.poll()
            if code is not None:
                job=next(j for j in jobs if j['id']==name)
                job.update(exit_code=code,elapsed_seconds=time.time()-job['started']);active.pop(name)
    try:
        assert len(names)==len(c['gpus']) and len(set(names))==len(names)
        assert c['gpu_seconds_limit']==len(names)*c['feature_cap_seconds']
        assert c['prior_gpu_seconds']+c['gpu_seconds_limit']<=c['original_total_gpu_seconds_limit']
        if time.time()+c['feature_cap_seconds']+c['analysis_cap_seconds']>c['deadline']:
            raise TimeoutError('Insufficient inherited deadline')
        free={int(i):int(m) for i,m in (line.split(',') for line in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines())}
        assert all(free[g]>=c['minimum_free_mib']*c['gpus'].count(g) for g in set(c['gpus']))
        for name,gpu in zip(names,c['gpus']):
            child=dict(c,job=name,checkpoint=c['checkpoints'][name])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),Q109_KIND='features',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            stream=(root/(name+'.private.log')).open('w')
            p=subprocess.Popen([c['python'],c['entry']],stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,
                               env=env,text=True,start_new_session=True)
            jobs.append(dict(id=name,gpu=gpu,pid=p.pid,started=time.time(),cap_seconds=c['feature_cap_seconds']))
            active[name]=p;record();p.stdin.write(json.dumps(child));p.stdin.close();stream.close()
        checked=False
        while active:
            reap();record()
            if any(j.get('exit_code',0)!=0 for j in jobs):raise RuntimeError('Feature job failed; no retry')
            if any(time.time()-j['started']>j['cap_seconds'] for j in jobs if j['id'] in active):
                raise TimeoutError('Feature process residence cap')
            if not checked and time.time()-started>10:
                ps=subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True)
                records=[line.strip().split(None,2) for line in ps.splitlines()]
                descendants={os.getpid()}
                for _ in range(8):
                    descendants.update(int(pid) for pid,ppid,args in records if int(ppid) in descendants)
                lines=[args for pid,ppid,args in records if int(pid) in descendants]
                forbidden=('wangbomin','LongTailedCL','Long-tailed','readout_cross','SHIFT','AFFINE')
                assert not any(word in arg for arg in lines for word in forbidden)
                save(public/'PROCESS_CHECK.json',dict(passed=True,commands=lines,checked_at=time.time()))
                checked=True
            if active:time.sleep(2)
        for j in jobs:
            s=json.loads((root/j['id']/'STATUS.json').read_text());assert s['status']=='COMPLETE' and s['optimizer_updates']==0
        storage=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if storage>c['storage_limit_bytes']:raise RuntimeError('Readout storage limit')
        phase='CPU_READOUT';record()
        child=dict(os.environ,CUDA_VISIBLE_DEVICES='',Q109_KIND='analyze',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (root/'analysis.private.log').open('w') as stream:
            result=subprocess.run([c['python'],c['entry']],input=json.dumps(c),stdout=stream,stderr=subprocess.STDOUT,
                env=child,text=True,timeout=c['analysis_cap_seconds'])
        if result.returncode:raise RuntimeError('Readout reconstruction/audit failed')
        audit=json.loads((public/'RESULTS.json').read_text())
        outcome='AWAITING_PUBLICATION';phase='COMPLETE'
        record(analysis_cpu_core_seconds=audit['analysis_cpu_core_seconds'],
               analysis_wall_seconds=audit['analysis_wall_seconds'],storage_bytes=storage)
    except BaseException as exc:
        for p in active.values():
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for p in active.values():
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        reap();outcome='INCOMPLETE';phase='STOPPED';record(error=str(exc));raise
