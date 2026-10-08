"""One frozen T4 covariance intervention using existing private feature caches."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

import prototype_coherent as method
from run_prototype_single import save
from run_readout_cross import rebuild, metrics, diagonal_guard


def run(c):
    root=Path(c['root']); cache=Path(c['cache_root']); public=root/'public'
    started=time.monotonic(); cpu=time.process_time(); start_time=time.time()
    if (root/'PROGRAM_STATE.json').exists():
        raise ValueError('No automatic restart or overwrite')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or c['task']!=4:
        raise ValueError('Frozen CPU-only T4 diagnostic required')
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    def record(status, **extra):
        value=dict(status=status,started=start_time,finished=time.time(),
            wall_seconds=time.monotonic()-started,cpu_core_seconds=time.process_time()-cpu,
            gpu_seconds=0,optimizer_updates=0,policy_updates=0,test_accessed=False,
            prior_gpu_seconds=c['prior_gpu_seconds'],cumulative_gpu_seconds=c['prior_gpu_seconds'],
            deadline=c['deadline'],source_commit=c['source_commit'],publication_verified=False,**extra)
        save(root/'PROGRAM_STATE.json',value);save(public/'BUDGET_LEDGER.json',value)
        return value
    def stop(signum,frame):
        raise TimeoutError('Frozen CPU readout budget reached')
    record('RUNNING')
    try:
        remaining=min(c['analysis_cap_seconds'],c['deadline']-time.time())
        if remaining<=0:raise TimeoutError('Inherited deadline expired')
        signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,remaining)
        command=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','args='],text=True).strip()
        if any(word in command for word in ('wangbomin','LongTailedCL','Long-tailed','readout','AFFINE','SHIFT')):
            raise ValueError('Non-neutral process command')
        save(public/'PROCESS_CHECK.json',dict(passed=True,pid=os.getpid(),command=command,gpu_process=False))
        parent=json.loads((cache/'PROGRAM_STATE.json').read_text())
        assert parent['status']=='COMPLETE_DIAGNOSTIC' and parent['publication_verified']
        previous=json.loads((cache/'public'/'RESULTS.json').read_text())
        order=c['order'];old_count=6
        assert order==[4,0,3,7,5,6,2,1] and c['seed']==74002
        cells={};references={};invariants={};rows={}
        for row in ('SHIFT','AFFINE'):
            prefix=cache/('PREFIX_'+row)
            state=torch.load(prefix/'STATE.private.pt',map_location='cpu',weights_only=False)
            endpoint=torch.load(cache/row/'STATE.private.pt',map_location='cpu',weights_only=False)
            assert state['steps']==318 and endpoint['steps']==476
            assert state['seen']==order[:6] and endpoint['seen']==order
            assert state['fit_only'] and endpoint['fit_only']
            with np.load(prefix/'FEATURES.private.npz') as p,np.load(cache/row/'FEATURES.private.npz') as x:
                assert np.array_equal(p['labels'],x['labels'])
                assert np.array_equal(p['fit_positions'],x['fit_positions'])
                before=torch.from_numpy(p['fit']).double();after=torch.from_numpy(x['fit']).double()
                labels=torch.tensor([order.index(int(y)) for y in p['labels']])
                assert [int((labels==k).sum()) for k in (6,7)]==[2310,2710]
                val=x['val'];raw_truth=x['val_labels']
            truth=np.asarray([order.index(int(y)) for y in raw_truth])
            assert np.bincount(truth,minlength=8).tolist()==[41,44,40,24,37,26,46,37]
            seeds=method.seed_components(before,labels);banks={};heads={};scores={}
            for drift in ('SHIFT','AFFINE','AFFINE_MEAN'):
                key=row+'_'+drift
                bank,W,G,H,audit=rebuild(state['bank'],before,after,labels,drift,seeds)
                score=val@W.float().numpy()
                if not np.isfinite(score).all():raise ValueError('Nonfinite scores')
                cells[key]=dict(metrics(score,raw_truth,order),map=audit)
                banks[drift]=bank;heads[drift]=(W,G,H);scores[drift]=score
                if drift!='AFFINE_MEAN':
                    with np.load(cache/(key+'.private.npz')) as ref:
                        reference_scores=ref['scores']
                        check=dict(labels_equal=bool(np.array_equal(raw_truth,ref['labels'])),
                            scores_close=bool(np.allclose(score,ref['scores'],atol=c['score_atol'],rtol=c['score_rtol'])),
                            predictions_equal=bool(np.array_equal(score.argmax(1),ref['scores'].argmax(1))),
                            head_close=bool(np.allclose(W.numpy(),ref['head'],atol=c['head_atol'],rtol=c['head_rtol'])))
                        references[key]=dict(checks=check,passed=all(check.values()),
                            max_score_error=float(np.max(np.abs(score-ref['scores']))))
                    assert cells[key]['details']==previous['cells'][key]['details']
                    if row==drift:
                        guard=diagonal_guard(score,reference_scores,W,endpoint['head'],bank,endpoint['bank'],c)
                        references[key]['saved_endpoint_guard']=guard
                        references[key]['passed'] &= guard['passed']
                    save(public/'REFERENCE_CHECK.json',references)
                    if not references[key]['passed']:raise ValueError('Frozen reference reconstruction failed')
                else:
                    np.savez_compressed(root/(key+'.private.npz'),scores=score,labels=raw_truth,head=W.numpy())
            a,m=banks['AFFINE'],banks['AFFINE_MEAN'];old=state['bank']
            old_cov=old['Q']-old['mu'][:,:,None]*old['mu'][:,None,:]
            new_cov=m['Q'][:6]-m['mu'][:6,:,None]*m['mu'][:6,None,:]
            covariance_error=float((new_cov-old_cov).abs().max())
            mu_error=float((m['mu']-a['mu']).abs().max())
            new_Q_error=float((m['Q'][6:]-a['Q'][6:]).abs().max())
            spectrum=torch.linalg.eigvalsh((new_cov+new_cov.transpose(1,2))/2)
            Wm,Gm,Hm=heads['AFFINE_MEAN'];Wa,Ga,Ha=heads['AFFINE']
            eigen=torch.linalg.eigvalsh(Gm)
            invariant=dict(old_covariance_max_error=covariance_error,affine_means_max_error=mu_error,
                new_moments_max_error=new_Q_error,covariance_min_eigenvalue=float(spectrum.min()),
                mean_rhs_max_error=float((Hm-Ha).abs().max()),G_min_eigenvalue=float(eigen[0]),
                G_condition=float(eigen[-1]/eigen[0]),head_relative_residual=float((Gm@Wm-Hm).norm()/Hm.norm()),
                old_second_traces=m['Q'][:6].diagonal(dim1=1,dim2=2).sum(1).tolist())
            invariant['passed']=(covariance_error<1e-12 and mu_error==0 and new_Q_error==0
                and invariant['mean_rhs_max_error']==0 and invariant['covariance_min_eigenvalue']>=-1e-8
                and invariant['G_min_eigenvalue']>0 and invariant['head_relative_residual']<1e-8)
            invariants[row]=invariant;save(public/'INVARIANT_CHECK.json',invariants)
            if not invariant['passed']:raise ValueError('Covariance intervention invariant failed')
            deltas={base:{k:cells[row+'_AFFINE_MEAN'][k]-cells[row+'_'+base][k]
                          for k in ('BA','old_BA','new_BA')} for base in ('SHIFT','AFFINE')}
            paired={};new=truth>=6
            for base in ('SHIFT','AFFINE'):
                pa,pb=scores[base].argmax(1),scores['AFFINE_MEAN'].argmax(1)
                paired[base]={}
                for group,mask in dict(all=np.ones(len(truth),dtype=bool),old=~new,new=new).items():
                    ca,cb=(pa==truth)&mask,(pb==truth)&mask
                    paired[base][group]=dict(n=int(mask.sum()),both_correct=int((ca&cb).sum()),
                        reference_only_correct=int((ca&~cb).sum()),candidate_only_correct=int((~ca&cb).sum()),
                        both_wrong=int((~ca&~cb&mask).sum()),prediction_changed=int(((pa!=pb)&mask).sum()))
            z=torch.from_numpy(val).double();ids=torch.from_numpy(np.flatnonzero(new))
            target=torch.from_numpy(truth[new]);baseline=z@heads['SHIFT'][0]
            competitor=baseline[ids,:6].argmax(1)
            margins={}
            for drift,(W,_,_) in heads.items():
                values=z@W;margins[drift]=float((values[ids,target]-values[ids,competitor]).mean())
            rows[row]=dict(deltas=deltas,paired=paired,new_sample_margin_means=margins,
                margin_reference='same strongest old competitor under SHIFT; sample weighted',
                new_not_below_shift=deltas['SHIFT']['new_BA']>=0,
                BA_above_shift=deltas['SHIFT']['BA']>0)
            rows[row]['gate_passed']=rows[row]['new_not_below_shift'] and rows[row]['BA_above_shift']
            print(json.dumps(dict(row=row,deltas=deltas,gate_passed=rows[row]['gate_passed'])),flush=True)
        passed=all(row['gate_passed'] for row in rows.values())
        result=dict(status='COMPLETE_DIAGNOSTIC',cycle='MEAN_ONLY_T4',cells=cells,rows=rows,
            gate_passed=passed,next_decision='FREEZE_FOR_FULL_TRAJECTORY_CONFIRMATION' if passed else 'KEEP_SHIFT_STOP_THIS_CANDIDATE',
            readout_cells=6,new_candidates=1,optimizer_updates=0,policy_updates=0,gpu_seconds=0,
            test_accessed=False,meta_used_for_fit=0,independent_confirmation=False,
            shared_history_across_rows=False,full_trajectory_attribution=False)
        save(public/'RESULTS.json',result)
        with (public/'SIX_CELL_METRICS.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['fixed_trajectory','readout','BA','old_BA','new_BA'])
            for row in ('SHIFT','AFFINE'):
                for drift in ('SHIFT','AFFINE','AFFINE_MEAN'):
                    writer.writerow([row,drift]+[cells[row+'_'+drift][k] for k in ('BA','old_BA','new_BA')])
        storage=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if storage>c['storage_limit_bytes']:raise ValueError('Storage cap exceeded')
        receipt=record('AWAITING_PUBLICATION',phase='CPU_READOUT_COMPLETE',storage_bytes=storage,
            gate_passed=passed,next_decision=result['next_decision'])
        save(public/'COMPLETION_RECEIPT.json',receipt)
    except BaseException as exc:
        record('INCOMPLETE',phase='STOPPED',error=str(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
