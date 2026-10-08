"""Audit completed FDRL1 receipts and saved predictions; no images or model forwards."""
import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import time

import numpy as np


def run(root):
    started=time.process_time();root=Path(root);public=root/'public'
    def read(p):return json.loads(p.read_text())
    def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    result=read(public/'main/RESULTS.json');ledger=read(public/'BUDGET_LEDGER.json');lock=read(public/'PROTOCOL_LOCK.json');config=read(root/'cycle.private.json')
    arms=lock['arms'];names=['main_'+a for a in arms];jobs=ledger['jobs']
    assert ledger['formal_ids']==names and len(jobs)==19 and ledger['gpu_reserved_seconds']==0
    assert not result['failures'] and all(j['exit_code']==0 for j in jobs)
    training=[j for j in jobs if j['kind']=='train'];evaluation=[j for j in jobs if j['kind']=='eval']
    assert len(training)==len(evaluation)==9
    latest_train=max(j['started']+j['elapsed_seconds'] for j in training)
    assert min(j['started'] for j in evaluation)>=latest_train
    assert ledger['total_gpu_seconds']<=lock['gpu_seconds_limit'] and ledger['diagnostic_gpu_seconds']<=lock['diagnostic_seconds_limit']
    assert abs(sum(j['elapsed_seconds'] for j in jobs)-ledger['total_gpu_seconds'])<1e-9
    with (Path(config['base']['manifest'])/'train.csv').open() as f:train=list(csv.DictReader(f))
    membership=read(Path(config['split_file']))['membership'];counts=Counter(int(r['original_label']) for r in train)
    tail=set(sorted(lock['order'],key=lambda c:(-counts[c],c))[4:])
    fit=[r for r in train if membership[r['relative_path']]=='fit'];meta=[r for r in train if membership[r['relative_path']]=='meta']
    assert len(fit)==15122 and len(meta)==3596 and not ({r['identity_component'] for r in fit}&{r['identity_component'] for r in meta})
    decisions={};metrics={};audits={};predictions={}
    def reward(before,after,old,tail_ids):
        gain=np.asarray(before)-np.asarray(after)
        return 100*(gain.mean()+np.minimum(gain[:old],0).mean()+np.minimum(gain[old:],0).mean()+np.minimum(gain[tail_ids],0).mean())
    for name in names:
        status=read(root/name/'STATUS.json');ev=read(root/('eval_'+name)/'STATUS.json')
        assert status['status']=='TRAINED' and ev['status']=='COMPLETE' and ev['optimizer_updates']==0
        assert (status['steps'],status['actual_updates'],status['retained_updates'],status['rollout_updates'],status['warmup_updates'])==(476,928,200,728,128)
        assert status['policy_updates']==(10 if name in ('main_FDRL','main_SHUFFLED') else 4)
        rows=read(root/name/'CONTROLLER.json')['decisions'];decisions[name]=rows
        assert len(rows)==28 and sum(r['horizon'] for r in rows)==200
        residual=0.;max_reward_error=0.;entropy=[]
        for i,r in enumerate(rows):
            task=r['task'];k=2*task;old=k-2;tail_ids=[j for j,c in enumerate(lock['order'][:k]) if c in tail]
            p=np.asarray(r['probabilities']);assert p.shape==(3,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-12
            entropy.append(-sum(float(v)*math.log(float(v)) for v in p if v>0)/math.log(3))
            risk=np.asarray(r['branch_risks']);assert risk.shape==(3,k) and np.isfinite(risk).all()
            reconstructed=reward(r['before_risk'],r['after_risk'],old,tail_ids)+r['terminal_reward']
            error=abs(reconstructed-r['reward']);max_reward_error=max(max_reward_error,error);assert error<1e-10
            assert abs(r['reward']-r['branch_rewards'][r['selected_action']])<=1e-8
            assert np.allclose(r['after_risk'],risk[r['selected_action']],atol=1e-10,rtol=1e-10)
            if not (r['epoch']==2 and (i+1==len(rows) or rows[i+1]['task']!=task)):
                assert np.allclose([reward(r['before_risk'],v,old,tail_ids) for v in risk],r['branch_rewards'],atol=1e-10,rtol=1e-10)
            residual=max(residual,r['max_residual']);assert residual<=1e-8
            if name=='main_SHUFFLED':assert any(r['policy_state']==v['state'] for v in rows[:i+1])
            if name=='main_GREEDY':assert r['reward']>=max(r['branch_rewards'])-1e-12
        boundary=[]
        for task in (2,3,4):
            b=read(root/name/f'boundary_{task}.json');seen=lock['order'][:2*task]
            assert b['fit_only'] and b['meta_prototypes']==0
            assert b['fit_counts']==[sum(int(r['original_label'])==c for r in fit) for c in seen]
            assert b['meta_counts']==[sum(int(r['original_label'])==c for r in meta) for c in seen]
            residual=max(residual,b['solve']['relative_residual']);assert residual<=1e-8
            boundary.append(dict(task=task,fit_counts=b['fit_counts'],meta_counts=b['meta_counts']))
        warm=read(root/name/'WARMUP.json');assert warm['adapter_updates']==128 and len(warm['episodes'])==4 and not warm['validation_accessed']
        diag=read(root/name/'diagnostics.json')
        for episode in warm['episodes']+[d['policy_update'] for d in diag if d['policy_update'] is not None]:
            # Online reward vectors live in the corresponding task/epoch controller rows.
            if 'rewards' in episode:rewards=episode['rewards']
            else:
                d=next(d for d in diag if d['policy_update'] is episode)
                rewards=[r['reward'] for r in rows if (r['task'],r['epoch'])==(d['task'],d['epoch'])]
            returns=[sum(.95**(j-i)*rewards[j] for j in range(i,len(rewards))) for i in range(len(rewards))]
            assert np.allclose(returns,episode['returns'],atol=1e-12,rtol=1e-12)
        record=result['records'][name];m=record['metrics'];last=m['stages'][-1];all_predictions=[]
        assert abs(m['final_balanced_accuracy']-last['balanced_accuracy'])<1e-12
        for stage in m['stages']:
            t=stage['task']
            with np.load(root/('eval_'+name)/(name+f'_stage_{t}.private.npz')) as z:
                y=z['labels'];prediction=z['prediction'];all_predictions.append(prediction.copy())
                recalls={str(c):float(np.mean(prediction[y==c]==c)) for c in stage['seen']}
            assert recalls==stage['per_class_recall'] and abs(np.mean(list(recalls.values()))-stage['balanced_accuracy'])<1e-12
        predictions[name]=all_predictions
        forgetting=[]
        for c in lock['order']:
            hist=[s['per_class_recall'][str(c)] for s in m['stages'] if str(c) in s['per_class_recall']]
            if len(hist)>1:forgetting.append(max(hist[:-1])-hist[-1])
        assert abs(np.mean(forgetting)-m['forgetting'])<1e-12
        assert abs(np.mean([s['balanced_accuracy'] for s in m['stages']])-m['average_incremental_balanced_accuracy'])<1e-12
        assert abs(np.mean([last['per_class_recall'][str(c)] for c in tail])-m['final_tail_recall'])<1e-12
        metrics[name]=dict(ba=m['final_balanced_accuracy'],average_ba=m['average_incremental_balanced_accuracy'],tail=m['final_tail_recall'],forgetting=m['forgetting'],new_recall=float(np.mean([last['per_class_recall'][str(c)] for c in last['seen'][-2:]])))
        audits[name]=dict(decisions=len(rows),action_counts=dict(Counter(str(int(r['fd_weight'])) for r in rows)),action_sequence=[int(r['fd_weight']) for r in rows],policy_updates=status['policy_updates'],
            last_behavior_probabilities=rows[-1]['probabilities'],mean_normalized_entropy=float(np.mean(entropy)),last_normalized_entropy=entropy[-1],
            max_solve_residual=residual,max_reward_reconstruction_error=max_reward_error,
            chosen_proxy_best_count=sum(r['reward']>=max(r['branch_rewards'])-1e-12 for r in rows),
            mean_proxy_best_gap=float(np.mean([max(r['branch_rewards'])-r['reward'] for r in rows])),boundary_counts=boundary)
    for a in ('FD5','FD10','FD20'):
        assert all(r['fd_weight']==float(a[2:]) for r in decisions['main_'+a])
    rl=decisions['main_FDRL'];shuffle=decisions['main_SHUFFLED']
    same_actions=sum(a['selected_action']==b['selected_action'] for a,b in zip(rl,shuffle))
    same_predictions=[bool(np.array_equal(a,b)) for a,b in zip(predictions['main_FDRL'],predictions['main_SHUFFLED'])]
    differences={name:{k:100*(metrics['main_FDRL'][k]-v[k]) for k in v} for name,v in metrics.items() if name!='main_FDRL'}
    d=differences['main_R'];f=differences['main_FD10']
    checks=dict(ba_gain=d['ba']>=1,tail=d['tail']>=-.5,forgetting=d['forgetting']<=1,new_recall=d['new_recall']>=-1,
        fd10_tail=f['tail']>=-.5,fd10_forgetting=f['forgetting']<=1,fd10_new_recall=f['new_recall']>=-1)
    checks.update({'above_'+a:differences['main_'+a]['ba']>0 for a in lock['screen']['strictly_above']})
    assert all(checks.values())==result['screen_success']==False
    completed=max(j['started']+j['elapsed_seconds'] for j in jobs);assert completed<lock['deadline']
    summary=dict(cycle='FDRL1',status='STOPPED_SCREEN_FAILED',phase='CYCLE_CLOSED',source_commit=config['source_commit'],
        completed_at=completed,completed_at_shanghai=datetime.fromtimestamp(completed,timezone(timedelta(hours=8))).isoformat(),
        screen_success=False,metrics=metrics,differences_pp=differences,screen_checks=checks,failed_checks=[k for k,v in checks.items() if not v],
        controllers=audits,shuffled_comparison=dict(same_actions=same_actions,total_actions=28,identical_predictions_by_stage=same_predictions,
            changed_inputs=sum(a['state']!=b['policy_state'] for a,b in zip(rl,shuffle)),
            max_probability_difference=max(abs(x-y) for a,b in zip(rl,shuffle) for x,y in zip(a['probabilities'],b['probabilities']))),
        gpu_seconds_used=ledger['total_gpu_seconds'],gpu_reserved_seconds=0,formal_attempts=9,actual_optimizer_updates=8352,
        retained_updates=1800,counterfactual_updates=5400,warmup_updates=1152,policy_updates=sum(v['policy_updates'] for v in audits.values()),
        failures=[],unrun=['new seeds','method revision','HyperKvasir transfer'],unrun_reason='frozen main screen failed',
        test_accessed=False,independent_confirmation=False,paper_evidence_ready=False)
    cost=time.process_time()-started
    save(public/'COMPLETION_AUDIT.json',dict(status='PASS',cpu_core_seconds=cost,gpu_seconds=0,new_scientific_updates=0,
        actual_update_accounting=True,all_evaluations_after_all_training=True,fit_meta_identity_disjoint=True,boundary_counts_match_split=True,
        independent_block_reward_and_discounted_return_recalculation=True,saved_prediction_metrics_recalculated=True,
        audit_limit='Terminal full-data reward components checked against recorded values and runtime branch assertions; no fresh image/model forward.',
        controllers=audits,source_commit=config['source_commit']))
    summary['completion_audit_cpu_core_seconds']=cost
    save(public/'FINAL_SUMMARY.json',summary);save(public/'FINAL_BUDGET_LEDGER.json',ledger)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('controllers','metrics')},indent=2))


if __name__=='__main__':run(json.load(sys.stdin)['root'])
