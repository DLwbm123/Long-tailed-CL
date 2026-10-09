"""Read completed paired traces; no image access or model/policy updates."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from diagnose_edge_learning import paired


CUTS=('immediate','encoder_only','trained','fit_refit','all_refit')


def ranks(values):
    x=np.asarray(values);return np.asarray([1+np.sum(x<v)+.5*(np.sum(x==v)-1) for v in x],dtype=float)


def correlation(a,b):
    a,b=ranks(a),ranks(b)
    if np.std(a)<1e-12 or np.std(b)<1e-12:return None
    return float(np.corrcoef(a,b)[0,1])


def choose(candidates):
    best=min(candidates,key=lambda r:(-r['initial_reward'],r['candidate']))
    return best['candidate'] if best['initial_reward']>1e-8 else 0


def readout(root):
    root=Path(root);lock=json.loads((root/'PROTOCOL_LOCK.json').read_text())
    public={};comparisons=[];costs={};baseline={};initial={};sequences={};residual=0.;updates=0;actual=0
    for dataset in ('ISIC','HK'):
        suite=json.loads((root/'preflight'/dataset/'STATUS.json').read_text())
        probe=json.loads((root/'preflight'/(dataset+'_probe')/'PREFLIGHT.json').read_text())
        assert suite['status']=='COMPLETE' and probe['status']=='PASS' and probe['adapter_updates']==probe['policy_updates']==0
        costs['preflight_'+dataset]=suite['elapsed_seconds']
    for job in lock['jobs']:
        name=job['name'];p=root/'runs'/name
        suite=json.loads((root/'formal'/name/'STATUS.json').read_text());status=json.loads((p/'STATUS.json').read_text())
        r=json.loads((p/'RESULTS.json').read_text());marker=json.loads((p/'TRAIN_COMPLETE.json').read_text())
        assert suite['status']==status['status']=='COMPLETE'
        assert status['completed_candidates']==status['evaluated_candidates']==marker['completed_candidates']==job['candidate_ids']
        assert status['adapter_updates']==len(job['candidate_ids'])*job['expected_steps_per_candidate']
        assert status['policy_updates']==r['policy_updates']==0 and not status['test_accessed'] and not r['test_accessed']
        assert r['development_after_all_training'] and r['initial_checkpoint_development_reproduced']
        assert sorted(map(int,r['candidates']))==job['candidate_ids'] and r['dataset']==job['dataset']
        costs['formal_'+name]=suite['elapsed_seconds'];updates+=status['adapter_updates'];actual+=len(r['candidates'])
        residual=max(residual,status['max_solve_residual']);public[name]=r
        initial[name]=torch.load(p/'INITIAL.private.pt',map_location='cpu',weights_only=False)
        sequences[name]=torch.load(p/'BATCH_ORDER.private.pt',map_location='cpu',weights_only=False)
        baseline[name]=torch.load(p/'candidate_00/DEVELOPMENT.private.pt',map_location='cpu',weights_only=False)
        base=r['candidates']['0'];labels=baseline[name]['labels'];seen=r['seen'];new=r['current_classes']
        for index,record in r['candidates'].items():
            assert record['adapter_updates']==job['expected_steps_per_candidate'] and record['training_batch_order_verified']
            if int(index)==0:assert record['initial_feedback']['reward']==0
            row=dict(dataset=job['dataset'],shard=name,candidate=int(index),initial_reward=record['initial_feedback']['reward'],cuts={},paired_to_zero={})
            values=torch.load(p/f'candidate_{int(index):02d}'/'DEVELOPMENT.private.pt',map_location='cpu',weights_only=False)
            assert np.array_equal(labels,values['labels'])
            for cut in CUTS:
                row['cuts'][cut]={};row['paired_to_zero'][cut]={}
                for view,keep in [('all',np.ones(len(labels),dtype=bool)),('current',np.isin(labels,new)),('old',~np.isin(labels,new))]:
                    a=record['development'][view]['heads'][cut];b=base['development'][view]['heads'][cut]
                    assert a['n']==b['n']==int(keep.sum()) and set(a['per_class'])==set(b['per_class'])
                    row['cuts'][cut][view]={metric:a[metric]-b[metric] for metric in ('balanced_accuracy','macro_square_risk')}
                    row['paired_to_zero'][cut][view]=paired(baseline[name]['scores'][cut][keep],values['scores'][cut][keep],labels[keep],seen)
            row['gain_changes']={a+'_to_'+b:{view:row['cuts'][b][view]['balanced_accuracy']-row['cuts'][a][view]['balanced_accuracy']
                                for view in ('all','current','old')} for a,b in zip(CUTS[:-1],CUTS[1:])}
            comparisons.append(row);del values
    assert updates==lock['expected_adapter_updates']==682 and actual==lock['expected_actual_forks']==35 and residual<=1e-8
    aa,bb=initial['ISIC_A'],initial['ISIC_B']
    duplicate=dict(reference_head_max_abs=float((aa['reference_head']-bb['reference_head']).abs().max()),
        initial_features_max_abs=float((aa['features']-bb['features']).abs().max()),
        training_sequence_identical=sequences['ISIC_A']==sequences['ISIC_B'],cuts={})
    assert np.array_equal(aa['labels'],bb['labels']) and aa['fit_indices']==bb['fit_indices'] and aa['meta_indices']==bb['meta_indices']
    for cut in CUTS:
        a,b=baseline['ISIC_A']['scores'][cut],baseline['ISIC_B']['scores'][cut]
        assert a.shape==b.shape and np.array_equal(baseline['ISIC_A']['labels'],baseline['ISIC_B']['labels'])
        duplicate['cuts'][cut]=dict(max_score_abs_difference=float(np.max(np.abs(a-b))),
            prediction_identical=bool(np.array_equal(a.argmax(1),b.argmax(1))),
            balanced_accuracy_difference=public['ISIC_A']['candidates']['0']['development']['all']['heads'][cut]['balanced_accuracy']-
                public['ISIC_B']['candidates']['0']['development']['all']['heads'][cut]['balanced_accuracy'])
    duplicate['mergeable']=bool(duplicate['training_sequence_identical'] and
        torch.allclose(aa['reference_head'],bb['reference_head'],atol=1e-6,rtol=1e-5) and
        torch.allclose(aa['features'],bb['features'],atol=1e-6,rtol=1e-5) and
        all(v['prediction_identical'] for v in duplicate['cuts'].values()))
    summaries={}
    for dataset in ('ISIC','HK'):
        rows=[r for r in comparisons if r['dataset']==dataset and not (r['shard']=='ISIC_B' and r['candidate']==0)]
        assert sorted(r['candidate'] for r in rows)==list(range(17))
        selected=choose(rows);oracle=min(rows,key=lambda r:(-r['cuts']['all_refit']['all']['balanced_accuracy'],r['candidate']))
        selected_row=next(r for r in rows if r['candidate']==selected)
        summaries[dataset]=dict(mergeable=dataset!='ISIC' or duplicate['mergeable'],proxy_selected=selected,
            finite_set_oracle_candidate=oracle['candidate'],oracle_is_privileged_diagnostic=True,
            proxy_selected_gains=selected_row['cuts'],finite_set_oracle_gains=oracle['cuts'],
            spearman_proxy_vs_final_BA=correlation([r['initial_reward'] for r in rows],[r['cuts']['all_refit']['all']['balanced_accuracy'] for r in rows]),
            reward_positive=sum(r['initial_reward']>1e-8 for r in rows),
            reward_positive_final_BA_positive=sum(r['initial_reward']>1e-8 and r['cuts']['all_refit']['all']['balanced_accuracy']>1e-12 for r in rows),
            reward_nonpositive_final_BA_positive=[r['candidate'] for r in rows if r['initial_reward']<=1e-8 and r['cuts']['all_refit']['all']['balanced_accuracy']>1e-12],
            pareto_candidates=[r['candidate'] for r in rows if r['cuts']['all_refit']['all']['balanced_accuracy']>1e-12 and
                min(r['cuts']['all_refit'][v]['balanced_accuracy'] for v in ('old','current'))>=-1e-12],
            immediate_gain_lost=[r['candidate'] for r in rows if r['cuts']['immediate']['all']['balanced_accuracy']>1e-12 and r['cuts']['all_refit']['all']['balanced_accuracy']<=1e-12])
        if not summaries[dataset]['mergeable']:
            summaries[dataset]['global_oracle_interpretation']='Unavailable: cross-GPU zero-action control disagrees; use within-shard contrasts only'
    return dict(results=public,comparisons=comparisons,summary=summaries,
        audit=dict(status='PASS',actual_forks=actual,unique_forks=34,adapter_updates=updates,policy_updates=0,
            maximum_solve_residual=residual,test_accessed=False,development_after_all_training=True,
            initial_checkpoint_development_reproduced=True,duplicate_zero=duplicate),
        budget=dict(prior_closed_gpu_seconds=lock['prior_closed_gpu_seconds'],suite_gpu_seconds=costs,
            round_gpu_seconds=sum(costs.values()),cumulative_gpu_seconds=lock['prior_closed_gpu_seconds']+sum(costs.values()),
            cpu_selfchecks_seconds=lock['cpu_selfchecks_seconds'],readout_selfcheck_cpu_seconds=10.586600667999999,failed_or_repair_attempts=[],
            original_deadline=lock['original_deadline'],note='Suite residence includes barrier waiting and evaluation; do not add worker elapsed again. CPU selfchecks/readout separate.'))


def self_check():
    assert np.array_equal(ranks([2,1,2]),[2.5,1,2.5])
    assert correlation([1,1],[1,2]) is None and abs(correlation([1,2,3],[3,2,1])+1)<1e-12
    assert choose([dict(candidate=0,initial_reward=0.),dict(candidate=1,initial_reward=-1.)])==0
    assert choose([dict(candidate=1,initial_reward=.1),dict(candidate=2,initial_reward=.1)])==1


if __name__=='__main__':
    config=json.load(sys.stdin);self_check()
    if config.get('self_check'):print('PASS: tied ranking, constant correlation, gated selection and deterministic ties')
    else:
        started=time.process_time();data=readout(config['root']);data['budget']['readout_cpu_seconds']=time.process_time()-started
        out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
        for file,key in [('RESULTS','results'),('COMPARISONS','comparisons'),('SUMMARY','summary'),('AUDIT','audit'),('BUDGET_LEDGER','budget')]:
            text=json.dumps(data[key],indent=2,ensure_ascii=False,allow_nan=False)+'\n'
            assert not any(x in text for x in ['/remote-home/','/Users/','identity_component'])
            (out/(file+'.json')).write_text(text)
        print(json.dumps({k:v for k,v in data.items() if k not in ('results','comparisons')},indent=2))
