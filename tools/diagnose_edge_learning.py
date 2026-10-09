"""Frozen medical checkpoints: learning support and paired head diagnostics only."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import core1_competition as competition
import prototype_coherent as native
from run_multilabel import ApartFeatures
from run_prototype_coherent import fit_split
from run_prototype_single import Images, extract, manifests, save


def summarize(scores, labels, seen, identities):
    if scores.shape != (len(labels), len(seen)) or not np.isfinite(scores).all():
        raise ValueError('Invalid score matrix')
    truth = np.asarray([seen.index(int(c)) for c in labels])
    prediction = scores.argmax(1)
    own = scores[np.arange(len(scores)), truth]
    other = scores.copy(); other[np.arange(len(scores)), truth] = -np.inf
    margins = own-other.max(1)
    ranks = 1+(scores > own[:, None]).sum(1)
    loss = .5*((scores-np.eye(len(seen))[truth])**2).sum(1)
    per_class = {}
    for i, c in enumerate(seen):
        keep = truth == i
        if not keep.any():
            continue
        wrong = keep & (prediction != i)
        per_class[str(c)] = dict(n=int(keep.sum()),
            identity_groups=len({identities[j] for j in np.flatnonzero(keep)}),
            correct=int((prediction[keep] == i).sum()),
            top3=int((ranks[keep] <= min(3, len(seen))).sum()),
            mean_true_rank=float(ranks[keep].mean()), mean_true_score=float(own[keep].mean()),
            mean_margin=float(margins[keep].mean()),
            margin_quantiles=np.quantile(margins[keep], [.1, .5, .9]).tolist(),
            mean_square_risk=float(loss[keep].mean()),
            wrong_destinations={str(seen[j]): int((wrong & (prediction == j)).sum())
                                for j in range(len(seen)) if (wrong & (prediction == j)).any()})
    return dict(n=len(labels), per_class=per_class,
        balanced_accuracy=float(np.mean([v['correct']/v['n'] for v in per_class.values()])),
        macro_square_risk=float(np.mean([v['mean_square_risk'] for v in per_class.values()])))


def paired(reference, candidate, labels, seen):
    truth = np.asarray([seen.index(int(c)) for c in labels])
    a, b = reference.argmax(1), candidate.argmax(1)
    return {str(c): dict(n=int((truth == i).sum()),
        helped=int(((truth == i) & (a != i) & (b == i)).sum()),
        hurt=int(((truth == i) & (a == i) & (b != i)).sum()),
        prediction_changed=int(((truth == i) & (a != b)).sum())) for i, c in enumerate(seen)}


def run(config):
    if config['dataset'] not in ('HK', 'ISIC'):
        raise ValueError('Medical diagnostics only; no test evaluation')
    source = Path(config['run'])
    original = json.loads((source/'INPUT.private.json').read_text())
    if original['evaluation_split'] != 'development_validation':
        raise ValueError('Unexpected evaluation split')
    root = Path(config['output']); root.mkdir(parents=True, exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    completed = []
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds'] or time.time() >= config['original_deadline']:
            raise TimeoutError('Original deadline or diagnostic residence exceeded')
    def status(state, **extra):
        save(root/'STATUS.json', dict(status=state, completed_states=completed,
            elapsed_seconds=time.monotonic()-started, cpu_process_seconds=time.process_time()-cpu,
            optimizer_updates=0, policy_updates=0, test_accessed=False, **extra))
    try:
        status('RUNNING', phase='initialize')
        torch.set_num_threads(4); seed=original['seed']
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        encoder=ApartFeatures(original['legacy_repo'], original['weight'], len(original['order']), 'cuda:0', seed)
        encoder.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in encoder.parameters())
        train, val=manifests(original)
        expected=json.loads((source/'metrics.json').read_text())
        partition, introduced={}, {}
        cursor=0
        for task, size in enumerate(original['task_sizes'], 1):
            classes=original['order'][cursor:cursor+size]; cursor+=size
            rows=[r for r in train if r['label'] in classes]
            fi, mi=fit_split(rows, original['split_seed'])
            assert not ({rows[i]['identity_component'] for i in fi} & {rows[i]['identity_component'] for i in mi})
            for i in fi: partition[id(rows[i])]='fit'
            for i in mi: partition[id(rows[i])]='meta'
            introduced.update({c: task for c in classes})
        from run_medical_v2 import transform
        def get(rows, task):
            loader=DataLoader(Images(rows, original['images'], transform(False), seed+task*100003),
                batch_size=64, num_workers=4, multiprocessing_context='spawn',
                generator=torch.Generator().manual_seed(seed+task*2003))
            return extract(encoder, loader, budget)
        result=dict(dataset=config['dataset'], arm=original['method'], states={}, optimizer_updates=0,
            policy_updates=0, test_accessed=False, independent_confirmation=False,
            meta_already_used_in_original_refit=True, historical_training_images_diagnostic_only=True,
            scope='Fixed saved encoders and banks; head-only interventions, not retrained methods',
            unavailable='No epoch encoder/features saved: selection-to-update causal chain cannot be reconstructed')
        for task in config['stages']:
            budget(); status('RUNNING', phase='checkpoint', task=task)
            state=torch.load(source/f'stage_{task}.pt', map_location='cpu', weights_only=False)
            seen=state['seen']; assert seen==original['order'][:sum(original['task_sizes'][:task])]
            encoder.load_state_dict(state['adapter'], strict=False)
            assert not any(p.requires_grad for p in encoder.parameters())
            bank={k:v.cuda() if torch.is_tensor(v) else v for k,v in state['bank'].items()}
            bank['components']=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in c.items()}
                                for c in state['bank']['components']]
            with torch.no_grad():
                native_head, metric=native.head(bank, 1.)
                base=competition.competition(bank, native_head)
                fixed, fixed_audit=competition.bank_head(bank, base, regularizer=metric)
                replay, replay_audit=competition.bank_head(bank, state['boundary_pair_weights'].cuda(), regularizer=metric)
            saved=state['head'].numpy()
            error=float(np.max(np.abs(replay.float().cpu().numpy()-saved)))
            if not np.allclose(replay.float().cpu().numpy(), saved, atol=1e-6, rtol=1e-5):
                raise ValueError('Saved action head reconstruction failed')
            heads=dict(saved=saved, native=native_head.float().cpu().numpy(),
                       static_pc=fixed.float().cpu().numpy(), action_replay=replay.float().cpu().numpy())
            record=dict(task=task, seen=seen, reconstruction_max_head_error=error,
                solves=dict(static_pc=fixed_audit, action_replay=replay_audit), splits={},
                prototype_support={str(c):dict(introduced_task=introduced[c],
                    training_n=bank['n'][i], component_counts=[p['count'] for p in bank['components'] if p['label']==i])
                    for i,c in enumerate(seen)})
            training=[r for r in train if r['label'] in seen]
            validation=[r for r in val if r['label'] in seen]
            for split, rows in [('train',training), ('development',validation)]:
                status('RUNNING', phase=split+'_features', task=task)
                z,y=get(rows,task)
                assert np.array_equal(y, np.asarray([r['label'] for r in rows]))
                scores={name:z@head for name,head in heads.items()}
                if not np.array_equal(scores['saved'].argmax(1), scores['action_replay'].argmax(1)):
                    raise ValueError('Saved action predictions differ after reconstruction')
                views={'development':np.ones(len(rows),dtype=bool)} if split=='development' else {
                    'fit':np.asarray([partition[id(r)]=='fit' for r in rows]),
                    'meta':np.asarray([partition[id(r)]=='meta' for r in rows])}
                for view, keep in views.items():
                    identities=[rows[i]['identity_component'] for i in np.flatnonzero(keep)]
                    record['splits'][view]=dict(heads={name:summarize(s[keep],y[keep],seen,identities)
                        for name,s in scores.items()}, paired_to_saved={name:paired(scores['saved'][keep],s[keep],y[keep],seen)
                        for name,s in scores.items() if name!='saved'})
                if split=='development':
                    observed=record['splits']['development']['heads']['saved']['per_class']
                    reference=expected['stages'][task-1]
                    for c,v in observed.items():
                        assert v['n']==reference['per_class_n'][c]
                        assert abs(v['correct']/v['n']-reference['per_class_recall'][c])<1e-12
                    record['saved_development_reproduced']=True
                else:
                    transport={}
                    with torch.no_grad():
                        w=torch.from_numpy(saved).cuda().double()
                        for i,c in enumerate(seen):
                            budget(); zz=torch.from_numpy(z[y==c]).cuda().double()
                            mu=zz.mean(0); q=zz.T@zz/len(zz)
                            def risk(mean,second):
                                return .5*((w*(second@w)).sum()-2*(mean@w[:,i])+1)
                            transport[str(c)]=dict(is_old=introduced[c]<task,
                                mean_relative_error=float((mu-bank['mu'][i]).norm()/mu.norm().clamp_min(1e-12)),
                                second_relative_error=float((q-bank['Q'][i]).norm()/q.norm().clamp_min(1e-12)),
                                observed_square_risk=float(risk(mu,q)),
                                stored_square_risk=float(risk(bank['mu'][i],bank['Q'][i])))
                    record['stored_vs_observed_training_moments']=transport
                del z,scores
            result['states'][str(task)]=record; completed.append(task)
            save(root/'RESULTS.json',result)
            del state, bank, heads, native_head, metric, fixed, replay
            torch.cuda.empty_cache()
        status('COMPLETE', ended=time.time(), peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status('INCOMPLETE', error=str(exc)); raise


def self_check():
    scores=np.array([[2.,1.,0.],[0.,2.,1.],[2.,1.,0.],[0.,1.,2.]])
    labels=np.array([10,20,30,30]);seen=[10,20,30]
    s=summarize(scores,labels,seen,['a','b','c','c'])
    assert s['per_class']['30']['correct']==1 and s['per_class']['30']['identity_groups']==1
    assert s['per_class']['30']['mean_true_rank']==2 and s['per_class']['30']['mean_margin']==-.5
    assert s['per_class']['30']['wrong_destinations']=={'10':1}
    changed=scores.copy();changed[2]=[0.,1.,3.]
    p=paired(scores,changed,labels,seen)['30']
    assert p==dict(n=2,helped=1,hurt=0,prediction_changed=1)
    try:summarize(scores*np.nan,labels,seen,['a','b','c','c'])
    except ValueError:pass
    else:raise AssertionError('Nonfinite input accepted')
    print('PASS: class scores/ranks/margins, identity counts, paired changes, nonfinite rejection')


if __name__=='__main__':
    config=json.load(sys.stdin)
    if config.get('self_check'):self_check()
    else:run(config)
