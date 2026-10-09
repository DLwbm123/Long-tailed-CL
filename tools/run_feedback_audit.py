"""Frozen-feature reward audit and paired 2x2 final-head interventions."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import edge_competition as edge
import prototype_coherent as native
from diagnose_edge_learning import paired
from diagnose_holdout_feedback import rebuild, to_device
from run_action_trace import actions, head_for
from run_prototype_coherent import common_shift
from run_prototype_single import save


def parts(values, baseline, old, tail):
    gain = (values.new_tensor(baseline)-values)/values.new_tensor(baseline).clamp_min(.1)
    negative = (-gain).clamp_min(0)
    terms = dict(mean_gain=float(gain.mean()), old_penalty=float(negative[:old].mean()),
                 new_penalty=float(negative[old:].mean()),
                 tail_penalty=float(negative[tail].mean()) if tail else 0.)
    terms['risk_reward'] = terms['mean_gain']-sum(terms[k] for k in ('old_penalty', 'new_penalty', 'tail_penalty'))
    terms['class_normalized_gains'] = gain.tolist()
    return terms


def metrics(scores, labels, seen, current):
    truth = np.asarray([seen.index(int(c)) for c in labels]); scores = np.asarray(scores)
    assert scores.shape == (len(labels), len(seen)) and np.isfinite(scores).all()
    target = np.eye(len(seen))[truth]; risks = .5*np.square(scores-target).sum(1)
    own = scores[np.arange(len(scores)), truth]; other = scores.copy()
    other[np.arange(len(scores)), truth] = -np.inf
    classes = {}
    for i, c in enumerate(seen):
        keep = truth == i
        if keep.any():
            classes[str(c)] = dict(n=int(keep.sum()), correct=int((scores[keep].argmax(1) == i).sum()),
                recall=float((scores[keep].argmax(1) == i).mean()), risk=float(risks[keep].mean()),
                true_score=float(own[keep].mean()), score_energy=float((.5*scores[keep]**2).sum(1).mean()),
                margin=float((own-other.max(1))[keep].mean()))
    views = {}
    for name, ids in [('all', seen), ('old', [c for c in seen if c not in current]), ('current', current)]:
        rows = [classes[str(c)] for c in ids if str(c) in classes]
        if rows:
            views[name] = dict(BA=float(np.mean([c['recall'] for c in rows])),
                               risk=float(np.mean([c['risk'] for c in rows])))
    return dict(per_class=classes, views=views)


def contrasts(cells):
    result = {}
    for name, a, b in [('total', 'B11', 'B00'), ('head_baseline', 'B01', 'B00'),
                       ('path_zero_head', 'B10', 'B00'), ('head_action_encoder', 'B11', 'B10')]:
        result[name] = {view: {key: cells[a]['development']['views'][view][key]-cells[b]['development']['views'][view][key]
                       for key in ('BA', 'risk')} for view in ('all', 'old', 'current')}
    result['interaction'] = {view: {key: result['total'][view][key]-result['head_baseline'][view][key]-result['path_zero_head'][view][key]
                             for key in ('BA', 'risk')} for view in ('all', 'old', 'current')}
    return result


def self_check():
    v = torch.tensor([.2, .4, .1], dtype=torch.float64); b = torch.tensor([.3, .3, .2], dtype=torch.float64)
    p = parts(v, b.tolist(), 2, [1, 2])
    assert abs(p['risk_reward']-float(edge.reward(v, b, 2, [1, 2]))) < 1e-14
    x = np.array([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    m = metrics(x, np.array([0, 0, 1, 1]), [0, 1], [1])
    assert m['views']['all']['BA'] == .25 and m['views']['all']['risk'] == .75
    cells = {key: {'development': {'views': {g: {'BA': val, 'risk': val*2} for g in ('all', 'old', 'current')}}}
             for key, val in [('B00', .2), ('B01', .4), ('B10', .3), ('B11', .6)]}
    c = contrasts(cells)
    assert abs(c['interaction']['all']['BA']-.1) < 1e-12
    # The same moments can imply different classification rates.
    assert np.isclose(np.mean([-1., 1.]), .8*(-.5)+.2*2)
    assert np.isclose(np.mean(np.square([-1., 1.])), .8*.25+.2*4)


@torch.no_grad()
def run(config):
    started, cpu = time.monotonic(), time.process_time(); torch.set_num_threads(4)
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=False)
    source = Path(config['source_run']); completed = []; max_residual = 0.; checks = []
    def status(state, **extra):
        save(out/'STATUS.json', dict(status=state, completed_candidates=completed,
             elapsed_seconds=time.monotonic()-started, cpu_process_seconds=time.process_time()-cpu,
             adapter_updates=0, policy_updates=0, image_reads=0, test_accessed=False, **extra))
    def budget():
        if time.monotonic()-started >= config['max_wall_seconds'] or time.time() >= config['original_deadline']:
            raise TimeoutError('Frozen diagnostic deadline exceeded')
    def load(path):
        budget(); return torch.load(path, map_location='cpu', weights_only=False)
    def scores(features, head):
        return np.asarray(features) @ head.float().cpu().numpy()
    def checked(head, saved, observed, expected, label):
        error = float((head.cpu()-saved).abs().max())
        assert torch.allclose(head.cpu(), saved, atol=1e-6, rtol=1e-5), label
        assert np.allclose(observed, expected, atol=1e-6, rtol=1e-5), label
        assert np.array_equal(observed.argmax(1), expected.argmax(1)), label
        checks.append(dict(label=label, head_max_error=error, score_max_error=float(np.abs(observed-expected).max())))
    try:
        status('RUNNING', phase='initialize')
        original_config = json.loads((source/'INPUT.private.json').read_text())
        original = json.loads((Path(original_config['run'])/'INPUT.private.json').read_text())
        results = json.loads((source/'RESULTS.json').read_text())
        assert json.loads((source/'STATUS.json').read_text())['status'] == 'COMPLETE'
        assert original['evaluation_split'] == 'development_validation' and results['dataset'] in ('ISIC', 'HK')
        assert sorted(map(int, results['candidates'])) == config['candidate_ids']
        seen, current = results['seen'], results['current_classes']; old_count = len(seen)-len(current)
        with (Path(original['manifest'])/'train.csv').open() as stream:
            counts = {c: 0 for c in original['order']}
            for row in csv.DictReader(stream):
                assert row['split'] == 'train'; counts[int(row['original_label'])] += 1
        ranked = sorted(original['order'], key=lambda c: (-counts[c], c))
        tail = [i for i, c in enumerate(seen) if c in ranked[len(ranked)//2:]]
        previous = load(Path(original_config['run'])/'stage_1.pt')
        old = to_device(previous['bank'], 'cuda'); del previous
        initial = load(source/'INITIAL.private.pt'); before = initial['features'].cuda()
        raw_y = initial['labels']; y = torch.tensor([seen.index(int(c)) for c in raw_y], device='cuda')
        fit = torch.tensor(initial['fit_indices'], device='cuda'); meta = torch.tensor(initial['meta_indices'], device='cuda')
        zero = load(source/'candidate_00/TRACE.private.pt')
        zero_dev = load(source/'candidate_00/DEVELOPMENT.private.pt')
        initial_dev = load(source/'INITIAL_DEVELOPMENT.private.pt')
        assert np.array_equal(initial_dev['labels'], zero_dev['labels'])
        baseline_banks = {}; base_cells = {}; base_scores = {}
        for partition, include in [('fit', False), ('all', True)]:
            bank, _ = rebuild(old, before, zero['post_features'].cuda(), y, fit, include)
            w, ref, _, base, pairs, residual = head_for(bank, actions()[0], old_count)
            max_residual = max(max_residual, residual); baseline_banks[partition] = bank
            s = scores(zero_dev['features'], w)
            checked(w, zero[partition+'_refit'], s, zero_dev['scores'][partition+'_refit'], 'zero_'+partition)
            use = torch.arange(len(y), device='cuda') if include else fit
            shifted = native.translate(old, common_shift(before[use], zero['post_features'].cuda()[use], y[use]))
            proxy = edge.risks(shifted, w, zero['post_features'].cuda()[meta], y[meta]).tolist()
            base_cells[partition] = dict(development=metrics(s, zero_dev['labels'], seen, current),
                meta=metrics(scores(zero['post_features'][meta.cpu()], w), raw_y[meta.cpu()], seen, current), proxy_class_risks=proxy)
            base_scores[partition] = s
        records = {}
        for index in config['candidate_ids']:
            budget(); status('RUNNING', phase='cross_readout', active_candidate=index)
            trace = load(source/f'candidate_{index:02d}/TRACE.private.pt')
            dev = load(source/f'candidate_{index:02d}/DEVELOPMENT.private.pt')
            assert np.array_equal(dev['labels'], zero_dev['labels']) and np.array_equal(trace['labels'], raw_y)
            after = trace['post_features'].cuda(); saved = results['candidates'][str(index)]
            initial_feedback = saved['initial_feedback']
            # Preserve original double precision when checking near-zero rewards.
            initial_parts = parts(torch.tensor(initial_feedback['class_risks'], dtype=torch.float64),
                 initial_feedback['reference_class_risks'], old_count, tail)
            initial_parts['kl_penalty'] = .01*initial_feedback['allocation_kl']
            initial_parts['reward'] = initial_parts['risk_reward']-initial_parts['kl_penalty']
            assert abs(initial_parts['reward']-initial_feedback['reward']) < 1e-12
            initial_parts['class_risks'] = initial_feedback['class_risks']
            initial_parts['reference_class_risks'] = initial_feedback['reference_class_risks']
            initial_parts['development'] = metrics(dev['scores']['immediate'], dev['labels'], seen, current)
            initial_parts['reference_development'] = metrics(zero_dev['scores']['immediate'], dev['labels'], seen, current)
            record = dict(initial=initial_parts, partitions={})
            for partition, include in [('fit', False), ('all', True)]:
                budget(); bank, _ = rebuild(old, before, after, y, fit, include)
                w11, w10, _, base, pairs, residual = head_for(bank, actions()[index], old_count)
                w01, _, _, _, _, r01 = head_for(baseline_banks[partition], actions()[index], old_count)
                max_residual = max(max_residual, residual, r01)
                all_scores = dict(B00=base_scores[partition], B01=scores(zero_dev['features'], w01),
                                  B10=scores(dev['features'], w10), B11=scores(dev['features'], w11))
                checked(w11, trace[partition+'_refit'], all_scores['B11'], dev['scores'][partition+'_refit'], f'{index}_{partition}')
                use = torch.arange(len(y), device='cuda') if include else fit
                shifted = native.translate(old, common_shift(before[use], after[use], y[use]))
                shifted0 = native.translate(old, common_shift(before[use], zero['post_features'].cuda()[use], y[use]))
                cells = dict(B00=base_cells[partition])
                for name, w, features, history in [('B01', w01, zero['post_features'].cuda(), shifted0),
                                                  ('B10', w10, after, shifted), ('B11', w11, after, shifted)]:
                    cells[name] = dict(development=metrics(all_scores[name], dev['labels'], seen, current),
                        meta=metrics(scores(features[meta].cpu(), w), raw_y[meta.cpu()], seen, current),
                        proxy_class_risks=edge.risks(history, w, features[meta], y[meta]).tolist())
                within = parts(torch.tensor(cells['B11']['proxy_class_risks'], dtype=torch.float64), cells['B10']['proxy_class_risks'], old_count, tail)
                within['kl_penalty'] = .01*float(edge.action_kl(pairs, base))
                within['reward'] = within['risk_reward']-within['kl_penalty']
                assert abs(within['reward']-saved['post_feedback'][partition+'_refit']['reward']) < 1e-10
                branch = parts(torch.tensor(cells['B11']['proxy_class_risks'], dtype=torch.float64), cells['B00']['proxy_class_risks'], old_count, tail)
                record['partitions'][partition] = dict(cells=cells, contrasts=contrasts(cells),
                    same_encoder_feedback=within, branch_risk_feedback_without_KL=branch,
                    meta_is_held_out=not include, paired={key: paired(all_scores[b], all_scores[a], dev['labels'], seen)
                    for key, a, b in [('total','B11','B00'), ('head_baseline','B01','B00'), ('path_zero_head','B10','B00'), ('head_action_encoder','B11','B10')]})
                if index == 0:
                    assert all(np.array_equal(all_scores['B00'], s) for s in all_scores.values())
                    assert within['reward'] == branch['risk_reward'] == 0
            records[str(index)] = record; completed.append(index)
            save(out/'RESULTS.json', dict(dataset=results['dataset'], seen=seen, current=current, tail=[seen[i] for i in tail],
                candidates=records, checks=checks, maximum_solve_residual=max_residual,
                adapter_updates=0, policy_updates=0, image_reads=0, test_accessed=False,
                independent_confirmation=False, old_proxy_vs_development_gap_is_not_transport_error=True))
        assert max_residual <= 1e-8
        status('COMPLETE', maximum_solve_residual=max_residual, reconstruction_checks=len(checks))
    except BaseException as exc:
        status('INCOMPLETE', error=str(exc)); raise


if __name__ == '__main__':
    config = json.load(sys.stdin)
    if config.get('self_check'):
        started = time.process_time(); self_check()
        print(json.dumps(dict(status='PASS', cpu_seconds=time.process_time()-started)))
    else:
        run(config)
