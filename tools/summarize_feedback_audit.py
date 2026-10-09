"""Aggregate frozen feedback audits; only anonymous scalar results are read."""
import json
import math
from pathlib import Path
import sys


def mean(values):
    return sum(values)/len(values)


def correlation(a, b):
    def ranks(x):
        return [1+sum(z < v for z in x)+.5*(sum(z == v for z in x)-1) for v in x]
    a, b = ranks(a), ranks(b); ma, mb = mean(a), mean(b)
    denominator = math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return sum((x-ma)*(y-mb) for x, y in zip(a, b))/denominator if denominator else None


def risk_terms(values, baseline, seen, current, tail):
    gain = {c: (b-v)/max(b, .1) for c, v, b in zip(seen, values, baseline)}
    result = dict(mean_gain=mean(list(gain.values())))
    for name, group in [('old', [c for c in seen if c not in current]), ('new', current), ('tail', tail)]:
        result[name+'_penalty'] = mean([max(-gain[c], 0.) for c in group]) if group else 0.
    result['risk_reward'] = result['mean_gain']-sum(result[k+'_penalty'] for k in ('old', 'new', 'tail'))
    return result


def protections(gains, pairs):
    return dict(positive=gains['all']['BA'] > 1e-12,
        group=min(gains[g]['BA'] for g in ('old', 'current')) >= -1e-12,
        per_class=all(v['helped'] >= v['hurt'] for v in pairs.values()),
        per_sample=all(v['hurt'] == 0 for v in pairs.values()))


def best(rows, predicate, gain):
    eligible = [r for r in rows if predicate(r)]
    return min(eligible, key=lambda r: (-gain(r), r['candidate']))['candidate'] if eligible else None


def summarize(data, lock):
    actual = sum(len(r['candidates']) for r in data['results'].values())
    assert actual == lock['expected_actual_candidate_records'] == 35
    assert len(data['suites']) == len(data['statuses']) == 3
    for job in lock['jobs']:
        name = job['name']; s = data['statuses'][name]; r = data['results'][name]
        assert s['status'] == data['suites'][name]['status'] == 'COMPLETE'
        assert s['completed_candidates'] == job['candidate_ids'] == sorted(map(int, r['candidates']))
        assert s['adapter_updates'] == s['policy_updates'] == s['image_reads'] == 0 and not s['test_accessed']
        assert r['maximum_solve_residual'] <= 1e-8
    aa, bb = (data['results'][n]['candidates']['0'] for n in ('ISIC_A', 'ISIC_B'))
    assert aa == bb, 'Cross-GPU baseline aggregate mismatch; do not merge'
    output = {}; tables = {}; identities = 0
    for dataset in ('ISIC', 'HK'):
        rows = []
        for job in lock['jobs']:
            name = job['name']; result = data['results'][name]
            if job['dataset'] != dataset:
                continue
            seen, current, tail = result['seen'], result['current'], result['tail']
            for key, record in result['candidates'].items():
                index = int(key)
                if name == 'ISIC_B' and index == 0:
                    continue
                initial = record['initial']; d = initial['development']; ref = initial['reference_development']
                empirical = risk_terms([d['per_class'][str(c)]['risk'] for c in seen],
                    [ref['per_class'][str(c)]['risk'] for c in seen], seen, current, tail)
                row = dict(candidate=index, initial_reward=initial['reward'], initial_mean_gain=initial['mean_gain'],
                    initial_risk_reward=initial['risk_reward'], initial_development_mean_gain=empirical['mean_gain'],
                    initial_development_risk_reward=empirical['risk_reward'],
                    initial_development_BA_gain=d['views']['all']['BA']-ref['views']['all']['BA'], partitions={})
                for partition in ('fit', 'all'):
                    p = record['partitions'][partition]; c = p['contrasts']; cells = p['cells']
                    for view in ('all', 'old', 'current'):
                        for metric in ('BA', 'risk'):
                            assert abs(c['total'][view][metric]-sum(c[k][view][metric] for k in ('head_baseline','path_zero_head','interaction'))) < 1e-12
                            identities += 1
                    full = protections(c['total'], p['paired']['total'])
                    head = protections(c['head_baseline'], p['paired']['head_baseline'])
                    dev_risk = risk_terms([cells['B11']['development']['per_class'][str(k)]['risk'] for k in seen],
                        [cells['B00']['development']['per_class'][str(k)]['risk'] for k in seen], seen, current, tail)
                    row['partitions'][partition] = dict(contrasts=c, total_protection=full, head_only_protection=head,
                        head_vs_total_max_class_recall_difference=max(abs(cells['B01']['development']['per_class'][str(k)]['recall']-
                            cells['B11']['development']['per_class'][str(k)]['recall']) for k in seen),
                        same_encoder_reward=p['same_encoder_feedback']['reward'],
                        branch_proxy_risk_reward=p['branch_risk_feedback_without_KL']['risk_reward'],
                        branch_proxy_mean_gain=p['branch_risk_feedback_without_KL']['mean_gain'],
                        branch_development_mean_gain=dev_risk['mean_gain'],
                        branch_development_risk_reward=dev_risk['risk_reward'],
                        total_helped=sum(v['helped'] for v in p['paired']['total'].values()),
                        total_hurt=sum(v['hurt'] for v in p['paired']['total'].values()),
                        head_only_helped=sum(v['helped'] for v in p['paired']['head_baseline'].values()),
                        head_only_hurt=sum(v['hurt'] for v in p['paired']['head_baseline'].values()))
                rows.append(row)
        rows.sort(key=lambda r: r['candidate']); assert [r['candidate'] for r in rows] == list(range(17))
        selected = min(rows, key=lambda r: (-r['initial_reward'], r['candidate']))
        selected_id = selected['candidate'] if selected['initial_reward'] > 1e-8 else 0
        summary = dict(proxy_selected=selected_id, partitions={}, initial_correlations={})
        for measure in ('initial_reward','initial_mean_gain','initial_risk_reward','initial_development_mean_gain','initial_development_risk_reward'):
            summary['initial_correlations'][measure] = correlation([r[measure] for r in rows], [r['initial_development_BA_gain'] for r in rows])
        for partition in ('fit', 'all'):
            gains = lambda r: r['partitions'][partition]['contrasts']['total']['all']['BA']
            head_gains = lambda r: r['partitions'][partition]['contrasts']['head_baseline']['all']['BA']
            psummary = dict(oracles={}, head_only_oracles={}, positive_total=[r['candidate'] for r in rows if gains(r)>1e-12],
                positive_head_only=[r['candidate'] for r in rows if head_gains(r)>1e-12], correlations={})
            psummary['total_and_head_BA_differ'] = [r['candidate'] for r in rows if abs(gains(r)-head_gains(r))>1e-12]
            psummary['positive_total_and_head_BA_match'] = [r['candidate'] for r in rows if gains(r)>1e-12 and abs(gains(r)-head_gains(r))<=1e-12]
            psummary['max_abs_path_zero_head_BA'] = max(abs(r['partitions'][partition]['contrasts']['path_zero_head']['all']['BA']) for r in rows)
            for tier in ('unconstrained','group','per_class','per_sample'):
                ok = lambda r: tier == 'unconstrained' or r['partitions'][partition]['total_protection'][tier]
                h_ok = lambda r: tier == 'unconstrained' or r['partitions'][partition]['head_only_protection'][tier]
                psummary['oracles'][tier] = best(rows, ok, gains)
                psummary['head_only_oracles'][tier] = best(rows, h_ok, head_gains)
            for measure in ('same_encoder_reward','branch_proxy_risk_reward','branch_proxy_mean_gain',
                            'branch_development_mean_gain','branch_development_risk_reward'):
                psummary['correlations'][measure+'_vs_total_BA'] = correlation([r['partitions'][partition][measure] for r in rows], [gains(r) for r in rows])
            psummary['correlations']['head_only_vs_total_BA'] = correlation([head_gains(r) for r in rows], [gains(r) for r in rows])
            summary['partitions'][partition] = psummary
        output[dataset] = summary; tables[dataset] = rows
    costs = {k: v['elapsed_seconds'] for k, v in data['suites'].items()}
    return dict(summary=output, comparisons=tables,
        audit=dict(status='PASS',actual_candidate_records=actual,unique_candidates=34,logical_readouts=280,
            decomposition_identities=identities,duplicate_zero_aggregates_equal=True,
            reconstruction_checks=sum(len(r['checks']) for r in data['results'].values()),
            maximum_solve_residual=max(r['maximum_solve_residual'] for r in data['results'].values()),
            adapter_updates=0,policy_updates=0,image_reads=0,test_accessed=False),
        budget=dict(prior_closed_gpu_seconds=lock['prior_closed_gpu_seconds'],suite_gpu_seconds=costs,
            round_gpu_seconds=sum(costs.values()),cumulative_gpu_seconds=lock['prior_closed_gpu_seconds']+sum(costs.values()),
            worker_cpu_seconds={k:v['cpu_process_seconds'] for k,v in data['statuses'].items()},
            original_deadline=lock['original_deadline'],failed_or_repair_attempts=[],
            note='Sum suite residence once; worker CPU reported separately, not added again.'))


def self_check():
    assert correlation([1,2,3],[3,2,1]) == -1 and correlation([1,1],[1,2]) is None
    p = protections({'all':{'BA':.1},'old':{'BA':0},'current':{'BA':.2}}, {'a':{'helped':1,'hurt':1}})
    assert p['positive'] and p['group'] and p['per_class'] and not p['per_sample']
    terms = risk_terms([.2,.4],[.3,.3],[0,1],[1],[0])
    assert abs(terms['mean_gain'])<1e-12 and abs(terms['risk_reward']+1/3)<1e-12


if __name__ == '__main__':
    config=json.load(sys.stdin);self_check()
    if config.get('self_check'):
        print('PASS: ranking, protection tiers, and reward decomposition')
    else:
        data=json.loads(Path(config['input']).read_text());lock=json.loads(Path(config['lock']).read_text())
        result=summarize(data,lock);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
        for name,value in [('RESULTS',data['results']),('SUMMARY',result['summary']),('COMPARISONS',result['comparisons']),('AUDIT',result['audit']),('BUDGET_LEDGER',result['budget'])]:
            text=json.dumps(value,indent=2,allow_nan=False)+'\n'
            assert not any(s in text for s in ('/remote-home/','/Users/','identity_component'))
            (out/(name+'.json')).write_text(text)
        print(json.dumps(result['audit'],indent=2))
