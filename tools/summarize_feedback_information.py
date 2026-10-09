"""Read frozen anonymous aggregates; no model, data, or candidate tuning."""
import json
import math
from pathlib import Path
import statistics as st
import sys
import time

from summarize_decision_probability import KEYS
from summarize_feedback_audit import correlation

FAMILIES = dict(edge=KEYS[:17], norm=[KEYS[0]]+KEYS[17:])
HEADS = KEYS+['linear_ridge', 'prototype_ridge']
mean = st.mean


def groups(r, classes):
    return dict(all=classes, old=[c for c in classes if c not in r['current']],
                current=[c for c in classes if c in r['current']], tail=[c for c in classes if c in r['tail']])


def select(values, r, keys):
    audit = {}; classes = r['covered_classes']
    for key in keys:
        i = KEYS.index(key); terms = {}
        for g, cs in groups(r, classes).items():
            assert cs
            delta = [mean(b[i][classes.index(c)]-b[0][classes.index(c)] for c in cs) for b in values]
            terms[g] = dict(gain=mean(delta), numerical_se=st.stdev(delta)/math.sqrt(len(delta)))
        protected = all(v['gain'] >= -1e-12 for g,v in terms.items() if g != 'all')
        resolution = max(1e-8, 2*terms['all']['numerical_se'])
        audit[key] = dict(groups=terms, protection_pass=protected, resolution=resolution,
                          eligible=protected and terms['all']['gain'] > resolution)
    eligible = [k for k in keys if audit[k]['eligible']]
    return (min(eligible, key=lambda k: (-audit[k]['groups']['all']['gain'], KEYS.index(k))) if eligible else KEYS[0]), audit


def development(r, split):
    table = r['evaluation'][split]; classes = r['seen'] if split == 'full' else r['covered_classes']
    zero = table[KEYS[0]]['metrics']['per_class']; rows = {}
    assert set(table) == set(HEADS)
    for key in HEADS:
        per = table[key]['metrics']['per_class']; pairs = table[key]['paired_to_strong']
        assert set(per) == {str(c) for c in classes}
        delta = {str(c): per[str(c)]['recall']-zero[str(c)]['recall'] for c in classes}
        for c in classes:
            p = pairs[str(c)]; v = per[str(c)]
            assert p['n'] == v['n'] and v['n'] > 0
            assert abs(v['recall']-v['correct']/v['n']) < 1e-12
            assert abs(delta[str(c)]-(p['helped']-p['hurt'])/p['n']) < 1e-12
        absolute = {g: mean(per[str(c)]['recall'] for c in cs) for g,cs in groups(r, classes).items()}
        gain = {g: mean(delta[str(c)] for c in cs) for g,cs in groups(r, classes).items()}
        assert abs(absolute['all']-table[key]['metrics']['views']['all']['BA']) < 1e-12
        rows[key] = dict(BA=absolute, BA_gain=gain, per_class_recall_gain=delta,
            helped=sum(v['helped'] for v in pairs.values()), hurt=sum(v['hurt'] for v in pairs.values()),
            group_protection=all(v >= -1e-12 for g,v in gain.items() if g != 'all'),
            per_class_protection=min(delta.values()) >= -1e-12,
            per_sample_protection=all(v['hurt'] == 0 for v in pairs.values()))
    return rows


def ranks(xs):
    return [mean(j for j,y in enumerate(sorted(xs)) if y == x) for x in xs]


def calibration(pred, truth, r):
    result = {}
    for family, keys in FAMILIES.items():
        indices = [KEYS.index(k) for k in keys]; dp = [[p-pred[0][c] for c,p in enumerate(pred[i])] for i in indices]
        dt = [[p-truth[0][c] for c,p in enumerate(truth[i])] for i in indices]
        result[family] = dict(absolute_MAE=mean(abs(pred[i][c]-truth[i][c]) for i in indices for c in range(len(pred[0]))),
            delta_MAE=mean(abs(dp[i][c]-dt[i][c]) for i in range(1,len(indices)) for c in range(len(pred[0]))),
            group_spearman={})
        for g, cs in groups(r, r['covered_classes']).items():
            pp = [mean(v[r['covered_classes'].index(c)] for c in cs) for v in dp]
            tt = [mean(v[r['covered_classes'].index(c)] for c in cs) for v in dt]
            result[family]['group_spearman'][g] = correlation(ranks(pp), ranks(tt))
    return result


def conditional(r):
    result = {}
    for c, raw in r['conditional_transport'].items():
        assert raw['target_excluded_from_shift'] and raw['target_was_used_in_encoder_training']
        direct = [mean(b[i] for b in raw['direct_batches']) for i in range(33)]
        transported = [mean(b[i] for b in raw['transported_batches']) for i in range(33)]
        fit, meta = raw['empirical_fit'], raw['empirical_meta']; result[c] = {}
        for family, keys in FAMILIES.items():
            ids = [KEYS.index(k) for k in keys]; nonzero = ids[1:]
            delta_error = lambda a,b: mean(abs((a[i]-a[0])-(b[i]-b[0])) for i in nonzero)
            result[c][family] = dict(same_fit_gaussian_absolute_MAE=mean(abs(direct[i]-fit[i]) for i in ids),
                same_fit_gaussian_delta_MAE=delta_error(direct,fit), transport_direct_delta_MAE=delta_error(transported,direct),
                fit_meta_delta_MAE=delta_error(fit,meta),
                transport_direct_paired_numerical_SE=mean(st.stdev([(t[i]-t[0])-(d[i]-d[0]) for t,d in
                    zip(raw['transported_batches'],raw['direct_batches'])])/math.sqrt(8) for i in nonzero))
    return result


def summarize(data, lock):
    summary = {}; comparisons = {}; checks = 0
    for job in lock['jobs']:
        name = job['name']; status = data['statuses'][name]; expected = [s['name'] for s in job['states']]
        assert status['status'] == data['suites'][name]['status'] == 'COMPLETE'
        assert status['prepared_states'] == status['evaluated_states'] == expected
        assert status['adapter_updates'] == status['policy_updates'] == 0 and not status['test_accessed']
        assert status['maximum_solve_residual'] <= lock['max_solve_residual']
        assert data['barriers'][name]['states'] == expected and not data['barriers'][name]['B_metrics_evaluated']
        assert set(data['results'][name]) == set(expected)
        for key,r in data['results'][name].items():
            assert r['B_evaluated_after_all_selections'] and not r['independent_confirmation']
            assert r['zero_head_reconstruction_error'] <= lock['reconstruction_atol']
            assert set(r['excluded_classes']) == set(r['seen']) & set(lock['known_excluded_development_classes'][r['dataset']])
            assert set(r['covered_classes']) == set(r['seen'])-set(r['excluded_classes'])
            for c,s in r['coverage'].items():
                assert s['A_identities']+s['B_identities'] == s['identities']
                assert s['covered'] == (int(c) in r['covered_classes']) == (s['identities'] >= 4)
                if s['covered']: assert min(s['A_identities'],s['B_identities']) >= 2
            gaussian = r['gaussian_A_batches']; assert len(gaussian) == 8
            assert all(len(b) == 33 and all(len(p) == len(r['covered_classes']) for p in b) for b in gaussian)
            assert all(0 <= p <= 1 for b in gaussian for row in b for p in row)
            for source, values in [('empirical_A',[r['empirical_A']]*8),('gaussian_A',gaussian)]:
                for family, keys in FAMILIES.items():
                    label = source+'_'+family; chosen, audit = select(values,r,keys)
                    assert chosen == r['selected'][label]
                    for k,v in audit.items():
                        saved = r['choice_audits'][label][k]
                        assert v['eligible'] == saved['eligible'] and v['protection_pass'] == saved['protection_pass']
                        assert abs(v['resolution']-saved['resolution']) < 1e-12
                        for g,terms in v['groups'].items():
                            assert all(abs(x-saved['groups'][g][t]) < 1e-12 for t,x in terms.items())
                        checks += 1
            comparisons[key] = {s:development(r,s) for s in ('full','A','B')}
            for k in HEADS:
                for c in r['covered_classes']:
                    for field in ('n','correct'):
                        get = lambda s:r['evaluation'][s][k]['metrics']['per_class'][str(c)][field]
                        assert get('full') == get('A')+get('B')
                    for field in ('n','helped','hurt','prediction_changed'):
                        get = lambda s:r['evaluation'][s][k]['paired_to_strong'][str(c)][field]
                        assert get('full') == get('A')+get('B')
            parts = {}
            for split,rows in comparisons[key].items():
                oracles = {}
                for family,keys in FAMILIES.items():
                    oracles[family] = {}
                    for tier in ('unconstrained','group','per_class','per_sample'):
                        eligible = [k for k in keys if tier == 'unconstrained' or rows[k][tier+'_protection']]
                        best = min(eligible,key=lambda k:(-rows[k]['BA_gain']['all'],KEYS.index(k)))
                        oracles[family][tier] = dict(key=best, **rows[best])
                choices = {**{'original_'+k:v for k,v in r['original_selected'].items()}, **r['selected']}
                parts[split] = dict(selected={k:dict(key=v,**rows[v]) for k,v in choices.items()},oracles=oracles)
            full = comparisons[key]['full']; baselines = {}
            for base in (KEYS[0],'linear_ridge','prototype_ridge'):
                baselines[base] = dict(BA=full[base]['BA'],headroom={})
                for family,keys in FAMILIES.items():
                    best = parts['full']['oracles'][family]['unconstrained']['key']
                    static = full[KEYS[0]]['BA']['all']-full[base]['BA']['all']
                    adaptive = full[best]['BA_gain']['all']; total = full[best]['BA']['all']-full[base]['BA']['all']
                    assert abs(total-static-adaptive) < 1e-12
                    baselines[base]['headroom'][family] = dict(best=best,static_difference=static,
                        adaptive_headroom=adaptive,raw_library_difference=total,with_stay_option=max(0,total))
            empirical_B = [[r['evaluation']['B'][k]['metrics']['per_class'][str(c)]['recall'] for c in r['covered_classes']] for k in KEYS]
            gm = [[mean(b[i][c] for b in gaussian) for c in range(len(r['covered_classes']))] for i in range(33)]
            for i,k in enumerate(KEYS):
                for j,c in enumerate(r['covered_classes']):
                    assert abs(r['empirical_A'][i][j]-r['evaluation']['A'][k]['metrics']['per_class'][str(c)]['recall']) < 1e-12
            summary[key] = dict(dataset=r['dataset'],task=r['task'],cohort=r['cohort'],covered_classes=r['covered_classes'],
                excluded_classes=r['excluded_classes'],baselines=baselines,partitions=parts,
                calibration=dict(gaussian_A_vs_empirical_A=calibration(gm,r['empirical_A'],r),
                    empirical_A_vs_B=calibration(r['empirical_A'],empirical_B,r),gaussian_A_vs_B=calibration(gm,empirical_B,r)),
                conditional_transport=conditional(r))
    assert len(summary) == lock['states'] == 5 and checks == 340
    assert sum(s['full_head_readouts'] for s in data['statuses'].values()) == lock['full_head_readouts'] == 525
    images = sum(s['image_rows'] for s in data['statuses'].values()); assert images == lock['expected_current_training_image_passes']
    costs = {k:v['elapsed_seconds'] for k,v in data['suites'].items()}; total = sum(costs.values())
    return dict(summary=summary,comparisons=comparisons,
        audit=dict(status='PASS',states=5,head_readouts=525,selector_candidate_checks=checks,image_rows=images,
            maximum_solve_residual=max(s['maximum_solve_residual'] for s in data['statuses'].values()),
            adapter_updates=0,policy_updates=0,test_accessed=False,independent_confirmation=False),
        budget=dict(prior_closed_gpu_seconds=lock['prior_closed_gpu_seconds'],suite_seconds=costs,round_gpu_seconds=total,
            cumulative_gpu_seconds=lock['prior_closed_gpu_seconds']+total,
            worker_cpu_seconds={k:v['cpu_process_seconds'] for k,v in data['statuses'].items()},
            original_deadline=lock['original_deadline'],note='Suite residence includes barrier; worker CPU reported separately, not added.'))


def self_check():
    r = dict(covered_classes=[0,1],current=[1],tail=[0])
    values = [[[.5,.5],[.6,.5],[.7,.4]]]*8
    assert select(values,r,KEYS[:3])[0] == KEYS[1]
    assert select(values,r,[KEYS[0],KEYS[2]])[0] == KEYS[0]
    assert ranks([2,1,2]) == [1.5,0,1.5]


def table(summary):
    lines = ['# FEEDBACK-INFORMATION1 完整汇总', '', 'BA 与增量均以百分点表示。A/B 仅覆盖身份足够的类别；full 包含全部已见类别。所有 oracle 都是事后诊断，不是可部署策略。', '']
    for key,r in summary.items():
        lines += ['## '+key, '', 'A/B 排除类别：'+str(r['excluded_classes']), '',
                  '| 固定编码器的终端读出 | full BA | old | current | tail | edge 静态差异 | edge 自适应空间 | edge 总空间 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for k,v in r['baselines'].items():
            h = v['headroom']['edge']; numbers = list(v['BA'].values())+[h[x] for x in ('static_difference','adaptive_headroom','raw_library_difference')]
            lines.append('| '+k+' | '+' | '.join(f'{100*x:.4f}' for x in numbers)+' |')
        for split,p in r['partitions'].items():
            lines += ['', '### '+split, '', '| 选择来源 | 候选 | ΔBA | Δold | Δcurrent | Δtail | helped | hurt |', '|---|---|---:|---:|---:|---:|---:|---:|']
            for name,v in p['selected'].items():
                lines.append('| '+name+' | '+v['key']+' | '+' | '.join(f'{100*x:.4f}' for x in v['BA_gain'].values())+f" | {v['helped']} | {v['hurt']} |")
            lines += ['', '| oracle 家族 | 保护层级 | 候选 | ΔBA | hurt |', '|---|---|---|---:|---:|']
            for family,tiers in p['oracles'].items():
                for tier,v in tiers.items(): lines.append(f"| {family} | {tier} | {v['key']} | {100*v['BA_gain']['all']:.4f} | {v['hurt']} |")
        lines += ['', '| 反馈比较 | 家族 | 绝对 recall MAE | Δrecall MAE | all Spearman |', '|---|---|---:|---:|---:|']
        for name,fs in r['calibration'].items():
            for family,v in fs.items():lines.append(f"| {name} | {family} | {100*v['absolute_MAE']:.4f} | {100*v['delta_MAE']:.4f} | {v['group_spearman']['all']} |")
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    started = time.process_time(); self_check()
    config = json.load(sys.stdin); data = json.loads(Path(config['input']).read_text()); lock = json.loads(Path(config['lock']).read_text())
    result = summarize(data,lock); result['budget']['readout_cpu_seconds'] = time.process_time()-started
    result['budget']['cpu_selfcheck_seconds'] = json.loads((Path(config['lock']).parent/'SELFCHECK.json').read_text())['cpu_seconds']
    out = Path(config['output']); out.mkdir(parents=True,exist_ok=True)
    for name,value in result.items(): (out/(name.upper()+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    for suite in data['suites'].values():
        for field in ('pid','child_pid'): suite.pop(field,None)
    (out/'RESULTS.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (out/'TABLE_ZH.md').write_text(table(result['summary']))
    print(json.dumps(result['audit']))
