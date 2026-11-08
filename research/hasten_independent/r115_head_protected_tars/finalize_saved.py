"""Saved-only verification of risk guards, parameter updates and final scores."""
import csv
import json
from pathlib import Path


def run(cfg):
    root = Path(cfg['output'])
    load = lambda p: json.loads((root / p).read_text())
    assert load('CONTROL_EXIT.json')['exit_code'] == 0
    data = load('FINAL_RESULTS.json')
    curves = load('TRAINING_CURVES.json')
    base = Path(cfg['base_source'])
    weights = json.loads((base / 'TARS_WEIGHTS.json').read_text())
    n = weights['counts']
    summaries, table, errors = {}, [], []
    alphas = [1., .5, .25, .125, .0625, 0.]
    for arm in ['PROJECTED_TARS', 'SHRINK_TARS']:
        rr = [r for r in curves if r['arm'] == arm]
        assert len(rr) == 512 and [r['update'] for r in rr] == list(range(512))
        assert load(arm + '_SEALED.json')['updates'] == 512
        for r in rr:
            expected = next(i for i, loss in enumerate(r['guard_losses']) if loss <= r['guard_losses'][-1])
            assert r['accepted_alpha'] == alphas[expected]
            assert r['head_loss_after'] == r['guard_losses'][expected] <= r['head_loss_before']
            assert r['head_loss_before'] == r['guard_losses'][-1]
            assert r['unchanged_policy_logp_max_difference'] == 0
        summaries[arm] = {'attempted_AdamW_updates': 512,
                          'nonzero_parameter_updates': sum(r['nonzero_parameter_update'] for r in rr),
                          'cone_active_steps': sum(r['cone_active'] for r in rr),
                          'zero_alpha_steps': sum(r['accepted_alpha'] == 0 for r in rr),
                          'positive_actual_head_dot_steps': sum(r['actual_head_dot'] > 0 for r in rr),
                          'guard_violations': 0, 'groups': sum(r['groups'] for r in rr),
                          'constant_groups': sum(r['constant_groups'] for r in rr),
                          'sampled_responses': sum(sum(r['responses_by_class']) for r in rr),
                          'mean_reward': sum(r['reward_mean'] for r in rr) / 512}
    for r in data['results']:
        assert sum(r['denominator']) == 1698 and len(r['correct']) == len(r['denominator']) == 23
        recall = []
        for c, (good, den) in enumerate(zip(r['correct'], r['denominator'])):
            assert isinstance(good, int) and 0 <= good <= den and den > 0
            recall.append(good / den)
            table.append({'arm': r['name'], 'class': c, 'name': weights['class_names'][str(c)],
                          'TRAIN_count': n[c], 'correct': good, 'denominator': den, 'recall_percent': 100*good/den})
        for metric, classes in {'macro': range(23), 'head': [c for c in range(23) if n[c]>=250],
                                'tail': [c for c in range(23) if n[c]<250], 'tiny': [c for c in range(23) if n[c]<25]}.items():
            errors.append(abs(r[metric] - sum(recall[c] for c in classes)/len(classes)))
        errors.append(abs(r['micro'] - sum(r['correct'])/1698))
    assert max(errors) < 1e-7
    main, shrink = data['results']
    sft = next(r for r in data['historical_R114_anchors'] if r['name']=='SFT')
    gates = {'macro_over_SFT_ge_1pp':main['macro']-sft['macro']>=.01,
             'tail_over_SFT_ge_2pp':main['tail']-sft['tail']>=.02,
             'head_over_SFT_ge_minus1pp':main['head']-sft['head']>=-.01,
             'macro_over_SHRINK_nonnegative':main['macro']>=shrink['macro'],
             'tail_over_SHRINK_positive':main['tail']>shrink['tail']}
    assert gates == data['gates'] and all(gates.values()) == data['all_gates_pass']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    report = ['# R115 头类风险约束 TARS-RL：最终诊断', '',
              '**冻结成功门：'+('通过' if all(gates.values()) else '未通过')+'。**', '',
              '|方法|macro %|head %|tail %|tiny %|micro %|', '|---|---:|---:|---:|---:|---:|']
    for r in data['historical_R114_anchors'] + data['results']:
        report.append('|'+r['name']+'|'+'|'.join(f'{100*r[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report.extend(['', 'R114六臂是已有固定锚点，非本轮新独立重复。新两臂共同使用额外TRAIN头类反馈，方向修正与仅缩步的反馈/调用次数匹配；生成token长度和最终应用步长可不同。', '',
                   '逐项门：'+'；'.join(k+'='+str(v) for k,v in gates.items())+'。', '',
                   '每臂512次AdamW尝试，实际非零参数更新：'+'；'.join(k+'='+str(v['nonzero_parameter_updates']) for k,v in summaries.items())+'。', '',
                   'TRAIN minibatch NLL守卫不等于未见样本准确率或每类保护保证；约束真实Adam位移，保留未投影梯度的优化器矩状态。几何约束借鉴A-GEM，非新颖性或完整持续学习方法声明。', '',
                   '单开发seed74005，静态HK23，VAL反复开发使用，TEST0，患者独立性未知。最小类分母仍很小；全23类整数计数、零步和守卫记录完整公开，不挑epoch/seed。', '',
                   '复用R114基础模型、SFT及缓存，无新bootstrap/VALencoder；复用成本另引R114，不当作本轮新执行。权重/图像/特征/逐例预测/私有路径不公开。'])
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    result = {'passed':True,'rows':len(table),'max_metric_error':max(errors),'learning':summaries,
              'gates':gates,'all_gates_pass':all(gates.values()),'new_model_queries_updates':0,'scope':'saved_only'}
    (root/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2))
    return result
