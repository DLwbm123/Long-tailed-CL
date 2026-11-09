"""Saved-only independent checks for common support and incremental RL."""
import csv
import json
from pathlib import Path


def run(cfg):
    root = Path(cfg['output']);load = lambda n: json.loads((root / n).read_text())
    assert load('CONTROL_EXIT.json')['exit_code'] == 0
    final, curves, cost = load('FINAL_RESULTS.json'), load('TRAINING_CURVES.json'), load('COSTS.json')
    info = json.loads((Path(cfg['base_source']) / 'TARS_WEIGHTS.json').read_text())
    counts = info['counts'];allocation = load('ALLOCATION.json');queries = allocation['realized_queries_by_class']
    support_records = [r for r in curves if r['phase'] == 'SHARED_SUPPORT']
    assert len(support_records) == 853 and sum(r['queries'] for r in support_records) == 6821
    assert [r['update'] for r in support_records] == list(range(853))
    assert load('SUPPORT_SFT_SEALED.json')['updates'] == 853
    assert sum(r['queries'] for r in load('SUPPORT_EXPOSURE.json')['rows']) == 6821
    arms = ['SUPPORT_TARS_FULL_IS', 'SUPPORT_GRPO_FULL_IS', 'SUPPORT_CE_FULL_IS']
    assert load('ALL_BRANCHES_SEALED.json')['arms'] == ['SUPPORT_SFT'] + arms
    learning = {}
    for arm in arms:
        rr = [r for r in curves if r['arm'] == arm]
        assert len(rr) == 256 and [r['update'] for r in rr] == list(range(256))
        assert load(arm + '_SEALED.json')['common_initialization_and_reference_verified']
        learning[arm] = {'updates': 256, 'nonzero_gradient_records': sum(r['gradient_norm'] > 0 for r in rr),
                         'actual_nonzero_parameter_updates': sum(r['parameter_delta_L2'] > 0 for r in rr),
                         'cumulative_parameter_path_L2': sum(r['parameter_delta_L2'] for r in rr)}
        if arm == arms[-1]:continue
        assert all(r['unchanged_policy_logp_max_difference'] == 0 for r in rr)
        classes = []
        for c in range(23):
            wrong, right, mixed = [sum(r[k][c] for r in rr) for k in
                                   ['allwrong_groups_by_class', 'allright_groups_by_class', 'mixed_groups_by_class']]
            success = sum(r['correct_by_class'][c] for r in rr)
            assert wrong + right + mixed == queries[c]
            assert sum(r['responses_by_class'][c] for r in rr) == 8 * queries[c]
            assert 8 * right + mixed <= success <= 8 * right + 7 * mixed
            classes.append({'class': c, 'queries': queries[c], 'successful_responses': int(success),
                            'allwrong_groups': wrong, 'allright_groups': right, 'mixed_groups': mixed})
        assert sum(r['constant_groups'] for r in rr) == sum(c['allwrong_groups'] + c['allright_groups'] for c in classes)
        learning[arm]['classes'] = classes
        for c in [5, 6, 17, 19]:
            assert final['successful_responses_formerly_zero_classes'][arm][str(c)] == classes[c]['successful_responses']
    expected = {'engineering_optimizer_updates': 4, 'SFT_updates': 853, 'posttraining_optimizer_updates': 768,
                'generation_responses': 39592, 'TRAIN_encoder_rows': 2, 'VAL_encoder_rows': 0}
    assert cost['counts'] == expected and cost['shared_support_queries'] == 6821
    repair_path = root/'PRE_SCIENCE_REPAIR.json'
    attempts = {'completed_run': cost}
    if repair_path.exists():
        repair = load('PRE_SCIENCE_REPAIR.json')
        assert repair['failed_native_actual_optimizer_attempts'] == 2 and repair['failed_native_science_updates'] == 0
        attempts['failed_native'] = repair
    (root/'ALL_ATTEMPT_COSTS.json').write_text(json.dumps(attempts, indent=2)+'\n')
    rows = [];errors = []
    assert [r['name'] for r in final['results']] == ['SUPPORT_SFT'] + arms
    for result in final['results']:
        assert len(result['correct']) == len(result['denominator']) == 23 and sum(result['denominator']) == 1698
        recall = []
        for c, (good, den) in enumerate(zip(result['correct'], result['denominator'])):
            assert isinstance(good, int) and 0 <= good <= den and den > 0
            recall.append(good/den);rows.append({'arm': result['name'], 'class': c, 'name': info['class_names'][str(c)],
                'TRAIN_count': counts[c], 'correct': good, 'denominator': den, 'recall_percent': 100*good/den})
        for metric, ids in {'macro': list(range(23)), 'head': [c for c in range(23) if counts[c] >= 250],
                            'tail': [c for c in range(23) if counts[c] < 250], 'tiny': [c for c in range(23) if counts[c] < 25]}.items():
            errors.append(abs(result[metric] - sum(recall[c] for c in ids)/len(ids)))
        errors.append(abs(result['micro']-sum(result['correct'])/1698))
    assert max(errors) < 1e-7
    support, main, ordinary, ce = final['results']
    gates = {'macro_over_support_ge_1pp': main['macro']-support['macro'] >= .01,
             'tail_over_support_ge_2pp': main['tail']-support['tail'] >= .02,
             'head_over_support_ge_minus1pp': main['head']-support['head'] >= -.01,
             'macro_over_GRPO_positive': main['macro'] > ordinary['macro'],
             'tail_over_GRPO_positive': main['tail'] > ordinary['tail'],
             'macro_over_CE_positive': main['macro'] > ce['macro'], 'tail_over_CE_positive': main['tail'] > ce['tail']}
    original = final['historical_original_SFT']
    gates.update(end_to_end_macro_over_original_ge_1pp=main['macro']-original['macro'] >= .01,
                 end_to_end_tail_over_original_ge_2pp=main['tail']-original['tail'] >= .02,
                 end_to_end_head_over_original_ge_minus1pp=main['head']-original['head'] >= -.01)
    assert gates == final['gates'] and all(gates.values()) == final['all_gates_pass']
    assert any(final['successful_responses_formerly_zero_classes'][arms[0]].values()) == final['any_formerly_zero_class_discovered_by_main']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result = {'passed': True, 'scope': 'saved_only', 'new_model_queries_updates': 0, 'rows': len(rows),
              'max_metric_error': max(errors), 'learning': learning, 'gates': gates, 'all_gates_pass': all(gates.values())}
    (root/'FINAL_VERIFICATION.json').write_text(json.dumps(result, indent=2)+'\n')
    report = ['# R118 共同反馈支持冷启动与RL增量', '', '**原冻结成功门：'+('通过' if all(gates.values()) else '未通过')+'。**', '',
              '|方法|macro %|head %|tail %|tiny %|micro %|', '|---|---:|---:|---:|---:|---:|']
    for r in [final['historical_original_SFT']] + final['results']:
        report.append('|'+r['name']+'|'+'|'.join(f'{100*r[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report += ['', '新增均衡监督的贡献与后续RL增量分别报告。三个后训练臂共享同一新增起点与固定参考；CE无KL且计算更少。', '',
               '单seed静态开发VAL，非独立患者或持续学习验证；不按结果选择epoch/seed、修改保护阈值或support长度。', '',
               '原门：'+json.dumps(gates), '', '原来零成功类的实际RL正确响应：'+json.dumps(final['successful_responses_formerly_zero_classes'])]
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    return result
