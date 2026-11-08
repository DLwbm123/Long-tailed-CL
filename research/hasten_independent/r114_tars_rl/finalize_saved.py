"""Independent stdlib count/gate verification and an aggregate-only report."""
import csv
import json
from pathlib import Path


def finalize(root):
    root = Path(root)
    load = lambda name: json.loads((root / name).read_text())
    final, weights, curves = load('FINAL_RESULTS.json'), load('TARS_WEIGHTS.json'), load('TRAINING_CURVES.json')
    arms = ['SFT', 'CE_CONT', 'TARS_CE', 'GRPO', 'CONST_GRPO', 'TARS_GRPO']
    assert load('CONTROL_EXIT.json')['exit_code'] == 0
    assert load('ALL_BRANCHES_SEALED.json')['arms'] == arms
    result = {r['name']: r for r in final['results']}
    assert set(result) == set(arms)
    n = weights['counts']
    sets = {'macro': list(range(23)), 'head': [c for c in range(23) if n[c] >= 250],
            'tail': [c for c in range(23) if n[c] < 250], 'tiny': [c for c in range(23) if n[c] < 25]}
    table, differences = [], []
    for arm in arms:
        r = result[arm]
        assert len(r['correct']) == len(r['denominator']) == 23
        assert sum(r['denominator']) == 1698
        assert r['denominator'] == result['SFT']['denominator']
        rates = []
        for c, (good, den) in enumerate(zip(r['correct'], r['denominator'])):
            assert isinstance(good, int) and isinstance(den, int) and 0 <= good <= den and den > 0
            rates.append(good / den)
            table.append({'arm': arm, 'class': c, 'name': weights['class_names'][str(c)],
                          'TRAIN_count': n[c], 'correct': good, 'denominator': den, 'recall_percent': 100 * good / den})
        for metric, classes in sets.items():
            differences.append(abs(r[metric] - sum(rates[c] for c in classes) / len(classes)))
        differences.append(abs(r['micro'] - sum(r['correct']) / sum(r['denominator'])))
    assert max(differences) < 1e-7
    assert len([r for r in curves if r['phase'] == 'SFT']) == 854
    learning = {}
    for arm in arms[1:]:
        rr = [r for r in curves if r.get('arm') == arm]
        assert len(rr) == 512 and [r['update'] for r in rr] == list(range(512))
        assert load(arm + '_SEALED.json')['updates'] == 512
        if arm.endswith('GRPO') or arm == 'GRPO':
            assert sum(r['groups'] for r in rr) == 4096
            assert all(sum(r['responses_by_class']) == 64 for r in rr)
            assert all(r['unchanged_policy_logp_max_difference'] == 0 for r in rr)
            learning[arm] = {'groups': 4096, 'constant_groups': sum(r['constant_groups'] for r in rr),
                             'sampled_responses': 32768, 'truncated_responses': sum(r['truncated_responses'] for r in rr),
                             'mean_reward': sum(r['reward_mean'] for r in rr) / 512,
                             'maximum_gradient_norm': max(r['gradient_norm'] for r in rr),
                             'mean_KL': sum(r['KL'] for r in rr) / 512,
                             'maximum_sampling_forward_logp_difference': max(r['cached_sampling_vs_behavior_forward_max_difference'] for r in rr)}
    t, s = result['TARS_GRPO'], result['SFT']
    gates = {'macro_over_SFT_ge_1pp': t['macro']-s['macro'] >= .01,
             'tail_over_SFT_ge_2pp': t['tail']-s['tail'] >= .02,
             'head_over_SFT_ge_minus1pp': t['head']-s['head'] >= -.01}
    for other in ['GRPO', 'CONST_GRPO']:
        for metric in ['macro', 'tail']:
            gates[f'{metric}_over_{other}_positive'] = t[metric] > result[other][metric]
    gates['macro_over_TARS_CE_nonnegative'] = t['macro'] >= result['TARS_CE']['macro']
    gates['tail_over_TARS_CE_positive'] = t['tail'] > result['TARS_CE']['tail']
    assert gates == final['gates'] and all(gates.values()) == final['all_gates_pass']
    with (root / 'CLASS_RESULTS.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    summary = {'passed': True, 'count_rows': len(table), 'max_metric_error': max(differences),
               'gates': gates, 'all_gates_pass': all(gates.values()), 'learning': learning,
               'new_images_queries_optimizer_updates': 0, 'scope': 'saved_only',
               'TARS_amplified_classes': sum(w > 1 for w in weights['weights']),
               'TARS_constant_strength': weights['constant_strength']}
    (root / 'FINAL_VERIFICATION.json').write_text(json.dumps(summary, indent=2))
    costs = load('COSTS.json')
    exit_receipt = load('CONTROL_EXIT.json')
    previous_exit = load('attempt1.private/CONTROL_EXIT.json')
    combined = {'completed_attempt': costs,
                'failed_attempt1_SFT_updates': {'lower': 11, 'upper': 12},
                'failed_attempt1_engineering_optimizer_updates': 2,
                'failed_attempt1_encoded_TRAIN_rows': 6823,
                'numerical_probe_same_native_TRAIN_images_encoded_again': 2,
                'numerical_probe_sampled_responses': 16, 'numerical_probe_optimizer_updates': 0,
                'all_attempts_SFT_updates': {'lower': 865, 'upper': 866},
                'all_attempts_engineering_optimizer_updates': 4,
                'all_attempts_posttraining_optimizer_updates': 2560,
                'controller_worker_CPU_seconds': sum(x['worker']['CPU_seconds'] for x in [exit_receipt, previous_exit]),
                'numerical_probe_CPU_seconds': None, 'preparation_CPU_seconds': None,
                'GPU_kernel_seconds': None, 'original_elapsed_seconds': exit_receipt['original_elapsed_seconds'],
                'all_attempts_peak_disk_bytes': exit_receipt['peak_disk_bytes'],
                'new_cache_copies': 1, 'shared_model_weights_not_copied': True}
    (root / 'ALL_ATTEMPT_COSTS.json').write_text(json.dumps(combined, indent=2))
    lines = ['# R114 生成式识别 TARS-RL：最终诊断', '',
             '**已完成。冻结成功门：' + ('通过' if all(gates.values()) else '未通过') + '。**', '',
             '单个开发种子74005，静态HK23，TRAIN6821/VAL1698/TEST0；不是持续学习结果或独立患者确认。', '',
             '|方法|micro %|macro %|head %|tail %|tiny %|无效预测|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        r = result[arm]
        lines.append('|' + arm + '|' + '|'.join(f'{100*r[k]:.4f}' for k in ['micro','macro','head','tail','tiny']) + f"|{r['invalid_predictions']}|")
    lines.extend(['', 'TARS_GRPO相对公共SFT：' + '；'.join(f'{k} {100*(t[k]-s[k]):+.4f}pp' for k in ['macro','tail','head']) + '。', '',
                  '各冻结门逐项：' + '；'.join(k + '=' + str(v) for k,v in gates.items()) + '。', '',
                  '同一SFT起点、相同自然查询序列与更新次数。CE控制的生成和反向次数更少，不称计算量匹配。原始TARS、常数强度RL和加权CE分别检查类别分配、总放大尺度与RL增量。', '',
                  'TARS仅放大已有优势，不制造奖励多样性。逐臂常量组、奖励、KL和梯度见FINAL_VERIFICATION.json；权重来自固定SFT的TRAIN一次贪心识别，可能含拟合偏差。', '',
                  '共享Qwen2.5-VL-3B，冻结视觉/基础参数，语言q/v LoRA rank8 alpha16，224分辨率。与原文全语言参数/四H100/三seed不同，不是原论文精确复现。', '',
                  'BF16缓存生成与训练前向存在数值差异；行为概率在相同训练微批路径重新冻结，未更新策略概率比严格1，原始缓存差异保留。称数值近似on-policy，不称比特精确采样概率。', '',
                  '首次实现于正式RL前修复，失败11–12次SFT更新和工程成本保留，原时钟不重置。完整新增成本还须合并attempt1、两图概率诊断及控制器账；未知CPU/GPU kernel费用不补0。', '',
                  'VAL已用于历史开发；tiny分母小，所有类整数计数公开。没有选择最好epoch/seed，不据结果改旧协议。图片、身份、特征、适配器、逐例预测和私有运行路径不公开。', '',
                  '本报告是科学证据收口；自主研究须根据全部对照与奖励多样性选择新假设，不能仅重复监测、改名重跑或把静态收益包装成持续学习成功。'])
    (root / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    return summary


def run(cfg):
    return finalize(cfg['output'])


if __name__ == '__main__':
    print(json.dumps(finalize(Path(__file__).parent), indent=2))
