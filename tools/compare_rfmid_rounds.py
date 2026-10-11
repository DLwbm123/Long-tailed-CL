"""Compare completed anonymous RFMiD rounds without changing the live experiment."""
import json
from pathlib import Path
import sys

from prepare_rfmid_ml import BENCHMARK_LABELS
from summarize_rfmid import FIELDS


def compare(old, new, old_lock, new_lock):
    labels = list(BENCHMARK_LABELS)
    assert old_lock['steps_per_arm'] == 66 and new_lock['steps_per_arm'] == 60
    tables, final, rows = {}, {}, {}
    for name, result, lock in (('previous', old, old_lock), ('recurrent', new, new_lock)):
        assert lock['labels'] == labels and lock['frequency_groups'] == old_lock['frequency_groups']
        assert lock['test_accessed'] is False and lock['threshold'] == .5
        for arm in ('base', 'local_transport'):
            key = name + '_' + arm
            metrics = result['runs']['RFMiD_' + arm]['metrics']
            assert metrics['evaluation_split'] == 'official_validation' and metrics['test_accessed'] is False
            stages = metrics['stages']; assert len(stages) == 5
            assert all(s['visible_labels'] == lock['label_order'][:count]
                       for s, count in zip(stages, (9, 14, 19, 24, 29)))
            final[key] = stages[-1]
            tables[key] = {v['label']: v for v in stages[-1]['per_label']}
            assert list(tables[key]) == labels
            assert all(v['observed'] == 640 and v['positive'] > 0 and v['negative'] > 0 and v['AP'] is not None for v in tables[key].values())
            rows[key] = dict(result['rows'][arm], final_macro_F1=stages[-1]['aggregate']['macro_F1'],
                             final_micro_F1=stages[-1]['aggregate']['micro_F1'])
    reference = tables['previous_base']
    assert all([(t[k]['positive'], t[k]['negative']) for k in labels] ==
               [(reference[k]['positive'], reference[k]['negative']) for k in labels] for t in tables.values())
    fields = (*FIELDS, 'final_macro_AUROC', 'final_macro_F1', 'final_micro_F1')
    cross = {arm: {k: 100 * (rows['recurrent_' + arm][k] - rows['previous_' + arm][k]) for k in fields}
             for arm in ('base', 'local_transport')}
    within = {name: {k: 100 * (rows[name + '_local_transport'][k] - rows[name + '_base'][k]) for k in fields}
              for name in ('previous', 'recurrent')}
    interaction = {k: within['recurrent'][k] - within['previous'][k] for k in fields}
    lines = ['# RFMiD 新旧划分完整对比', '',
             '两轮同一官方validation640图、固定29标签和train频数组；test封存。旧轮每臂66步，新轮60步。标签引入组、样本分配、known mask与更新量均变化，以下为描述性比较，不能将差异单独归因于共病监督或旧阳性。', '',
             '|划分/方法|最终mAP %|平均增量mAP %|tail AP %|AP遗忘 pp|旧label AP %|新label AP %|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in rows.items():
        lines.append('|' + '|'.join([name] + [f'{100 * values[k]:.4f}' for k in FIELDS]) + '|')
    for arm, values in cross.items():
        lines.append('|新−旧 ' + arm + ' pp|' + '|'.join(f'{values[k]:+.4f}' for k in FIELDS) + '|')
    for name, values in within.items():
        lines.append('|' + name + '候选−base pp|' + '|'.join(f'{values[k]:+.4f}' for k in FIELDS) + '|')
    lines.append('|候选收益变化 pp|' + '|'.join(f'{interaction[k]:+.4f}' for k in FIELDS) + '|')
    lines += ['', '最终mAP、固定tail/head/middle、29标签AP采用同一集合与分母，可直接描述跨轮变化。平均增量mAP、遗忘和原old/new AP对应不同课程/标签集合，数值保留但不是相同任务指标；下表另用固定集合比较old/new。候选收益变化为(new candidate−new base)−(old candidate−old base)，仍非独立确认或单因素因果证据。', '',
              '|最终补充指标 %|旧base|旧candidate|新base|新candidate|base新−旧 pp|candidate新−旧 pp|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for field in ('final_macro_AUROC', 'final_macro_F1', 'final_micro_F1'):
        lines.append('|' + '|'.join([field] + [f'{100 * rows[k][field]:.4f}' for k in rows] +
                                   [f'{cross[a][field]:+.4f}' for a in cross]) + '|')
    fixed_groups = dict(old_lock['frequency_groups'])
    fixed_groups.update(previous_old=old_lock['label_order'][:24], previous_new=old_lock['label_order'][24:],
                        recurrent_old=new_lock['label_order'][:24], recurrent_new=new_lock['label_order'][24:])
    group_rows = {}
    lines += ['', '|固定标签集合 AP %|旧base|旧candidate|新base|新candidate|base新−旧 pp|candidate新−旧 pp|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for group, members in fixed_groups.items():
        values = {key: sum(table[k]['AP'] for k in members) / len(members) for key, table in tables.items()}
        group_rows[group] = dict(labels=members, values=values)
        lines.append('|' + '|'.join([group] + [f'{100 * values[k]:.4f}' for k in values] +
            [f'{100 * (values["recurrent_" + a] - values["previous_" + a]):+.4f}' for a in cross]) + '|')
    lines += ['', '|label|val正n|val负n|旧base AP %|旧candidate AP %|新base AP %|新candidate AP %|base新−旧 pp|candidate新−旧 pp|新candidate−base pp|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    label_rows = []
    signs = {arm: dict(positive=0, negative=0, same=0) for arm in cross}
    for label in labels:
        values = {key: table[label]['AP'] for key, table in tables.items()}
        delta = {a: 100 * (values['recurrent_' + a] - values['previous_' + a]) for a in cross}
        for arm, d in delta.items(): signs[arm]['positive' if d > 0 else 'negative' if d < 0 else 'same'] += 1
        label_rows.append(dict(label=label, positive=reference[label]['positive'], negative=reference[label]['negative'],
                               AP=values, cross_delta_pp=delta))
        lines.append('|' + '|'.join([label, str(reference[label]['positive']), str(reference[label]['negative'])] +
            [f'{100 * v:.4f}' for v in values.values()] + [f'{v:+.4f}' for v in delta.values()] +
            [f'{100 * (values["recurrent_local_transport"] - values["recurrent_base"]):+.4f}']) + '|')
    common_stages = []
    lines += ['', '|task|两轮共用已见标签数|旧base共集AP %|旧candidate共集AP %|新base共集AP %|新candidate共集AP %|',
              '|---|---:|---:|---:|---:|---:|']
    for index in range(5):
        common = [k for k in labels if k in old['runs']['RFMiD_base']['metrics']['stages'][index]['visible_labels']
                  and k in new['runs']['RFMiD_base']['metrics']['stages'][index]['visible_labels']]
        values = {}
        for name, result in (('previous', old), ('recurrent', new)):
            for arm in cross:
                table = {v['label']: v['AP'] for v in result['runs']['RFMiD_' + arm]['metrics']['stages'][index]['per_label']}
                values[name + '_' + arm] = sum(table[k] for k in common) / len(common) if common else None
        common_stages.append(dict(task=index + 1, labels=common, values=values))
        lines.append('|' + '|'.join([str(index + 1), str(len(common))] +
                                   ['null' if v is None else f'{100 * v:.4f}' for v in values.values()]) + '|')
    lines += ['', '共集stage仅补充同标签集合的描述，不代表相同训练课程。新轮完整五stage/29label AP、AUROC、F1及支持度保留于RESULTS，旧轮原结果不重写。',
              f'跨轮最终AP：base {signs["base"]}，candidate {signs["local_transport"]}，全部正负均报告。',
              f'最终mAP新−旧：base {cross["base"]["final_mAP"]:+.4f}pp，candidate {cross["local_transport"]["final_mAP"]:+.4f}pp。新轮候选−base {within["recurrent"]["final_mAP"]:+.4f}pp，候选收益变化 {interaction["final_mAP"]:+.4f}pp。',
              '旧分配未来阳性监督损失324次、后续旧标签阳性0；新分配损失0、后续旧阳性676次/560图、共病650对均保留。支持修复本身不能预测或证明AP改善；不把单标签BA门套入多标签，不夸大极少val TV/PT/RP各2阳性。',
              '单seed/order、重复official val、缺patient/eye ID与内容重复核对，仅开发证据，不是临床泛化、test benchmark或APPLE/MONICA精确复现。新旧成本分别列于各轮AUDIT，旧轨迹不重跑也不重新收费。']
    result = dict(status='DESCRIPTIVE_CROSS_ROUND_COMPARISON', rows=rows, cross_delta_pp=cross,
                  within_delta_pp=within, module_gain_change_pp=interaction, fixed_group_AP=group_rows,
                  per_label=label_rows, common_label_stage_AP=common_stages, label_change_counts=signs,
                  paired_validation_denominators_equal=True, individual_predictions_compared=False,
                  causal_single_factor_attribution=False, previous_steps_per_arm=66, recurrent_steps_per_arm=60)
    return result, '\n'.join(lines) + '\n'


if __name__ == '__main__':
    config = json.load(sys.stdin)
    read = lambda path: json.loads(Path(path).read_text())
    result, text = compare(read(config['previous_results']), read(config['current_results']),
                           read(config['previous_protocol']), read(config['current_protocol']))
    output = Path(config['output']); assert output.is_dir()
    for name, content in (('CROSS_ROUND.json', json.dumps(result, indent=2, allow_nan=False) + '\n'),
                           ('CROSS_ROUND_ZH.md', text)):
        target = output / name; assert not target.exists(); target.write_text(content)
    print(json.dumps(result['cross_delta_pp'], indent=2))
