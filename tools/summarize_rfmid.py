"""Anonymous RFMiD paired report; no single-label accuracy gates."""
import json
from pathlib import Path
import sys

import numpy as np

from prepare_rfmid_ml import BENCHMARK_LABELS

FIELDS = ('final_mAP', 'average_incremental_mAP', 'final_tail_AP', 'forgetting_AP', 'final_old_AP', 'final_new_AP')


def summarize(stages):
    if len(stages) != 5 or [len(s['visible_labels']) for s in stages] != [9, 14, 19, 24, 29]:
        raise ValueError('RFMiD stage scope differs')
    final = stages[-1]
    by_label = [{v['label']: v for v in s['per_label']} for s in stages]
    old = stages[-2]['visible_labels']
    forgetting = np.mean([max(table[k]['AP'] for table in by_label[:-1] if table[k]['AP'] is not None)
                          - by_label[-1][k]['AP'] for k in old])
    return dict(final_mAP=final['aggregate']['mAP'],
                average_incremental_mAP=float(np.mean([s['aggregate']['mAP'] for s in stages])),
                final_tail_AP=final['groups']['tail']['AP'], forgetting_AP=float(forgetting),
                final_old_AP=final['groups']['old']['AP'], final_new_AP=final['groups']['new']['AP'],
                final_macro_AUROC=final['aggregate']['macro_AUROC'])


def report(snapshot, output, audit):
    output = Path(output)
    if output.exists():
        raise ValueError('Do not overwrite RFMiD results')
    runs = snapshot['runs']
    data = {arm: runs['RFMiD_' + arm]['metrics'] for arm in ('base', 'local_transport')}
    for arm, metrics in data.items():
        if metrics['dataset'] != 'RFMiD' or metrics['test_accessed'] or metrics['evaluation_split'] != 'official_validation':
            raise ValueError('RFMiD validation permission differs')
        for stage in metrics['stages']:
            if [v['label'] for v in stage['per_label']] != list(BENCHMARK_LABELS):
                raise ValueError('RFMiD all-label report is missing labels')
            visible = set(stage['visible_labels'])
            for v in stage['per_label']:
                if v['label'] in visible:
                    if (v['observed'] != 640 or v['positive'] <= 0 or v['negative'] <= 0
                            or v['positive'] + v['negative'] != 640 or v['AP'] is None or v['AUROC'] is None
                            or not 0 <= v['AP'] <= 1 or not 0 <= v['AUROC'] <= 1):
                        raise ValueError('RFMiD introduced label evaluation support differs')
                elif v['observed'] != 0 or v['AP'] is not None or v['AUROC'] is not None:
                    raise ValueError('Future RFMiD label was evaluated as known')
    for b, c in zip(data['base']['stages'], data['local_transport']['stages']):
        if b['visible_labels'] != c['visible_labels'] or [(v['positive'], v['negative']) for v in b['per_label']] != [(v['positive'], v['negative']) for v in c['per_label']]:
            raise ValueError('RFMiD paired class denominators differ')
    rows = {arm: summarize(v['stages']) for arm, v in data.items()}
    delta = {k: 100 * (rows['local_transport'][k] - rows['base'][k]) for k in (*FIELDS, 'final_macro_AUROC')}
    lines = ['# RFMiD 29标签完整多标签配对结果', '',
             '仅official validation，test封存；多hot共病保留，原非RL A为基线，固定binary局部先验/区域运输为候选。', '',
             '|方法|最终mAP %|平均增量mAP %|tail AP %|AP遗忘 pp|旧label AP %|新label AP %|',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in rows:
        lines.append('|' + '|'.join([arm] + [f'{100 * rows[arm][k]:.4f}' for k in FIELDS]) + '|')
    lines.append('|候选−base pp|' + '|'.join(f'{delta[k]:+.4f}' for k in FIELDS) + '|')
    lines += ['', f'最终mAP {"有正向差异" if delta["final_mAP"] > 0 else "没有正向差异"}，Δ={delta["final_mAP"]:+.4f}pp；不套单标签BA门或称自然医学目标通过。',
              f'新label AP：{"无下降" if delta["final_new_AP"] >= 0 else "仍下降"}。没有为本轮RF设定单标签−1pp保护门，不伪称多标签无损。', '',
              '|task|seen|base mAP %|candidate mAP %|Δpp|', '|---|---:|---:|---:|---:|']
    for b, c in zip(data['base']['stages'], data['local_transport']['stages']):
        lines.append(f'|{b["task"]}|{len(b["visible_labels"])}|{100*b["aggregate"]["mAP"]:.4f}|{100*c["aggregate"]["mAP"]:.4f}|{100*(c["aggregate"]["mAP"]-b["aggregate"]["mAP"]):+.4f}|')
    lines += ['', '|train频数组|base AP %|candidate AP %|Δpp|', '|---|---:|---:|---:|']
    for group in ('head', 'middle', 'tail'):
        b, c = [data[arm]['stages'][-1]['groups'][group]['AP'] for arm in data]
        lines.append(f'|{group}|{100*b:.4f}|{100*c:.4f}|{100*(c-b):+.4f}|')
    lines += ['', f'最终macro AUROC：base {100*rows["base"]["final_macro_AUROC"]:.4f}%，candidate {100*rows["local_transport"]["final_macro_AUROC"]:.4f}%，Δ{delta["final_macro_AUROC"]:+.4f}pp。', '',
              '|label|val正n|val负n|base AP %|candidate AP %|Δpp|base AUROC %|candidate AUROC %|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    positive = negative = same = 0
    for b, c in zip(data['base']['stages'][-1]['per_label'], data['local_transport']['stages'][-1]['per_label']):
        d = c['AP'] - b['AP']; positive += d > 0; negative += d < 0; same += d == 0
        lines.append(f'|{b["label"]}|{b["positive"]}|{b["negative"]}|{100*b["AP"]:.4f}|{100*c["AP"]:.4f}|{100*d:+.4f}|{100*b["AUROC"]:.4f}|{100*c["AUROC"]:.4f}|')
    lines += ['', f'最终AP正向{positive}、负向{negative}、相同{same}标签，全部报告，不只展示改善标签。',
              'CPU/真实预检/解析残差和预算审计见AUDIT；数学/模块激活不等于分类收益。固定threshold.5的F1、全部五stage/29label AP与支持度及无定义项在RESULTS中完整保留。',
              ('新划分每task384图、每图仅一次，累计known监督；2470次阳性及650对共病监督保留，后续每个旧label都有正负支持。未来负例仍unknown，未声称全部负例监督保留。' if data['base'].get('partition_variant') in ('balanced_recurrent_v2', 'negative_coverage_v3') else '任务按train少数阳性优先、图像只使用一次，早期共病图上的未来阳性被unknown mask屏蔽；完整监督损失见TASK_SUPPORT_AUDIT，不能称全部阳性监督均保留。'),
              '原代码没有内部meta，本轮当前task图全部训练及边界聚合，val不作meta。clip1保持原多标签A；没有强套单标签softmax/竞争损失。',
              '单seed/order、重复official val仅开发证据；极少val阳性TV/PT/RP各2，不夸大AP变化。无patient/eye ID，未做内容重复检测，非患者泛化/临床验证或精确MONICA复现。',
              f'本轮GPU驻留{audit["round_gpu_process_residence_seconds"]:.6f}秒；墙钟{audit["parallel_wall_seconds"]:.6f}秒单列，既有CPU数据准备与各次CPU检查不重复算GPU。全部闭合GPU累计{audit["cumulative_gpu_process_residence_seconds"]:.6f}秒，独立round各计一次。']
    output.mkdir()
    (output / 'RESULTS.json').write_text(json.dumps(dict(rows=rows, delta_pp=delta, runs=runs, audit=audit), indent=2, allow_nan=False) + '\n')
    (output / 'AUDIT.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    (output / 'REPORT_ZH.md').write_text('\n'.join(lines) + '\n')
    return dict(rows=rows, delta_pp=delta)


if __name__ == '__main__':
    config = json.load(sys.stdin)
    print(json.dumps(report(json.loads(Path(config['snapshot']).read_text()), config['output'], config['audit']), indent=2, allow_nan=False))
