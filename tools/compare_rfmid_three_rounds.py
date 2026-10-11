"""Three-round anonymous RFMiD comparison, with fixed focus/delayed-label sets."""
import json
from pathlib import Path
import sys

from compare_rfmid_rounds import compare
from summarize_rfmid import FIELDS


def compare_three(results, locks):
    assert len(results) == len(locks) == 3
    comparisons = [compare(results[i], results[2], locks[i], locks[2]) for i in (0, 1)]
    labels = locks[2]['labels']
    intro = [{c: t for t, group in enumerate(lock['introduced_groups']) for c in group}
             for lock in locks[1:]]
    sets = dict(focus=['AION', 'ODP', 'LS', 'CSR'],
                delayed_vs_v2=[c for c in labels if intro[1][c] > intro[0][c]],
                earlier_vs_v2=[c for c in labels if intro[1][c] < intro[0][c]],
                unchanged_vs_v2=[c for c in labels if intro[1][c] == intro[0][c]])
    tables = {f'round{i+1}_{arm}': {v['label']: v for v in result['runs']['RFMiD_'+arm]['metrics']['stages'][-1]['per_label']}
              for i, result in enumerate(results) for arm in ('base', 'local_transport')}
    group_AP = {name: dict(labels=members, AP={key: sum(table[c]['AP'] for c in members)/len(members)
                                            for key, table in tables.items()}) for name, members in sets.items()}
    rows = {f'round{i+1}_{arm}': {**result['rows'][arm],
            'final_macro_F1':result['runs']['RFMiD_'+arm]['metrics']['stages'][-1]['aggregate']['macro_F1'],
            'final_micro_F1':result['runs']['RFMiD_'+arm]['metrics']['stages'][-1]['aggregate']['micro_F1']}
            for i,result in enumerate(results) for arm in ('base','local_transport')}
    lines=['# RFMiD 三轮完整对比', '',
           '相同官方validation640图、29标签、固定train频数组；test封存。第一轮66步/arm，第二与第三轮均60步/arm。第三轮重点标签与常见锚点前移、Normal分散到五task、其余组和样本分配改变。跨轮为描述性证据，不能将变化归于负例覆盖单一因素。', '',
           '|轮次/方法|最终mAP %|平均增量mAP %|tail AP %|AP遗忘 pp|旧AP %|新AP %|AUROC %|macro F1 %|micro F1 %|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fields=(*FIELDS,'final_macro_AUROC','final_macro_F1','final_micro_F1')
    for key,value in rows.items():lines.append('|'+ '|'.join([key]+[f'{100*value[k]:.4f}' for k in fields])+'|')
    lines += ['', '平均增量mAP、遗忘及各轮old/new AP依赖课程与标签集合，不能当同一任务指标。固定集合与共用已见stage对比见后面两份逐轮完整报告。F1固定threshold.5，不调整阈值。', '',
              '|固定集合|标签|第一base AP %|第一candidate|第二base|第二candidate|第三base|第三candidate|',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,row in group_AP.items():
        lines.append('|'+ '|'.join([name,', '.join(row['labels'])]+[f'{100*v:.4f}' for v in row['AP'].values()])+'|')
    lines += ['', '|label|val正n|val负n|相对第二轮进入时机|第一base AP %|第一candidate|第二base|第二candidate|第三base|第三candidate|第三base−第二 pp|第三candidate−第二 pp|',
              '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    per_label=[]
    for c in labels:
        values={key:table[c]['AP'] for key,table in tables.items()}
        n=[(table[c]['positive'],table[c]['negative']) for table in tables.values()];assert len(set(n))==1
        timing='推迟' if c in sets['delayed_vs_v2'] else '提前' if c in sets['earlier_vs_v2'] else '相同'
        per_label.append(dict(label=c,positive=n[0][0],negative=n[0][1],AP=values,timing_vs_v2=timing))
        lines.append('|'+ '|'.join([c,str(n[0][0]),str(n[0][1]),timing]+[f'{100*v:.4f}' for v in values.values()]+[
            f'{100*(values["round3_"+arm]-values["round2_"+arm]):+.4f}' for arm in ('base','local_transport')])+'|')
    for i,(comparison,report) in enumerate(comparisons):
        lines += ['', f'## 第三轮与第{i+1}轮：全部指标、固定组与stage对比', '', report]
    lines += ['', '支持度/负例组成见冻结NEGATIVE_COVERAGE_AUDIT与TASK_SUPPORT_AUDIT。四重点标签全部训练负例覆盖不等于净收益；推迟标签的得失全部报告。单seed、重复validation、少量阳性及患者ID缺失限制保留。三轮原成本分别记账，不重跑或重计旧轨迹。']
    output=dict(status='DESCRIPTIVE_THREE_ROUND_COMPARISON',rows=rows,per_label=per_label,
                fixed_focus_and_timing_sets=group_AP,comparisons_to_round1_and_round2=[v[0] for v in comparisons],
                all_29_labels_reported=True,denominators_equal=True,causal_single_factor_attribution=False,
                old_runs_reused_without_gpu=True,test_accessed=False)
    return output,'\n'.join(lines)+'\n'


if __name__=='__main__':
    config=json.load(sys.stdin);read=lambda p:json.loads(Path(p).read_text())
    result,report=compare_three([read(p) for p in config['results']],[read(p) for p in config['protocols']])
    for name,text in [('THREE_ROUND.json',json.dumps(result,indent=2,allow_nan=False)+'\n'),('THREE_ROUND_ZH.md',report)]:
        target=Path(config['output'])/name;assert not target.exists();target.write_text(text)
    print(json.dumps({'all_29_labels_reported':True,'fixed_sets':result['fixed_focus_and_timing_sets']},indent=2))
