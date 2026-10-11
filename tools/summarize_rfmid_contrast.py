"""Four-cell RFMiD contributions: two published controls and two new arms."""
import json
from pathlib import Path
import sys

from prepare_rfmid_ml import BENCHMARK_LABELS
from summarize_rfmid import FIELDS, summarize

ARMS = dict(A0='base', AL='local_transport', AC='difference_transport',
            ACS='difference_shared_removed_transport')
PAIRS = (('AC','A0'),('ACS','A0'),('AC','AL'),('ACS','AL'),('ACS','AC'))
EXTRA = ('final_macro_AUROC','final_macro_F1','final_micro_F1')


def report(snapshot, output, audit):
    output = Path(output)
    if output.exists(): raise ValueError('Do not overwrite contrast results')
    runs = snapshot['runs']; data = {a:runs['RFMiD_'+m]['metrics'] for a,m in ARMS.items()}
    reference = data['A0']['stages']
    for a,m in data.items():
        if (m['dataset']!='RFMiD' or m['module']!=ARMS[a] or m['test_accessed'] is not False or
                m['evaluation_split']!='official_validation' or m['partition_variant']!='negative_coverage_v3' or
                len(m['stages'])!=5): raise ValueError('Contrast data/scope differs')
        for s,b in zip(m['stages'],reference):
            if s['visible_labels']!=b['visible_labels'] or [v['label'] for v in s['per_label']]!=list(BENCHMARK_LABELS):
                raise ValueError('Contrast stage label set differs')
            for v,w in zip(s['per_label'],b['per_label']):
                if (v['positive'],v['negative'],v['observed'])!=(w['positive'],w['negative'],w['observed']):
                    raise ValueError('Contrast paired label support differs')
                if v['label'] in s['visible_labels']:
                    if v['positive']<=0 or v['negative']<=0 or v['observed']!=640 or any(
                            v[k] is None or not 0<=v[k]<=1 for k in ('AP','AUROC')):
                        raise ValueError('Visible label metric invalid')
                elif v['AP'] is not None or v['AUROC'] is not None or v['observed']!=0:
                    raise ValueError('Future unknown label evaluated')
    rows={a:summarize(m['stages']) for a,m in data.items()}
    for a in rows:
        aggregate=data[a]['stages'][-1]['aggregate']
        rows[a].update(final_macro_F1=aggregate['macro_F1'],final_micro_F1=aggregate['micro_F1'])
    delta={a+'-'+b:{k:100*(rows[a][k]-rows[b][k]) for k in (*FIELDS,*EXTRA)} for a,b in PAIRS}
    contribution={}
    for a,b in PAIRS:
        contribution[a+'-'+b]=dict(summary_pp=delta[a+'-'+b],stages=[])
        for x,y in zip(data[a]['stages'],data[b]['stages']):
            per_label=[dict(label=v['label'],positive=v['positive'],negative=v['negative'],
                AP_delta_pp=None if v['AP'] is None else 100*(v['AP']-w['AP']),
                AUROC_delta_pp=None if v['AUROC'] is None else 100*(v['AUROC']-w['AUROC']))
                for v,w in zip(x['per_label'],y['per_label'])]
            contribution[a+'-'+b]['stages'].append(dict(task=x['task'],
                mAP_delta_pp=100*(x['aggregate']['mAP']-y['aggregate']['mAP']),per_label=per_label,
                groups_AP_delta_pp={k:None if x['groups'][k]['AP'] is None else 100*(x['groups'][k]['AP']-y['groups'][k]['AP'])
                                    for k in x['groups']}))
    lines=['# RFMiD 正负差异与去共享贡献对照','',
        'A0/AL严格复用第三轮已公开结果，AC/ACS为两条新真实训练；仅official validation，test封存。',
        '四格课程/样本/mask/60更新相同。AC−AL同时改变列数、方向与能量；ACS−AC同时去共享并降低能量，不作等强度或单因素因果归因。','',
        '|格|方法|final mAP %|avg mAP %|tail AP %|forget pp|old AP %|new AP %|AUROC %|macro F1 %|micro F1 %|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,m in ARMS.items(): lines.append('|'+a+'|'+m+'|'+'|'.join(f'{100*rows[a][k]:.6f}' for k in (*FIELDS,*EXTRA))+'|')
    for name,v in delta.items(): lines.append('|'+name+'|全部pp|'+'|'.join(f'{v[k]:+.6f}' for k in (*FIELDS,*EXTRA))+'|')
    lines+=['','|task|seen|A0 mAP %|AL mAP %|AC mAP %|ACS mAP %|','|---|---:|---:|---:|---:|---:|']
    for i,s in enumerate(reference): lines.append(f'|{i+1}|{len(s["visible_labels"])}|'+'|'.join(f'{100*data[a]["stages"][i]["aggregate"]["mAP"]:.6f}' for a in ARMS)+'|')
    for i,s in enumerate(reference):
        lines+=['',f'## Stage {i+1} 全29标签（未来标签为 —）','',
            '|label|val P/N|A0 AP %|AL AP %|AC AP %|ACS AP %|AC−A0 pp|ACS−A0 pp|AC−AL pp|ACS−AL pp|ACS−AC pp|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for j,v in enumerate(s['per_label']):
            values=[data[a]['stages'][i]['per_label'][j]['AP'] for a in ARMS]
            text=['—' if x is None else f'{100*x:.6f}' for x in values]
            diffs=[contribution[a+'-'+b]['stages'][i]['per_label'][j]['AP_delta_pp'] for a,b in PAIRS]
            lines.append('|'+v['label']+f'|{v["positive"]}/{v["negative"]}|'+'|'.join(text+['—' if x is None else f'{x:+.6f}' for x in diffs])+'|')
    lines+=['','## 固定训练频数组与旧/新标签','',
        '|组|A0 AP %|AL AP %|AC AP %|ACS AP %|AC−A0 pp|ACS−A0 pp|AC−AL pp|ACS−AL pp|ACS−AC pp|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in ('head','middle','tail','old','new'):
        v={a:data[a]['stages'][-1]['groups'][g]['AP'] for a in ARMS}
        lines.append('|'+g+'|'+'|'.join([f'{100*v[a]:.6f}' for a in ARMS]+[f'{100*(v[a]-v[b]):+.6f}' for a,b in PAIRS])+'|')
    lines+=['','CSR是机制重点，AION/ODP/LS和其余标签全部正负均保留；不把小分母AP、先验激活或头参数变化当成功。',
        '完整AUROC/F1和所有支持度保存在RESULTS；未来标签不当negative或计入macro，AP遗忘允许负。单seed/order重复val，无patient/eyeID及内容重叠核对，仅开发证据。',
        f'新增GPU驻留{audit["round_gpu_process_residence_seconds"]:.6f}秒；复用两控制不重新收费。全部闭合GPU累计{audit["cumulative_gpu_process_residence_seconds"]:.6f}秒，独立round各计一次，CPU与并行墙钟另列。']
    output.mkdir()
    (output/'RESULTS.json').write_text(json.dumps(dict(rows=rows,delta_pp=delta,runs=runs,audit=audit),indent=2,allow_nan=False)+'\n')
    (output/'CONTRIBUTIONS.json').write_text(json.dumps(contribution,indent=2,allow_nan=False)+'\n')
    (output/'AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return dict(rows=rows,delta_pp=delta)


if __name__=='__main__':
    c=json.load(sys.stdin)
    print(json.dumps(report(json.loads(Path(c['snapshot']).read_text()),c['output'],c['audit']),allow_nan=False))
