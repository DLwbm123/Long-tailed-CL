"""Anonymous diagnostic collection and complete three-mode, paired, w reporting."""
import csv
import json
from pathlib import Path
import subprocess
import sys

from apart_batch_diagnostic import MODES
from summarize_external_methods import FIELDS


def report(s, out):
    out=Path(out);out.mkdir(exist_ok=False)
    lock=s['lock'];program=s['program'];runs=s['runs']
    costs=sum(c['residence_seconds'] for c in program['costs'])
    if program['status']!='COMPLETE' or program['active'] or program['failures']:
        audit=dict(status='INCOMPLETE',round_gpu_process_residence_seconds=costs,
            optimizer_updates=0,test_accessed=False,complete_comparison=False)
        (out/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
        (out/'PARTIAL_RESULTS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
        (out/'REPORT_ZH.md').write_text('# APART诊断未完整结束\n\n全部匿名部分结果/费用保留，缺失不填0；私有核对错误后补具体去标识原因。不重训、不自动重试。\n')
        return audit
    assert len(program['costs'])==2 and all(c['returncode']==0 and c['gpu'] in (0,1) for c in program['costs'])
    reproduction={};rows=[];contrasts=[]
    for ds in ('ISIC','HK'):
        r=runs[ds];data=lock['data'][ds]
        assert r['optimizer_updates']==0 and not r['train_accessed'] and not r['test_accessed']
        assert r['parameters_unchanged'] and r['checkpoint_metadata_all_pass']
        assert r['model_source_commit']==lock['checkpoint_source_commit']
        for mode in MODES:
            m=r['metrics'][mode]
            assert len(m['stages'])==len(data['task_sizes'])
            for stage in m['stages']:
                seen=data['order'][:sum(data['task_sizes'][:stage['task']])]
                assert stage['seen']==seen and stage['per_class_n']=={str(c):data['val_counts'][str(c)] for c in seen}
            rows.append(dict(dataset=ds,mode=mode,**{f:100*m[f] for f in FIELDS},
                final_accuracy=100*m['final_accuracy'],average_incremental_accuracy=100*m['average_incremental_accuracy'],
                final_macro_F1=100*m['final_macro_F1']))
        original=s['original_results']['runs'][ds+'_APART_full']['metrics']
        mixed=r['metrics']['mixed_batch']
        reproduction[ds]=dict(exact_per_class_recalls=all(a['per_class_recall']==b['per_class_recall']
            for a,b in zip(original['stages'],mixed['stages'])),max_six_metric_abs_difference=max(abs(original[f]-mixed[f]) for f in FIELDS))
        for a,b in [('mixed_batch','class_concat'),('mixed_batch','individual_routing'),('class_concat','individual_routing')]:
            contrasts.append(dict(dataset=ds,reference=a,candidate=b,
                delta_pp={f:100*(r['metrics'][b][f]-r['metrics'][a][f]) for f in FIELDS}))
    ledger=s['closed_round_cost_ledger'].copy();ledger[s['round_name']]=costs
    audit=dict(status='PASS' if all(v['exact_per_class_recalls'] for v in reproduction.values()) else 'BASELINE_REPRODUCTION_MISMATCH',
        complete_comparison=True,checkpoint_stages=9,mode_stage_evaluations=27,
        validation_image_exposures=sum(v['validation_image_exposures'] for v in runs.values()),
        optimizer_updates=0,train_accessed=False,test_accessed=False,parameters_unchanged=True,
        checkpoint_source_commit=lock['checkpoint_source_commit'],diagnostic_source_commit=s['source_commit'],
        successful_cost_records=2,failures=[],dispatch_gpus=[0,1],mixed_reproduction=reproduction,
        round_gpu_process_residence_seconds=costs,cumulative_gpu_process_residence_seconds=83019.61353103025+sum(ledger.values()),
        closed_round_cost_ledger=ledger,parallel_wall_seconds=program['elapsed_seconds'],cpu_checks=s['cpu_checks'],
        engineering_events=s.get('engineering_events',[]))
    result=dict(rows=rows,runs=runs,contributions=contrasts,audit=audit,costs=program['costs'],lock=lock)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (out/'CONTRIBUTIONS.json').write_text(json.dumps(contrasts,indent=2)+'\n')
    lines=['# APART既有checkpoint：评价批次/逐样本路由/w冻结诊断','',
        '仅已有val和9个既有checkpoint，无train访问/optimizer/调参/test。按类拼接利用已知val标签组织batch，仅诊断作者实现，不作为临床部署方案。逐样本模式只关闭两个池的多数票，仍batch48；不是batch1重训。w使用真实val类别对应train频数，仅诊断，不参与任何预测。','',
        '|数据集|模式|最终BA|平均BA|尾类|遗忘↓|旧类|新类|最终Acc|平均Acc|macroF1|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:lines.append('|'+row['dataset']+'|'+row['mode']+'|'+'|'.join(f'{row[f]:.6f}' for f in (*FIELDS,'final_accuracy','average_incremental_accuracy','final_macro_F1'))+'|')
    lines+=['','## 全部模式增量（candidate−reference，pp）','','|数据集|比较|BA|平均BA|尾类|遗忘|旧类|新类|','|---|---|---:|---:|---:|---:|---:|---:|']
    for d in contrasts:lines.append('|'+d['dataset']+'|'+d['candidate']+'−'+d['reference']+'|'+'|'.join(f'{d["delta_pp"][f]:+.6f}' for f in FIELDS)+'|')
    lines+=['','## 各stage完整Acc/BA/F1','','|数据集|模式|task|n|Acc|BA|macroF1|','|---|---|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for mode in MODES:
            for st in runs[ds]['metrics'][mode]['stages']:
                lines.append(f'|{ds}|{mode}|{st["task"]}|{st["validation_n"]}|{100*st["accuracy"]:.6f}|{100*st["balanced_accuracy"]:.6f}|{100*st["macro_F1"]:.6f}|')
    lines+=['','## 全部31类最终召回及n','','|数据集|类|n|mixed|concat|individual|concat−mixed|individual−mixed|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ('ISIC','HK'):
        for c,n in sorted(lock['data'][ds]['val_counts'].items(),key=lambda x:int(x[0])):
            v=[100*runs[ds]['metrics'][m]['stages'][-1]['per_class_recall'][c] for m in MODES]
            lines.append(f'|{ds}|{c}|{n}|'+ '|'.join(f'{x:.6f}' for x in v)+f'|{v[1]-v[0]:+.6f}|{v[2]-v[0]:+.6f}|')
    lines+=['','## 配对预测变化：救回和损伤均保留','','|数据集|task|比较|n|预测改变|救回|损伤|任一池路由改变|','|---|---:|---|---:|---:|---:|---:|---:|']
    for ds,r in runs.items():
        for stage in r['paired_stages']:
            for name,values in stage['comparisons'].items():
                p=values['all'];lines.append(f'|{ds}|{stage["task"]}|{name}|{p["n"]}|{p["predictions_changed"]}|{p["rescued"]}|{p["harmed"]}|{p["either_pool_route_changed"]}|')
    lines+=['','## w：各阶段全部/旧/新/native rare/common','','|数据集|task|组|n|mean|median|≤.01个数|≤.05个数|≥.95个数|近零screen|','|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for ds,r in runs.items():
        for st in r['w_stages']:
            for group,w in st['groups'].items():
                if w is None:continue
                lines.append(f'|{ds}|{st["task"]}|{group}|{w["n"]}|{w["mean"]:.8f}|{w["quantiles"]["0.5"]:.8f}|{w["n_le_001"]}|{w["n_le_005"]}|{w["n_ge_095"]}|{w["near_zero_screen"]}|')
    lines+=['','## 每类首次引入/最终w','','|数据集|类|n|首次mean|最终mean|最终≤.05比例|近零screen|','|---|---:|---:|---:|---:|---:|---|']
    for ds,r in runs.items():
        for c,n in sorted(lock['data'][ds]['val_counts'].items(),key=lambda x:int(x[0])):
            first=next(s['per_class'][c] for s in r['w_stages'] if c in s['per_class']);last=r['w_stages'][-1]['per_class'][c]
            lines.append(f'|{ds}|{c}|{n}|{first["mean"]:.8f}|{last["mean"]:.8f}|{last["n_le_005"]/n:.6f}|{last["near_zero_screen"]}|')
    lines+=['',f'审计{audit["status"]}；mixed复核{reproduction}。本轮GPU驻留{costs:.6f}秒，协调器墙钟{program["elapsed_seconds"]:.6f}秒，CPU另列；旧训练/评价不重复收费。','',
        '近零screen为冻结的≥90%样本w≤.05且mean≤.05，仅描述当前checkpoint在val上的条件权重，不证明历史训练何时坍缩；rare阈值train≤100。全部分位数/每stage每类分母/救回与损伤见RESULTS.json及STAGE_CLASS_RECALLS.csv。','',
        '不选择更高模式覆盖原结果，不把类分组的标签上下文当作部署性能。single seed/order、反复val、小n不是临床或封存test证据。批次诊断改善或未改善均完整报告；不自动修复、重训或扩展诊断。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    with (out/'STAGE_CLASS_RECALLS.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['dataset','mode','task','class','val_n','recall_percent'])
        for ds,r in runs.items():
            for mode in MODES:
                for st in r['metrics'][mode]['stages']:
                    for c in st['seen']:w.writerow([ds,mode,st['task'],c,st['per_class_n'][str(c)],100*st['per_class_recall'][str(c)]])
    return audit


def collect(round_dir):
    r=Path(round_dir);h=json.loads((r/'local_runtime/HANDOFF.private.json').read_text())
    if (r/'results').exists():raise ValueError('No repeat collection')
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
runs={}
for ds in ('ISIC','HK'):
 folder=root/'runs'/ds
 if (folder/'RESULTS.json').exists():runs[ds]=json.loads((folder/'RESULTS.json').read_text())
 elif (folder/'PARTIAL_AGGREGATES.json').exists():runs[ds]=json.loads((folder/'PARTIAL_AGGREGATES.json').read_text())
print(json.dumps(dict(program=p,runs=runs),allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),
        original_results=json.loads(Path('docs/air_apart_isic_hk1_20261012/results/RESULTS.json').read_text()),
        source_commit=h['source_commit'],round_name=r.name,cpu_checks=json.loads((r/'CPU_CHECK.json').read_text()),
        engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'],
        closed_round_cost_ledger={f.parent.parent.name:json.loads(f.read_text())['round_gpu_process_residence_seconds']
            for f in Path('docs').glob('*/results/AUDIT.json') if 'round_gpu_process_residence_seconds' in json.loads(f.read_text())})
    (r/'local_runtime/SNAPSHOT_ANONYMOUS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    return report(s,r/'results')


if __name__=='__main__':print(json.dumps(collect(json.load(sys.stdin)['round_dir']),indent=2))
