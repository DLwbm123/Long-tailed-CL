"""Collect only anonymous fixed-state aggregate diagnostics and complete cost records."""
import csv
import json
from pathlib import Path
import subprocess

JOBS=[ds+'_'+arm for ds in ('ISIC','HK') for arm in ('lca_supported','local_transport_lca_supported')]
METRICS=('balanced_accuracy','accuracy','tail_recall','old_recall','new_recall','macro_f1')


def report(s,out):
    out=Path(out);p=s['program'];runs=s['runs'];lock=s['lock']
    complete=p['status']=='COMPLETE' and not p['active'] and not p['failures'] and set(runs)==set(JOBS)
    if complete:
        assert len(p['costs'])==4 and {x['job'] for x in p['costs']}==set(JOBS)
        assert all(x['returncode']==0 and x['gpu'] in (0,1) for x in p['costs'])
        for name,r in runs.items():
            assert r['status']['status']=='COMPLETE'
            assert r['on_published_parity'] and r['encoder_state_unchanged'] and r['checkpoint_metadata_verified']
            assert r['optimizer_updates']==r['policy_updates']==0 and not r['train_accessed'] and not r['test_accessed']
            assert r['checkpoint_source_commit']==lock['checkpoint_source_commit']
            assert r['diagnostic_source_commit']==s['source_commit']
            assert r['mechanical']['solver_relative_residual_max']<=1e-8
            data=lock['data'][r['dataset']]
            assert r['validation_image_exposures']==sum(data['val_counts'].values())
            for mode in ('OFF','ON'):
                m=r['metrics'][mode];assert m['per_class_n']==data['val_counts']
                assert sum(r['confusion'][mode].values())==m['validation_n']
                assert abs(m['balanced_accuracy']-sum(m['per_class_recall'].values())/len(data['order']))<1e-12
            assert r['metrics']['ON']['per_class_recall']==lock['on_references'][name]['per_class_recall']
            for c in data['order']:
                a=r['paired'][str(c)];assert a['n']==data['val_counts'][str(c)]
                assert a['net_correct']==a['rescued']-a['harmed']==a['correct_on']-a['correct_off']
                assert a['changed']==a['rescued']+a['harmed']+a['wrong_to_different_wrong']
    cost=sum(c['residence_seconds'] for c in p['costs']);ledger=dict(s['closed_round_cost_ledger'])
    assert s['round_name'] not in ledger
    history=83019.61353103025+sum(ledger.values());ledger[s['round_name']]=cost
    audit=dict(status='PASS' if complete else 'INCOMPLETE',diagnostic_source_commit=s['source_commit'],
        source_commit=s['source_commit'],checkpoint_source_commit=lock['checkpoint_source_commit'],
        complete_comparison=complete,optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,
        completed_checkpoints=sum(r.get('status',{}).get('status')=='COMPLETE' for r in runs.values()),
        validation_image_exposures=sum(r.get('validation_image_exposures',0) for r in runs.values()),
        cost_records=p['costs'],failures=p['failures'],round_gpu_process_residence_seconds=cost,
        historical_closed_gpu_process_residence_seconds=history,closed_round_cost_ledger=ledger,
        cumulative_gpu_process_residence_seconds=history+cost,whole_round_wall_seconds=p.get('whole_round_wall_seconds'),
        cpu_checks=s['cpu_checks'],engineering_events=s.get('engineering_events',[]),
        new_training=False,new_checkpoint_selection=False,local_counterfactual_not_retraining=True,
        historical_costs_recharged=0)
    out.mkdir(exist_ok=False)
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json')).write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    lines=['# 最终 checkpoint 同状态模块开关诊断','',
        'OFF：保持 checkpoint 的 encoder、统计量、局部先验、竞争权重/RHS不变，仅令新增S为0后重解；ON：原保存float32解析头，double重建验收。差异统一为ON−OFF。OFF不是历史base/local_transport。',
        '仅最终stage，因此不制造平均增量或遗忘指标。单seed反复development val；没有重训、调参、train/test/reserved读取。',
        '本诊断不恢复去重前风险矩阵、不验证旧类漂移真值，也不能隔离训练过程中的全部历史作用。','']
    if not complete:
        lines += ['四状态未完整成功；missing不填0，不自动retry。需私有核对错误并在本报告补充去标识原因。','']
    rows=[];class_rows=[];confusion=[]
    lines += ['|状态|模式|BA|Acc|tail|old|new|macroF1|','|---|---|---:|---:|---:|---:|---:|---:|']
    for name in JOBS:
        r=runs.get(name,{})
        if 'metrics' not in r:continue
        for mode in ('OFF','ON'):
            v=r['metrics'][mode];rows.append(dict(job=name,mode=mode,**{m:100*v[m] for m in METRICS}))
            lines.append('|'+name+'|'+mode+'|'+'|'.join(f'{100*v[m]:.6f}' for m in METRICS)+'|')
        delta={m:100*(r['metrics']['ON'][m]-r['metrics']['OFF'][m]) for m in METRICS}
        lines.append('|'+name+'|ON−OFF pp|'+'|'.join(f'{delta[m]:+.6f}' for m in METRICS)+'|')
    lines+=['','## 全部纠正与破坏','', '|状态|组|n|预测改变|纠正|破坏|错→另一错|净正确|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in JOBS:
        r=runs.get(name,{})
        if 'paired' not in r:continue
        for group in ('all','old','new'):
            a=r['paired'][group]
            lines.append('|'+name+'|'+group+'|'+'|'.join(str(a[k]) for k in ('n','changed','rescued','harmed','wrong_to_different_wrong','net_correct'))+'|')
        data=lock['data'][r['dataset']]
        for c in data['order']:
            a=r['paired'][str(c)];class_rows.append(dict(job=name,original_label=c,
                group='new' if c in data['order'][-data['task_sizes'][-1]:] else 'old',**a,
                recall_off=100*r['metrics']['OFF']['per_class_recall'][str(c)],
                recall_on=100*r['metrics']['ON']['per_class_recall'][str(c)],
                delta_recall_pp=100*(r['metrics']['ON']['per_class_recall'][str(c)]-r['metrics']['OFF']['per_class_recall'][str(c)])))
    lines+=['','## 旧新类错误流向（ON−OFF为计数变化）','', '|状态|流向|OFF|ON|变化|','|---|---|---:|---:|---:|']
    for name in JOBS:
        r=runs.get(name,{})
        if 'confusion' not in r:continue
        for flow,v in r['confusion']['OFF'].items():
            w=r['confusion']['ON'][flow];confusion.append(dict(job=name,flow=flow,off=v,on=w,change=w-v))
            lines.append(f'|{name}|{flow}|{v}|{w}|{w-v:+d}|')
    lines+=['','## 实际作用','', '|状态|力/RHS|相对头变化|新类头变化范数|额外项新类直接力|分数RMS变化|相对分数RMS|', '|---|---:|---:|---:|---:|---:|---:|']
    for name in JOBS:
        r=runs.get(name,{})
        if 'mechanical' not in r:continue
        a=r['mechanical'];b=r['score_change']
        v=[a['force_at_off_relative_rhs'],a['delta_head_relative'],a['delta_new_head_norm'],a['new_direct_force_norm'],b['rms'],b['relative_rms']]
        lines.append('|'+name+'|'+'|'.join(f'{x:.9g}' for x in v)+'|')
    lines+=['','## 全部逐类正负','', '|状态|类|组|n|OFF recall|ON recall|差pp|纠正|破坏|改变|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in class_rows:
        lines.append(f"|{a['job']}|{a['original_label']}|{a['group']}|{a['n']}|{a['recall_off']:.6f}|{a['recall_on']:.6f}|{a['delta_recall_pp']:+.6f}|{a['rescued']}|{a['harmed']}|{a['changed']}|")
    lines+=['','## 费用与范围','',f'本轮GPU进程驻留 {cost:.9f} 秒；墙钟 {p.get("whole_round_wall_seconds")} 秒。各次进程（包括失败）逐条计费，原训练和旧评价不重复收费。',
        f'历史闭合GPU {history:.9f} 秒；含本诊断累计 {history+cost:.9f} 秒。',
        'CPU检查独立列于AUDIT；pipeline包含在驻留中，不重复相加。影像/模型/头/逐样本分数/特征/私有配置/raw日志不外传。',
        '全部标量、目标函数变化、每类和old/new间隔变化见RESULTS.json。完成后不自动改模块或追加实验。','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines))
    for filename,data in [('MAIN.csv',rows),('CLASS_CHANGES.csv',class_rows),('CONFUSION_FLOWS.csv',confusion)]:
        if data:
            with (out/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    return audit


def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime';h=json.loads((rt/'HANDOFF.private.json').read_text())
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('No repeat collection')
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\nnames='+repr(JOBS)+'\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
runs={}
for name in names:
 folder=root/'runs'/name
 if (folder/'STATUS.json').exists():
  st=json.loads((folder/'STATUS.json').read_text())
  value=json.loads((folder/'RESULTS.json').read_text()) if (folder/'RESULTS.json').exists() else {}
  value['status']={k:v for k,v in st.items() if k in ('status','optimizer_updates','policy_updates','train_accessed','test_accessed','error_type')}
  runs[name]=value
print(json.dumps(dict(program=p,runs=runs),allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout)
    # Runtime PIDs are not part of the public scientific output (terminal active is empty).
    s.update(round_name=r.name,source_commit=h['source_commit'],lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),
        cpu_checks=json.loads((r/'CPU_CHECK.json').read_text()),
        engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'],
        closed_round_cost_ledger={p.parent.parent.name:json.loads(p.read_text())['round_gpu_process_residence_seconds']
            for p in Path('docs').glob('*/results/AUDIT.json') if 'round_gpu_process_residence_seconds' in json.loads(p.read_text())})
    (rt/'SNAPSHOT_ANONYMOUS.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    return report(s,r/'results')


if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
