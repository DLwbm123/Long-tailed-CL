"""One-shot anonymous main-grid collection; all signs, margins and process costs."""
import csv
import json
from pathlib import Path
import subprocess
from lca_pair_metrics import FIELDS,aggregate

JOBS=[ds+'_'+a for ds in ('ISIC','HK') for a in ('base','local_transport')]
METHODS=['A0','AL','B','P','AIR']


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate(s):
    p=s['program'];lock=s['lock'];runs=s['runs']
    complete=p['status']=='COMPLETE' and not p['active'] and not p['failures'] and set(runs)==set(JOBS)
    if not complete:return False
    assert len(p['costs'])==8 and len(p['cpu_costs'])==4
    assert {(v['job'],v['operation']) for v in p['costs']}=={(n,op) for n in JOBS for op in ('preflight','evaluate')}
    assert all(v['returncode']==0 and v['gpu'] in (0,1) for v in p['costs'])
    assert all(v['returncode']==0 and v['gpu'] is None for v in p['cpu_costs'])
    assert set(s['prepared'])==set(s['preflights'])==set(JOBS)
    assert p['sidecars_sealed_at']<=p['preflights_sealed_at']
    for name,r in runs.items():
        d=lock['data'][r['dataset']];f=s['preflights'][name];a=s['prepared'][name]
        assert a['status']=='PREPARED' and len(a['stages'])==len(d['task_sizes'])
        assert f['status']=='PASS' and f['parameters_unchanged'] and f['CPU_GPU_guard_exact']
        assert r['status']['status']=='COMPLETE' and r['source_commit']==f['source_commit']==s['source_commit']
        assert r['version']==lock['version'] and r['control_source']==lock['controls'][r['control']]['source']
        assert r['optimizer_updates']==r['policy_updates']==f['optimizer_updates']==0
        assert not r['train_accessed'] and not r['test_accessed'] and not f['test_accessed']
        assert len(r['stages'])==len(d['task_sizes'])
        for mode in ('OFF','ON'):
            rebuilt=aggregate([t['metrics'][mode] for t in r['stages']],d)
            assert rebuilt==r['metrics'][mode]
        for i,t in enumerate(r['stages']):
            ref=lock['control_references'][name]['stages'][i]
            assert t['published_control_exact'] and t['encoder_state_unchanged']
            assert t['metrics']['OFF']['per_class_n']==ref['per_class_n']
            assert t['metrics']['OFF']['per_class_recall']==ref['per_class_recall']
            assert t['metrics']['OFF']['accuracy']==ref['accuracy']
            assert t['effect']['new_margin_changed']==t['effect']['new_prediction_changed']==t['effect']['predicted_group_changed']==0
            for group,v in t['paired'].items():
                assert v['net_correct']==v['correct_on']-v['correct_off']==v['rescued']-v['harmed']
                assert v['changed']==v['rescued']+v['harmed']+v['wrong_to_different_wrong']
                if group not in ('all','old','new'):assert v['n']==d['val_counts'][group]
            m=t['mechanical'];assert m['native_residual']<=1e-8 and m.get('relative_residual',0)<=1e-8
            assert m.get('relative_native_energy_step',0)<=.100000001
    return True


def report(s,out):
    complete=validate(s);out=Path(out);out.mkdir(exist_ok=False)
    lock=s['lock'];p=s['program'];metrics={};rows=[];contrasts=[];gaps=[];gates=[];classes=[];paired=[];margins=[];flows=[];effects=[]
    all_fields=(*FIELDS,'final_accuracy','average_incremental_accuracy','final_macro_F1')
    for ds in ('ISIC','HK'):
        metrics[ds]={'AIR':lock['AIR_references'][ds]}
        for arm,off,on in [('base','A0','B'),('local_transport','AL','P')]:
            r=s['runs'].get(ds+'_'+arm,{})
            metrics[ds][off]=r.get('metrics',{}).get('OFF',lock['control_references'][ds+'_'+arm])
            metrics[ds][on]=r.get('metrics',{}).get('ON')
        m=metrics[ds]
        for method in METHODS:
            v=m[method];rows.append(dict(dataset=ds,method=method,**{f:None if v is None or v.get(f) is None else 100*v[f] for f in all_fields}))
            if v:
                for t in v['stages']:
                    for label,n in t['per_class_n'].items():classes.append(dict(dataset=ds,method=method,task=t['task'],label=label,
                        train_n=lock['data'][ds]['train_counts'][label],val_n=n,correct_n=round(n*t['per_class_recall'][label]),recall_percent=100*t['per_class_recall'][label]))
        for a,b in [('B','A0'),('P','AL'),('P','B'),('P','A0'),('B','AIR'),('P','AIR')]:
            contrasts.append(dict(dataset=ds,contrast=a+'−'+b,delta_pp={f:100*(m[a][f]-m[b][f]) if m[a] and m[b] else None for f in FIELDS}))
        contrasts.append(dict(dataset=ds,contrast='(P−AL)−(B−A0)',delta_pp={f:100*((m['P'][f]-m['AL'][f])-(m['B'][f]-m['A0'][f])) if m['P'] and m['B'] else None for f in FIELDS}))
        for a,b in [('B','A0'),('P','AL')]:
            gaps.append(dict(dataset=ds,candidate=a,control=b,AIR_gap_before_pp=100*(m['AIR'][FIELDS[0]]-m[b][FIELDS[0]]),
                AIR_gap_after_pp=100*(m['AIR'][FIELDS[0]]-m[a][FIELDS[0]]) if m[a] else None,
                gap_shrink_pp=100*(m[a][FIELDS[0]]-m[b][FIELDS[0]]) if m[a] else None))
            delta={f:100*(m[a][f]-m[b][f]) for f in FIELDS} if m[a] else None
            gates.append(dict(dataset=ds,candidate=a,own_control_gates=None if not delta else
                dict(BA_ge_1=delta[FIELDS[0]]>=1,old_ge_1=delta[FIELDS[4]]>=1,tail_ge_minus_05=delta[FIELDS[2]]>=-.5,forget_le_1=delta[FIELDS[3]]<=1)))
    for name,r in s['runs'].items():
        for t in r.get('stages',[]):
            base=dict(job=name,dataset=r['dataset'],task=t['task'])
            for group,a in t['paired'].items():paired.append(dict(base,group=group,**a))
            for group,a in t['margins'].items():
                for mode,stats in a.items():margins.append(dict(base,group=group,mode=mode,**stats))
            for flow,a in t['flows']['OFF'].items():flows.append(dict(base,flow=flow,off=a,on=t['flows']['ON'][flow],delta=t['flows']['ON'][flow]-a))
            effects.append(dict(base,**t['effect'],**t['mechanical']))
    cost=sum(v['residence_seconds'] for v in p['costs']);cpu=sum(v['residence_seconds'] for v in p['cpu_costs'])
    ledger=dict(s['closed_round_cost_ledger']);assert s['round_name'] not in ledger
    history=83019.61353103025+sum(ledger.values());ledger[s['round_name']]=cost
    primary=[g for g in gates if g['candidate']=='P']
    audit=dict(status='PASS' if complete else 'INCOMPLETE',source_commit=s['source_commit'],version=lock['version'],
        completed_trajectories=sum(r.get('status',{}).get('status')=='COMPLETE' for r in s['runs'].values()),
        prepared_stage_checkpoints=sum(len(v.get('stages',[])) for v in s['prepared'].values()),
        real_zero_update_preflights=sum(v.get('status')=='PASS' for v in s['preflights'].values()),
        optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,complete_comparison=complete,
        cost_records=p['costs'],cpu_preparation_cost_records=p['cpu_costs'],failures=p['failures'],
        round_gpu_process_residence_seconds=cost,cpu_preparation_process_residence_seconds=cpu,
        historical_closed_gpu_process_residence_seconds=history,cumulative_gpu_process_residence_seconds=history+cost,
        closed_round_cost_ledger=ledger,whole_round_wall_seconds=p.get('whole_round_wall_seconds'),
        production_cpu_checks=s['cpu_checks'],historical_design_cpu_recharged=0,historical_control_training_or_AIR_cost_recharged=0,
        engineering_events=s['engineering_events'],gates=gates,
        primary_target_met=bool(complete and all(all(g['own_control_gates'].values()) for g in primary)),
        guard_preservation_passed=bool(complete),old_drift_truth_measured=False,independent_confirmation=False)
    dump(out/'AUDIT.json',audit);dump(out/'CONTRIBUTIONS.json',contrasts);dump(out/'AIR_GAPS.json',gaps)
    dump(out/'PREFLIGHT_CHECKS.json',s['preflights']);dump(out/'MECHANICAL.json',effects)
    dump(out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json'),dict(metrics=metrics,runs=s['runs'],prepared=s['prepared'],rows=rows,audit=audit))
    for filename,data in [('MAIN.csv',rows),('STAGE_CLASS_RECALLS.csv',classes),('PAIRED_COUNTS.csv',paired),('MARGINS.csv',margins),('CONFUSION_FLOWS.csv',flows)]:
        if data:
            with (out/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    fmt=lambda x: '未完成/未提供' if x is None else f'{x:.6f}'
    lines=['# 旧类对残差校正与新类保护：主四格','',
        'A0=原base；AL=原local_transport（固定主控制）；B=A0+新版；P=AL+新版。18个原阶段checkpoint只读复用；四条共享encoder的OFF/ON评价。AIR复用公开指标。',
        '无重训、调参、训练影像/test/reserved读取。新版仅边界readout，不反馈下一阶段。raw仅守卫前间隔/幅度诊断，不是新增评价臂。',
        '原checkpoint未嵌入完整source/order/seed元数据；来源由原CONFIG及独立DELIVERY、固定路径、seen/steps/module/n与重建共同验收。浮点头容差前瞻冻结，控制召回和准确率必须精确复现。',
        '守卫保持逐阶段新类logit、旧类最高分、每个输入旧新组别、每个新类预测集合/正确性与原始间隔；不保证新类softmax概率。真新类仍错分旧类时可改变错误旧标签，计入错→另一错。只能改变旧类内部排名，不能修复跨旧新组错误。早期新类在后续成为旧类后可变。','']
    if not complete:lines+=['**未完成四格比较；缺失不填0，不自动重试，不宣布模块贡献。具体停止原因另见工程事件。**','']
    lines+=['|数据集|方法|BA|平均BA|tail|forget↓|old|new|Acc|平均Acc|F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['dataset']+'|'+r['method']+'|'+'|'.join(fmt(r[f]) for f in all_fields)+'|')
    lines+=['','## 全部贡献，pp（遗忘负值有利）','','|数据集|比较|BA|平均BA|tail|forget|old|new|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:lines.append('|'+r['dataset']+'|'+r['contrast']+'|'+'|'.join('缺失' if r['delta_pp'][f] is None else f'{r["delta_pp"][f]:+.6f}' for f in FIELDS)+'|')
    lines+=['','## AIR最终BA差距','','|数据集|臂|AIR−控制|AIR−新版|缩小pp|','|---|---|---:|---:|---:|']
    for r in gaps:lines.append('|'+r['dataset']+'|'+r['candidate']+'|'+'|'.join(fmt(r[k]) for k in ('AIR_gap_before_pp','AIR_gap_after_pp','gap_shrink_pp'))+'|')
    lines+=['','## 全stage评价','','|数据集|方法|stage|n|Acc|BA|F1|','|---|---|---:|---:|---:|---:|---:|']
    for ds,m in metrics.items():
        for name,v in m.items():
            if v:
                for t in v['stages']:lines.append(f'|{ds}|{name}|{t["task"]}|{t["validation_n"]}|{100*t["accuracy"]:.6f}|{100*t["balanced_accuracy"]:.6f}|{fmt(None if t.get("macro_F1") is None else 100*t["macro_F1"])}|')
    lines+=['','## 每阶段全部纠正和破坏（重复val曝光按stage分别计数）','','|轨迹|stage|组|n|改变|纠正|破坏|错→另一错|净正确|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:lines.append('|'+r['job']+'|'+str(r['task'])+'|'+r['group']+'|'+'|'.join(str(r[k]) for k in ('n','changed','rescued','harmed','wrong_to_different_wrong','net_correct'))+'|')
    lines+=['','## 新类间隔与实际作用','','|轨迹|stage|raw RMS|guard RMS|raw新类间隔下降数|guard新类间隔变化|新类决策集合变化|组别变化|真新类错旧标签变化|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in effects:lines.append('|'+r['job']+'|'+str(r['task'])+f'|{r["raw_rms"]:.9g}|{r["guarded_rms"]:.9g}|'+ '|'.join(str(r[k]) for k in ('raw_new_margin_decreased','new_margin_changed','new_prediction_changed','predicted_group_changed','new_sample_wrong_label_changed'))+'|')
    for ds in ('ISIC','HK'):
        lines+=['',f'## {ds}全部最终类别recall','','|类|train n|val n|A0|AL|B|P|AIR|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for c in lock['data'][ds]['order']:
            values=[None if metrics[ds][a] is None else 100*metrics[ds][a]['stages'][-1]['per_class_recall'][str(c)] for a in METHODS]
            lines.append('|'+str(c)+'|'+str(lock['data'][ds]['train_counts'][str(c)])+'|'+str(lock['data'][ds]['val_counts'][str(c)])+'|'+'|'.join(fmt(v) for v in values)+'|')
    lines+=['','## 费用与证据边界','',f'本轮GPU进程驻留 {cost:.9f}s，CPU实际checkpoint解析准备驻留 {cpu:.9f}s；墙钟 {p.get("whole_round_wall_seconds")}s。',
        f'历史闭合GPU {history:.9f}s；累计 {history+cost:.9f}s。成功与失败逐次计费，CPU pipeline包含在各自驻留中。CPU生产合成检查另见AUDIT；历史设计23.633729566s不重计。',
        'CPU准备0医学影像但读取18个私有checkpoint统计；GPU预检与评价只读现有val。原训练和AIR费用0新增，当前推理与失败均收费。费用是进程驻留，不是纯GPU计算或货币账单。',
        '每条轨迹的实际checkpoint/sidecar字节、GPU峰值及val曝光见RESULTS；无旧图回放、无额外optimizer更新。',
        '全部margin正零负计数见MARGINS.csv；n<5类别连续值抑制，计数/召回完整。CONFUSION_FLOWS保存全部旧新错误流，STAGE_CLASS_RECALLS保存全部stage支持度。',
        '本轮以当前已反复使用的development val给出单seed/order证据；旧均值不是漂移真值。主四格不能单独归因曲率归一化、能量限幅或守卫作用，也不能作为原版LCA复现、独立患者或临床证据。',
        '不因参数或分数非零声称有贡献。双数据集门须同时满足：相对各自控制BA和old均≥1pp、tail≥−0.5pp、forget≤1pp，并且新类预测/间隔/组别变化精确为0。结果后不追加实验。','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit


def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime';h=json.loads((rt/'HANDOFF.private.json').read_text())
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('No repeat collection')
    code='import json,os\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\nnames='+repr(JOBS)+'\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
for d in Path('/proc').iterdir():
 if d.name.isdigit():
  try:cmd=(d/'cmdline').read_bytes().split(b'\\0')
  except (FileNotFoundError,PermissionError):continue
  assert b'/tmp/q167w.py' not in cmd
value=dict(program=p,runs={},prepared={},preflights={})
for name in names:
 for target,folder,file in [('runs','runs','RESULTS.json'),('prepared','prepared','PREPARED.json'),('preflights','preflight','RESULTS.json')]:
  path=root/folder/name/file
  if path.exists():
   v=json.loads(path.read_text())
   if target=='runs':v['status']=json.loads((path.parent/'STATUS.json').read_text())
   value[target][name]=v
print(json.dumps(value,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(round_name=r.name,source_commit=h['source_commit'],
        lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),
        engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'],
        closed_round_cost_ledger={p.parent.parent.name:json.loads(p.read_text())['round_gpu_process_residence_seconds']
            for p in Path('docs').glob('*/results/AUDIT.json') if 'round_gpu_process_residence_seconds' in json.loads(p.read_text())})
    dump(rt/'SNAPSHOT_ANONYMOUS.json',s)
    return report(s,r/'results')


if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
