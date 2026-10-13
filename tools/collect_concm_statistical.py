"""Anonymous results from three prespecified variants; no selection by val."""
import csv
import json
from pathlib import Path
import subprocess
from concm_statistical import VARIANTS,VERSION
from concm_statistical_metrics import FIELDS,aggregate
from lca_pair_worker import JOBS


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate(s):
    p=s['program'];lock=s['lock']
    if p['active']:raise ValueError('Active processes cannot be collected')
    if p['status']!='COMPLETE':return False
    if p['failures'] or len(p['costs'])!=24 or len(p['cpu_costs'])!=4:raise ValueError('Cost coverage differs')
    expected={(v,n,op) for v in VARIANTS for n in JOBS for op in ('preflight','evaluate')}
    if {(a['variant'],a['job'],a['operation']) for a in p['costs']}!=expected:raise ValueError('Missing or duplicate GPU cost')
    if any(a['returncode'] or a['gpu'] not in (0,1) for a in p['costs']):raise ValueError('Invalid GPU admission/cost')
    if any(a['returncode'] or a['gpu'] is not None for a in p['cpu_costs']):raise ValueError('Invalid CPU cost')
    for name in JOBS:
        a=s['prepared'][name];ds=name.split('_')[0];d=lock['data'][ds]
        if a['status']!='PREPARED' or a['source_commit']!=s['source_commit'] or len(a['stages'])!=len(d['task_sizes']):raise ValueError('Prepared coverage mismatch')
        for v in VARIANTS:
            r=s['runs'][v][name];f=s['preflights'][v][name]
            if (r['status']['status']!='COMPLETE' or f['status']!='PASS' or not f['parameters_unchanged']
                    or not f['CPU_GPU_scores_close'] or f['classification_metrics_read']
                    or r['source_commit']!=s['source_commit'] or f['source_commit']!=s['source_commit']
                    or r['version']!=VERSION or r['variant']!=v
                    or r['optimizer_updates'] or f['optimizer_updates'] or r['policy_updates']
                    or r['test_accessed'] or r['train_accessed'] or f['test_accessed'] or f['train_accessed']):
                raise ValueError('Run/preflight scope mismatch')
            for mode in ('OFF','ON'):
                if aggregate([t['metrics'][mode] for t in r['stages']],d)!=r['metrics'][mode]:raise ValueError('Aggregate mismatch')
            for i,t in enumerate(r['stages']):
                ref=lock['control_references'][name]['stages'][i]
                if not t['published_control_exact'] or not t['encoder_state_unchanged']:raise ValueError('Parity or frozen state failed')
                for key in ('seen','per_class_n','per_class_recall','accuracy'):
                    if t['metrics']['OFF'][key]!=ref[key]:raise ValueError('Control parity differs')
                if t['mechanical']['solver']['relative_residual']>1e-8:raise ValueError('Solve residual too high')
                for z in t['paired'].values():
                    if z['net_correct']!=z['rescued']-z['harmed'] or z['changed']!=z['rescued']+z['harmed']+z['wrong_to_different_wrong']:
                        raise ValueError('Paired count identity failed')
    return True


def report(s,out):
    complete=validate(s);out=Path(out);out.mkdir(exist_ok=False)
    lock=s['lock'];program=s['program'];fields=(*FIELDS,'final_accuracy','average_incremental_accuracy','final_macro_F1')
    metrics={};rows=[];contrasts=[];paired=[];margins=[];flows=[];classes=[];mechanical=[];gaps=[];gates=[]
    for ds in ('ISIC','HK'):
        m={'A0':lock['control_references'][ds+'_base'],'AL':lock['control_references'][ds+'_local_transport'],'AIR':lock['AIR_references'][ds]}
        for v in VARIANTS:
            for arm,label in [('base','B'),('local_transport','P')]:
                r=s['runs'].get(v,{}).get(ds+'_'+arm)
                m[v+'_'+label]=None if not r else r['metrics']['ON']
        metrics[ds]=m
        for name,value in m.items():
            rows.append(dict(dataset=ds,method=name,**{f:None if not value or value.get(f) is None else 100*value[f] for f in fields}))
            if value:
                for t in value['stages']:
                    for c,n in t['per_class_n'].items():
                        classes.append(dict(dataset=ds,method=name,task=t['task'],label=c,train_n=lock['data'][ds]['train_counts'][c],val_n=n,correct_n=round(n*t['per_class_recall'][c]),recall_percent=100*t['per_class_recall'][c]))
        for v in VARIANTS:
            b,p=v+'_B',v+'_P'
            for a,c in [(b,'A0'),(p,'AL'),(p,b),(p,'A0'),(b,'AIR'),(p,'AIR')]:
                contrasts.append(dict(dataset=ds,contrast=a+'−'+c,delta_pp={f:100*(m[a][f]-m[c][f]) if m[a] and m[c] else None for f in FIELDS}))
            contrasts.append(dict(dataset=ds,contrast=f'({p}−AL)−({b}−A0)',delta_pp={f:100*((m[p][f]-m['AL'][f])-(m[b][f]-m['A0'][f])) if m[p] and m[b] else None for f in FIELDS}))
            for a,c in [(b,'A0'),(p,'AL')]:
                delta={f:100*(m[a][f]-m[c][f]) for f in FIELDS} if m[a] else None
                gates.append(dict(dataset=ds,method=a,control=c,checks=None if delta is None else dict(BA_ge_1=delta[FIELDS[0]]>=1,tail_ge_minus05=delta[FIELDS[2]]>=-.5,forget_le_1=delta[FIELDS[3]]<=1,old_ge_minus1=delta[FIELDS[4]]>=-1,new_ge_minus1=delta[FIELDS[5]]>=-1)))
                gaps.append(dict(dataset=ds,method=a,control=c,AIR_gap_before_pp=100*(m['AIR'][FIELDS[0]]-m[c][FIELDS[0]]),AIR_gap_after_pp=None if not m[a] else 100*(m['AIR'][FIELDS[0]]-m[a][FIELDS[0]]),gap_shrink_pp=None if not m[a] else delta[FIELDS[0]]))
        for arm in ('B','P'):
            for single in ('MPC','DSM'):
                a,c='MPC_DSM_'+arm,single+'_'+arm
                contrasts.append(dict(dataset=ds,contrast=a+'−'+c,delta_pp={f:100*(m[a][f]-m[c][f]) if m[a] and m[c] else None for f in FIELDS}))
    for v,jobs in s['runs'].items():
        for name,r in jobs.items():
            for t in r.get('stages',[]):
                base=dict(variant=v,job=name,task=t['task'])
                for group,z in t['paired'].items():paired.append(dict(base,group=group,**z))
                for group,z in t['margins'].items():
                    for mode,a in z.items():margins.append(dict(base,group=group,mode=mode,**a))
                for flow,z in t['flows']['OFF'].items():flows.append(dict(base,flow=flow,off=z,on=t['flows']['ON'][flow],delta=t['flows']['ON'][flow]-z))
                mechanical.append(dict(base,effect=t['effect'],mechanical=t['mechanical']))
    cost=sum(a['residence_seconds'] for a in program['costs']);cpu=sum(a['residence_seconds'] for a in program['cpu_costs'])
    ledger=dict(s['closed_round_cost_ledger']);assert s['round_name'] not in ledger
    history=83019.61353103025+sum(ledger.values());ledger[s['round_name']]=cost
    audit=dict(status='PASS' if complete else 'INCOMPLETE',version=VERSION,source_commit=s['source_commit'],
        complete_comparison=complete,optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,
        completed_trajectories=sum(r.get('status',{}).get('status')=='COMPLETE' for runs in s['runs'].values() for r in runs.values()),
        real_zero_update_preflights=sum(f.get('status')=='PASS' for runs in s['preflights'].values() for f in runs.values()),
        cost_records=program['costs'],cpu_preparation_cost_records=program['cpu_costs'],failures=program['failures'],
        round_gpu_process_residence_seconds=cost,cpu_preparation_process_residence_seconds=cpu,historical_closed_gpu_process_residence_seconds=history,
        cumulative_gpu_process_residence_seconds=history+cost,closed_round_cost_ledger=ledger,whole_round_wall_seconds=program.get('whole_round_wall_seconds'),
        CPU_PRODUCTION_CHECK=s['cpu_checks'],engineering_events=s['engineering_events'],gates=gates,
        reused_control_and_AIR_cost=0,independent_confirmation=False,official_ConCM_reproduction=False,hard_new_class_guard=False)
    for file,value in [('AUDIT.json',audit),('CONTRIBUTIONS.json',contrasts),('AIR_GAPS.json',gaps),('MECHANICAL.json',mechanical),('PREFLIGHT_CHECKS.json',s['preflights']),('RESOURCES.json',dict(runs={v:{n:{k:r.get(k) for k in ('validation_image_exposures','source_checkpoint_bytes','sidecar_bytes','peak_gpu_bytes','worker_wall_seconds')} for n,r in jobs.items()} for v,jobs in s['runs'].items()}))]:dump(out/file,value)
    dump(out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json'),dict(metrics=metrics,rows=rows,runs=s['runs'],prepared=s['prepared'],audit=audit))
    for file,data in [('MAIN.csv',rows),('STAGE_CLASS_RECALLS.csv',classes),('PAIRED_COUNTS.csv',paired),('MARGINS.csv',margins),('CONFUSION_FLOWS.csv',flows)]:
        if data:
            with (out/file).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    fmt=lambda x:'缺失/未提供' if x is None else f'{x:.6f}'
    lines=['# ConCM-inspired统计校准与结构匹配：三轮固定主四格','',
        'A0=原base/Core；AL=固定主方法local_transport。各轮B/P分别是在A0/AL现有阶段checkpoint上的边界解析头，原训练不重跑。MPC→DSM→MPC_DSM固定执行，结果不决定后续参数或是否跳过负结果。',
        '非官方视觉统计版本：无WordNet/语义属性、元学习补全网络、非线性projector或高斯特征回放。支持量收缩不是可信旧漂移估计。DSM的最大匹配ETF是head目标，不是作者训练projector的原机制。',
        '三个方法均重新求解全类别头，允许改变新类分数和旧新决策；无硬保护保证。边界sidecar不反馈下一阶段，不能称端到端持续学习训练收益或无损。',
        '每阶段使用同一encoder特征得到OFF/ON；OFF支持度、类recall和Acc须精确等于公开控制。全18原stage只读复用，统计CPU准备；预检和正式评价仅既有development val。','']
    if not complete:lines+=['**不完整比较：缺失不填0；保留所有失败和驻留，停止依赖，不自动重试。具体原因需去标识补充。**','']
    lines+=['|数据集|方法|BA|平均BA|tail|forget↓|old|new|Acc|平均Acc|F1|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+r['dataset']+'|'+r['method']+'|'+'|'.join(fmt(r[f]) for f in fields)+'|')
    lines+=['','## 全部正零负贡献，pp','','|数据集|比较|BA|平均BA|tail|forget|old|new|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in contrasts:lines.append('|'+r['dataset']+'|'+r['contrast']+'|'+'|'.join('缺失' if r['delta_pp'][f] is None else f'{r["delta_pp"][f]:+.6f}' for f in FIELDS)+'|')
    lines+=['','## 全stage指标','','|数据集|方法|stage|n|Acc|BA|F1|','|---|---|---:|---:|---:|---:|---:|']
    for ds,methods in metrics.items():
        for method,m in methods.items():
            if m:
                for t in m['stages']:lines.append(f'|{ds}|{method}|{t["task"]}|{t["validation_n"]}|{100*t["accuracy"]:.6f}|{100*t["balanced_accuracy"]:.6f}|{fmt(None if t.get("macro_F1") is None else 100*t["macro_F1"])}|')
    for ds,m in metrics.items():
        names=list(m);lines+=['',f'## {ds}全部最终类别','','|类|train n|val n|'+'|'.join(names)+'|','|---|---:|---:|'+'|'.join(['---:']*len(names))+'|']
        for c in lock['data'][ds]['order']:lines.append('|'+str(c)+'|'+str(lock['data'][ds]['train_counts'][str(c)])+'|'+str(lock['data'][ds]['val_counts'][str(c)])+'|'+'|'.join(fmt(None if not m[a] else 100*m[a]['stages'][-1]['per_class_recall'][str(c)]) for a in names)+'|')
    lines+=['','## 全部纠正与破坏','','|版本|轨迹|stage|组|n|改变|纠正|破坏|错→另一错|净正确|','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:lines.append('|'+r['variant']+'|'+r['job']+'|'+str(r['task'])+'|'+r['group']+'|'+'|'.join(str(r[k]) for k in ('n','changed','rescued','harmed','wrong_to_different_wrong','net_correct'))+'|')
    lines+=['','## 资源与限制','',f'GPU进程驻留{cost:.9f}s；CPU checkpoint准备驻留{cpu:.9f}s；墙钟{program.get("whole_round_wall_seconds")}s。历史闭合GPU{history:.9f}s，累计{history+cost:.9f}s。',
        '成功/失败每次驻留计一次；CPU pipeline包含在驻留中，CPU合成兼容检查另列。原控制和AIR无新增训练/评价收费。原有训练预算及资源不等于本轮边界推理预算。',
        '全部AIR差距见AIR_GAPS；全部新类间隔正零负及旧新错误流见MARGINS/CONFUSION_FLOWS；小于5例类别抑制连续margin摘要，但保留n、recall及纠正/破坏计数。',
        '各变体双数据集相对各自控制的开发门：BA≥+1pp、tail≥−0.5pp、forget≤+1pp、old/new≥−1pp；保护门允许损失，不是无损保证。无按val挑主方法、参数、种子、checkpoint或阈值。',
        'single seed/order、反复development val、缺乏独立患者验证；不能称官方ConCM复现、等资源胜负、统计显著、临床或封存test证据。三轮负结果全部保留，不自动追加实验。','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit


def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime';h=json.loads((rt/'HANDOFF.private.json').read_text())
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('No repeat collection')
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\nnames='+repr(JOBS)+'\nvariants='+repr(VARIANTS)+'\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
for d in Path('/proc').iterdir():
 if d.name.isdigit():
  try:cmd=(d/'cmdline').read_bytes().split(b'\\0')
  except (FileNotFoundError,PermissionError):continue
  assert b'/tmp/q169w.py' not in cmd
value=dict(program=p,runs={v:{} for v in variants},preflights={v:{} for v in variants},prepared={})
for name in names:
 path=root/'prepared'/name/'PREPARED.json'
 if path.exists():value['prepared'][name]=json.loads(path.read_text())
 for v in variants:
  for target,folder in [('runs','runs'),('preflights','preflight')]:
   path=root/v/folder/name/'RESULTS.json'
   if path.exists():
    z=json.loads(path.read_text())
    if target=='runs':z['status']=json.loads((path.parent/'STATUS.json').read_text())
    value[target][v][name]=z
print(json.dumps(value,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError(x.stderr)
    s=json.loads(x.stdout);s.update(round_name=r.name,source_commit=h['source_commit'],lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),
        cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'],
        closed_round_cost_ledger=json.loads(Path('docs/lca_pair_guard1_20261013/results/AUDIT.json').read_text())['closed_round_cost_ledger'])
    dump(rt/'SNAPSHOT_ANONYMOUS.json',s);return report(s,r/'results')


if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
