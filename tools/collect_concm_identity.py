"""Terminal anonymous collection and signed reporting; no Torch or model access."""
import csv,json,subprocess,math
from pathlib import Path
from concm_statistical_metrics import FIELDS
HEADS=('MPC','TAIL','TAIL_REG','DSM','MPC_TAIL','MPC_TAIL_REG','MPC_DSM')
JOBS=[d+'_'+a for d in ('ISIC','HK') for a in ('base','local_transport')]
EXTRA=('final_accuracy','average_incremental_accuracy','final_macro_F1')
VERSION='identity_covariance_tail_direction_v2'
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def finite(v):
    if isinstance(v,float) and not math.isfinite(v):raise ValueError('Nonfinite aggregate')
    if isinstance(v,dict):
        for x in v.values():finite(x)
    if isinstance(v,list):
        for x in v:finite(x)

def validate(s):
    finite(s);p=s['program'];lock=s['lock'];sha=s['source_commit']
    if p['active'] or p['status'] not in ('COMPLETE','INCOMPLETE'):raise ValueError('Not terminal')
    if p['status']=='INCOMPLETE':return False
    if len(p['costs'])!=8 or {(z['job'],z['operation']) for z in p['costs']}!={(n,o) for n in JOBS for o in ('preflight','evaluate')} or len(p['cpu_costs'])!=4 or p['failures']:raise ValueError('Cost coverage differs')
    if any(z['returncode'] or not z['process_started'] or z['gpu'] not in (0,1) for z in p['costs']):raise ValueError('GPU scope differs')
    if any(z['gpu'] is not None or z['operation']!='prepare' or z['returncode'] or not z['process_started'] for z in p['cpu_costs']):raise ValueError('CPU scope differs')
    total=0;images=0
    for n in JOBS:
        d=lock['data'][n.split('_')[0]];prepared=s['prepared'][n];run=s['runs'][n];f=s['preflights'][n];ds=n.split('_')[0];arm=n[len(ds)+1:]
        if prepared['status']!='PREPARED' or prepared['source_commit']!=sha or prepared['optimizer_updates'] or prepared['medical_images_read'] or len(prepared['stages'])!=len(d['task_sizes']):raise ValueError('Preparation coverage differs')
        if run['status']['status']!='COMPLETE' or f['status']!='PASS' or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or not f['guard_checked'] or f['classification_metrics_read'] or f['real_validation_images']!=64:raise ValueError('Zero-update preflight differs')
        for a in (run,f):
            if a['source_commit']!=sha or a['version']!=VERSION or a['optimizer_updates'] or a['policy_updates'] or a['train_accessed'] or a['test_accessed']:raise ValueError('Provenance differs')
        if set(run['metrics'])!=set(HEADS) or len(run['stages'])!=len(d['task_sizes']):raise ValueError('Report coverage differs')
        for task,(prep,stage) in enumerate(zip(prepared['stages'],run['stages']),1):
            seen=d['order'][:sum(d['task_sizes'][:task])];old=sum(d['task_sizes'][:task-1])
            if prep['task']!=task or stage['task']!=task or set(prep['variants'])!=set(HEADS) or set(stage['heads'])!=set(HEADS) or set(stage['raw_heads'])!=set(HEADS):raise ValueError('Head/task coverage differs')
            if prep['native_solve']['relative_residual']>1e-8:raise ValueError('Native solve differs')
            for v in HEADS:
                md=prep['variants'][v]['metadata'];expected=dict(dataset=ds,control=arm,control_source=lock['controls'][arm]['source'],order=d['order'],seen=seen,seed=74002,task=task,source=sha,variant=v,version=VERSION,control_steps=d['cumulative_steps'][task-1],optimizer_updates=0)
                if md!=expected or prep['variants'][v]['mechanical']['relative_residual']>1e-8:raise ValueError('Sidecar metadata/solve differs')
                total+=1
                a=stage['heads'][v]
                if not a['published_control_exact'] or not a['encoder_state_unchanged'] or a['paired']['new']['harmed'] or a['margins']['new']['applied_delta']['negative'] or not a['guard_diagnostic']['new_logits_exact'] or task==1 and a['paired']['all']['changed']:raise ValueError('Protection/identity failed')
                for report in (a,stage['raw_heads'][v]):
                    for mode,m in report['metrics'].items():
                        if m['seen']!=seen or m['per_class_n']!={str(c):d['val_counts'][str(c)] for c in seen}:raise ValueError('Denominator differs')
                    for z in report['paired'].values():
                        if z['changed']!=z['rescued']+z['harmed']+z['wrong_to_different_wrong'] or z['net_correct']!=z['rescued']-z['harmed']:raise ValueError('Signed counts differ')
                ref=lock['control_references'][n]['stages'][task-1]
                for key in ('seen','per_class_n','per_class_recall','accuracy'):
                    if a['metrics']['OFF'][key]!=ref[key]:raise ValueError('OFF parity differs')
        images+=run['validation_image_exposures']
    if total!=126 or images!=15966:raise ValueError('Total sidecars/exposure differs')
    return True

def delta(a,b):return {k:None if a.get(k) is None or b.get(k) is None else 100*(a[k]-b[k]) for k in FIELDS+EXTRA}
def csvwrite(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def table(lines,cols,rows):
    lines.extend(['','|'+'|'.join(cols)+'|','|'+'|'.join('---' for _ in cols)+'|'])
    for z in rows:lines.append('|'+'|'.join('未提供' if z.get(k) is None else f'{z[k]:.6f}' if isinstance(z[k],float) else str(z.get(k)) for k in cols)+'|')

def report(s,r):
    complete=validate(s);out=r/'results';out.mkdir(exist_ok=False);main=[];rawrows=[];classes=[];paired=[];margins=[];confusions=[];risks=[];contrasts=[];gates=[];summary_by_job={}
    for n,run in s['runs'].items():
        ds=run['dataset'];control=run['control'];d=s['lock']['data'][ds];off=run['control_metrics'];summary_by_job[n]=run['metrics']
        def mainrow(v,mode,m):return dict(dataset=ds,control=control,head=v,readout=mode,**{k:None if m.get(k) is None else 100*m[k] for k in FIELDS+EXTRA})
        main.append(mainrow('OFF','original',off))
        for v in HEADS:
            for mode,target in [('guarded',main),('raw',rawrows)]:target.append(mainrow(v,mode,run['metrics'][v][mode]))
            a=run['metrics'][v]['guarded'];change=delta(a,off);contrasts.append(dict(job=n,contrast=v+'−OFF',delta_pp=change))
            contrasts.append(dict(job=n,contrast=v+'_guarded−raw',delta_pp=delta(a,run['metrics'][v]['raw'])))
            gates.append(dict(job=n,head=v,main=v=='MPC_DSM',BA=change[FIELDS[0]]>=1,tail=change[FIELDS[2]]>=-.5,forget=change[FIELDS[3]]<=1,old=change[FIELDS[4]]>=-1,new=change[FIELDS[5]]>=0,delta_pp=change))
            air=s['lock']['AIR_references'][ds];contrasts.append(dict(job=n,contrast=v+'−AIR',delta_pp=delta(a,air),control_minus_AIR=delta(off,air),BA_gap_reduction_pp=100*(a[FIELDS[0]]-off[FIELDS[0]])))
        for on,before in [('TAIL_REG','TAIL'),('MPC_TAIL_REG','MPC_TAIL'),('DSM','TAIL_REG'),('MPC_DSM','MPC_TAIL_REG'),('MPC_DSM','MPC'),('MPC_DSM','DSM'),('DSM','TAIL'),('MPC_TAIL','MPC'),('MPC_TAIL','TAIL')]:contrasts.append(dict(job=n,contrast=on+'−'+before,delta_pp=delta(run['metrics'][on]['guarded'],run['metrics'][before]['guarded'])))
        for st in run['stages']:
            task=st['task']
            for v in HEADS:
                for mode,a in [('guarded',st['heads'][v]),('raw',st['raw_heads'][v])]:
                    for c,count in a['metrics']['ON']['per_class_n'].items():
                        classes.append(dict(job=n,task=task,head=v,readout=mode,label=c,train_n=d['train_counts'][c],val_n=count,OFF_recall=a['metrics']['OFF']['per_class_recall'][c],ON_recall=a['metrics']['ON']['per_class_recall'][c],**a['paired'][c]))
                    for g,z in a['paired'].items():paired.append(dict(job=n,task=task,head=v,readout=mode,group=g,**z))
                    for g,z in a['margins'].items():
                        for kind,value in z.items():margins.append(dict(job=n,task=task,head=v,readout=mode,group=g,margin=kind,**value))
                for z in st['heads'][v]['class_confusion']:confusions.append(dict(job=n,task=task,head=v,**z))
                for z in st['heads'][v]['risk']['all_edges']:risks.append(dict(job=n,task=task,head=v,**z))
    interactions=[]
    if complete:
        for ds in ('ISIC','HK'):
            b=s['runs'][ds+'_base'];p=s['runs'][ds+'_local_transport']
            for v in HEADS:
                x=delta(p['metrics'][v]['guarded'],p['control_metrics']);y=delta(b['metrics'][v]['guarded'],b['control_metrics']);interactions.append(dict(dataset=ds,head=v,interaction_pp={k:None if x[k] is None or y[k] is None else x[k]-y[k] for k in x}))
    gpu=sum(z['residence_seconds'] for z in s['program']['costs']);cpu=sum(z['residence_seconds'] for z in s['program']['cpu_costs']);ledger=dict(s['ledger']);assert r.name not in ledger;history=83019.61353103025+sum(ledger.values());ledger[r.name]=gpu
    passed=complete and all(all(g[k] for k in ('BA','tail','forget','old','new')) for g in gates if g['main'])
    audit=dict(status='PASS' if complete else 'INCOMPLETE',scientific_development_gate_passed=passed,source_commit=s['source_commit'],reporting_source_commit=s['reporting_source_commit'],design_source_commit=s['lock']['design_source_commit'],optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,automatic_retry=False,round_gpu_process_residence_seconds=gpu,cpu_preparation_process_residence_seconds=cpu,CPU_PRODUCTION_CHECK=s['cpu_checks'],cpu_production_all_attempt_process_residence_seconds=s['cpu_checks'].get('all_CPU_attempt_residence_seconds',s['cpu_checks']['process_residence_seconds']),historical_design_CPU_residence_seconds=40.24965241225436,historical_design_recharged=False,cost_records=s['program']['costs'],cpu_cost_records=s['program']['cpu_costs'],failures=s['program']['failures'],whole_round_wall_seconds=s['program'].get('whole_round_wall_seconds'),historical_closed_gpu_process_residence_seconds=history,cumulative_gpu_process_residence_seconds=history+gpu,closed_round_cost_ledger=ledger,prior_results_reused_cost=0,engineering_events=s['engineering_events'],source_stage_checkpoints=18 if complete else None,sidecars=126 if complete else None,shared_val_exposures=15966 if complete else None,preflight_val_exposures=sum(a['real_validation_images'] for a in s['preflights'].values()),main_head='MPC_DSM',primary_control='local_transport')
    dump(out/'AUDIT.json',audit);dump(out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json'),dict(rows=main,raw_rows=rawrows,runs=s['runs'],prepared=s['prepared'],program=s['program'],preflights=s['preflights'],AIR_references=s['lock']['AIR_references'],missing_jobs=[n for n in JOBS if n not in s['runs']]));dump(out/'CONTRIBUTIONS.json',dict(contrasts=contrasts,interactions=interactions,gates=gates))
    for filename,rows in [('MAIN.csv',main),('RAW_MAIN.csv',rawrows),('STAGE_CLASS_RECALLS.csv',classes),('PAIRED_COUNTS.csv',paired),('MARGINS.csv',margins),('CLASS_CONFUSION.csv',confusions),('PROXY_CONFUSION.csv',risks)]:csvwrite(out/filename,rows)
    lines=['# MPC/DSM-inspired v2 冻结验证完整报告','',f'工程状态{audit["status"]}；固定开发门通过={passed}。主方法local_transport；主四格A0/AL/B/P的B/P固定MPC_DSM，不按单项结果换主方法。','原18阶段checkpoint只读复用，0训练图像/0optimizer/0policy，test/reserved封存。七性能头共享每stage编码器特征；REG仅CPU严格identity。两数据集必须都过门才报告主组合通过；新类无损是机械验收，不是功效结论。','非官方ConCM-inspired；统计共享形状高斯尾部是代理，不是测得的旧类漂移。local_transport改变R与W0，本模块保持这些几何，以协方差和旧旧错误尾部补充。所有raw/guarded完整保留。','重复development val、单seed/order；不称独立患者、临床、统计显著或等资源胜负。旧旧边对齐仅事后诊断，不回馈图/参数。']
    table(lines,['dataset','control','head']+list(FIELDS+EXTRA),main)
    lines+=['','## AIR参考（复用公开结果、计费0）']
    airrows=[dict(dataset=ds,**{k:None if a.get(k) is None else 100*a[k] for k in FIELDS+EXTRA}) for ds,a in s['lock']['AIR_references'].items()];table(lines,['dataset']+list(FIELDS+EXTRA),airrows)
    lines+=['','## 全部正、零、负贡献（pp；forget降低有利）'];table(lines,['job','contrast']+list(FIELDS+EXTRA),[dict(job=z['job'],contrast=z['contrast'],**z['delta_pp']) for z in contrasts]);table(lines,['dataset','head']+list(FIELDS+EXTRA),[dict(dataset=z['dataset'],head=z['head'],**z['interaction_pp']) for z in interactions])
    lines+=['','## 固定raw完整表（诊断，不切换主读出）'];table(lines,['dataset','control','head']+list(FIELDS+EXTRA),rawrows)
    lines+=['','## 全阶段新类纠正、破坏、margin及守卫作用']
    newrows=[]
    for n,run in s['runs'].items():
        for st in run['stages']:
            for v in HEADS:
                a=st['heads'][v];b=st['raw_heads'][v];newrows.append(dict(job=n,task=st['task'],head=v,guarded_rescued=a['paired']['new']['rescued'],guarded_harmed=a['paired']['new']['harmed'],raw_rescued=b['paired']['new']['rescued'],raw_harmed=b['paired']['new']['harmed'],new_margin_delta=a['margins']['new']['applied_delta']['mean'],**{k:val for k,val in a['guard_diagnostic'].items() if k!='new_logits_exact'}))
    if newrows:table(lines,list(newrows[0]),newrows)
    lines+=['','## 全阶段逐类n/recall/纠正/破坏（raw和guarded）'];table(lines,['job','task','head','readout','label','train_n','val_n','OFF_recall','ON_recall','rescued','harmed','wrong_to_different_wrong'],classes)
    lines+=['','## 代理可靠性与作用量','完整全边预测质量/经验混淆/纠正破坏见PROXY_CONFUSION.csv；全stage Spearman/加权重合/top3见RESULTS各head.risk（零质量或常数相关留null）。全头/判别/公共范数、增量、REG/ETF力、原始/借用trace和rank、无供体与有效供体数、所有solver见RESULTS.prepared。n<5不公开连续margin摘要，离散变化完整。','不保证旧类precision/F1或tail；守卫允许旧真值转新false positive，被消除的潜在旧救回与新救回已上表。','', '## 费用与完整性',f'GPU {gpu:.9f}s，CPU四准备 {cpu:.9f}s，CPU生产fixture {s["cpu_checks"]["process_residence_seconds"]:.9f}s（含pipeline {s["cpu_checks"]["pipeline_seconds"]:.9f}s）。历史设计CPU40.249652412s另列且不重计。并行wall {s["program"].get("whole_round_wall_seconds")}s。闭合历史GPU{history:.9f}s，本轮后{history+gpu:.9f}s；独立账本每次驻留一次，旧控制/AIR重用0。', '4预检真实256图，统一评价15966图；七头分数不重复收费影像曝光。全部失败保留驻留/missing；INCOMPLETE不补0、不宣称完整比较，不自动retry。']
    if not complete:lines+=['','本轮未完成；去标识原因及缺失见AUDIT.failures和PARTIAL_RESULTS，完整科学配对缺失。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit

def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('Already collected/delivered')
    h=json.loads((rt/'HANDOFF.private.json').read_text())
    code='import json\nfrom pathlib import Path\nroot=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((root/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:args=(proc/'cmdline').read_bytes().split(bytes([0]))
  except (FileNotFoundError,PermissionError):continue
  assert b'/tmp/q175w.py' not in args and b'/tmp/q175c.py' not in args
s=dict(program=p,prepared={},runs={},preflights={})
for n in ['ISIC_base','ISIC_local_transport','HK_base','HK_local_transport']:
 path=root/'prepared'/n/'PREPARED.json'
 if path.exists():s['prepared'][n]=json.loads(path.read_text())
 for key,folder in [('runs','runs'),('preflights','preflight')]:
  path=root/'IDENTITY'/folder/n/'RESULTS.json'
  if path.exists():
   a=json.loads(path.read_text())
   if key=='runs':a['status']=json.loads((path.parent/'STATUS.json').read_text())
   s[key][n]=a
print(json.dumps(s,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError('Private collection receipt failed')
    s=json.loads(x.stdout);s.update(source_commit=h['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),ledger=json.loads(Path('docs/concm_mechanism1_20261013/results/AUDIT.json').read_text())['closed_round_cost_ledger'],cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'])
    dump(rt/'SNAPSHOT_ANONYMOUS.json',s);return report(s,r)
if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
