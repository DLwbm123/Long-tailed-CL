"""Terminal anonymous collection and signed reporting; no Torch or model access."""
import csv,json,subprocess,math
from pathlib import Path
from concm_statistical_metrics import FIELDS
HEADS=('GAIN','GAIN_NO_BORROW','GAIN_RESCUE_ONLY','NEUTRAL_MATCHED')
JOBS=[d+'_'+a for d in ('ISIC','HK') for a in ('base','local_transport')]
EXTRA=('final_accuracy','average_incremental_accuracy','final_macro_F1')
VERSION='guarded_class_gain_v1'
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
    if total!=72 or images!=15966:raise ValueError('Total sidecars/exposure differs')
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
    complete=validate(s);out=r/'results';out.mkdir(exist_ok=False);main=[];rawrows=[];classes=[];rawclasses=[];paired=[];margins=[];confusions=[];risks=[];contrasts=[];diagnostic_contrasts=[];gates=[];interactions=[]
    for n,run in s['runs'].items():
        ds=run['dataset'];control=run['control'];d=s['lock']['data'][ds];off=run['control_metrics']
        def row(v,mode,m):return dict(dataset=ds,control=control,head=v,readout=mode,**{k:None if m.get(k) is None else 100*m[k] for k in FIELDS+EXTRA})
        main.append(row('OFF','original',off))
        for v in HEADS:
            a=run['metrics'][v]['guarded'];main.append(row(v,'guarded',a));rawrows.append(row(v,'raw',run['metrics'][v]['raw']))
            change=delta(a,off);contrasts.append(dict(job=n,contrast=v+'−OFF',delta_pp=change))
            diagnostic_contrasts.append(dict(job=n,contrast=v+'_guarded−raw',delta_pp=delta(a,run['metrics'][v]['raw'])))
            gates.append(dict(job=n,head=v,main=v=='GAIN',BA=change[FIELDS[0]]>=1,tail=change[FIELDS[2]]>=-.5,forget=change[FIELDS[3]]<=1,old=change[FIELDS[4]]>=-1,new=change[FIELDS[5]]>=0,delta_pp=change))
        for v in HEADS[1:]:contrasts.append(dict(job=n,contrast='GAIN−'+v,delta_pp=delta(run['metrics']['GAIN']['guarded'],run['metrics'][v]['guarded'])))
        for v,m in [('OFF',off),('GAIN',run['metrics']['GAIN']['guarded'])]:
            air=s['lock']['AIR_references'][ds];contrasts.append(dict(job=n,contrast=v+'−AIR',delta_pp=delta(m,air),BA_gap_reduction_pp=100*(m[FIELDS[0]]-off[FIELDS[0]])))
        for st in run['stages']:
            task=st['task']
            reference=st['heads']['GAIN']
            for c,count in reference['metrics']['OFF']['per_class_n'].items():
                z=dict(reference['paired'][c]);z.update(changed=0,rescued=0,harmed=0,wrong_to_different_wrong=0,correct_on=z['correct_off'],net_correct=0)
                recall=reference['metrics']['OFF']['per_class_recall'][c]
                classes.append(dict(job=n,task=task,head='OFF',readout='original',label=c,train_n=d['train_counts'][c],val_n=count,OFF_recall=recall,ON_recall=recall,**z))
            for v in HEADS:
                for mode,a in [('guarded',st['heads'][v]),('raw',st['raw_heads'][v])]:
                    target=classes if mode=='guarded' else rawclasses
                    for c,count in a['metrics']['ON']['per_class_n'].items():target.append(dict(job=n,task=task,head=v,readout=mode,label=c,train_n=d['train_counts'][c],val_n=count,OFF_recall=a['metrics']['OFF']['per_class_recall'][c],ON_recall=a['metrics']['ON']['per_class_recall'][c],**a['paired'][c]))
                    for g,z in a['paired'].items():paired.append(dict(job=n,task=task,head=v,readout=mode,group=g,**z))
                    for g,z in a['margins'].items():
                        for kind,value in z.items():margins.append(dict(job=n,task=task,head=v,readout=mode,group=g,margin=kind,**value))
                for z in st['heads'][v]['class_confusion']:confusions.append(dict(job=n,task=task,head=v,**z))
                for z in st['heads'][v]['risk']['classes']:risks.append(dict(job=n,task=task,head=v,**z))
    if complete:
        for ds in ('ISIC','HK'):
            b=s['runs'][ds+'_base'];p=s['runs'][ds+'_local_transport'];x=delta(p['metrics']['GAIN']['guarded'],p['control_metrics']);y=delta(b['metrics']['GAIN']['guarded'],b['control_metrics']);interaction={k:None if x[k] is None or y[k] is None else x[k]-y[k] for k in x}
            contrasts.append(dict(job=ds,contrast='P−B',delta_pp=delta(p['metrics']['GAIN']['guarded'],b['metrics']['GAIN']['guarded'])));contrasts.append(dict(job=ds,contrast='(P−AL)−(B−A0)',delta_pp=interaction));interactions.append(dict(dataset=ds,head='GAIN',interaction_pp=interaction))
        if len(contrasts)!=40 or len(main)!=20 or len(classes)!=1260 or len(rawclasses)!=1008:raise ValueError('Frozen comparison and class coverage required')
    gpu=sum(z['residence_seconds'] for z in s['program']['costs']);cpu=sum(z['residence_seconds'] for z in s['program']['cpu_costs']);ledger=dict(s['ledger']);assert r.name not in ledger;history=83019.61353103025+sum(ledger.values());ledger[r.name]=gpu
    passed=complete and all(all(g[k] for k in ('BA','tail','forget','old','new')) for g in gates if g['main'])
    identity=[dict(job=n,task=z['task'],head=v,exact_identity=a['mechanical']['exact_identity']) for n,run in s['runs'].items() for z in run['stages'] for v,a in z['heads'].items()]
    audit=dict(status='PASS' if complete else 'INCOMPLETE',scientific_development_gate_passed=passed,source_commit=s['source_commit'],reporting_source_commit=s['reporting_source_commit'],design_source_commit=s['lock']['design_source_commit'],optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,automatic_retry=False,round_gpu_process_residence_seconds=gpu,cpu_preparation_process_residence_seconds=cpu,CPU_PRODUCTION_CHECK=s['cpu_checks'],cpu_production_all_attempt_process_residence_seconds=s['cpu_checks']['all_CPU_attempt_residence_seconds'],historical_design_CPU_residence_seconds=43.3347933748737,historical_design_recharged=False,cost_records=s['program']['costs'],cpu_cost_records=s['program']['cpu_costs'],failures=s['program']['failures'],whole_round_wall_seconds=s['program'].get('whole_round_wall_seconds'),historical_closed_gpu_process_residence_seconds=history,cumulative_gpu_process_residence_seconds=history+gpu,closed_round_cost_ledger=ledger,prior_results_reused_cost=0,engineering_events=s['engineering_events'],source_stage_checkpoints=18 if complete else None,sidecars=72 if complete else None,shared_val_exposures=15966 if complete else None,preflight_val_exposures=sum(a['real_validation_images'] for a in s['preflights'].values()),main_head='GAIN',primary_control='local_transport',identity_sidecars=identity,model_certificate_calibrated=False)
    airrows=[dict(dataset=ds,control='reference',head='AIR',readout='public_reuse',**{k:None if a.get(k) is None else 100*a[k] for k in FIELDS+EXTRA}) for ds,a in s['lock']['AIR_references'].items()]
    dump(out/'AUDIT.json',audit);dump(out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json'),dict(rows=main+airrows,raw_rows=rawrows,runs=s['runs'],prepared=s['prepared'],program=s['program'],preflights=s['preflights'],AIR_references=s['lock']['AIR_references'],missing_jobs=[n for n in JOBS if n not in s['runs']]));dump(out/'CONTRIBUTIONS.json',dict(contrasts=contrasts,guarded_minus_raw=diagnostic_contrasts,interactions=interactions,gates=gates))
    for filename,rows in [('MAIN.csv',main+airrows),('RAW_MAIN.csv',rawrows),('STAGE_CLASS_RECALLS.csv',classes),('RAW_STAGE_CLASS_RECALLS.csv',rawclasses),('PAIRED_COUNTS.csv',paired),('MARGINS.csv',margins),('CLASS_CONFUSION.csv',confusions),('GAIN_ESTIMATES.csv',risks),('IDENTITY.csv',identity)]:csvwrite(out/filename,rows)
    lines=['# GAIN 冻结验证完整报告','',f'工程状态{audit["status"]}；固定开发门通过={passed}。主方法local_transport；主四格A0/AL/B/P的B/P固定GAIN，不按贡献头换主方法。','原18stage checkpoint只读复用，0训练图像/0optimizer/0policy，test/reserved封存。四读出共享各stage编码器特征；全部identity也保留。重复development val、单seed/order，非独立患者、临床或统计显著证据。','救回与破坏是原/候选联合赢家模型界。统计借用不增加支持量，不改原Q/R/pairs；矩误差范围未经医学校准，development是否落入区间不是置信覆盖证明。GAIN_NO_BORROW改变提案及情景完整路径，非准入门单因素；neutral匹配守卫前native能量，不保证守卫后效果相同。']
    table(lines,['dataset','control','head']+list(FIELDS+EXTRA),main+airrows)
    lines+=['','## 全部40组正、零、负贡献（pp；forget降低有利）'];table(lines,['job','contrast']+list(FIELDS+EXTRA),[dict(job=z['job'],contrast=z['contrast'],**z['delta_pp']) for z in contrasts])
    lines+=['','## 原始读出及守卫贡献'];table(lines,['dataset','control','head']+list(FIELDS+EXTRA),rawrows);table(lines,['job','contrast']+list(FIELDS+EXTRA),[dict(job=z['job'],contrast=z['contrast'],**z['delta_pp']) for z in diagnostic_contrasts])
    lines+=['','## 各stage逐类n/recall/救回与破坏'];table(lines,['job','task','head','readout','label','train_n','val_n','OFF_recall','ON_recall','rescued','harmed','wrong_to_different_wrong'],classes)
    lines+=['','## 模型净收益与实际可修复范围'];table(lines,['job','task','head','label','n','rescue_lower','rescue_upper','harm_lower','harm_upper','actual_rescue','actual_harm','original_errors','permitted_original_old_errors','unrepairable_original_new_errors','new_below_old_mean_errors'],risks)
    lines+=['','## 作用、守卫、新类间隔与拒绝','完整raw/guarded新旧间隔及正零负见MARGINS；n<5连续margin抑制，离散n/recall/救回破坏保留。CLASS_CONFUSION包含全部旧旧/旧新/新旧/新新流。RESULTS各stage含normalized fixed-competitor margin、守卫移除/救回/破坏计数、模型macro区间与实际macro净值；prepared含供体权重/支持量、五候选区间/步长、原曲率能量及范数。','新类无损仅机械验收，不保护全部旧类或新类precision。原新赢家的错误、以及新类分数低于原旧均值的错误不计为当前可救回质量。forget需结合历史峰值和终局逐类变化，峰值下降不称保持改善。','',f'非零/identity完整见IDENTITY.csv：{sum(z["exact_identity"] for z in identity)}/{len(identity)} identity。不能把identity与neutral同为零称方向成功。','', '## 实际费用',f'GPU本轮 {gpu:.9f}s；CPU四准备 {cpu:.9f}s；生产CPU所有尝试 {s["cpu_checks"]["all_CPU_attempt_residence_seconds"]:.9f}s，pipeline包含其中不相加。设计CPU43.334793375s历史单列不重计。并行wall {s["program"].get("whole_round_wall_seconds")}s。历史闭合GPU{history:.9f}s；本轮后{history+gpu:.9f}s。','4真实零更新预检256图，统一val15966图；共享4头不重复收费图像曝光，旧控制/AIR费用0。8GPU进程每次驻留保留；失败额外全计。']
    if not complete:lines+=['','本轮未完成：missing不补0、不称完整配对。去标识失败原因见AUDIT及ENGINEERING_AUDIT；无自动重跑。']
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
  assert b'/tmp/q179w.py' not in args and b'/tmp/q179c.py' not in args
s=dict(program=p,prepared={},runs={},preflights={})
for n in ['ISIC_base','ISIC_local_transport','HK_base','HK_local_transport']:
 path=root/'prepared'/n/'PREPARED.json'
 if path.exists():s['prepared'][n]=json.loads(path.read_text())
 for key,folder in [('runs','runs'),('preflights','preflight')]:
  path=root/'GAIN'/folder/n/'RESULTS.json'
  if path.exists():
   a=json.loads(path.read_text())
   if key=='runs':a['status']=json.loads((path.parent/'STATUS.json').read_text())
   s[key][n]=a
print(json.dumps(s,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError('Private collection receipt failed')
    s=json.loads(x.stdout);s.update(source_commit=h['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),ledger=json.loads(Path('docs/concm_identity1_20261013/results/AUDIT.json').read_text())['closed_round_cost_ledger'],cpu_checks=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),engineering_events=json.loads((r/'ENGINEERING_AUDIT.json').read_text())['events'])
    dump(rt/'SNAPSHOT_ANONYMOUS.json',s);return report(s,r)
if __name__=='__main__':print(json.dumps(collect(json.load(__import__('sys').stdin)['round_dir']),indent=2))
