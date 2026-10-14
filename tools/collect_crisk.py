"""Terminal anonymous CRISK collection. Standard library only, no medical IO."""
import csv
import json
import math
import subprocess
from pathlib import Path

VERSION='crisk_training_evidence_evaluation_v4'
JOBS=[d+'_'+a for d in ('ISIC','HK') for a in ('base','local_transport')]
CANDIDATES=('C0125','C025','C05','C1')
HEADS=('CRISK','NO_RESERVOIR','NO_DRIFT','NO_DONOR_TRUST','NO_EDGE_GUARD')
FIELDS=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall','final_accuracy','average_incremental_accuracy','final_macro_F1')


def finite(v):
    if isinstance(v,float) and not math.isfinite(v):raise ValueError('Nonfinite anonymous aggregate')
    if isinstance(v,dict):
        for x in v.values():finite(x)
    if isinstance(v,list):
        for x in v:finite(x)


def csvwrite(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for z in rows for k in z))
    with p.open('w') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def delta(a,b):return {k:None if a.get(k) is None or b.get(k) is None else 100*(a[k]-b[k]) for k in FIELDS}


def validate(s):
    finite(s);p=s['program'];lock=s['lock'];sha=s['source_commit']
    if p['active'] or p['status'] not in ('COMPLETE','INCOMPLETE'):raise ValueError('Not terminal')
    if p['status']=='INCOMPLETE':return False
    if len(p['costs'])!=12 or len(p['cpu_costs'])!=8 or p['failures']:raise ValueError('Exact process coverage required')
    for key,operations,gpus in [('costs',('preflight','statistics','evaluate'),(0,1)),('cpu_costs',('prepare','risk_prepare'),(None,))]:
        if {(z['job'],z['operation']) for z in p[key]}!={(n,o) for n in JOBS for o in operations}:raise ValueError('Every process instance required')
        if any(z['returncode'] or not z['process_started'] or z['gpu'] not in gpus for z in p[key]):raise ValueError('Process admission differs')
    train=val=0
    for n in JOBS:
        d=lock['data'][n.split('_')[0]];run=s['runs'][n];f=s['preflights'][n];stats=s['statistics'][n];risk=s['risk_prepared'][n];prep=s['prepared'][n]
        for z,status in [(run,'COMPLETE'),(f,'PASS'),(stats,'STATS_READY'),(risk,'RISK_READY'),(prep,'PREPARED')]:
            if z['status']!=status or z['source_commit']!=sha or z.get('optimizer_updates',0) or z.get('policy_updates',0) or z.get('test_accessed',False):raise ValueError('Source/zero-update terminal differs')
        if f['val_accessed'] or f['classification_metrics_read'] or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or f['real_training_images']!=64:raise ValueError('Four genuine train-only checks required')
        if stats['val_accessed'] or risk['val_accessed'] or set(run['metrics'])!=set(CANDIDATES+HEADS):raise ValueError('Val before sealed selection or missing methods')
        if any(len(z['stages'])!=len(d['task_sizes']) for z in [run,stats,risk,prep]):raise ValueError('18 stage coverage differs')
        for task,(rs,st,rk) in enumerate(zip(run['stages'],stats['stages'],risk['stages']),1):
            seen=d['order'][:sum(d['task_sizes'][:task])]
            if rs['task']!=task or st['task']!=task or rk['task']!=task or set(rk['selections'])!=set(HEADS):raise ValueError('Order/task differs')
            expected_audit={(h,v,c) for h in HEADS for v in CANDIDATES for c in seen} if task>1 else set()
            if {(z['mode'],z['candidate'],z['label']) for z in rk['class_audit']}!=expected_audit or len(rk['class_audit'])!=len(expected_audit):raise ValueError('Every conditional class/candidate/mode required')
            for z in rk['class_audit']:
                count=d['train_counts'][str(z['label'])];tr=z['training_counts']
                if z['target_actual_n']!=count or tr['n']!=count or z['classifier_independent'] or tr['rescued']>count-tr['correct_off'] or tr['harmed']>tr['correct_off']:raise ValueError('Training reservoir/source differs')
            expected={str(c):d['val_counts'][str(c)] for c in seen};ref=lock['control_references'][n]['stages'][task-1]
            for h in CANDIDATES+HEADS:
                a=rs['heads'][h]
                for z in (a,rs['raw_heads'][h]):
                    for m in z['metrics'].values():
                        if m['seen']!=seen or m['per_class_n']!=expected:raise ValueError('Bad val denominator')
                if a['paired']['new']['harmed'] or task==1 and a['paired']['all']['changed']:raise ValueError('New class/identity guard failed')
                for k in ('seen','per_class_n','per_class_recall','accuracy'):
                    if a['metrics']['OFF'][k]!=ref[k]:raise ValueError('Original OFF mismatch')
            if rs['selections']!=rk['selections']:raise ValueError('Selection modified after val')
            for h,choice in rk['selections'].items():
                if choice['status']=='NOT_READY':raise ValueError('Missing evidence evaluated')
                scale=choice['chosen_scale'];expected_metric=rs['heads'][CANDIDATES[(.125,.25,.5,1.).index(scale)]]['metrics']['ON'] if scale else rs['heads'][h]['metrics']['OFF']
                if rs['heads'][h]['metrics']['ON']!=expected_metric:raise ValueError('Fixed selected candidate mismatch')
        train+=stats['image_exposures'];val+=run['image_exposures']
    if train!=219448 or val!=15966:raise ValueError('Frozen training/val exposures differ')
    return True


def report(s,r):
    complete=validate(s);out=r/'results';out.mkdir(exist_ok=False);main=[];contrasts=[];paired=[];classes=[];guards=[];confusion=[];margins=[];riskrows=[];selections=[];drift=[];stage_actual=[];bias=[]
    for n,run in s['runs'].items():
        ds=n.split('_')[0];arm=n[len(ds)+1:];allm=dict(OFF=run['control_metrics'],**{h:run['metrics'][h]['guarded'] for h in CANDIDATES+HEADS})
        for h,m in allm.items():
            main.append(dict(job=n,dataset=ds,control=arm,head=h,**{k:None if m.get(k) is None else 100*m[k] for k in FIELDS}))
            if h!='OFF':contrasts.append(dict(job=n,contrast=h+'−OFF',**delta(m,allm['OFF'])))
            contrasts.append(dict(job=n,contrast=h+'−AIR',**delta(m,s['lock']['AIR_references'][ds])))
        for h in HEADS[1:]:contrasts.append(dict(job=n,contrast='CRISK−'+h,**delta(allm['CRISK'],allm[h])))
        for st in run['stages']:
            task=st['task'];rk=s['risk_prepared'][n]['stages'][task-1];selections.append(dict(job=n,task=task,selections=rk['selections']))
            riskrows.extend(dict(job=n,task=task,**z) for z in rk['class_audit']);drift.extend(dict(job=n,task=task,**z) for z in rk['drift'])
            for z in rk['class_audit']:
                actual=st['heads'][z['candidate']]['paired'][str(z['label'])];train=z['training_counts']
                bias.append(dict(job=n,task=task,mode=z['mode'],candidate=z['candidate'],label=z['label'],train_n=train['n'],val_n=actual['n'],
                                 posterior_rescue=z['rescue_mean'],posterior_harm=z['harm_mean'],
                                 train_rescue=train['rescued']/train['n'],train_harm=train['harmed']/train['n'],
                                 val_rescue=actual['rescued']/actual['n'],val_harm=actual['harmed']/actual['n'],
                                 posterior_net=z['rescue_mean']-z['harm_mean'],conditional_sensitive_net=z['conditional_net_mean'],
                                 train_net=(train['rescued']-train['harmed'])/train['n'],val_net=actual['net_correct']/actual['n'],
                                 classifier_independent=False))
            for h in CANDIDATES+HEADS:
                a=st['heads'][h];seen=a['metrics']['OFF']['seen'];net=sum(a['paired'][str(c)]['net_correct']/a['paired'][str(c)]['n'] for c in seen)/len(seen)
                stage_actual.append(dict(job=n,task=task,head=h,actual_macro_net_pp=100*net,rescued=a['paired']['all']['rescued'],harmed=a['paired']['all']['harmed'],new_rescued=a['paired']['new']['rescued'],new_harmed=a['paired']['new']['harmed']))
                for readout,z in [('guarded',a),('raw',st['raw_heads'][h])]:
                    paired.extend(dict(job=n,task=task,head=h,readout=readout,group=g,**v) for g,v in z['paired'].items())
                    for c in seen:classes.append(dict(job=n,task=task,head=h,readout=readout,label=c,train_n=s['lock']['data'][ds]['train_counts'][str(c)],recall=z['metrics']['ON']['per_class_recall'][str(c)],**z['paired'][str(c)]))
                    for g,v in z['margins'].items():
                        for kind,value in v.items():margins.append(dict(job=n,task=task,head=h,readout=readout,group=g,kind=kind,**value))
                guards.append(dict(job=n,task=task,head=h,**a['guard_diagnostic']));confusion.extend(dict(job=n,task=task,head=h,**v) for v in a['class_confusion'])
    if complete:
        for ds in ('ISIC','HK'):
            b=s['runs'][ds+'_base'];l=s['runs'][ds+'_local_transport'];x=delta(l['metrics']['CRISK']['guarded'],l['control_metrics']);y=delta(b['metrics']['CRISK']['guarded'],b['control_metrics'])
            contrasts.append(dict(job=ds,contrast='CRISK×local_transport interaction',**{k:None if x[k] is None or y[k] is None else x[k]-y[k] for k in FIELDS}))
    for name,rows in [('MAIN',main),('CONTRIBUTIONS',contrasts),('STAGE_ACTUAL',stage_actual),('PAIRED_COUNTS',paired),('STAGE_CLASS_RECALLS',classes),('GUARD_COUNTS',guards),('CLASS_CONFUSION',confusion),('MARGINS',margins),('CLASS_RISK',riskrows),('DRIFT',drift),('ESTIMATE_BIAS',bias)]:csvwrite(out/(name+'.csv'),rows)
    gpu=sum(z['residence_seconds'] for z in s['program']['costs']);cpu=sum(z['residence_seconds'] for z in s['program']['cpu_costs'])
    audit=dict(status='PASS' if complete else 'INCOMPLETE',source_commit=s['source_commit'],reporting_source_commit=s['reporting_source_commit'],design_source_commit=s['lock']['design_source_commit'],fixed_sidecar_source=s['lock']['previous_sidecar_source'],round_gpu_process_residence_seconds=gpu,round_CPU_process_residence_seconds=cpu,CPU_production_check=s['CPU_check'],program_cost_records=s['program']['costs'],CPU_cost_records=s['program']['cpu_costs'],failures=s['program']['failures'],wall_seconds=s['program'].get('whole_round_wall_seconds'),optimizer_updates=0,policy_updates=0,test_accessed=False,train_inference_only=True,independent_calibration=False,validation_retuning=False,planned_train_image_exposures=219448,planned_preflight_image_exposures=512,planned_val_image_exposures=15966,prior_results_recharged=False)
    for name,value in [('AUDIT',audit),('SELECTIONS',selections),('RESULTS' if complete else 'PARTIAL_RESULTS',s)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    if s.get('partial_stages'):(out/'PARTIAL_STAGE_RESULTS.json').write_text(json.dumps(s['partial_stages'],indent=2,allow_nan=False)+'\n')
    lines=['# CRISK 完整主四格与贡献报告','',f'完成状态：{audit["status"]}。主方法固定local_transport+CRISK；零训练/策略更新、test封存、没有val调参。训练内条件预测不是独立校准证书。','', '|job|head|'+'|'.join(FIELDS)+'|','|---|---|'+'|'.join(['---:']*len(FIELDS))+'|']
    lines+=['|'+'|'.join(str(z[k]) for k in ['job','head']+list(FIELDS))+'|' for z in main]
    lines+=['','全部正负贡献（pp；forget下降有利）：','', '|job|contrast|'+'|'.join(FIELDS)+'|','|---|---|'+'|'.join(['---:']*len(FIELDS))+'|']
    lines+=['|'+'|'.join(str(z[k]) for k in ['job','contrast']+list(FIELDS))+'|' for z in contrasts]
    lines+=['','全部四候选、五选择器的逐stage/31类救回破坏、边界侵蚀准入与拒绝、条件储量与有效支持、漂移/供体评分及RAW/新类守卫见同目录CSV和JSON。未识别条件概率留null，零积分事件不当已识别零风险。','',f'GPU进程驻留{gpu:.9f}秒；正式CPU准备驻留{cpu:.9f}秒；生产CPU预检全部尝试单列于AUDIT。旧控制/方向/AIR费用不重复计；各pipeline包含于驻留，不相加。失败保留，missing不填0。','', '单seed/order反复development val，机制由此前诊断启发，不构成独立患者/临床/封存test或统计显著证据。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit


def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('Already collected/delivered')
    h=json.loads((rt/'HANDOFF.private.json').read_text())
    code='import json,subprocess\nfrom pathlib import Path\nr=Path('+repr(h['current_root'])+')\n'+'''
p=json.loads((r/'PROGRAM_STATE.json').read_text());assert p['status'] in ('COMPLETE','INCOMPLETE') and not p['active']
ps=subprocess.check_output(['ps','-eo','args='],text=True);assert not any('/tmp/q224w.py' in z or '/tmp/q224c.py' in z for z in ps.splitlines())
s=dict(program=p,partial_stages={})
for key,folder,file in [('prepared','prepared','PREPARED.json'),('preflights','preflight','RESULTS.json'),('statistics','statistics','RESULTS.json'),('risk_prepared','risk_prepared','RESULTS.json'),('runs','runs','RESULTS.json')]:
 s[key]={}
 for n in ['ISIC_base','ISIC_local_transport','HK_base','HK_local_transport']:
  path=r/folder/n/file
  if path.exists():s[key][n]=json.loads(path.read_text())
  elif (path.parent/'PARTIAL_RESULTS.json').exists():s['partial_stages'][key+'_'+n]=json.loads((path.parent/'PARTIAL_RESULTS.json').read_text())
print(json.dumps(s,allow_nan=False))
'''
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError('Private terminal collection failed')
    s=json.loads(x.stdout);s.update(source_commit=h['source_commit'],reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),CPU_check=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()))
    return report(s,r)
