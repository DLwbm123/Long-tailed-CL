"""Collect only anonymous training diagnostic aggregates; never fetch tensors."""
import csv,json,math,subprocess
from pathlib import Path
VERSION='gain_training_paired_diagnostic_v1'
CANDIDATES=('C0125','C025','C05','C1')
JOBS=[ds+'_'+arm for ds in ('ISIC','HK') for arm in ('base','local_transport')]


def csvwrite(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def validate(s):
    p=s['lock'];program=s['program'];complete=program['status']=='COMPLETE'
    if program['active'] or program['status'] not in ('COMPLETE','INCOMPLETE'):raise ValueError('Not terminal')
    if not complete:return False
    if len(program['costs'])!=8 or len(program['cpu_costs'])!=4 or program['failures']:raise ValueError('Cost coverage differs')
    if any(z['returncode'] or not z['process_started'] or z['gpu'] is not None or z['operation']!='prepare' for z in program['cpu_costs']):raise ValueError('CPU metadata costs differ')
    if {(z['job'],z['operation']) for z in program['costs']}!={(n,o) for n in JOBS for o in ('preflight','diagnose')}:raise ValueError('Every operation residence required')
    if any(z['returncode'] or not z['process_started'] or z['gpu'] not in (0,1) for z in program['costs']):raise ValueError('GPU01 success required')
    total=0
    for n in JOBS:
        run=s['runs'][n];f=s['preflights'][n];d=p['data'][n.split('_')[0]]
        if run['status']!='COMPLETE' or f['status']!='PASS' or f['classification_metrics_read'] or not f['parameters_unchanged'] or not f['CPU_GPU_scores_close'] or f['real_training_images']!=64:raise ValueError('Real preflight/result incomplete')
        for value in (run,f):
            if value['source_commit']!=s['source_commit'] or value['version']!=VERSION or value['optimizer_updates'] or value['policy_updates'] or value['val_accessed'] or value['test_accessed'] or value['independent_calibration']:raise ValueError('Training-only source/scope differs')
        if len(run['stages'])!=len(d['task_sizes']):raise ValueError('Stage coverage differs')
        exposures=0
        for task,z in enumerate(run['stages'],1):
            seen=d['order'][:sum(d['task_sizes'][:task])];expected={str(c):d['train_counts'][str(c)] for c in seen}
            if z['task']!=task or set(z['heads'])!=set(CANDIDATES) or z['independent_calibration']:raise ValueError('All four fixed heads required')
            count=sum(expected.values());exposures+=count*(1 if task==1 else 2)
            for head in z['heads'].values():
                a=head['guarded']
                if a['paired']['new']['harmed'] or task==1 and a['paired']['all']['changed']:raise ValueError('Identity/new guard failed')
                for report in (a,head['raw']):
                    for metric in report['metrics'].values():
                        if metric['per_class_n']!=expected or metric['seen']!=seen:raise ValueError('Wrong train denominator')
                for row in a['paired'].values():
                    if row['net_correct']!=row['rescued']-row['harmed']:raise ValueError('Paired counts differ')
            if len(z['drift'])!=len(seen):raise ValueError('Drift coverage differs')
        if run['image_exposures']!=exposures:raise ValueError('Paired exposure differs')
        total+=exposures
    if total!=219448:raise ValueError('Frozen exposure budget differs')
    return True


def report(s,r):
    complete=validate(s);out=r/'results';out.mkdir(exist_ok=False);main=[];classes=[];estimates=[];folds=[];drift=[];boundaries=[];confusion=[];macro=[];foldcheck=[];rankings=[]
    for job,run in s['runs'].items():
        for st in run.get('stages',[]):
            prefix=dict(job=job,task=st['task'])
            for name,head in st['heads'].items():
                a=head['guarded'];paired=a['paired'];seen=a['metrics']['ON']['seen']
                R=sum(paired[str(c)]['rescued']/paired[str(c)]['n'] for c in seen)/len(seen)
                H=sum(paired[str(c)]['harmed']/paired[str(c)]['n'] for c in seen)/len(seen)
                main.append(dict(prefix,head=name,rescued=paired['all']['rescued'],harmed=paired['all']['harmed'],macro_rescue_pp=100*R,macro_harm_pp=100*H,actual_net_pp=100*(R-H),OFF_BA_pp=100*a['metrics']['OFF']['balanced_accuracy'],ON_BA_pp=100*a['metrics']['ON']['balanced_accuracy'],new_rescued=paired['new']['rescued'],new_harmed=paired['new']['harmed'],independent_calibration=False))
                classes.extend(dict(prefix,head=name,label=c,**paired[str(c)]) for c in seen)
                estimates.extend(dict(prefix,head=name,**v) for v in a['estimates'])
                folds.extend(dict(prefix,head=name,**v) for v in a['fold_counts'])
                confusion.extend(dict(prefix,head=name,**v) for v in a['class_confusion'])
                kinds={(v['estimator'],v['scenario']) for v in a['estimates']}
                for kind,scenario in sorted(kinds):
                    rows=[v for v in a['estimates'] if v['estimator']==kind and v['scenario']==scenario]
                    if len(rows)!=len(seen):raise ValueError('Estimator class coverage differs')
                    rknown=sum(v['predicted_known_rescue'] for v in rows)/len(seen);hknown=sum(v['predicted_known_harm'] for v in rows)/len(seen);unknown=sum(v['unknown_mass'] for v in rows)/len(seen)
                    macro.append(dict(prefix,head=name,estimator=kind,scenario=scenario,predicted_known_net=rknown-hknown,actual_net=R-H,unknown_mass=unknown,full_bias=None if unknown else rknown-hknown-R+H,known_partial_minus_actual=rknown-hknown-R+H,sign_error=(rknown-hknown)*(R-H)<0,independent_calibration=False))
                fc={(v['fold'],v['label']):v for v in a['fold_counts']}
                for c in seen:
                    for f in (0,1):
                        fit=fc.get((f,c));check=fc.get((1-f,c));missing=fit is None or check is None or fit['missing'] or check['missing']
                        foldcheck.append(dict(prefix,head=name,label=c,fit_fold=f,fit_n=None if fit is None else fit['n'],check_n=None if check is None else check['n'],missing=missing,rescue_rate_difference=None if missing else fit['rescue_rate']-check['rescue_rate'],harm_rate_difference=None if missing else fit['harm_rate']-check['harm_rate'],classifier_independent=False))
            drift.extend(dict(prefix,**v) for v in st['drift']);boundaries.extend(dict(prefix,**v) for v in st['boundaries'])
    for i,a in enumerate(macro):
        for b in macro[i+1:]:
            if (a['job'],a['task'],a['estimator'],a['scenario'])==(b['job'],b['task'],b['estimator'],b['scenario']) and a['head']!=b['head']:
                predicted=a['predicted_known_net']-b['predicted_known_net'];actual=a['actual_net']-b['actual_net']
                rankings.append(dict(job=a['job'],task=a['task'],estimator=a['estimator'],scenario=a['scenario'],head_a=a['head'],head_b=b['head'],predicted_difference=predicted,actual_difference=actual,rank_reversal=predicted*actual<0,unknown_mass_present=bool(a['unknown_mass'] or b['unknown_mass']),selection_enabled=False))
    for name,rows in [('MAIN',main),('PAIRED_CLASSES',classes),('ESTIMATE_BIAS',estimates),('ESTIMATE_MACRO',macro),('RANKING_ERRORS',rankings),('FOLD_COUNTS',folds),('FOLD_CONSISTENCY',foldcheck),('DRIFT',drift),('BOUNDARIES',boundaries),('CONFUSION',confusion)]:csvwrite(out/(name+'.csv'),rows)
    gpu=sum(z['residence_seconds'] for z in s['program']['costs']);cpu=sum(z['residence_seconds'] for z in s['program']['cpu_costs'])
    audit=dict(status='PASS' if complete else 'INCOMPLETE',source_commit=s['source_commit'],version=VERSION,optimizer_updates=0,policy_updates=0,val_accessed=False,test_accessed=False,independent_calibration=False,round_gpu_process_residence_seconds=gpu,round_cpu_prepare_residence_seconds=cpu,prelaunch_CPU_residence_seconds=s['CPU_check'].get('CPU_process_residence_total_seconds',s['CPU_check']['process_residence_seconds']),pipeline_included_not_added=True,program_cost_records=s['program']['costs'],CPU_cost_records=s['program']['cpu_costs'],planned_diagnostic_image_exposures=219448,preflight_image_exposures=512,main_rows=len(main),class_rows=len(classes),estimator_rows=len(estimates),estimator_macro_rows=len(macro),ranking_rows=len(rankings),failure_records=s['program']['failures'],wall_seconds=s['program']['whole_round_wall_seconds'])
    (out/'AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (out/('RESULTS.json' if complete else 'PARTIAL_RESULTS.json')).write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    lines=['# 训练侧配对诊断','', '完成状态：'+('COMPLETE' if complete else 'INCOMPLETE')+'。仅训练图像冻结推理，0optimizer/policy；val/test未读取。不是独立校准、泛化指标或新方法选择。','', '|job|task|head|救回|破坏|宏救回pp|宏破坏pp|净pp|新类救回|新类破坏|','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    lines += ['|'+ '|'.join(str(v) for v in [z['job'],z['task'],z['head'],z['rescued'],z['harmed'],round(z['macro_rescue_pp'],6),round(z['macro_harm_pp'],6),round(z['actual_net_pp'],6),z['new_rescued'],z['new_harmed']])+'|' for z in main]
    lines+=['','完整逐类/两fold/预测偏差/漂移/竞争边界/纠正破坏流见同目录CSV。成对分数矩阵、shift、特征、样本ID和模型留NAS私有，不外传。未知质量只比较已识别部分与实际值，不能填完整总体偏差；两fold不消除分类器曝光。','',f'GPU进程驻留 {gpu} 秒；CPU元数据准备 {cpu} 秒；预启动合成CPU {audit["prelaunch_CPU_residence_seconds"]} 秒；并行墙钟 {audit["wall_seconds"]} 秒，各项单列。旧控制/原方向与旧评价费用不重复收费。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');return audit


def collect(round_dir):
    r=Path(round_dir);rt=r/'local_runtime'
    if (r/'results').exists() or (rt/'DELIVERY.private.json').exists():raise ValueError('Already collected/delivered')
    h=json.loads((rt/'HANDOFF.private.json').read_text());root=h['current_root']
    code='import json,subprocess\nfrom pathlib import Path\nr=Path('+repr(root)+')\ns=json.loads((r/"PROGRAM_STATE.json").read_text());assert s["status"] in ("COMPLETE","INCOMPLETE") and not s["active"]\nps=subprocess.check_output(["ps","-eo","args="],text=True);assert not any("/tmp/q220w.py" in z or "/tmp/q220c.py" in z for z in ps.splitlines())\na=dict(program=s,runs={},preflights={})\nfor n in '+repr(JOBS)+':\n for key,folder in [("runs","runs"),("preflights","preflight")]:\n  p=r/folder/n/"RESULTS.json"\n  if not p.exists() and key=="runs":p=r/folder/n/"PARTIAL_RESULTS.json"\n  if p.exists():a[key][n]=json.loads(p.read_text())\nprint(json.dumps(a,allow_nan=False))\n'
    x=subprocess.run(['ssh','my-gpu','/tmp/q107v/bin/python -'],input=code,text=True,capture_output=True)
    if x.returncode:raise RuntimeError('Terminal collection failed; retain private receipts')
    s=json.loads(x.stdout);s.update(lock=json.loads((r/'PROTOCOL_LOCK.json').read_text()),source_commit=h['source_commit'],CPU_check=json.loads((r/'CPU_PRODUCTION_CHECK.json').read_text()),reporting_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    return report(s,r)
