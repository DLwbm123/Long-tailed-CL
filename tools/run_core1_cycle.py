"""Bounded CORE1 waves. Private config on stdin; public aggregates only."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from run_prototype_single import save
from next1_support import screening
from evaluate_next1 import pair_tables
from evaluate_next2h import transitions


def screen(candidate, reference):
    result = screening(candidate['metrics'], reference['metrics'])
    order = reference['metrics']['stages'][-1]['seen']
    def new(value):
        recalls = value['metrics']['stages'][-1]['per_class_recall']
        return sum(recalls[str(c)] for c in order[-2:])/2
    result['new_recall'] = new(candidate)-new(reference)
    result['passed'] = result['passed'] and result['new_recall'] >= -.01
    return result


def report(root, records, result):
    public = root/'public'; mechanisms = {}
    for p in sorted(root.glob('*/diagnostics.json')):
        mechanisms[p.parent.name] = json.loads(p.read_text())
    boundaries = {p.parent.name+'/'+p.name:json.loads(p.read_text()) for p in root.glob('*/boundary_*.json')}
    save(public/'MECHANISM_DIAGNOSTICS.json', dict(blocks=mechanisms,boundaries=boundaries))
    with (public/'STAGE_METRICS.csv').open('w') as stream:
        writer = csv.writer(stream);writer.writerow(['setting_arm','task','n','BA','accuracy','tail'])
        for name, value in records.items():
            for row in value['metrics']['stages']:
                writer.writerow([name,row['task'],row['validation_n'],row['balanced_accuracy'],row['accuracy'],row['tail_recall']])
    with (public/'CLASS_DIAGNOSTICS.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['setting_arm','class','n','first_task','first','historical_max','final','first_minus_final','standard_forgetting'])
        for name,value in records.items():
            for label,row in value['class_learning'].items():
                history=[r['per_class_recall'][label] for r in value['metrics']['stages'] if label in r['per_class_recall']]
                writer.writerow([name,label,value['metrics']['stages'][-1]['per_class_n'][label],row['first_task'],row['first_recall'],max(history),row['final_recall'],row['first_minus_final'],row['standard_forgetting']])
    lines=['# CORE1 原型引导竞争约束：阶段结果','',
        '完整候选PC；PC_uniform与PC_mean为预先固定消融。原型决定类对竞争权重；完整二阶矩计算平方间隔代理；适配器持续训练。推理仍为线性头。','',
        '|设置|最终BA %|平均BA %|尾类 %|遗忘 pp|新两类 %|','|---|---:|---:|---:|---:|---:|']
    for name,value in records.items():
        m=value['metrics'];last=m['stages'][-1];new=np.mean([last['per_class_recall'][str(c)] for c in last['seen'][-2:]])
        lines.append(f"|{name}|{m['final_balanced_accuracy']*100:.4f}|{m['average_incremental_balanced_accuracy']*100:.4f}|{m['final_tail_recall']*100:.4f}|{m['forgetting']*100:.4f}|{new*100:.4f}|")
    lines+=['', '判定与下一阶段：`'+json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False)+'`','',
        '固定初筛要求：PC−R最终BA至少+1pp，尾类不低于−0.5pp、遗忘增加不超过1pp、最后任务新类不低于−1pp。消融不参与胜者选择。',
        '初轮复用NEXT1同一T1完整前缀，不能称独立重复。新增seed从头训练；官方val为反复使用的开发集，meta亦是训练来源。test封存，没有独立确认声明。',
        '旧统计共同平移是假设；二次损失在给定统计下精确，不等于真实旧类风险。头残差、实际更新、权重和梯度诊断见MECHANISM_DIAGNOSTICS，失败和驻留见BUDGET_LEDGER。',
        '所有正负结果和失败保留。只有完整候选满足门槛才进入预注册复核；否则等待预算内有明确证据的机制修订。', '']
    (public/'REPORT_ZH.md').write_text('\n'.join(lines))


def run(c):
    root=Path(c['root']);public=root/'public'
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((root/'PROGRAM_STATE.json').read_text())
    if previous['status'] not in ('READY','AWAITING_ANALYSIS'):
        raise ValueError('Coordinator requires READY or a frozen follow-up')
    ledger_path=public/'BUDGET_LEDGER.json'
    old_ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    jobs=old_ledger.get('jobs',[]);active={};phase='PREFLIGHT';records={}
    failures=[];formal=list(old_ledger.get('formal_ids',[]));result={}
    if (public/'RESULTS.json').exists(): records=json.loads((public/'RESULTS.json').read_text()).get('records',{})
    def ledger():
        rows=[];total=0.;reserved=0.;diagnostic=0.
        for j in jobs:
            elapsed=j.get('elapsed_seconds',max(0,time.time()-j['started']))
            sfile=root/j['id']/'STATUS.json'
            try:s=json.loads(sfile.read_text())
            except (FileNotFoundError,json.JSONDecodeError):s={}
            extra=elapsed if j['kind']=='preflight' else min(elapsed,s.get('diagnostic_gpu_seconds',0.))
            total+=elapsed;diagnostic+=extra
            if j['id'] in active:reserved+=max(0,j['cap']-elapsed)
            rows.append(dict(j,elapsed_seconds=elapsed,diagnostic_seconds=extra,worker_status=s.get('status'),worker_phase=s.get('phase'),
                logical_updates=s.get('steps'),actual_updates=s.get('actual_updates'),peak_gpu_bytes=s.get('peak_gpu_bytes')))
        return dict(started=c['started'],deadline=c['deadline'],phase=phase,total_gpu_seconds=total,
            gpu_reserved_seconds=reserved,gpu_limit_seconds=115200,diagnostic_gpu_seconds=diagnostic,diagnostic_limit_seconds=3600,
            formal_ids=formal,formal_used=len(formal),formal_limit=20,storage_limit_bytes=c['storage_limit_bytes'],jobs=rows,
            test_accessed=False,independent_confirmation=False)
    def state(status='RUNNING',**fields):
        value=ledger();save(ledger_path,value)
        save(root/'PROGRAM_STATE.json',dict(status=status,phase=phase,started=c['started'],deadline=c['deadline'],
            gpu_seconds_used=value['total_gpu_seconds'],gpu_seconds_reserved=value['gpu_reserved_seconds'],formal_used=len(formal),
            revision_used=c.get('revision',0),publication_verified=False,test_accessed=False,**fields))
        save(root/'EVALUATION_GATE.json',dict(phase=phase))
    def reap():
        for key,p in list(active.items()):
            code=p.poll()
            if code is not None:
                job=next(j for j in jobs if j['id']==key)
                job.update(elapsed_seconds=time.time()-job['started'],exit_code=code);active.pop(key)
                if code: failures.append(dict(id=key,exit_code=code))
    def terminate():
        for p in active.values():
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        until=time.monotonic()+5
        while active and time.monotonic()<until:reap();time.sleep(.1)
        for p in list(active.values()):
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
        reap()
    def poll():
        reap();value=ledger()
        if time.time()>=c['deadline']:raise RuntimeError('DEADLINE')
        if value['total_gpu_seconds']>=115200 or value['diagnostic_gpu_seconds']>=3600:raise RuntimeError('BUDGET')
        for j in jobs:
            if j['id'] in active and time.time()-j['started']>=j['cap']:
                p=active[j['id']]
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                j['timeout']=True
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL);p.wait()
        reap()
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())>c['storage_limit_bytes']:
            raise RuntimeError('STORAGE_QUOTA')
        state()
    def launch(spec,gpu):
        value=ledger()
        if any(j['id']==spec['id'] for j in jobs):raise RuntimeError('NO_RETRY')
        if value['total_gpu_seconds']+value['gpu_reserved_seconds']+spec['cap']>115200:raise RuntimeError('NO_RESERVATION')
        if spec['kind']=='preflight' and value['diagnostic_gpu_seconds']+sum(max(0,j['cap']-(time.time()-j['started'])) for j in jobs if j['id'] in active and j['kind']=='preflight')+spec['cap']>3600:
            raise RuntimeError('NO_DIAGNOSTIC_RESERVATION')
        if time.time()+spec['cap']>c['deadline']:raise RuntimeError('NO_DEADLINE_RESERVATION')
        if spec['kind']=='train':
            if len(formal)>=20:raise RuntimeError('FORMAL_LIMIT')
            formal.append(spec['id'])
        config=dict(c['base'],**spec['config'],max_wall_seconds=spec['cap']-10,output=str(root/spec['id']))
        cfg=root/(spec['id']+'.private.json');save(cfg,config)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),Q98_SOURCE=str(root/'source/tools'),Q98_KIND='train' if spec['kind']=='preflight' else spec['kind'],
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with cfg.open('rb') as inp,(root/(spec['id']+'.private.log')).open('wb') as out:
            process=subprocess.Popen([c['python'],'/tmp/q98w.py'],stdin=inp,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        jobs.append(dict(id=spec['id'],kind=spec['kind'],cap=spec['cap'],gpu=gpu,pid=process.pid,started=time.time()))
        active[spec['id']]=process;state()
        command=subprocess.check_output(['ps','-p',str(process.pid),'-o','args='],text=True)
        if any(word in command for word in ('wangbomin','LongTailed','core1','prototype')):raise RuntimeError('NON_NEUTRAL_PROCESS')
    def schedule(specs):
        pending=list(specs)
        while pending or active:
            poll();used={j['gpu'] for j in jobs if j['id'] in active}
            if pending:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                for line in free.strip().splitlines():
                    index,memory=map(int,line.split(','))
                    if index not in c['gpus'] or index in used or memory<18432:continue
                    if pending:launch(pending.pop(0),index);used.add(index)
            time.sleep(2)
        poll()
    def spec(name,arm,seed,prefix=None,preflight=False):
        config=dict(method=arm,seed=seed,split_file=c['split_file'])
        if prefix:config['prefix']=prefix
        if preflight:config['preflight']=True
        return dict(id=name,kind='preflight' if preflight else 'train',cap=600 if preflight else 5400,config=config)
    def evaluate(names):
        specs=[]
        for name in names:
            s=json.loads((root/name/'STATUS.json').read_text()) if (root/name/'STATUS.json').exists() else {}
            if s.get('status')!='TRAINED':continue
            if s.get('steps')!=476:raise ValueError('Incorrect ISIC training steps')
            cfg=json.loads((root/(name+'.private.json')).read_text())
            specs.append(dict(id='eval_'+name,kind='eval',cap=600,config=dict(models={name:str(root/name)},seed=cfg['seed'],
                evaluation_gate=str(root/'EVALUATION_GATE.json'))))
        schedule(specs)
        for item in specs:
            name=item['id'][5:];p=root/item['id']/(name+'.json')
            if not p.exists():continue
            records[name]=json.loads(p.read_text())
            stages=[];labels=None
            for t in range(1,5):
                with np.load(root/item['id']/(name+f'_stage_{t}.private.npz')) as data:
                    stages.append(data['scores']);labels=data['labels'].copy()
            save(root/item['id']/'TRANSITIONS.json',transitions(stages,labels))
    try:
        state()
        schedule([spec('preflight_'+a,a,74002,c['prefix'],True) for a in ('PC','PC_mean')])
        preflights={a:json.loads((root/('preflight_'+a)/'PREFLIGHT.json').read_text()) for a in ('PC','PC_mean')}
        if failures or any(p['optimizer_updates']!=0 or p['status']!='PASS' for p in preflights.values()):raise RuntimeError('PREFLIGHT_FAILED')
        save(public/'PREFLIGHT.json',dict(cpu=c['cpu_checks'],native=preflights))
        phase='TRAIN_MAIN';state()
        names=['main_'+a for a in ('R','PC','PC_uniform','PC_mean')]
        schedule([spec(n,n[5:],74002,c['prefix']) for n in names])
        phase='EVALUATE_MAIN';state();evaluate(names)
        result=dict(records=records,failures=failures,independent_confirmation=False,test_accessed=False)
        if all(n in records for n in names):
            result['main_screen']={a:screen(records['main_'+a],records['main_R']) for a in ('PC','PC_uniform','PC_mean')}
            result['paired_predictions']=pair_tables({n:root/('eval_'+n) for n in names},names)
            result['screen_success']=result['main_screen']['PC']['passed']
        else:result['screen_success']=False
        save(public/'RESULTS.json',result);report(root,records,result)
        if result['screen_success'] and not failures:
            phase='TRAIN_ROBUSTNESS';state(screen_success=True)
            names=[f'seed{s}_{a}' for s in (74003,74004) for a in ('R','PC')]
            schedule([spec(f'seed{s}_{a}',a,s) for s in (74003,74004) for a in ('R','PC')])
            phase='EVALUATE_ROBUSTNESS';state();evaluate(names)
            robust={}
            for s in (74003,74004):
                if all(f'seed{s}_{a}' in records for a in ('R','PC')):
                    d=screen(records[f'seed{s}_PC'],records[f'seed{s}_R'])
                    d['passed']=d['ba']>0 and d['tail']>=-.005 and d['forgetting']<=.01 and d['new_recall']>=-.01
                    robust[str(s)]=d
            result['robustness']=robust
            result['macro_ba_difference']=float(np.mean([result['main_screen']['PC']['ba']]+[d['ba'] for d in robust.values()]))
            result['robustness_success']=len(robust)==2 and all(d['passed'] for d in robust.values()) and result['macro_ba_difference']>=.01
        result.update(records=records,failures=failures)
        phase='WAVE_COMPLETE';save(public/'RESULTS.json',result);report(root,records,result)
        state('READY_FOR_TRANSFER' if result.get('robustness_success') else 'AWAITING_ANALYSIS',
            screen_success=result['screen_success'],robustness_success=result.get('robustness_success',False),failures=failures)
    except BaseException as exc:
        terminate();phase='WAVE_STOPPED'
        result.update(records=records,failures=failures,error=str(exc),screen_success=False,independent_confirmation=False,test_accessed=False)
        save(public/'RESULTS.json',result);report(root,records,result)
        hard=any(s in str(exc) for s in ('DEADLINE','BUDGET','RESERVATION','FORMAL_LIMIT','STORAGE_QUOTA'))
        state('STOPPED_BUDGET' if hard else 'AWAITING_ANALYSIS',error=str(exc),failures=failures)
        raise


if __name__=='__main__':run(json.load(sys.stdin))
