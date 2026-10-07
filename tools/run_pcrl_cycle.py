"""Bounded PCRL1 validation. Private config on stdin; public aggregates only."""
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


def report(root, records, result, public=None):
    from run_core1_cycle import report as tables
    public = public or root/'public/main'
    tables(root,records,result,public)
    controllers={p.parent.name:json.loads(p.read_text()) for p in root.glob('*/CONTROLLER.json')}
    save(public/'CONTROLLER_DIAGNOSTICS.json',controllers)
    rows=['# 原型竞争预算 RL 验证','',
        '七动作小型组相对策略；相同预算 GREEDY 控制。旧类风险仅为共同平移二阶统计代理。',
        '每个决策四个采样分支加一个 PC 参考，各进行最多两步真实适配器更新；全部计入总成本。',
        '主路径采用更新后策略最大概率动作；GREEDY 才在采样分支与参考中选择最高奖励。',
        'PCRL2：RESPONSE0 使用固定解析预测；RL_RESPONSE 加共享可学习残差；RL_SHARED 去掉固定评分；FIXED1 始终动作1。',
        '主候选、复用对照和追加波以本周期 PROTOCOL_LOCK.json 为准。','',
        '|设置|最终 BA %|平均 BA %|尾类 %|遗忘 pp|最后新类 %|',
        '|---|---:|---:|---:|---:|---:|']
    for name,value in records.items():
        m=value['metrics'];last=m['stages'][-1]
        new=np.mean([last['per_class_recall'][str(c)] for c in last['seen'][-2:]])
        rows.append(f"|{name}|{m['final_balanced_accuracy']*100:.4f}|{m['average_incremental_balanced_accuracy']*100:.4f}|{m['final_tail_recall']*100:.4f}|{m['forgetting']*100:.4f}|{new*100:.4f}|")
    rows += ['', '## 判定', '', '```json', json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2), '```', '',
        '初筛：RL 相对匹配 R 最终 BA ≥ +1 pp、尾类 ≥ −0.5 pp、遗忘增加 ≤ 1 pp、新两类 ≥ −1 pp；还需最终 BA 严格高于协议列出的全部对照。',
        '通过才追加 seed74003/74004 的全部四臂。每个新 seed 对 R 的 BA > 0 且保护通过，三 seed 平均 BA 差 ≥ 1 pp，平均候选 BA 高于全部预注册稳健性对照，才进入固定模块消融。',
        '共享 T1 前缀不算独立重复；meta 属于训练来源且参与任务末 refit，官方 val 为反复复用的开发集。test 封存。',
        '历史代理风险不是真实召回或遗忘保证；短视 contextual bandit，不宣称长期 RL 收益。全负结果、失败、分支及策略更新均保留。',
        'HK 迁移需先核验完整23类资产、泛化评估及成本并冻结附录；未经此步骤不会启动。', '']
    (public/'REPORT_ZH.md').write_text('\n'.join(rows))


def run(c):
    root=Path(c['root']);public=root/'public'/c.get('wave','main');public.mkdir(exist_ok=True)
    wave=c.get('wave','main');primary=c.get('primary','RL');arms=c.get('arms',['R','PC','GREEDY','RL'])
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((root/'PROGRAM_STATE.json').read_text())
    if previous['status'] not in ('READY','AWAITING_ANALYSIS'):
        raise ValueError('Coordinator requires READY or a frozen follow-up')
    ledger_path=root/'public/BUDGET_LEDGER.json'
    old_ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    jobs=old_ledger.get('jobs',[]);active={};phase='PREFLIGHT';records={}
    failures=[];formal=list(old_ledger.get('formal_ids',[]));result={}
    if (public/'RESULTS.json').exists(): records=json.loads((public/'RESULTS.json').read_text()).get('records',{})
    if c.get('reference_result'):
        records[c['reference_id']]=json.loads(Path(c['reference_result']).read_text())
    for name,path in c.get('reference_records',{}).items():
        records[name]=json.loads(Path(path).read_text())
    controls=c.get('controls',['PC','GREEDY'])
    replication_arms=c.get('replication_arms',arms)
    replication_controls=c.get('replication_controls',['PC','GREEDY'])
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
                logical_updates=s.get('steps'),actual_updates=s.get('actual_updates'),retained_updates=s.get('retained_updates'),rollout_updates=s.get('rollout_updates'),policy_updates=s.get('policy_updates'),peak_gpu_bytes=s.get('peak_gpu_bytes')))
        return dict(started=c['started'],deadline=c['deadline'],phase=phase,total_gpu_seconds=total,
            gpu_reserved_seconds=reserved,gpu_limit_seconds=115200,diagnostic_gpu_seconds=diagnostic,diagnostic_limit_seconds=3600,
            formal_ids=formal,formal_used=len(formal),formal_limit=20,storage_limit_bytes=c['storage_limit_bytes'],jobs=rows,
            test_accessed=False,independent_confirmation=False,diagnostic_cpu_core_seconds=c.get('diagnostic_cpu_core_seconds',0.))
    def state(status='RUNNING',**fields):
        value=ledger();save(ledger_path,value)
        save(root/'PROGRAM_STATE.json',dict(status=status,phase=phase,started=c['started'],deadline=c['deadline'],
            gpu_seconds_used=value['total_gpu_seconds'],gpu_seconds_reserved=value['gpu_reserved_seconds'],formal_used=len(formal),
            revision_used=0,wave=wave,primary=primary,main_publication_sha=c.get('main_publication_sha'),publication_verified=False,test_accessed=False,**fields))
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
        if value['total_gpu_seconds']+value['gpu_reserved_seconds']+spec['cap']+40>115200:raise RuntimeError('NO_RESERVATION')
        if spec['kind']=='preflight' and value['diagnostic_gpu_seconds']+sum(max(0,j['cap']-(time.time()-j['started'])) for j in jobs if j['id'] in active and j['kind']=='preflight')+spec['cap']>3600:
            raise RuntimeError('NO_DIAGNOSTIC_RESERVATION')
        if time.time()+spec['cap']>c['deadline']:raise RuntimeError('NO_DEADLINE_RESERVATION')
        if spec['kind']=='train':
            if len(formal)>=20:raise RuntimeError('FORMAL_LIMIT')
            formal.append(spec['id'])
        config=dict(c['base'],**spec['config'],max_wall_seconds=spec['cap']-10,output=str(root/spec['id']))
        cfg=root/(spec['id']+'.private.json');save(cfg,config)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),Q101_SOURCE=c.get('source_dir',str(root/'source/tools')),Q101_KIND='train' if spec['kind']=='preflight' else spec['kind'],
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with cfg.open('rb') as inp,(root/(spec['id']+'.private.log')).open('wb') as out:
            process=subprocess.Popen([c['python'],c.get('worker_entry','/tmp/q101w.py')],stdin=inp,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        jobs.append(dict(id=spec['id'],kind=spec['kind'],cap=spec['cap'],gpu=gpu,pid=process.pid,started=time.time(),source_commit=c.get('source_commit')))
        active[spec['id']]=process;state()
        command=subprocess.check_output(['ps','-p',str(process.pid),'-o','args='],text=True)
        if any(word in command for word in ('wangbomin','LongTailed','core1','prototype','pcrl')):raise RuntimeError('NON_NEUTRAL_PROCESS')
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
        return dict(id=name,kind='preflight' if preflight else 'train',cap=900 if preflight else 7200,config=config)
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
        preflight_arms=c.get('preflight_arms',['RL'])
        schedule([spec(wave+'_preflight_'+a,a,74002,c['prefix'],True) for a in preflight_arms])
        preflights={a:json.loads((root/(wave+'_preflight_'+a)/'PREFLIGHT.json').read_text()) for a in preflight_arms}
        if failures or any(p['optimizer_updates']!=0 or p['status']!='PASS' for p in preflights.values()):raise RuntimeError('PREFLIGHT_FAILED')
        save(public/'PREFLIGHT.json',dict(cpu=c['cpu_checks'],native=preflights))
        phase='TRAIN_MAIN';state()
        names=[wave+'_'+a for a in arms]
        schedule([spec(wave+'_'+a,a,74002,c['prefix']) for a in arms])
        phase='EVALUATE_MAIN';state();evaluate(names)
        reference=c.get('reference_id',wave+'_R');primary_id=wave+'_'+primary
        compared=[n for n in records if n!=reference]
        result=dict(records=records,failures=failures,independent_confirmation=False,test_accessed=False,primary_candidate=primary,reference=reference,wave=wave)
        if reference in records:
            result['main_screen']={n:screen(records[n],records[reference]) for n in compared if n in records}
            paired=[reference]+[n for n in compared if n in records]
            result['paired_predictions']=pair_tables({n:Path(c['reference_records'][n]).parent if n in c.get('reference_records',{}) else root/('eval_'+n) for n in paired},paired)
            result['screen_success']=all(n in records for n in names) and primary_id in result['main_screen'] and result['main_screen'][primary_id]['passed']
        else:result['screen_success']=False
        save(public/'RESULTS.json',result);report(root,records,result,public)
        if result['screen_success']:
            rl_ba=records[primary_id]['metrics']['final_balanced_accuracy']
            result['rl_above_controls']={a:rl_ba>records[wave+'_'+a]['metrics']['final_balanced_accuracy'] for a in controls if wave+'_'+a in records}
            result['screen_success']=len(result['rl_above_controls'])==len(controls) and all(result['rl_above_controls'].values()) and not failures
        save(public/'RESULTS.json',result);report(root,records,result,public)
        if result['screen_success'] and not failures:
            phase='TRAIN_ROBUSTNESS';state(screen_success=True)
            names=[f'{wave}_seed{seed}_{a}' for seed in (74003,74004) for a in replication_arms]
            schedule([spec(f'{wave}_seed{seed}_{a}',a,seed) for seed in (74003,74004) for a in replication_arms])
            phase='EVALUATE_ROBUSTNESS';state();evaluate(names)
            robust={}
            for seed in (74003,74004):
                if all(f'{wave}_seed{seed}_{a}' in records for a in ('R',primary)):
                    d=screen(records[f'{wave}_seed{seed}_{primary}'],records[f'{wave}_seed{seed}_R'])
                    d['passed']=d['ba']>0 and d['tail']>=-.005 and d['forgetting']<=.01 and d['new_recall']>=-.01
                    robust[str(seed)]=d
            result['robustness']=robust
            result['macro_ba_difference']=float(np.mean([result['main_screen'][primary_id]['ba']]+[d['ba'] for d in robust.values()]))
            control_complete=all(p+'_'+a in records for p in (wave,wave+'_seed74003',wave+'_seed74004') for a in replication_arms)
            result['mean_rl_control_differences']={a:float(np.mean([
                records[p+'_'+primary]['metrics']['final_balanced_accuracy']-records[p+'_'+a]['metrics']['final_balanced_accuracy']
                for p in (wave, wave+'_seed74003', wave+'_seed74004')])) for a in replication_controls} if control_complete else {}
            result['robustness_success']=control_complete and len(robust)==2 and all(d['passed'] for d in robust.values()) and result['macro_ba_difference']>=.01 and all(v>0 for v in result['mean_rl_control_differences'].values()) and not failures
            if result['robustness_success']:
                phase='TRAIN_ABLATIONS';state()
                ablations=c.get('ablations',['RL_uniform','RL_mean'])
                names=[wave+'_'+a for a in ablations]
                schedule([spec(wave+'_'+a,a,74002,c['prefix']) for a in ablations])
                phase='EVALUATE_MAIN';state();evaluate(names)
                result['ablations_complete']=all(n in records for n in names)
        result.update(records=records,failures=failures)
        phase='WAVE_COMPLETE';save(public/'RESULTS.json',result);report(root,records,result,public)
        state('READY_FOR_TRANSFER_AUDIT' if result.get('robustness_success') and result.get('ablations_complete') and not failures else 'AWAITING_ANALYSIS',
            screen_success=result['screen_success'],robustness_success=result.get('robustness_success',False),failures=failures)
    except BaseException as exc:
        terminate();phase='WAVE_STOPPED'
        result.update(records=records,failures=failures,error=str(exc),screen_success=False,independent_confirmation=False,test_accessed=False)
        save(public/'RESULTS.json',result);report(root,records,result,public)
        hard=any(s in str(exc) for s in ('DEADLINE','BUDGET','RESERVATION','FORMAL_LIMIT','STORAGE_QUOTA'))
        state('STOPPED_BUDGET' if hard else 'AWAITING_ANALYSIS',error=str(exc),failures=failures)
        raise


if __name__=='__main__':run(json.load(sys.stdin))
