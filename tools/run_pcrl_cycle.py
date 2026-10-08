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
    if result.get('primary_candidate') == 'FDRL':
        text=(public/'REPORT_ZH.md').read_text()
        text=text.replace('# CORE1 原型引导竞争约束：阶段结果','# FDRL1 蒸馏强度时序调度：阶段结果')
        text=text.replace('完整候选及固定消融以本波协议为准。原型决定类对竞争权重，二阶矩计算历史代理；适配器持续训练，推理仍为线性头。',
            'FD 取 5/10/20；FDRL 主路径实际采样，用跨块折扣回报训练小型 actor–critic。拟合与奖励统计永久隔离。所有九臂均计费相同的三动作分支和四个短预热过程。')
        text=text.replace('固定初筛要求：PC−R最终BA至少+1pp','固定初筛要求：FDRL−匹配R最终BA至少+1pp')
        text=text.replace('所有正负结果和失败保留。只有完整候选满足门槛才进入预注册复核；否则等待预算内有明确证据的机制修订。',
            '还须超过 FD5/10/20、随机、梯度规则、贪心和因果历史状态打乱对照；通过后仅运行冻结的两组新种子。没有自动方法修订或 HK 迁移。')
        text += '\nmeta 仅提供训练反馈，不参与拟合；旧类只保存独立类级奖励矩统计。旧类共同平移仍为近似，官方 val 仍为开发集。\n'
        if result.get('cycle')=='FDRL2':
            text=text.replace('FDRL1 蒸馏强度时序调度','FDRL2 完整任务回报与奖励校准')
            text=text.replace('FD 取 5/10/20；FDRL 主路径实际采样，用跨块折扣回报训练小型 actor–critic。拟合与奖励统计永久隔离。所有九臂均计费相同的三动作分支和四个短预热过程。',
                'FD 取 5/10/20；FDRL 在同一任务两个 epoch 内固定行为策略，任务结束统一更新。R 底座无竞争项；COMP10/COMP_FDRL 单独检查竞争项。拟合与奖励永久隔离；每臂均计入16次不同采样比例与0/8步起点的预热，以及三动作分支成本。')
            text=text.replace('还须超过 FD5/10/20、随机、梯度规则、贪心和因果历史状态打乱对照；通过后仅运行冻结的两组新种子。没有自动方法修订或 HK 迁移。',
                '20个 T1 结束时的伪增量校准全部完成并通过冻结奖励门槛，才进入正式训练。候选还须超过 FD5/20、随机、贪心、状态打乱和竞争项对照；通过后仅追加两组冻结新种子。无自动修订或 HK 迁移。')
            if 'reward_calibration' in result:
                text+='\n## 奖励校准\n\n```json\n'+json.dumps(result['reward_calibration'],ensure_ascii=False,indent=2)+'\n```\n'
        (public/'REPORT_ZH.md').write_text(text)
        return
    if result.get('primary_candidate') in ('FINALHEAD', 'PAIRHEAD'):
        primary = result['primary_candidate']
        text = (public/'REPORT_ZH.md').read_text()
        text = text.replace('# CORE1 原型引导竞争约束：阶段结果', '# FINALHEAD 重拟合一致选择：阶段结果')
        text = text.replace('完整候选及固定消融以本波协议为准。原型决定类对竞争权重，二阶矩计算历史代理；适配器持续训练，推理仍为线性头。',
            '复用 FIXED1 完整编码器轨迹；当前 fit 拟合全部七个任务末分类头，meta 身份组重采样评分；候选无可信代理改善则保留动作1。没有新增适配器或策略更新。')
        text = text.replace('固定初筛要求：PC−R最终BA至少+1pp', '固定初筛要求：FINALHEAD−R最终BA至少+1pp')
        text = text.replace('所有正负结果和失败保留。只有完整候选满足门槛才进入预注册复核；否则等待预算内有明确证据的机制修订。',
            '还须最终 BA 严格高于 FIXED1 和 PC，且相对 FIXED1 的尾类、遗忘、新类保护通过。首轮只运行一个封存候选；通过后另行冻结新种子复核，不自动扩大本波。所有正负结果和失败保留。')
        if primary == 'PAIRHEAD':
            text = text.replace('FINALHEAD 重拟合一致选择', 'PAIRHEAD 类对风险重分配').replace('FINALHEAD−R', 'PAIRHEAD−R')
            text = text.replace('复用 FIXED1 完整编码器轨迹；当前 fit 拟合全部七个任务末分类头，meta 身份组重采样评分；候选无可信代理改善则保留动作1。没有新增适配器或策略更新。',
                '复用 FIXED1 编码器与训练统计；每类面向旧/新组的预算分别守恒，保留一半原权重，另一半按训练矩统计的标准化类对间隔风险分配。唯一固定候选，无 meta 选择、无参数搜索、无新增优化器更新。')
        (public/'REPORT_ZH.md').write_text(text)
        return
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
    gpu_limit=c.get('gpu_limit_seconds',115200);diagnostic_limit=c.get('diagnostic_limit_seconds',3600)
    formal_limit=c.get('formal_limit',20)
    lock=(root/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((root/'PROGRAM_STATE.json').read_text())
    if previous['status'] not in ('READY','AWAITING_ANALYSIS'):
        raise ValueError('Coordinator requires READY or a frozen follow-up')
    ledger_path=root/'public/BUDGET_LEDGER.json'
    old_ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    jobs=old_ledger.get('jobs',[]);active={};phase='PREFLIGHT';records={}
    failures=[];formal=list(old_ledger.get('formal_ids',[]));result=dict(cycle=c.get('cycle'),primary_candidate=primary)
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
            extra=elapsed if j['kind'] in ('preflight','calibration') else min(elapsed,s.get('diagnostic_gpu_seconds',0.))
            total+=elapsed;diagnostic+=extra
            if j['id'] in active:reserved+=max(0,j['cap']-elapsed)
            rows.append(dict(j,elapsed_seconds=elapsed,diagnostic_seconds=extra,worker_status=s.get('status'),worker_phase=s.get('phase'),
                logical_updates=s.get('steps'),actual_updates=s.get('actual_updates'),retained_updates=s.get('retained_updates'),rollout_updates=s.get('rollout_updates'),policy_updates=s.get('policy_updates'),peak_gpu_bytes=s.get('peak_gpu_bytes')))
        return dict(started=c['started'],deadline=c['deadline'],phase=phase,total_gpu_seconds=total,
            gpu_reserved_seconds=reserved,gpu_limit_seconds=gpu_limit,diagnostic_gpu_seconds=diagnostic,diagnostic_limit_seconds=diagnostic_limit,
            formal_ids=formal,formal_used=len(formal),formal_limit=formal_limit,storage_limit_bytes=c['storage_limit_bytes'],jobs=rows,
            test_accessed=False,independent_confirmation=False,diagnostic_cpu_core_seconds=c.get('diagnostic_cpu_core_seconds',0.))
    def state(status='RUNNING',**fields):
        value=ledger();save(ledger_path,value)
        save(root/'PROGRAM_STATE.json',dict(status=status,phase=phase,started=c['started'],deadline=c['deadline'],
            gpu_seconds_used=value['total_gpu_seconds'],gpu_seconds_reserved=value['gpu_reserved_seconds'],formal_used=len(formal),
            revision_used=0,engineering_recovery_used=c.get('engineering_recovery_used',0),wave=wave,primary=primary,main_publication_sha=c.get('main_publication_sha'),publication_verified=False,test_accessed=False,**fields))
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
        if value['total_gpu_seconds']>=gpu_limit or value['diagnostic_gpu_seconds']>=diagnostic_limit:raise RuntimeError('BUDGET')
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
        if value['total_gpu_seconds']+value['gpu_reserved_seconds']+spec['cap']+40>gpu_limit:raise RuntimeError('NO_RESERVATION')
        if spec['kind'] in ('preflight','calibration') and value['diagnostic_gpu_seconds']+sum(max(0,j['cap']-(time.time()-j['started'])) for j in jobs if j['id'] in active and j['kind'] in ('preflight','calibration'))+spec['cap']>diagnostic_limit:
            raise RuntimeError('NO_DIAGNOSTIC_RESERVATION')
        if time.time()+spec['cap']>c['deadline']:raise RuntimeError('NO_DEADLINE_RESERVATION')
        if spec['kind']=='train':
            if len(formal)>=formal_limit:raise RuntimeError('FORMAL_LIMIT')
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
        return dict(id=name,kind='preflight' if preflight else 'train',cap=900 if preflight else c.get('train_cap_seconds',7200),config=config)
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
        if c.get('prepared_prefix_arm'):
            c['prefix']=str(root/(wave+'_preflight_'+c['prepared_prefix_arm'])/'prepared_prefix.pt')
            if not Path(c['prefix']).is_file():raise ValueError('Missing prepared disjoint prefix')
        if c.get('calibration_specs'):
            from summarize_fdrl_calibration import summarize
            phase='CALIBRATE_REWARD';state()
            calibration_prefix=c.get('calibration_id_prefix','calibration')
            schedule([dict(id=f'{calibration_prefix}_{i:02d}',kind='calibration',cap=c['calibration_cap_seconds'],
                config=dict(x,prefix=c['prefix'],split_file=c['split_file'])) for i,x in enumerate(c['calibration_specs'])])
            calibration=[]
            for i in range(len(c['calibration_specs'])):
                path=root/f'{calibration_prefix}_{i:02d}'/'CALIBRATION.json'
                if path.exists():calibration.append(json.loads(path.read_text()))
            gate=summarize(calibration,len(c['calibration_specs']))
            gate['passed']=gate['passed'] and not failures
            save(public/'REWARD_CALIBRATION.json',dict(summary=gate,records=calibration))
            result.update(reward_calibration=gate,records=records,failures=failures,screen_success=False,
                independent_confirmation=False,test_accessed=False)
            if not gate['passed']:
                phase='WAVE_COMPLETE';save(public/'RESULTS.json',result);report(root,records,result,public)
                state('STOPPED_REWARD_GATE',screen_success=False,robustness_success=False,failures=failures)
                return
        phase='TRAIN_MAIN';state()
        names=[wave+'_'+a for a in arms]
        schedule([spec(wave+'_'+a,a,74002,c['prefix']) for a in arms])
        phase='EVALUATE_MAIN';state();evaluate(names)
        reference=c.get('reference_id',wave+'_R');primary_id=wave+'_'+primary
        compared=[n for n in records if n!=reference]
        result.update(records=records,failures=failures,independent_confirmation=False,test_accessed=False,primary_candidate=primary,reference=reference,wave=wave)
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
        for name in c.get('protection_references',[]):
            if name not in records or primary_id not in records:
                result['screen_success']=False
                continue
            guard=screen(records[primary_id],records[name])
            guard['passed']=guard['tail']>=-.005 and guard['forgetting']<=.01 and guard['new_recall']>=-.01
            result.setdefault('additional_protection',{})[name]=guard
            result['screen_success']=result['screen_success'] and guard['passed']
        save(public/'RESULTS.json',result);report(root,records,result,public)
        if result['screen_success'] and not failures and not c.get('stop_after_screen',False):
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
