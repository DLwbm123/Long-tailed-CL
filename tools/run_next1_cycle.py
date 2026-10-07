"""One bounded NEXT1 cycle; configurations and checkpoints remain private."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_prototype_single import save
from next1_support import screening
from evaluate_next1 import pair_tables


def run(c):
    root=Path(c['root']);public=root/'public';public.mkdir(exist_ok=True)
    base=c['base'];jobs=[];active={};formal=[];phase='AUDIT_PREFLIGHT'
    limits={'diagnostic':3600.,'main':14400.,'robustness':10800.}
    def ledger():
        spent={k:0. for k in limits};reserved={k:0. for k in limits}
        rows=[]
        for j in jobs:
            elapsed=j.get('elapsed_seconds',time.time()-j['started'])
            status_file=Path(j['output'])/'STATUS.json'
            try:status=json.loads(status_file.read_text())
            except (FileNotFoundError,json.JSONDecodeError):status={}
            extra=min(elapsed,status.get('diagnostic_gpu_seconds',0.)) if j['bucket']!='diagnostic' else 0.
            spent[j['bucket']]+=elapsed-extra;spent['diagnostic']+=extra
            if j['id'] in active:reserved[j['bucket']]+=max(0,j['cap']-elapsed)
            rows.append({k:v for k,v in j.items() if k!='output'}|dict(elapsed_seconds=elapsed,extra_diagnostic_seconds=extra,worker_status=status.get('status'),phase=status.get('phase'),logical_updates=status.get('steps'),actual_updates=status.get('actual_updates'),peak_gpu_bytes=status.get('peak_gpu_bytes')))
        return dict(started=c['started'],deadline=c['deadline'],phase=phase,formal_ids=formal,
            formal_used=len(formal),formal_limit=9,limits_seconds=limits,spent_seconds=spent,
            reserved_seconds=reserved,total_gpu_seconds=sum(spent.values()),jobs=rows,
            storage_limit_bytes=c['storage_limit_bytes'],test_accessed=False)
    def state(status='RUNNING',**fields):
        value=ledger();save(public/'BUDGET_LEDGER.json',value)
        save(root/'PROGRAM_STATE.json',dict(status=status,phase=phase,started=c['started'],deadline=c['deadline'],
            formal_used=len(formal),gpu_seconds_used=value['total_gpu_seconds'],gpu_seconds_reserved=sum(value['reserved_seconds'].values()),
            publication_verified=False,test_accessed=False,**fields))
        save(root/'EVALUATION_GATE.json',dict(phase=phase))
    def stop():
        for process in list(active.values()):
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        until=time.monotonic()+5
        while active and time.monotonic()<until:
            for key,process in list(active.items()):
                if process.poll() is not None:
                    next(j for j in jobs if j['id']==key)['elapsed_seconds']=time.time()-next(j for j in jobs if j['id']==key)['started'];active.pop(key)
            time.sleep(.1)
        for key,process in list(active.items()):
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait();j=next(j for j in jobs if j['id']==key);j['elapsed_seconds']=time.time()-j['started'];active.pop(key)
    def poll():
        for key,process in list(active.items()):
            code=process.poll()
            if code is not None:
                j=next(j for j in jobs if j['id']==key);j.update(elapsed_seconds=time.time()-j['started'],exit_code=code);active.pop(key)
                if code:raise RuntimeError('WORKER_FAILED:'+key)
        value=ledger()
        if time.time()>=c['deadline']:raise RuntimeError('DEADLINE')
        if any(value['spent_seconds'][k]>=limits[k] for k in limits):raise RuntimeError('GPU_BUDGET')
        if value['total_gpu_seconds']>=28800:raise RuntimeError('TOTAL_GPU_BUDGET')
        if any(time.time()-j['started']>=j['cap']-2 for j in jobs if j['id'] in active):raise RuntimeError('WORKER_RESIDENCE_CAP')
        if sum(p.stat().st_size for p in root.rglob('*.pt') if not p.is_symlink())>c['storage_limit_bytes']:raise RuntimeError('STORAGE_QUOTA')
        state()
    def launch(spec,gpu):
        value=ledger();bucket=spec['bucket']
        if value['spent_seconds'][bucket]+value['reserved_seconds'][bucket]+spec['cap']>limits[bucket]:raise RuntimeError('INSUFFICIENT_'+bucket.upper()+'_RESERVATION')
        if sum(value['spent_seconds'].values())+sum(value['reserved_seconds'].values())+spec['cap']>28800:raise RuntimeError('INSUFFICIENT_TOTAL_RESERVATION')
        if spec.get('formal'):
            if len(formal)>=9:raise RuntimeError('FORMAL_LIMIT')
            if spec['formal'] in formal:raise RuntimeError('NO_AUTOMATIC_RETRY')
            formal.append(spec['formal'])
        config=dict(base,**spec['config'],max_wall_seconds=spec['cap']-10)
        cfg=root/(spec['id']+'.private.json');save(cfg,config)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),Q95_SOURCE=str(root/'source/tools'),Q95_KIND=spec['kind'],
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with cfg.open('rb') as inp,(root/(spec['id']+'.private.log')).open('wb') as out:
            process=subprocess.Popen([c['python'],'/tmp/q95w.py'],stdin=inp,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        j=dict(id=spec['id'],bucket=bucket,cap=spec['cap'],gpu=gpu,pid=process.pid,started=time.time(),output=config['output'],formal=spec.get('formal'))
        jobs.append(j);active[spec['id']]=process;state()
        command=subprocess.check_output(['ps','-p',str(process.pid),'-o','args='],text=True).strip()
        if any(x in command for x in ('wangbomin','LongTailed','next1','prototype')):raise RuntimeError('NON_NEUTRAL_PROCESS')
    def schedule(specs):
        pending=list(specs)
        while pending or active:
            poll();used={j['gpu'] for j in jobs if j['id'] in active}
            if pending:
                free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                for line in free.strip().splitlines():
                    index,memory=[int(x.strip()) for x in line.split(',')]
                    if index not in c['gpus'] or index in used or memory<16000:continue
                    if pending:launch(pending.pop(0),index);used.add(index)
            time.sleep(2)
    def train(name,arm,seed,order,prefix,bucket='main',cap=2400,formal_id=None,stop_task=None):
        config=dict(method=arm,seed=seed,order=order,output=str(root/name),split_file=c['split_file'])
        if prefix:config['prefix']=str(prefix)
        if stop_task:config['stop_after_task']=stop_task
        return dict(id=name,kind='train',bucket=bucket,cap=cap,formal=formal_id,config=config)
    def evaluate(name,models,seed,order,bucket,historical=False):
        return dict(id=name,kind='eval',bucket=bucket,cap=450 if historical else 240,config=dict(
            models={k:str(v) for k,v in models.items()},seed=seed,order=order,output=str(root/name),
            historical_audit=historical,evaluation_gate=str(root/'EVALUATION_GATE.json')))
    order0=[4,0,3,7,5,6,2,1];order1=[4,0,5,6,2,1,3,7]
    records={};score_pairs=[]
    try:
        state()
        historical=c['historical']
        specs=[evaluate('audit_AB',{k:historical[k] for k in ('A','B')},74002,order0,'diagnostic',True),
            evaluate('audit_CD',{k:historical[k] for k in ('C','D')},74002,order0,'diagnostic',True),
            evaluate('audit_E',{'E':historical['E']},74002,order0,'diagnostic',True)]
        specs.append(dict(id='preflight',kind='train',bucket='diagnostic',cap=600,config=dict(method='R',seed=74002,
            order=order0,preflight=True,split_file=c['split_file'],output=str(root/'preflight'))))
        schedule(specs)
        paths={n:root/('audit_AB' if n in ('A','B') else 'audit_CD' if n in ('C','D') else 'audit_E') for n in historical}
        audit={n:json.loads((paths[n]/(n+'.json')).read_text()) for n in historical}
        save(public/'HISTORICAL_AUDIT.json',dict(models=audit,paired_predictions=pair_tables(paths,list(historical))))
        preflight=json.loads((root/'preflight/PREFLIGHT.json').read_text());assert preflight['status']=='PASS' and preflight['optimizer_updates']==0
        save(public/'PREFLIGHT.json',dict(cpu=c['cpu_checks'],native=preflight,zero_optimizer_updates=True))
        phase='TRAIN_MAIN_PREFIX';state()
        schedule([train('main_prefix','R',74002,order0,None,cap=1800,formal_id='main_R',stop_task=1)])
        prefix=root/'main_prefix/stage_1.pt'
        phase='TRAIN_MAIN';state()
        schedule([train('main_'+arm,arm,74002,order0,prefix,cap=900 if arm=='F' else 2400,
            formal_id=None if arm=='R' else 'main_'+arm) for arm in ('R','U','CB','UCB','F')])
        for arm in ('R','F','U','CB','UCB'):
            status=json.loads((root/('main_'+arm)/'STATUS.json').read_text())
            assert status['status']=='TRAINED' and status['steps']==(276 if arm=='F' else 476)
        phase='EVALUATE_MAIN';state()
        schedule([evaluate('eval_main_'+arm,{arm:root/('main_'+arm)},74002,order0,'main') for arm in ('R','F','U','CB','UCB')])
        for arm in ('R','F','U','CB','UCB'):records['main_'+arm]=json.loads((root/('eval_main_'+arm)/(arm+'.json')).read_text())
        score_pairs=pair_tables({a:root/('eval_main_'+a) for a in ('R','F','U','CB','UCB')},['R','F','U','CB','UCB'])
        control=records['main_R']['metrics'];screen={a:screening(records['main_'+a]['metrics'],control) for a in ('F','U','CB','UCB')}
        winner=next((a for a in ('F','U','CB','UCB') if screen[a]['passed']),None)
        save(public/'MAIN_SCREEN.json',dict(screen=screen,selected_candidate=winner))
        robustness={};passed=False
        if winner:
            phase='TRAIN_ROBUSTNESS_PREFIX';state(selected_candidate=winner)
            schedule([train('v1_prefix','R',74003,order0,None,'robustness',1800,'v1_R',1)])
            phase='TRAIN_ROBUSTNESS';state(selected_candidate=winner)
            specs=[]
            for setting,seed,order,p in [('v1',74003,order0,root/'v1_prefix/stage_1.pt'),('v2',74002,order1,prefix)]:
                for arm in ('R',winner):specs.append(train(setting+'_'+arm,arm,seed,order,p,'robustness',900 if arm=='F' else 1800,
                    None if setting=='v1' and arm=='R' else setting+'_'+arm))
            schedule(specs)
            phase='EVALUATE_ROBUSTNESS';state(selected_candidate=winner)
            schedule([evaluate('eval_'+setting,{a:root/(setting+'_'+a) for a in ('R',winner)},seed,order,'robustness')
                for setting,seed,order in [('v1',74003,order0),('v2',74002,order1)]])
            differences=[screen[winner]['ba']]
            for setting in ('v1','v2'):
                for arm in ('R',winner):records[setting+'_'+arm]=json.loads((root/('eval_'+setting)/(arm+'.json')).read_text())
                delta=screening(records[setting+'_'+winner]['metrics'],records[setting+'_R']['metrics'])
                delta['passed']=delta['ba']>0 and delta['tail']>=-.005 and delta['forgetting']<=.01
                robustness[setting]=delta;differences.append(delta['ba'])
            robustness['macro_ba_difference']=sum(differences)/3
            passed=robustness['v1']['passed'] and robustness['v2']['passed'] and robustness['macro_ba_difference']>=.01
        result=dict(records=records,main_screen=screen,selected_candidate=winner,robustness=robustness,
            screen_success=winner is not None,robustness_success=passed,independent_confirmation=False,test_accessed=False,
            paired_predictions=score_pairs,historical_R_ba_difference=control['final_balanced_accuracy']-audit['A']['metrics']['final_balanced_accuracy'])
        save(public/'RESULTS.json',result)
        phase='FINISHED';state('COMPLETE_PASS_SCREEN' if passed else 'ROBUSTNESS_FAILED' if winner else 'COMPLETE_NEGATIVE',
            screen_success=winner is not None,robustness_success=passed,selected_candidate=winner)
        report(root,records,result)
    except BaseException as exc:
        stop();phase='FINISHED';reason=str(exc)
        save(public/'RESULTS.json',dict(records=records,status='INCOMPLETE',reason=reason,screen_success=False,
            robustness_success=False,test_accessed=False,independent_confirmation=False))
        state('INCOMPLETE_'+reason.split(':')[0],error=reason,screen_success=False,robustness_success=False)
        report(root,records,dict(screen_success=False,robustness_success=False,error=reason))
        raise


def report(root,records,result):
    public=root/'public';mechanisms={}
    for path in sorted(root.glob('*/diagnostics.json')):
        mechanisms[path.parent.name]=json.loads(path.read_text())
    boundaries={}
    for directory in sorted(root.glob('main_*')):
        if directory.is_dir():boundaries[directory.name]=[json.loads(p.read_text()) for p in sorted(directory.glob('boundary_*.json'))]
    save(public/'MECHANISM_DIAGNOSTICS.json',dict(blocks=mechanisms,boundaries=boundaries))
    with (public/'STAGE_METRICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['setting_arm','task','validation_n','balanced_accuracy','accuracy','tail_recall']);w.writeheader()
        for name,value in records.items():
            for stage in value['metrics']['stages']:w.writerow(dict(setting_arm=name,**{k:stage[k] for k in w.fieldnames[1:]}))
    with (public/'CLASS_DIAGNOSTICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['setting_arm','class','validation_n','first_task','first_recall','final_recall','first_minus_final','standard_forgetting']);w.writeheader()
        for name,value in records.items():
            for label,row in value['class_learning'].items():w.writerow(dict(setting_arm=name,**{'class':label},validation_n=value['metrics']['stages'][-1]['per_class_n'][label],**row))
    lines=['# NEXT1 结果报告','',
        '本轮相对新 R 执行固定五臂及条件配对检查。完整第一任务前缀共享；官方 val 被历史与本轮反复使用，meta 在任务末参与拟合，均不构成独立确认。test 始终封存。','',
        '|设置/臂|最终 BA (%)|平均阶段 BA (%)|尾类 (%)|遗忘 (pp)|','|---|---:|---:|---:|---:|']
    for name,value in records.items():
        m=value['metrics'];lines.append(f"|{name}|{100*m['final_balanced_accuracy']:.4f}|{100*m['average_incremental_balanced_accuracy']:.4f}|{100*m['final_tail_recall']:.4f}|{100*m['forgetting']:.4f}|")
    lines+=['',f"效果初筛通过：{result.get('screen_success',False)}；追加稳健性通过：{result.get('robustness_success',False)}。",
        f"选定候选：{result.get('selected_candidate') or '无'}。",'','门槛为 BA≥+1pp、尾类≥−0.5pp、遗忘≤+1pp；多个通过者按 F→U→CB→UCB 冻结顺序选择。']
    for arm,delta in result.get('main_screen',{}).items():lines.append(f"- {arm}−R：BA {delta['ba']*100:+.4f}pp，尾类 {delta['tail']*100:+.4f}pp，遗忘 {delta['forgetting']*100:+.4f}pp；通过={delta['passed']}。")
    if all('main_'+a in records for a in ('R','F','U','CB','UCB')):
        ba={a:records['main_'+a]['metrics']['final_balanced_accuracy'] for a in ('R','F','U','CB','UCB')}
        lines+=['','预先指定的机制 BA 比较（描述性）：']
        for name,diff in [('U−R',ba['U']-ba['R']),('CB−R',ba['CB']-ba['R']),('UCB−U',ba['UCB']-ba['U']),('UCB−CB',ba['UCB']-ba['CB']),('交互 (UCB−U)−(CB−R)',ba['UCB']-ba['U']-ba['CB']+ba['R']),('F−R',ba['F']-ba['R'])]:lines.append(f'- {name}: {diff*100:+.4f}pp。')
    lines+=['',f"配对稳健性详情：{json.dumps(result.get('robustness',{}),ensure_ascii=False)}",'',
        '逐阶段、逐类分母和首次学习/正式遗忘见两个 CSV；完整混淆、已知任务掩码上界和逐样本正确性配对表见 RESULTS/HISTORICAL_AUDIT。掩码结果依赖任务信息，不能作为可部署性能。',
        '训练块首末梯度、每类曝光/重复及 fit/meta 平方损失见 MECHANISM_DIAGNOSTICS。额外梯度仅用于诊断；F 没有后续梯度，记 NA。共享前缀诊断实际只计算一次。',
        'Eshift/E0 仅检查当前类 meta 上的共同平移；T1 旧头方向记 NA，F 近零分母比值记 NA，不推断真实旧类保护。U 将 T4 从158更新降至68，不能宣称仅增加小任务计算。',
        '所有失败与初始化、提取、求解、评估均计入 BUDGET_LEDGER。诊断为驻留子区间，从主/稳健性费用移至诊断桶，总量不重复计费。无自动重试、未下载新权重、没有旧图像回放。',
        '公开源码、冻结协议、聚合结果、成本与日志摘要；私有配置、图像/身份、逐样本分数、checkpoint 与原始日志留在私有存储。','']
    if result.get('error'):lines+=['异常：'+result['error'],'']
    (public/'REPORT_ZH.md').write_text('\n'.join(lines))


if __name__=='__main__':run(json.load(sys.stdin))
