"""NEXT2-H bounded readout-only cycle. Private configuration on stdin."""
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

START_CPU=time.process_time()
import numpy as np
import torch
from next1_support import digest_file,state_digest
from run_prototype_single import save,manifests
from next2h_heads import validate_bank,build_cf_head,build_ncm_head,build_cblda_head,score,self_check
from evaluate_next2h import ORDER,evaluate,transitions,differences,bootstrap


def feature_extract(c):
    from run_multilabel import ApartFeatures
    from run_prototype_single import Images,extract
    from torch.utils.data import DataLoader
    start=time.monotonic();torch.set_num_threads(3);seed=74002;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    encoder=ApartFeatures(c['legacy_repo'],c['weight'],8,'cuda:0',seed)
    state=torch.load(c['f_prefix'],map_location='cpu',weights_only=False);encoder.load_state_dict(state['model'],strict=True);encoder.eval()
    del state
    from run_medical_v2 import transform
    _,val=manifests(c)
    ds=Images(val,c['images'],transform(False),seed+400012)
    loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(seed+8012))
    def limit():
        if time.monotonic()-start>=590:raise TimeoutError('Feature extraction residence cap')
    with torch.inference_mode():z,y=extract(encoder,loader,limit)
    np.savez_compressed(c['features_output'],features=z,labels=y)
    save(Path(c['feature_receipt']),dict(optimizer_updates=0,backward_calls=0,validation_n=len(y),
        feature_dim=z.shape[1],peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start))


def run(c):
    root=Path(c['root']);public=root/'public';started=c['started'];deadline=c['deadline'];gpu_elapsed=0.;active=None;formal=[];phase='PROTOCOL_LOCKED';metrics={};heads={};failure={}
    def ledger():
        child=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=time.process_time()-START_CPU+child.ru_utime+child.ru_stime
        if active is not None and active.poll() is None:
            try:
                stat=Path('/proc/'+str(active.pid)+'/stat').read_text().split();cpu+=(int(stat[13])+int(stat[14]))/os.sysconf('SC_CLK_TCK')
            except FileNotFoundError:pass
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        gpu=gpu_elapsed+(time.time()-c['gpu_started'] if active is not None else 0.)
        return dict(started=started,deadline=deadline,wall_seconds=time.time()-started,gpu_seconds=gpu,gpu_seconds_limit=7200,
            cpu_seconds=cpu,cpu_core_hours=cpu/3600,cpu_core_seconds_limit=28800,max_cpu_threads=4,
            persistent_bytes=size,persistent_bytes_limit=1024**3,formal_readouts=formal,readout_limit=3,new_candidates=2,
            neural_updates=0,backward_calls=0,phase=phase,feature_extractions=int(gpu>0),gpu_reserved_seconds=max(0,600-(time.time()-c['gpu_started'])) if active is not None else 0.)
    def budget():
        x=ledger();save(public/'BUDGET_LEDGER.json',x)
        if time.time()>=deadline or x['gpu_seconds']>=7200 or x['cpu_seconds']>=28800 or x['persistent_bytes']>=1024**3:raise RuntimeError('INCOMPLETE_BUDGET')
    def state(status='RUNNING',**kw):
        budget();save(root/'PROGRAM_STATE.json',dict(status=status,phase=phase,publication_verified=False,neural_updates=0,test_accessed=False,**kw))
    def hashed(path):
        budget();v=digest_file(path);budget();return v
    def table(name,rows):
        if not rows:return
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (public/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    try:
        torch.set_num_threads(4);state();self_check();phase='INPUT_AUDIT';state()
        old=Path(c['next1_root']);base=c['base'];train,val=manifests(base)
        assert len(train)==18718 and len(val)==295
        old_env=json.loads((old/'public/SOURCE_ENV_AUDIT.json').read_text());assert old_env['source_commit']=='c72353d5a487f6a6cb5c23eecde54c124d60a9c4'
        for filename,expected in old_env['source_files'].items():assert hashed(old/'source/tools'/filename)==expected
        for part in ('train','val'):assert hashed(Path(base['manifest'])/(part+'.csv'))==old_env['manifest_sha256'][part]
        assert hashed(base['weight'])==old_env['weight_sha256']
        assert hashed(old/'split.private.json')==old_env['split']['sha256']
        label_array=np.asarray([r['label'] for r in val]);groups=[r['identity_component'] for r in val]
        assert len(set(r['relative_path'] for r in val))==295
        previous=None;encoder_hash=None;inputs=[];cached={};file_hashes={};c0_checks=[];covariances={}
        old_results=json.loads((old/'public/RESULTS.json').read_text())['records']
        for arm in ('R','F'):
            cfg=json.loads((old/('eval_main_'+arm+'.private.json')).read_text());assert cfg['seed']==74002 and cfg['order']==ORDER
            assert Path(cfg['manifest'])==Path(base['manifest']) and Path(cfg['models'][arm])==old/('main_'+arm)
            cached[arm]=[]
            for task in range(1,5):
                path=old/('main_'+arm)/f'stage_{task}.pt';resolved=path.resolve()
                if str(resolved) not in file_hashes:file_hashes[str(resolved)]=hashed(resolved)
                data=np.load(old/('eval_main_'+arm)/(arm+f'_stage_{task}.private.npz'))
                rows=np.isin(label_array,ORDER[:2*task]);assert np.array_equal(data['labels'],label_array[rows])
                assert data['scores'].shape==(int(rows.sum()),2*task)
                assert np.array_equal(data['prediction'],np.asarray(ORDER[:2*task])[data['scores'].argmax(1)])
                cached[arm].append(data['scores'].copy());data.close()
                if arm=='R':continue
                state_value=torch.load(path,map_location='cpu',weights_only=False)
                assert state_value['seen']==ORDER[:2*task] and state_value['steps']==276
                model_hash=state_digest(state_value['model']);encoder_hash=encoder_hash or model_hash;assert model_hash==encoder_hash
                bank=state_value['bank'];assert bank['mu'].shape==(2*task,1536)
                C,audit=validate_bank(bank)
                for i,n in enumerate(bank['n']):
                    assert n==sum(r['label']==ORDER[i] for r in train)
                    mass=sum(float(comp['mass']) for comp in bank['components'] if comp['label']==i);assert abs(mass-1)<1e-8
                    assert abs(float(bank['Q'][i].trace())-1)<1e-5
                if previous:
                    n=len(previous['n']);assert torch.equal(bank['mu'][:n],previous['mu']) and torch.equal(bank['Q'][:n],previous['Q']) and bank['n'][:n]==previous['n']
                previous={k:bank[k] for k in ('mu','Q','n')}
                h=build_cf_head(bank);assert torch.allclose(h['weight'],state_value['head'],atol=1e-6,rtol=1e-5)
                c0_checks.append(dict(task=task,residual=h['residual'],head_max_absolute_difference=float((h['weight']-state_value['head']).abs().max()),parameter_equivalence=True))
                covariances[task]=C
                head=h if task>1 else dict(weight=state_value['head'],bias=torch.zeros(2))
                heads.setdefault('C0',[]).append(head)
                directory=root/'heads_C0';directory.mkdir(exist_ok=True);torch.save(head,directory/f'stage_{task}.pt')
                if task==1:
                    for name in ('C1','C2'):
                        heads[name]=[dict(weight=state_value['head'],bias=torch.zeros(2))]
                        directory=root/('heads_'+name);directory.mkdir(exist_ok=True);torch.save(heads[name][0],directory/'stage_1.pt')
                inputs.append(dict(task=task,seen=state_value['seen'],encoder_sha256=model_hash,checkpoint_sha256=file_hashes[str(resolved)],**audit))
                del state_value,bank,C
        for arm in ('R','F'):
            reconstructed=evaluate(cached[arm],label_array);expected=old_results['main_'+arm]['metrics']
            for a,b in zip(reconstructed['stages'],expected['stages']):assert a['per_class_recall']==b['per_class_recall'] and a['per_class_n']==b['per_class_n']
        save(public/'INPUT_AUDIT.json',dict(status='PASS',frozen_model=True,old_moments_unchanged=True,stages=inputs,
            checkpoint_hashes=list(file_hashes.values()),weight_sha256=old_env['weight_sha256'],manifest_sha256=old_env['manifest_sha256'],split_sha256=old_env['split']['sha256'],
            trusted_sample_alignment=dict(source_evaluator_sha256=old_env['source_files']['evaluate_next1.py'],
                manifest_rows_unique=True,manifest_hash_verified=True,evaluation_configs_verified=True,
                deterministic_unshuffled_manifest_order=True,cached_labels_match_manifest_subsets=True),
            preprocessing_source_sha256=hashed(Path(base['legacy_repo'])/'tools/run_medical_v2.py'),
            synthetic_tests='PASS',neural_updates=0,test_accessed=False))
        phase='C0_RECONSTRUCTION_PASS';state();save(public/'HEAD_EQUIVALENCE.json',dict(parameter_checks=c0_checks,prediction_check='PENDING_FROZEN_HEAD_FEATURE_EXTRACTION'))
        for task in (2,3,4):
            budget();state_value=torch.load(old/'main_F'/f'stage_{task}.pt',map_location='cpu',weights_only=False);bank=state_value['bank']
            for name in ('C1','C2'):
                if name in failure:continue
                try:head=build_ncm_head(bank) if name=='C1' else build_cblda_head(bank,covariances[task])
                except (torch.linalg.LinAlgError,ValueError):failure[name]='NUMERICAL_FAILURE';continue
                heads[name].append(head);torch.save(head,root/('heads_'+name)/f'stage_{task}.pt')
            del state_value,bank
        formal.extend(['C0','C1','C2']);head_hashes={name:{f'stage_{t}':hashed(root/('heads_'+name)/f'stage_{t}.pt') for t in range(1,len(values)+1)} for name,values in heads.items()}
        save(public/'HEADS_FROZEN.json',dict(head_hashes=head_hashes,failures=failure,freeze_time=time.time(),rho=.1,numerical_epsilon=1e-6))
        phase='HEADS_READY';state()
        available=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        gpus=[int(line.split(',')[0]) for line in available.splitlines() if int(line.split(',')[1])>=16000]
        if not gpus:raise RuntimeError('BLOCKED_INPUT: no GPU with sufficient memory')
        gpu=min(gpus);config=dict(base,f_prefix=str(old/'main_F/stage_1.pt'),features_output=str(root/'features.private.npz'),feature_receipt=str(root/'feature_receipt.json'))
        save(root/'feature.private.json',config);torch.set_num_threads(1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',Q97_MODE='features',Q97_SOURCE=str(root/'source/tools'))
        c['gpu_started']=time.time()
        with (root/'feature.private.json').open('rb') as inp,(root/'feature.private.log').open('wb') as log:
            active=subprocess.Popen([c['python'],'/tmp/q97w.py'],stdin=inp,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        save(root/'FEATURE_PROCESS.json',dict(pid=active.pid,gpu=gpu,started=c['gpu_started']))
        while active.poll() is None:
            budget()
            if time.time()-c['gpu_started']>=598:raise RuntimeError('INCOMPLETE_BUDGET: feature residence cap')
            time.sleep(1)
        code=active.returncode;gpu_elapsed=time.time()-c['gpu_started'];active=None;torch.set_num_threads(4)
        if code:raise RuntimeError('BLOCKED_INPUT: feature worker failure')
        data=np.load(root/'features.private.npz');z=data['features'];assert np.array_equal(data['labels'],label_array) and z.shape==(295,1536)
        assert np.isfinite(z).all();data.close();prediction_checks=[];new_scores={}
        for task in range(1,5):
            mask=np.isin(label_array,ORDER[:2*task]);sample=z[mask]
            # Original stored and reconstructed heads must agree on the same features.
            state_value=torch.load(old/'main_F'/f'stage_{task}.pt',map_location='cpu',weights_only=False)
            original=sample@state_value['head'].numpy();reconstructed=score(sample,heads['C0'][task-1])
            if not np.array_equal(original.argmax(1),reconstructed.argmax(1)) or not np.array_equal(original.argmax(1),cached['F'][task-1].argmax(1)):
                raise RuntimeError('BLOCKED_BASELINE: reconstructed or historical F predictions differ')
            prediction_checks.append(dict(task=task,n=len(sample),reconstructed_prediction_agreement=1.,historical_prediction_agreement=1.,
                historical_score_max_difference=float(np.abs(original-cached['F'][task-1]).max())))
            del state_value
        save(public/'HEAD_EQUIVALENCE.json',dict(parameter_checks=c0_checks,prediction_checks=prediction_checks,status='PASS'))
        phase='EVALUATION_UNLOCKED';state()
        for name,values in heads.items():
            if name in failure:continue
            new_scores[name]=[cached['F'][0]]+[score(z[np.isin(label_array,ORDER[:2*t])],values[t-1]) for t in (2,3,4)]
        new_scores['R']=cached['R'];metrics={name:evaluate(value,label_array) for name,value in new_scores.items()}
        decompositions={name:transitions(value,label_array) for name,value in new_scores.items()}
        for name,values in new_scores.items():
            for task,value in enumerate(values,1):np.savez_compressed(root/(name+f'_stage_{task}.private.npz'),scores=value,labels=label_array[np.isin(label_array,ORDER[:2*task])])
        comparisons={name:dict(versus_R=differences(metrics[name],metrics['R']),versus_C0=differences(metrics[name],metrics['C0'])) for name in ('C1','C2') if name in metrics}
        selected=next((name for name in ('C1','C2') if name in comparisons and comparisons[name]['versus_R']['passed']),None)
        uncertainty=bootstrap(new_scores,label_array,groups,budget);save(public/'BOOTSTRAP.json',uncertainty)
        status='READY_FOR_REPLICATION' if selected else 'COMPLETE_NEGATIVE'
        save(public/'MAIN_SCREEN.json',dict(comparisons=comparisons,selected_candidate=selected,status=status,failures=failure))
        save(public/'RESULTS.json',dict(metrics=metrics,decompositions=decompositions,comparisons=comparisons,selected_candidate=selected,status=status,
            failures=failure,neural_updates=0,independent_confirmation=False,test_accessed=False))
        table('STAGE_METRICS.csv',[dict(arm=name,**{k:v for k,v in row.items() if k not in ('seen','per_class_recall','per_class_n','score_diagnostics')}) for name,value in metrics.items() for row in value['stages']])
        table('CLASS_DIAGNOSTICS.csv',[dict(arm=name,class_id=label,val_n=value['stages'][-1]['per_class_n'][label],**row) for name,value in metrics.items() for label,row in value['class_learning'].items()])
        table('TRANSITION_DECOMPOSITION.csv',[dict(arm=name,**{k:v*100 if k.startswith('D_') and v is not None else v for k,v in row.items()}) for name,rows in decompositions.items() for row in rows])
        phase='REPORT';state(status,selected_candidate=selected);make_report(public,metrics,comparisons,status,selected,uncertainty,ledger(),decompositions)
        phase='STOP';state(status,selected_candidate=selected)
    except BaseException as exc:
        if active is not None:
            try:os.killpg(active.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:active.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
            gpu_elapsed+=time.time()-c['gpu_started'];active=None
        reason=str(exc);status=reason.split(':')[0] if reason.startswith(('BLOCKED_','INCOMPLETE_')) else 'BLOCKED_INPUT'
        save(public/'RESULTS.json',dict(status=status,reason=reason,metrics=metrics,failures=failure,neural_updates=0,test_accessed=False))
        save(public/'BUDGET_LEDGER.json',ledger());save(root/'PROGRAM_STATE.json',dict(status=status,phase='STOP',reason=reason,publication_verified=False,neural_updates=0,test_accessed=False))
        (public/'REPORT_ZH.md').write_text('# NEXT2-H 未完成报告\n\n状态：'+status+'。输入、数值或资源未完成不作为科学负结果。\n\n原因：'+reason+'\n\n无新增神经更新、无自动重试，私有原始输入与日志不公开。\n')
        raise


def make_report(public,metrics,comparisons,status,selected,uncertainty,cost,decompositions):
    lines=['# NEXT2-H 冻结表示读出实验','',f'状态：{status}；选定候选：{selected or "无"}。本轮新增神经更新0，沿用历史Task1的276更新；不称从零无训练。所有候选头冻结后共享一次F验证特征提取，T1保留原cf_linear分数。','',
        '|臂|最终BA (%)|平均BA (%)|尾类 (%)|遗忘 (pp)|旧六类 (%)|新两类 (%)|','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in metrics.items():lines.append('|'+name+'|'+'|'.join(f'{100*v[k]:.4f}' for k in ('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall'))+'|')
    lines+=['','C0为原生cf_linear重构，C1欧氏NCM（均值不再归一化），C2类平衡rho=0.1固定收缩共享协方差LDA，无先验/温度/偏置调参。相对R四条件：BA≥+1pp，尾类≥−0.5pp，遗忘≤+1pp，新两类≥−1pp；原始精度判断。两个都通过按C1→C2选择。','']
    for name,v in comparisons.items():
        for control,d in v.items():lines.append(f"- {name} {control}：BA{100*d['final_balanced_accuracy']:+.4f}pp，尾类{100*d['final_tail_recall']:+.4f}pp，遗忘{100*d['forgetting']:+.4f}pp，新类{100*d['final_new_recall']:+.4f}pp；通过={d['passed']}。")
    lines+=['','## 旧类退化分解','', '|臂|转换|内部变化pp|新增竞争pp|总下降pp|','|---|---:|---:|---:|---:|']
    for name,rows in decompositions.items():
        for r in rows:
            if r['group']=='old' and r['class_id']=='macro':lines.append(f"|{name}|{r['task']}|{100*r['D_internal']:.4f}|{100*r['D_competition']:.4f}|{100*r['D_total']:.4f}|")
    lines+=['','这只是条件于当前打分器的记账。F内部项描述固定表示的读出变化；R内部项混合表示/搬运/头变化。C1的T2还包含从cf_linear切换到NCM；T3/T4才检查固定旧均值的内部排序。掩码不是部署性能，且不是任意分类器普遍上界。内部项保留负值，不能与max(previous)−final标准遗忘混称。',
        '',f"配对身份组bootstrap固定2000次、seed74102，身份组{uncertainty.get('identity_groups','NA')}；无效抽样{uncertainty.get('invalid_draws','NA')}，不足1900有效样本的指标区间为NA，详见BOOTSTRAP。同一组权重用于所有臂/阶段，95%百分位区间只描述当前开发样本条件不确定性，不是独立确认。",'',
        f"GPU驻留{cost['gpu_seconds']:.4f}秒；CPU消耗{cost['cpu_seconds']:.4f}核秒；持久化{cost['persistent_bytes']}bytes（最终账本含报告自身）；正式读出3，新候选2，神经更新0。加载、审计、求解、评价与失败全部纳入账本；未复制历史checkpoint/bank。",'',
        '四阶段85/149/212/295分母、逐类首次/历史最高/最终召回及遗忘见CSV；最后两类标准遗忘NA。全部混淆与分组正确数在RESULTS。统计Q是类内归一化非中心二阶矩，C0无额外proximal，C2未改rho或裁剪特征值。',
        '官方val反复复用，meta历史参与拟合，均不是独立确认；本轮不读旧训练图像，不读test。通过仅READY_FOR_REPLICATION，失败COMPLETE_NEGATIVE，均停止，不自动新seed/顺序/数据集/训练或监测。',
        '公开源码、协议、头等价/输入审计、聚合指标/诊断/成本与报告；私有路径、身份键、逐样本特征/分数、checkpoint与原始日志不上传GitHub。']
    (public/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    config=json.load(sys.stdin)
    if os.environ.get('Q97_MODE')=='features':feature_extract(config)
    else:run(config)
