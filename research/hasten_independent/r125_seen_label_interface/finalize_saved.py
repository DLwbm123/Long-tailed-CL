"""CPU-only count, access, state, metric and gate audit after the real stream ends."""
import csv
import json
from pathlib import Path

from .core import ARMS,STAGES,score,summarize,weights_for_current,write,stage_prompt


def run(cfg):
    import torch
    torch.set_num_threads(1);root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
    y=[int(r['original_label']) for r in train];vy=[int(r['original_label']) for r in val]
    counts=[y.count(c) for c in range(23)];journal=load('ACCESS.private.json');curves=load('TRAINING_CURVES.json');evaluations=load('EVALUATIONS.json')
    assert len(curves)==2866 and len(evaluations)==25
    assert load('GENERIC_INITIALIZATION.json')['old_task_SFT_restored'] is False
    assert load('GENERIC_INITIALIZATION.json')['task_LoRA_B_zero'] is True
    for entry in journal:
        stage=entry['stage'];ids=entry['ids'];labels=y if entry['split']=='TRAIN' else vy
        allowed=STAGES[stage] if entry['split']=='TRAIN' else [c for task in STAGES[:stage+1] for c in task]
        assert ids and all(labels[i] in allowed for i in ids)
        if entry['split']=='VAL':
            assert entry['purpose']=='isolated_audit' and entry['all_stage_models_sealed']
            assert entry['epoch']>=load(f'STAGE_{stage}_ALL_MODELS_SEALED.json')['epoch']
    first=[i for i,c in enumerate(y) if c in STAGES[0]];g=torch.Generator().manual_seed(74005);expected_batches=[]
    for epoch in range(2):
        ids=torch.tensor(first)[torch.randperm(len(first),generator=g)]
        expected_batches.extend([ids[lo:lo+16].tolist() for lo in range(0,len(first),16)])
    actual=[r['ids'] for r in journal if r['stage']==0 and r['purpose']=='current_training'];assert actual==expected_batches
    bindings=load('STATE_BINDINGS.json');assert len(bindings)==20
    for stage in range(1,5):
        current=[i for i,c in enumerate(y) if c in STAGES[stage]];g=torch.Generator().manual_seed(12474005+stage)
        expected=torch.tensor(current)[torch.randint(len(current),(128,8),generator=g)]
        assert torch.equal(torch.load(root/f'QUERY_TAPE_{stage}.private.pt',map_location='cpu',weights_only=True),expected)
        for arm in ARMS:
            events=[r['ids'] for r in journal if r['stage']==stage and r['arm']==arm and r['purpose']=='current_training']
            assert events==expected.tolist()
            calibration=[i for r in journal if r['stage']==stage and r['arm']==arm and r['purpose']=='current_calibration' for i in r['ids']]
            assert calibration==current
            table=load(f'CALIBRATION_{stage}_{arm}.json');w,z,median=weights_for_current(table['counts'],table['correct'])
            assert table['classes']==STAGES[stage] and table['counts']==[y.count(c) for c in STAGES[stage]]
            assert all(0<=g<=n for g,n in zip(table['correct'],table['counts']))
            assert max(abs(a-b) for a,b in zip(w.tolist(),table['mean_normalized_TARS_weights']))<1e-12
            assert abs(z-table['Z'])<1e-12 and abs(median-table['raw_median'])<1e-12
            expected_used=w.tolist() if arm in ['TARS','TARS_RAPO','TARS_CE'] else [1.]*len(w)
            assert max(abs(a-b) for a,b in zip(expected_used,table['used_weights']))<1e-5
            state=torch.load(root/f'STATE_{stage}_{arm}.private.pt',map_location='cpu',weights_only=True)
            assert state['updates']==128*stage
            steps=[int(s['step']) for s in state['optimizer']['state'].values() if 'step' in s]
            assert steps and set(steps)=={128*stage}
            bind=next(r for r in bindings if r['stage']==stage and r['arm']==arm)
            assert bind['optimizer_steps_before']==128*(stage-1) and bind['previous_weights_restored'] and bind['anchor_from_own_previous_weights']
            seq=[r for r in curves if r['phase']=='POST' and r['stage']==stage and r['arm']==arm]
            assert len(seq)==128 and [r['update'] for r in seq]==list(range(128))
            sigma=bind['sigma_before']
            if stage==1:assert sigma==1.
            else:
                prior=torch.load(root/f'STATE_{stage-1}_{arm}.private.pt',map_location='cpu',weights_only=True)
                assert sigma==prior['sigma']
            for r in seq:
                assert r['nonzero_update']==(r['parameter_L2']>0)
                if arm!='TARS_CE':
                    assert r['unupdated_behavior_logp_difference']==0 and r['sigma_before']==sigma
                    if 'RAPO' in arm:sigma=.999*sigma+.001*r['total_reward_batch_std']
                    assert abs(r['sigma_after']-sigma)<1e-12
                    assert sum(v['queries'] for v in r['class_groups'].values())==8
                    successes=0
                    for c,v in r['class_groups'].items():
                        assert int(c) in STAGES[stage] and v['all_wrong']+v['all_correct']+v['mixed']==v['queries']
                        successes+=v['correct_responses']
                    assert abs(successes/64-r['task_reward_mean'])<1e-7
                    assert 0<r['retention_reward_mean']<=1. and r['drift_mean']>=0
            assert sigma==state['sigma']
    classes=[];old_new=[]
    for result in evaluations:
        stage=result['stage'];arm=result['arm'];filearm='COMMON' if stage==0 else arm
        saved=torch.load(root/f'VAL_{stage}_{filearm}.private.pt',map_location='cpu',weights_only=True)
        seen=[c for task in STAGES[:stage+1] for c in task];expected_ids=[i for i,c in enumerate(vy) if c in seen]
        assert saved['ids']==expected_ids
        good=[0]*23;den=[0]*23
        for i,p in zip(saved['ids'],saved['predictions']):den[vy[i]]+=1;good[vy[i]]+=int(p==vy[i])
        computed=score(good,den,seen,counts)
        for key,value in computed.items():assert result[key]==value
        assert result['evaluation_started_epoch']>=load(f'STAGE_{stage}_ALL_MODELS_SEALED.json')['epoch']
        old=[c for c in seen if c not in STAGES[stage]];new=STAGES[stage]
        old_new.append({'stage':stage,'arm':arm,'old_macro':sum(good[c]/den[c] for c in old)/len(old) if old else None,
                        'new_macro':sum(good[c]/den[c] for c in new)/len(new)})
        for c in seen:classes.append({'stage':stage,'arm':arm,'class':c,'name':train[y.index(c)]['official_class_name'],
            'TRAIN_count':counts[c],'correct':good[c],'denominator':den[c],'recall_percent':100*good[c]/den[c],
            'role':'new' if c in STAGES[stage] else 'old','shared_first_task_readout':stage==0})
    assert len(classes)==360
    final=load('FINAL_RESULTS.json');assert final==summarize(evaluations,counts)
    cost=load('COSTS.json');native=load('NATIVE_ADMISSION.json');c=cost['counts']
    expected={'engineering_optimizer_attempts':5,'engineering_teacherforced_queries':16,'SFT_updates':306,'SFT_queries':4868,
        'post_optimizer_attempts':2560,'post_completed_updates':2560,'SFT_optimizer_attempts':306,'post_CE_queries':4096,
        'post_RL_responses':131072,'calibration_responses':21935,'engineering_responses':80,'TRAIN_encoder_rows':4,'VAL_encoder_rows':0,
        'VAL_responses':native['VAL_responses_expected'],'generation_responses':native['total_generation_responses_expected'],
        'generation_attempts':native['total_generation_responses_expected']}
    assert c==expected,(c,expected)
    assert native['passed'] and native['generic_and_reference_restored']
    assert native['single_hidden_parity']<=.05 and native['same_shape_last_head_parity']<=.05
    # R114's frozen Experiment applies this same normalization to every task label.
    names={int(r['original_label']):r['official_class_name'].replace('-',' ') for r in train}
    prompts=load('PROMPT_BINDINGS.json');assert set(prompts)==set(map(str,range(5)))
    profile=native['conservative_longest_prompt_profile']
    assert native['native_processor_token_binding_exact'] and profile['classes']==STAGES[0]
    assert profile['question'].startswith(stage_prompt(0,names))
    assert profile['question'][len(stage_prompt(0,names)):].replace(' Category','')==''
    for stage in range(5):
        record=prompts[str(stage)]
        assert record['classes']==[c for task in STAGES[:stage+1] for c in task]
        assert record['question']==stage_prompt(stage,names) and record['visual_tokens']==64
        assert record['input_tokens']<=profile['input_tokens']
        assert record['training_generation_reference_and_audit_share_binding']
    qualification=[]
    for arm in ARMS[:-1]:
        seq=[r for r in curves if r['phase']=='POST' and r['stage']==1 and r['arm']==arm]
        success=sum(v['correct_responses'] for r in seq for v in r['class_groups'].values())
        mixed=sum(v['mixed'] for r in seq for v in r['class_groups'].values())
        nonzero=sum(r['nonzero_update'] for r in seq)
        qualification.append({'arm':arm,'stage':1,'correct_responses':success,'mixed_accuracy_groups':mixed,
            'nonzero_parameter_steps':nonzero,'startup_qualified':success>0 and mixed>0 and nonzero>0})
    write(root/'INTERFACE_QUALIFICATION.json',{'scope':'startup_not_efficacy_or_innovation','arms':qualification,
        'all_five_shared_cumulative_seen_prompts_verified':True,'no_future_categories_in_model_prompt_verified':True,
        'no_hard_decoding_mask':True,'fresh_prefix_supervision_and_interface_changes_not_RL_contribution':True})
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
    write(root/'FINAL_VERIFICATION.json',{'passed':True,'scope':'CPU_saved_only_no_new_model_calls','arm_stage_rows':25,
        'actual_unique_readouts':21,'class_rows':360,'curve_rows':2866,'current_only_TRAIN_and_seen_only_VAL_verified':True,
        'all_stage_prompt_bindings_and_native_token_parity_verified':True,
        'prompt_categories_match_frozen_Engine_hyphen_to_space_label_normalization':True,
        'zero_old_TRAIN_replay_verified':True,'same_current_query_tape_verified':True,'own_optimizer_and_sigma_continuity_verified':True,
        'initial_generic_no_old_task_SFT_verified':True,'all_counts_verified':True,'all_nine_gates_verified':True,
        'gates':final['gates'],'all_gates_pass':final['all_gates_pass']})
    write(root/'ALL_ATTEMPT_COSTS.json',{'successful_run':cost,'scientific_model_failed_attempts':0,
        'saved_only_verification_failed_attempts':int((root/'POSTPROCESS_REPAIR.json').exists()),'original_clock_not_reset':True,
        'CPU_software_preflight':load('CPU_SOFTWARE_RUN_COSTS.json'),
        'historical_cache_creation_costs_not_repeated_or_free':True,'partial_failures_must_be_added_if_any':True})
    # Preserve CPU-only interim analyses separately from model training costs.
    pulses=[json.loads(p.read_text()) for p in sorted((root/'INTEGRITY_PULSES').glob('*.json'))]
    costs=load('ALL_ATTEMPT_COSTS.json');costs['CPU_integrity_analyses']=pulses
    costs['CPU_integrity_analysis_body_seconds']=sum(p.get('CPU_seconds',p.get('CPU_analysis_body_seconds',0))
        +p.get('previous_inspection_CPU_body_seconds',0) for p in pulses)
    costs['CPU_integrity_body_times_are_not_whole_process_CPU']=True
    costs['controller_exit']=load('CONTROL_EXIT.json')
    if (root/'POSTPROCESS_REPAIR.json').exists():costs['saved_only_verification_repair']=load('POSTPROCESS_REPAIR.json')
    if (root/'POSTPROCESS_COSTS.json').exists():costs['successful_saved_only_postprocess']=load('POSTPROCESS_COSTS.json')
    write(root/'ALL_ATTEMPT_COSTS.json',costs)
    write(root/'OLD_NEW_SUMMARY.json',old_new)
    groups=[]
    for stage in range(1,5):
        for arm in ARMS[:-1]:
            seq=[r for r in curves if r['phase']=='POST' and r['stage']==stage and r['arm']==arm]
            for c in STAGES[stage]:
                totals={k:sum(r['class_groups'].get(str(c),{}).get(k,0) for r in seq)
                        for k in ['queries','all_wrong','all_correct','mixed','correct_responses']}
                groups.append({'stage':stage,'arm':arm,'class':c,**totals})
    with (root/'GROUP_DIAGNOSTICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(groups[0]));w.writeheader();w.writerows(groups)
    common=torch.load(root/'COMMON_PREFIX_SFT.private.pt',map_location='cpu',weights_only=True)
    zero=[]
    for arm in ARMS:
        seq=[r for r in curves if r['phase']=='POST' and r['arm']==arm]
        state=torch.load(root/f'STATE_4_{arm}.private.pt',map_location='cpu',weights_only=True)
        equal=all(torch.equal(x,a) and torch.equal(y,b) for (x,y),(a,b) in zip(state['weights'],common))
        record={'arm':arm,'attempts':len(seq),'nonzero_updates':sum(r['nonzero_update'] for r in seq),
                'final_weights_equal_common_SFT':equal,'final_sigma':state['sigma']}
        if arm!='TARS_CE':
            record['correct_RL_responses']=sum(v['correct_responses'] for r in seq for v in r['class_groups'].values())
            record['responses']=len(seq)*64
            record['retention_reward_range']=[min(r['retention_reward_mean'] for r in seq),max(r['retention_reward_mean'] for r in seq)]
            record['maximum_advantage_abs']=max(r['advantage_max_abs'] for r in seq)
        zero.append(record)
    write(root/'LEARNING_INTEGRITY.json',{'scope':'saved_only_full_run','new_model_queries_updates':0,'arms':zero})
    report=['# R125 replay-free生成式连续RL机制实验','', '**单顺序开发VAL、非独立患者；首任务共同SFT贡献不记RL。**', '',
        '|arm|post平均macro %|post平均tail %|平均forgetting pp|','|---|---:|---:|---:|']
    for arm in ARMS:report.append(f"|{arm}|{100*final['post_stage_means'][arm]['macro']:.4f}|{100*final['post_stage_means'][arm]['tail']:.4f}|{100*final['mean_classwise_forgetting'][arm]:.4f}|")
    report+=['','原九门：'+json.dumps(final['gates']),'全部通过：'+str(final['all_gates_pass']),
        '','CE控制为匹配TARS权重、同query和更新次数，但无G8 rollout/KL，计算更少。',
        '所有原正负结果与旧门保留，不把新连续条件改写为旧实验成功。',
        '25行含5行共享首任务读出；实际21个读出序列，不是25独立重复。原生cache后端存在不等于可读取旧/未来TRAIN；accessor日志与seal先后已核。',
        '校准保存了aggregate正确计数与数学表，未保存逐例校准预测；可核样本访问/类别分母/权重计算，不能事后独立重评分该校准正确计数。不追加生成补造。']
    report+=['','## 实际学习与保留的区别','',
        '|arm|post尝试|非零参数步|最终参数等首任务SFT|正确RL响应|', '|---|---:|---:|---|---:|']
    for row in zero:
        report.append(f"|{row['arm']}|{row['attempts']}|{row['nonzero_updates']}|{row['final_weights_equal_common_SFT']}|{row.get('correct_RL_responses','不适用')}|")
    if all(r['nonzero_updates']==0 and r['correct_RL_responses']==0 for r in zero if r['arm']!='TARS_CE'):
        report+=['','四RL臂全部新增任务共131072响应、0正确奖励、0实际参数更新；最终逐参数等于首任务共同SFT。组内优势恒零，RaPO的sigma变化没有产生学习。',
            '零遗忘与较高seen平均macro来自停留在首任务模型，不是主方法学会新类、保留奖励的净增量或持续学习成功。主对TARS/RAPO增量0；匹配CE可学习部分新类但严重忘旧，也不是整体成功。']
    report+=['','原九门通过数：'+str(sum(final['gates'].values()))+'/9。旧R117等结果与原失败门保持。',
        '本轮全方法统一累计seen词表、重新generic首任务SFT；历史R124开放问法与本轮的起点也不同，不能把历史差异全记成post提示单组件因果。接口与监督贡献不记RL或创新。',
        '这是事先冻结的唯一seen词表资格实验；若不启动或原九门未过，不扫描prompt/答案格式/seed、追加reasoning或监督救分。不称作者原样复现、独立患者证据或所有RL无效。',
        '所有182226响应、2866曲线、360类行、当前TRAIN/已见VAL访问与封存顺序核对通过。CPU软件和历次saved-only审计另计，body计时不是完整进程CPU。']
    report+=['','## 首个新增任务的启动资格（不替代原九门）','',
        '|arm|正确响应|mixed准确率组|非零参数步|启动资格|','|---|---:|---:|---:|---|']
    for row in qualification:
        report.append(f"|{row['arm']}|{row['correct_responses']}|{row['mixed_accuracy_groups']}|{row['nonzero_parameter_steps']}|{row['startup_qualified']}|")
    report+=['','后处理首次prompt校验误用CSV原始连字符类别名；训练与CE目标始终沿用冻结R114引擎的hyphen→space规范化。校验器改为同映射后重新做CPU-only保存验证，原失败源码/记录保留，0新模型调用/更新，未重跑或修改科学结果。类别身份/顺序不变；这是协议中canonical类别表的实际字符串规范化澄清，非事后更改提示。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
