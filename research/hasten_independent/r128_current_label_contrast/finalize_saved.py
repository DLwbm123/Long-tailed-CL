"""Saved-only independent reward, eligibility, stream, state and nine-gate audit."""
import csv
import json
from pathlib import Path
from .core import ARMS,STAGES,score,summarize,weights_for_current,write,stage_prompt,teacher_negatives


def run(cfg):
    import torch
    torch.set_num_threads(1);r=Path(cfg['output']);source=Path(cfg['prefix_source']);load=lambda n:json.loads((r/n).read_text());pt=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    assert load('CONTROL_EXIT.json')['exit_code']==0
    train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
    y=[int(x['original_label']) for x in train];vy=[int(x['original_label']) for x in val];counts=[y.count(c) for c in range(23)]
    names={int(x['original_label']):x['official_class_name'].replace('-',' ') for x in train}
    curves=load('TRAINING_CURVES.json');evals=load('EVALUATIONS.json');journal=load('ACCESS.private.json');bindings=load('STATE_BINDINGS.json');rollouts=load('ROLLOUT_RECORDS.private.json')
    assert len(curves)==2048 and len(rollouts)==2048 and len(evals)==20 and len(bindings)==16
    prefix=load('PREFIX_INITIALIZATION.json');assert prefix['R125_verified_chronological_first_task_only_prefix'] and prefix['prefix_weights_exact_after_restore'] and prefix['new_SFT_updates']==0
    original=json.loads((source/'EVALUATIONS.json').read_text());initial=next(x for x in original if x['stage']==0)
    for result in evals:
        if result['stage']==0:
            assert result['reused_historical_prefix_readout']
            for key in ['correct','denominator','macro','head','tail','tiny','micro']:assert result[key]==initial[key]
    for event in journal:
        s=event['stage'];ids=event['ids'];labels=y if event['split']=='TRAIN' else vy
        allowed=STAGES[s] if event['split']=='TRAIN' else sum(STAGES[:s+1],[])
        assert ids and all(labels[i] in allowed for i in ids)
        if event['split']=='VAL':assert event['purpose']=='isolated_audit' and event['all_stage_models_sealed'] and event['epoch']>=load(f'STAGE_{s}_ALL_MODELS_SEALED.json')['epoch']
    raw_by={(x['stage'],x['arm'],x['update']):x for x in rollouts};assert len(raw_by)==2048
    for stage in range(1,5):
        current=[i for i,c in enumerate(y) if c in STAGES[stage]];g=torch.Generator().manual_seed(12474005+stage)
        expected=torch.tensor(current)[torch.randint(len(current),(128,8),generator=g)];assert torch.equal(pt(r/f'QUERY_TAPE_{stage}.private.pt'),expected)
        for arm in ARMS:
            assert [e['ids'] for e in journal if e['stage']==stage and e['arm']==arm and e['purpose']=='current_training']==expected.tolist()
            assert [i for e in journal if e['stage']==stage and e['arm']==arm and e['purpose']=='current_calibration' for i in e['ids']]==current
            table=load(f'CALIBRATION_{stage}_{arm}.json');assert table['classes']==STAGES[stage] and table['counts']==[counts[c] for c in STAGES[stage]]
            assert all(0<=g<=n for g,n in zip(table['correct'],table['counts']))
            w,z,med=weights_for_current(table['counts'],table['correct']);assert max(abs(a-b) for a,b in zip(w.tolist(),table['mean_normalized_TARS_weights']))<1e-12
            assert abs(z-table['Z'])<1e-12 and abs(med-table['raw_median'])<1e-12
            assert max(abs(a-b) for a,b in zip(w.tolist(),table['used_weights']))<1e-5
            state=pt(r/f'STATE_{stage}_{arm}.private.pt');assert state['updates']==stage*128
            steps=[int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v];assert steps and set(steps)=={stage*128}
            bind=next(x for x in bindings if x['stage']==stage and x['arm']==arm)
            assert bind['previous_weights_restored'] and bind['anchor_from_own_previous_weights'] and not bind['old_TRAIN_access_allowed'] and bind['optimizer_steps_before']==(stage-1)*128
            sigma=bind['sigma_before'];assert sigma==(1. if stage==1 else pt(r/f'STATE_{stage-1}_{arm}.private.pt')['sigma'])
            seq=[x for x in curves if x['stage']==stage and x['arm']==arm];assert [x['update'] for x in seq]==list(range(128))
            for k,c in enumerate(seq):
                private=raw_by[stage,arm,k];labels=[y[i] for i in expected[k].tolist()];assert labels==private['labels']
                parsed,neg=teacher_negatives(private['text'],private['done'],labels,names,set(sum(STAGES[:stage+1],[])))
                assert parsed==private['parsed'] and neg==private['negatives']
                task=[int(parsed[i]==labels[i//8] and private['done'][i]) for i in range(64)];eligible=[x>=0 for x in neg]
                assert sum(eligible)==c['eligible_teacher_responses'] and abs(sum(task)/64-c['task_reward_mean'])<1e-7
                assert c['nonzero_update']==(c['parameter_L2']>0) and c['unupdated_behavior_logp_difference']==0
                assert c['sigma_before']==sigma;sigma=.999*sigma+.001*c['total_reward_batch_std'];assert abs(c['sigma_after']-sigma)<1e-12
                if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']:
                    assert c['teacher_last_sync_step']==(((stage-1)*128+k)//10)*10
                assert c['teacher_and_actor_anchor_restored']
                assert c['positive_teacher_rows']==(64 if arm=='POS_KD_TARS_RAPO' else sum(eligible) if arm=='CONTRAST_TARS_RAPO' else 0)
                assert c['negative_teacher_rows']==(sum(eligible) if arm=='CONTRAST_TARS_RAPO' else 0)
                assert sum(v['queries'] for v in c['class_groups'].values())==8
                for cls,value in c['class_groups'].items():
                    cls=int(cls);chosen=[i for i,x in enumerate(labels) if x==cls];group_sums=[sum(task[i*8:i*8+8]) for i in chosen]
                    assert value['queries']==len(chosen) and value['all_wrong']==sum(x==0 for x in group_sums) and value['all_correct']==sum(x==8 for x in group_sums) and value['mixed']==sum(0<x<8 for x in group_sums) and value['correct_responses']==sum(group_sums)
                    assert value['eligible_teacher_responses']==sum(sum(eligible[i*8:i*8+8]) for i in chosen)
                    assert value['all_wrong_queries_with_valid_teacher_pair']==sum(sum(task[i*8:i*8+8])==0 and any(eligible[i*8:i*8+8]) for i in chosen)
            assert state['sigma']==sigma
            if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']:
                assert state['teacher_sync_step']==((stage*128-1)//10)*10 and state['teacher'] is not None
    classes=[];oldnew=[]
    for result in evals:
        stage=result['stage'];arm=result['arm'];saved=pt(source/'VAL_0_COMMON.private.pt') if stage==0 else pt(r/f'VAL_{stage}_{arm}.private.pt')
        seen=sum(STAGES[:stage+1],[]);assert saved['ids']==[i for i,c in enumerate(vy) if c in seen]
        good=[0]*23;den=[0]*23;assert len(saved['ids'])==len(saved['predictions'])
        for i,p in zip(saved['ids'],saved['predictions']):den[vy[i]]+=1;good[vy[i]]+=int(vy[i]==p)
        for key,value in score(good,den,seen,counts).items():assert result[key]==value
        old=sum(STAGES[:stage],[]);new=STAGES[stage]
        oldnew.append({'stage':stage,'arm':arm,'old_macro':sum(good[c]/den[c] for c in old)/len(old) if old else None,'new_macro':sum(good[c]/den[c] for c in new)/len(new),'reused_historical_prefix':stage==0})
        for c in seen:classes.append({'stage':stage,'arm':arm,'class':c,'name':names[c],'TRAIN_count':counts[c],'correct':good[c],'denominator':den[c],'recall_percent':100*good[c]/den[c],'role':'new' if c in new else 'old','reused_historical_prefix':stage==0})
    assert len(classes)==288;final=load('FINAL_RESULTS.json');assert final==summarize(evals,counts)
    native=load('NATIVE_ADMISSION.json');cost=load('COSTS.json');c=cost['counts']
    expected={'engineering_optimizer_attempts':5,'engineering_teacherforced_queries':18,'SFT_updates':0,'SFT_queries':0,'post_optimizer_attempts':2048,'post_CE_queries':0,'post_RL_responses':131072,'calibration_responses':17548,'VAL_responses':22812,'engineering_responses':80,'generation_responses':171512,'generation_attempts':171512,'SFT_optimizer_attempts':0,'post_completed_updates':2048,'TRAIN_encoder_rows':4,'VAL_encoder_rows':0,'CE_aux_queries':4098,'native_profile_teacher_rows':32}
    for key,value in expected.items():assert c[key]==value,(key,c[key],value)
    assert c['positive_teacher_rows']==sum(x['positive_teacher_rows'] for x in curves)+sum(x.get('positive_teacher_rows',0) for x in native['native_training_paths'])+16
    assert c['negative_teacher_rows']==sum(x['negative_teacher_rows'] for x in curves)+sum(x.get('negative_teacher_rows',0) for x in native['native_training_paths'])+16
    assert native['passed'] and native['prefix_and_reference_restored'] and native['single_hidden_parity']<=.05 and native['same_shape_last_head_parity']<=.05
    prompts=load('PROMPT_BINDINGS.json');assert set(prompts)==set(map(str,range(5)))
    for stage in range(5):assert prompts[str(stage)]['question']==stage_prompt(stage,names) and prompts[str(stage)]['visual_tokens']==64
    groups=[]
    for stage in range(1,5):
        for arm in ARMS:
            seq=[x for x in curves if x['stage']==stage and x['arm']==arm]
            for cls in STAGES[stage]:groups.append({'stage':stage,'arm':arm,'class':cls,**{k:sum(x['class_groups'].get(str(cls),{}).get(k,0) for x in seq) for k in ['queries','all_wrong','all_correct','mixed','correct_responses','eligible_teacher_responses','all_wrong_queries_with_valid_teacher_pair']}})
    assert len(groups)==72
    def csvout(name,rows):
        with (r/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvout('CLASS_RESULTS.csv',classes);csvout('GROUP_DIAGNOSTICS.csv',groups);write(r/'OLD_NEW_SUMMARY.json',oldnew)
    learning=[]
    for arm in ARMS:
        seq=[x for x in curves if x['arm']==arm];learning.append({'arm':arm,'attempts':len(seq),'nonzero_updates':sum(x['nonzero_update'] for x in seq),'correct_responses':sum(v['correct_responses'] for x in seq for v in x['class_groups'].values()),'positive_teacher_rows':sum(x['positive_teacher_rows'] for x in seq),'negative_teacher_rows':sum(x['negative_teacher_rows'] for x in seq)})
    write(r/'LEARNING_INTEGRITY.json',learning)
    write(r/'FINAL_VERIFICATION.json',{'passed':True,'scope':'CPU_saved_only_no_new_model_calls','new_curve_rows':2048,'historical_prefix_curve_rows':306,'full_chain_rows':2354,'class_rows':288,'class_group_rows':72,'arm_stage_table_rows':20,'unique_readout_sequences':17,'new_readout_sequences':16,'all_prefix_access_seals_tapes_optimizer_sigma_teacher_sync_validity_negative_ties_reward_math_metrics_gates_and_costs_verified':True,'calibration_correct_counts_not_independently_rescorable':True,'teacher_logp_values_not_independently_recomputed':True,'gates':final['gates'],'all_gates_pass':final['all_gates_pass']})
    # Generation tensors were not logged per kind: retain honest padded-token bounds for base TF.
    rl_rows=131072+64;other_rows=c['generation_responses']-rl_rows;decoded=c['decoded_token_cells']
    rl_decoded_bounds=[max(rl_rows,decoded-24*other_rows),min(24*rl_rows,decoded-other_rows)]
    rl_prefill=32768*sum(prompts[str(stage)]['input_tokens'] for stage in range(1,5))+64*native['conservative_longest_prompt_profile']['input_tokens']
    accounting={'exact_generation_prefill_input_tokens':c['generation_input_tokens'],'exact_generation_active_tokens':c['generated_active_tokens'],'exact_generation_decoded_token_cells_including_padding':decoded,'exact_context_teacher_TF_input_cells':c['teacher_TF_input_token_cells'],'base_TF_response_rows_exact':3*rl_rows,'base_TF_input_token_cells_bounds':[3*(rl_prefill+x) for x in rl_decoded_bounds],'base_TF_per_kind_padded_lengths_not_logged_not_zero':True,'all_operator_wall_CPU_and_hard_budget_includes_these_calls':True}
    write(r/'COST_ACCOUNTING_SCOPE.json',accounting)
    pulses=[json.loads(path.read_text()) for path in sorted((r/'INTEGRITY_PULSES').glob('*.json'))]
    write(r/'ALL_ATTEMPT_COSTS.json',{'saved_only_integrity_pulses':pulses,'measured_integrity_pulse_whole_child_CPU_seconds':sum(x.get('whole_child_CPU_seconds',0.) for x in pulses),'successful_run':cost,'controller_exit':load('CONTROL_EXIT.json'),'CPU_software':load('CPU_SOFTWARE_RUN_COSTS.json'),'reused_historical_prefix':prefix,'original_23h_clock_not_reset':True,'private_rollout_outputs_remain_remote':True,'teacher_contexts_are_full_answer_auxiliary_supervision':True,'token_accounting_scope':accounting})
    report=['# R128 当前完整标签对比辅助监督资格实验','','已知SIPO机制的TARS/RaPO适配，不是作者原样复现、独立临床或创新结论。完整标签teacher项是辅助监督，不把它的收益自动记成纯RL必要性。','','|arm|postmacro %|posttail %|forgetting pp|','|---|---:|---:|---:|']
    for arm in ARMS:report.append(f"|{arm}|{100*final['post_stage_means'][arm]['macro']:.4f}|{100*final['post_stage_means'][arm]['tail']:.4f}|{100*final['mean_classwise_forgetting'][arm]:.4f}|")
    report+=['','九门通过'+str(sum(final['gates'].values()))+'/9：'+json.dumps(final['gates']),
        '原完整菜单/参数的首taskhead门保持；R127条件归因未用于更换head基线。全部旧实验失败与正信号保留，负不扫参数/提示/额外SFT/seed/teacher周期。',
        '所有4臂同query/update/学生G8生成预算；CE_AUX得到完整答案、全强度1.0 CE且额外TF/backward，POS_KD得到正教师条件答案，主得到正负条件。实际teacher行/token/时耗分别计费，不声称计算或信息强度完全相等。',
        'SIPO原1/G符号裕量不适用RaPO连续奖励；本轮只用显式token符号投影。无效/未来schema/truncated回答不造负例、不丢组；全错组也不是必有teacher信号，合法率/资格率由私有保存输出独立核算。',
        '复用唯一R125合法generic首taskSFT前缀，306历史步/4868queries/624历史VAL不当新调用或免费；本轮新2048步+131072RLresponses，校准17548/VAL22812/native80合171512新响应，0额外SFT/4TRAIN原生编码/0新VAL编码。2354是加历史前缀的完整链，非2354新更新。',
        '只当前TRAIN、无历史replay；动态seen菜单、全臂seal后seenVAL、各臂连续Adam/sigma/teacherK10、teacher/actor/reference还原与actualL2/资格数学均核。20表行含4共享前缀历史读出=17unique其中16新，288类行/72group类行。',
        'generation各kind未逐项保存padded长度，base TF物理response行数精确、总input-token cells给出由实际总decode cells推得的区间，不伪称全forward token精确值；所有这些算子的CPU/wall/硬预算均已实际计入。',
        '校准只有aggregate正确数，teacher logp/逐类梯度未保存，不补造独立重评分、variance或梯度效果；组份额/teacher项非零不等泛化。所有failed attempts若发生须另记且原钟不重置。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
