"""CPU-only audit of saved measurements; never imports a model or regenerates responses."""
import csv
import json
from pathlib import Path
from .core import ARMS,MAIN,STAGES,Evidence,score,summarize,weights_for_current,write,stage_prompt


def run(cfg):
    import torch
    torch.set_num_threads(1);root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    pt=lambda n:torch.load(root/n,map_location='cpu',weights_only=True)
    assert load('CONTROL_EXIT.json')['exit_code']==0
    train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
    y=[int(r['original_label']) for r in train];vy=[int(r['original_label']) for r in val];counts=[y.count(c) for c in range(23)]
    journal=load('ACCESS.private.json');curves=load('TRAINING_CURVES.json');evaluations=load('EVALUATIONS.json');bindings=load('STATE_BINDINGS.json')
    assert len(curves)==2866 and len(evaluations)==25 and len(bindings)==20
    generic=load('GENERIC_INITIALIZATION.json');assert not generic['old_task_SFT_restored'] and generic['task_LoRA_B_zero']
    events=load('BUFFER_EVENTS.private.json');pool=Evidence(y);birth={};selected={}
    for stage,event in enumerate(events):
        expected=pool.selected(stage);shard=pt(f'BUFFER_BIRTH_{stage}.private.pt')
        assert event['stage']==stage and event['ids']==expected and shard['ids']==expected
        assert shard['labels']==[y[i] for i in expected]==event['labels']
        assert shard['features'].dtype==torch.bfloat16 and shard['features'].shape==(len(expected),64,2048)
        assert event['epoch']>=load(f'STAGE_{stage}_ALL_MODELS_SEALED.json')['epoch']
        assert event['epoch']<=min(r['evaluation_started_epoch'] for r in evaluations if r['stage']==stage)
        for i,row in zip(expected,shard['features']):
            assert i not in birth;birth[i]=stage;selected[i]=y[i]
            pool.features[i]=row;pool.saved_labels[i]=y[i];pool.born[i]=stage
        pool.events.append(event)
        assert event['total_retained_rows']==len(birth)
    assert len(events)==5 and len(birth)==46 and all(list(selected.values()).count(c)==2 for c in range(23))
    summary=load('BUFFER_SUMMARY.json');assert summary['retained_components']==46 and summary['feature_body_bytes']==12058624
    for r in journal:
        stage=r['stage'];ids=r['ids'];split=r['split'];assert ids
        if split=='TRAIN':assert all(y[i] in STAGES[stage] for i in ids)
        elif split=='BUFFER':
            assert r['purpose']=='old_training' and r['arm']!='CURRENT_TARS_RAPO'
            assert all(i in birth and birth[i]<stage and r['epoch']>=events[birth[i]]['epoch'] for i in ids)
        else:
            assert split=='VAL' and r['purpose']=='isolated_audit' and r['all_stage_models_sealed']
            assert all(vy[i] in sum(STAGES[:stage+1],[]) for i in ids)
            assert r['epoch']>=load(f'STAGE_{stage}_ALL_MODELS_SEALED.json')['epoch']
    for stage,event in enumerate(events):
        actual=[r['ids'] for r in journal if r['stage']==stage and r['purpose']=='buffer_birth']
        assert actual==[event['ids']]
    first=[i for i,c in enumerate(y) if c in STAGES[0]];g=torch.Generator().manual_seed(74005);batches=[]
    for _ in range(2):
        ids=torch.tensor(first)[torch.randperm(len(first),generator=g)]
        batches.extend(ids[lo:lo+16].tolist() for lo in range(0,len(first),16))
    assert [r['ids'] for r in journal if r['stage']==0 and r['purpose']=='current_training']==batches
    assert sum(len(b) for b in batches)==4868
    for stage in range(1,5):
        current=[i for i,c in enumerate(y) if c in STAGES[stage]];g=torch.Generator().manual_seed(12474005+stage)
        expected=torch.tensor(current)[torch.randint(len(current),(128,8),generator=g)];old=pool.old_tape(stage)
        tape=pt(f'QUERY_TAPE_{stage}.private.pt');assert torch.equal(tape['current'],expected) and torch.equal(tape['old'],old)
        for arm in ARMS:
            current_events=[r for r in journal if r['stage']==stage and r['arm']==arm and r['purpose']=='current_training']
            assert [r['step'] for r in current_events]==list(range(128))
            assert [r['ids'] for r in current_events]==(expected if arm==ARMS[0] else expected[:,:4]).tolist()
            old_events=[r for r in journal if r['stage']==stage and r['arm']==arm and r['purpose']=='old_training']
            if arm==ARMS[0]:assert not old_events
            else:assert [r['step'] for r in old_events]==list(range(128)) and [r['ids'] for r in old_events]==old.tolist()
            calibration=[i for r in journal if r['stage']==stage and r['arm']==arm and r['purpose']=='current_calibration' for i in r['ids']]
            assert calibration==current
            table=load(f'CALIBRATION_{stage}_{arm}.json');w,z,median=weights_for_current(table['counts'],table['correct'])
            assert table['classes']==STAGES[stage] and table['counts']==[counts[c] for c in STAGES[stage]]
            assert all(0<=good<=n for good,n in zip(table['correct'],table['counts']))
            assert max(abs(a-b) for a,b in zip(w.tolist(),table['mean_normalized_TARS_weights']))<1e-12
            assert abs(z-table['Z'])<1e-12 and abs(median-table['raw_median'])<1e-12
            used=[1.]*len(w) if arm=='OLD_TASK_RAPO' else w.tolist()
            assert max(abs(a-b) for a,b in zip(used,table['used_weights']))<1e-5
            state=pt(f'STATE_{stage}_{arm}.private.pt');assert state['updates']==128*stage
            steps=[int(s['step']) for s in state['optimizer']['state'].values() if 'step' in s];assert steps and set(steps)=={128*stage}
            bind=next(r for r in bindings if r['stage']==stage and r['arm']==arm)
            assert bind['optimizer_steps_before']==128*(stage-1) and bind['previous_weights_restored'] and bind['anchor_from_own_previous_weights']
            assert not bind['old_TRAIN_access_allowed'] and bind['bounded_BUFFER_allowed']==(arm!=ARMS[0])
            seq=[r for r in curves if r['phase']=='POST' and r['stage']==stage and r['arm']==arm]
            assert len(seq)==128 and [r['update'] for r in seq]==list(range(128))
            sigma=bind['sigma_before'];assert sigma==(1. if stage==1 else pt(f'STATE_{stage-1}_{arm}.private.pt')['sigma'])
            for r in seq:
                assert r['nonzero_update']==(r['parameter_L2']>0)
                if arm=='OLD_TASK_CE':continue
                assert r['unupdated_behavior_logp_difference']==0 and r['sigma_before']==sigma
                sigma=.999*sigma+.001*r['total_reward_batch_std'];assert abs(r['sigma_after']-sigma)<1e-12
                actual=used=queries=old_queries=0
                for key,v in r['class_groups'].items():
                    c=v['class'];is_old=v['role']=='old';assert key==f"{c}:{v['role']}"
                    assert c in (sum(STAGES[:stage],[]) if is_old else STAGES[stage])
                    assert v['all_wrong']+v['all_correct']+v['mixed']==v['queries']
                    assert v['all_correct']*8+v['mixed']<=v['correct_responses']<=v['all_correct']*8+7*v['mixed']
                    assert v['used_correct_responses']==(0 if arm=='OLD_PROXY' and is_old else v['correct_responses'])
                    queries+=v['queries'];actual+=v['correct_responses'];used+=v['used_correct_responses'];old_queries+=v['queries'] if is_old else 0
                assert queries==8 and old_queries==(0 if arm==ARMS[0] else 4)
                assert r['current_queries']==8-old_queries and r['old_queries']==old_queries
                assert abs(actual/64-r['observed_task_reward_mean'])<1e-7 and abs(used/64-r['task_reward_mean'])<1e-7
                assert abs(r['total_reward_mean']-(r['task_reward_mean']+.5*r['retention_reward_mean']))<2e-7
                assert 0<r['retention_reward_mean']<=1 and r['drift_mean']>=0
            assert sigma==state['sigma']
    classes=[];old_new=[]
    for result in evaluations:
        stage=result['stage'];arm=result['arm'];saved=pt(f"VAL_{stage}_{'COMMON' if stage==0 else arm}.private.pt")
        seen=sum(STAGES[:stage+1],[]);assert saved['ids']==[i for i,c in enumerate(vy) if c in seen]
        assert len(saved['ids'])==len(saved['predictions'])
        good=[0]*23;den=[0]*23
        for i,p in zip(saved['ids'],saved['predictions']):den[vy[i]]+=1;good[vy[i]]+=int(p==vy[i])
        for key,value in score(good,den,seen,counts).items():assert result[key]==value
        oldclasses=sum(STAGES[:stage],[])
        old_new.append({'stage':stage,'arm':arm,'old_macro':sum(good[c]/den[c] for c in oldclasses)/len(oldclasses) if oldclasses else None,'new_macro':sum(good[c]/den[c] for c in STAGES[stage])/len(STAGES[stage])})
        for c in seen:classes.append({'stage':stage,'arm':arm,'class':c,'name':train[y.index(c)]['official_class_name'],'TRAIN_count':counts[c],'correct':good[c],'denominator':den[c],'recall_percent':100*good[c]/den[c],'role':'new' if c in STAGES[stage] else 'old','shared_first_task_readout':stage==0})
    assert len(classes)==360
    final=load('FINAL_RESULTS.json');assert final==summarize(evaluations,counts)
    cost=load('COSTS.json');native=load('NATIVE_ADMISSION.json');c=cost['counts']
    expected={'engineering_optimizer_attempts':5,'engineering_teacherforced_queries':16,'SFT_updates':306,'SFT_queries':4868,'post_optimizer_attempts':2560,'post_completed_updates':2560,'SFT_optimizer_attempts':306,'post_CE_queries':4096,'post_RL_responses':131072,'calibration_responses':21935,'engineering_responses':80,'TRAIN_encoder_rows':4,'VAL_encoder_rows':0,'VAL_responses':29139,'generation_responses':182226,'generation_attempts':182226,'buffer_feature_copy_rows':46,'post_current_queries':12288,'post_old_queries':8192}
    assert c==expected,(c,expected)
    assert cost['new_model_or_full_cache_copy_bytes']==0 and cost['bounded_buffer_feature_body_bytes']==12058624
    assert native['passed'] and native['generic_and_reference_restored'] and native['simulated_old_role_flags_use_only_legal_first_task_images']
    assert native['single_hidden_parity']<=.05 and native['same_shape_last_head_parity']<=.05
    names={int(r['original_label']):r['official_class_name'].replace('-',' ') for r in train}
    prompts=load('PROMPT_BINDINGS.json');assert set(prompts)==set(map(str,range(5)))
    profile=native['conservative_longest_prompt_profile'];assert native['native_processor_token_binding_exact'] and profile['classes']==STAGES[0]
    assert profile['question'].startswith(stage_prompt(0,names)) and profile['question'][len(stage_prompt(0,names)):].replace(' Category','')==''
    for stage in range(5):
        r=prompts[str(stage)];assert r['classes']==sum(STAGES[:stage+1],[]) and r['question']==stage_prompt(stage,names)
        assert r['visual_tokens']==64 and r['input_tokens']<=profile['input_tokens'] and r['training_generation_reference_and_audit_share_binding']
    groups=[];learning=[];common=pt('COMMON_PREFIX_SFT.private.pt')
    for arm in ARMS:
        seq=[r for r in curves if r['phase']=='POST' and r['arm']==arm];state=pt(f'STATE_4_{arm}.private.pt')
        same=all(torch.equal(x,a) and torch.equal(y,b) for (x,y),(a,b) in zip(state['weights'],common))
        learning.append({'arm':arm,'attempts':len(seq),'nonzero_updates':sum(r['nonzero_update'] for r in seq),'final_weights_equal_common_SFT':same,'final_sigma':state['sigma']})
        if arm=='OLD_TASK_CE':continue
        for stage in range(1,5):
            for role,cs in [('current',STAGES[stage]),('old',sum(STAGES[:stage],[]) if arm!=ARMS[0] else [])]:
                for cls in cs:
                    totals={k:sum(r['class_groups'].get(f'{cls}:{role}',{}).get(k,0) for r in seq if r['stage']==stage) for k in ['queries','all_wrong','all_correct','mixed','correct_responses','used_correct_responses']}
                    groups.append({'stage':stage,'arm':arm,'class':cls,'role':role,**totals})
    assert len(groups)==219
    def csvout(name,rows):
        with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csvout('CLASS_RESULTS.csv',classes);csvout('GROUP_DIAGNOSTICS.csv',groups)
    write(root/'OLD_NEW_SUMMARY.json',old_new);write(root/'LEARNING_INTEGRITY.json',{'scope':'saved_only_no_new_model_calls','arms':learning})
    write(root/'FINAL_VERIFICATION.json',{'passed':True,'scope':'CPU_saved_only_no_new_model_calls','curve_rows':2866,'class_rows':360,'arm_stage_rows':25,'actual_unique_readouts':21,'group_role_class_rows':219,'bounded_buffer_components':46,'all_birth_seal_access_tape_role_mask_weight_state_prompt_cost_metrics_and_nine_gates_verified':True,'calibration_correct_counts_not_independently_rescorable':True,'gates':final['gates'],'all_gates_pass':final['all_gates_pass']})
    ledger={'successful_run':cost,'controller_exit':load('CONTROL_EXIT.json'),'CPU_software_preflight':load('CPU_SOFTWARE_RUN_COSTS.json'),'original_clock_not_reset':True,'historical_cache_creation_not_free_or_repeated':True,'CPU_integrity_analyses':[json.loads(p.read_text()) for p in sorted((root/'INTEGRITY_PULSES').glob('*.json'))]}
    for name in ['ATTEMPT_COSTS.private.json','POSTPROCESS_COSTS.json']:
        if (root/name).exists():ledger[name]=load(name)
    write(root/'ALL_ATTEMPT_COSTS.json',ledger)
    report=['# R126 有界真实旧任务奖励资格实验','','单顺序开发VAL、非独立患者；共同首任务SFT贡献不记RL。','','|arm|post平均macro %|post平均tail %|平均forgetting pp|','|---|---:|---:|---:|']
    for arm in ARMS:report.append(f"|{arm}|{100*final['post_stage_means'][arm]['macro']:.4f}|{100*final['post_stage_means'][arm]['tail']:.4f}|{100*final['mean_classwise_forgetting'][arm]:.4f}|")
    report+=['','九门：'+json.dumps(final['gates'])+'；通过数'+str(sum(final['gates'].values()))+'/9。','首任务head变化pp：'+str(100*final['initial_task_head_change'])+'；最差oldtail对PROXY变化pp：'+str(100*final['worst_old_tail_over_OLD_PROXY']),
        '', '主与OLD_PROXY同4current+4old图像槽、生成上限与更新次数。差别是旧正确奖励是否进入组优势及持久sigma；自身policy分化后实际响应与token成本可不同，未声称输出完全相同。OLD_PROXY可获得相同历史图像/标签，其准确率仅供隔离诊断，不进入权重/优势/sigma。',
        'CURRENT用8current，混合臂只有4current；该比较同时改变当前暴露与旧访问，不能单独确定旧正确奖励因果。普通RaPO控制无TARS；CE同4+4查询/更新，答案监督更丰富且无G8/KL、计算更少，不据结果证明RL必要。',
        '每类出生固定2个TRAIN组件、46个上限，冻结视觉特征复制本体12058624B，容器/索引另计；旧图只从所选buffer读，全部TRAIN cache存在不允许任意旧访问。无未来类给模型。',
        '正确响应/mixed组/实际非零参数步分别报告，组份额非梯度/泛化效果；错误准确率组可因retention差异有PG。',
        '校准只保存aggregate正确计数：访问、类分母、权重数学可核，无法事后独立重评分正确计数；不追加生成补造。25表行包含5共享首任务，实际21个读出序列。',
        '这是已知经验回放与RaPO/TARS的机制资格诊断，不是CLEAR旧轨迹V-trace复现、顶会创新或独立临床验证。保留R125全部原失败门，失败不扫prompt/seed/参数/SFT/保护阈值救分。',
        '工程native仅以合法首任务图像模拟old角色测试优化路径，更新全部丢弃；实际旧buffer访问由CPU fixtures和科学访问日志验证。所有成本与失败共享首次准备原钟。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
