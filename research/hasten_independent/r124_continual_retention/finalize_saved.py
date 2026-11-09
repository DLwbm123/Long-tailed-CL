"""CPU-only count, access, state, metric and gate audit after the real stream ends."""
import collections
import csv
import json
from pathlib import Path

from .core import ARMS,STAGES,score,summarize,weights_for_current,write


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
    classes=[]
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
        for c in seen:classes.append({'stage':stage,'arm':arm,'class':c,'name':train[y.index(c)]['official_class_name'],
            'TRAIN_count':counts[c],'correct':good[c],'denominator':den[c],'recall_percent':100*good[c]/den[c],
            'shared_first_task_readout':stage==0})
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
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
    write(root/'FINAL_VERIFICATION.json',{'passed':True,'scope':'CPU_saved_only_no_new_model_calls','arm_stage_rows':25,
        'actual_unique_readouts':21,'class_rows':360,'curve_rows':2866,'current_only_TRAIN_and_seen_only_VAL_verified':True,
        'zero_old_TRAIN_replay_verified':True,'same_current_query_tape_verified':True,'own_optimizer_and_sigma_continuity_verified':True,
        'initial_generic_no_old_task_SFT_verified':True,'all_counts_verified':True,'all_nine_gates_verified':True,
        'gates':final['gates'],'all_gates_pass':final['all_gates_pass']})
    write(root/'ALL_ATTEMPT_COSTS.json',{'successful_run':cost,'failed_attempts':0,'original_clock_not_reset':True,
        'historical_cache_creation_costs_not_repeated_or_free':True,'partial_failures_must_be_added_if_any':True})
    report=['# R124 replay-free生成式连续RL机制实验','', '**单顺序开发VAL、非独立患者；首任务共同SFT贡献不记RL。**', '',
        '|arm|post平均macro %|post平均tail %|平均forgetting pp|','|---|---:|---:|---:|']
    for arm in ARMS:report.append(f"|{arm}|{100*final['post_stage_means'][arm]['macro']:.4f}|{100*final['post_stage_means'][arm]['tail']:.4f}|{100*final['mean_classwise_forgetting'][arm]:.4f}|")
    report+=['','原九门：'+json.dumps(final['gates']),'全部通过：'+str(final['all_gates_pass']),
        '','CE控制为匹配TARS权重、同query和更新次数，但无G8 rollout/KL，计算更少。',
        '所有原正负结果与旧门保留，不把新连续条件改写为旧实验成功。',
        '25行含5行共享首任务读出；实际21个读出序列，不是25独立重复。原生cache后端存在不等于可读取旧/未来TRAIN；accessor日志与seal先后已核。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
