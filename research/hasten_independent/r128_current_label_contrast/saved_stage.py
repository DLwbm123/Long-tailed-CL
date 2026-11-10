"""CPU-only interim audit of one fully sealed post-task stage; no generation."""
import csv
import json
import re
import os
from collections import Counter
from pathlib import Path
from .core import ARMS,STAGES,write


def run(cfg):
    import torch
    torch.set_num_threads(1);r=Path(cfg['output']);load=lambda n:json.loads((r/n).read_text());pt=lambda n:torch.load(r/n,map_location='cpu',weights_only=True)
    stage=int(os.environ['Q_AUDIT_STAGE']);assert 1<=stage<=4
    seen=sum(STAGES[:stage+1],[]);old=sum(STAGES[:stage],[]);new=STAGES[stage];sync=((stage*128-1)//10)*10
    sealed=load(f'STAGE_{stage}_ALL_MODELS_SEALED.json');assert sealed['arms']==ARMS
    vals=[x for x in load('EVALUATIONS.json') if x['stage']==stage];assert {x['arm'] for x in vals}==set(ARMS)
    train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
    y=[int(x['original_label']) for x in train];vy=[int(x['original_label']) for x in val];counts=[y.count(c) for c in range(23)]
    names={int(x['original_label']):x['official_class_name'].replace('-',' ') for x in train}
    norm=lambda s:re.sub(r'[\s_\-]+','',s.lower().strip());lookup={norm(v):c for c,v in names.items()}
    curves=[x for x in load('TRAINING_CURVES.json') if x['stage']==stage];raw=[x for x in load('ROLLOUT_RECORDS.private.json') if x['stage']==stage]
    assert len(curves)==len(raw)==512
    raw_by={(x['arm'],x['update']):x for x in raw};assert len(raw_by)==512
    journal=load('ACCESS.private.json');current=[i for i,c in enumerate(y) if c in STAGES[stage]]
    gen=torch.Generator().manual_seed(12474005+stage);expected=torch.tensor(current)[torch.randint(len(current),(128,8),generator=gen)]
    assert torch.equal(pt(f'QUERY_TAPE_{stage}.private.pt'),expected)
    classes=[];arms=[]
    for arm in ARMS:
        result=next(x for x in vals if x['arm']==arm);saved=pt(f'VAL_{stage}_{arm}.private.pt')
        ids=[i for i,c in enumerate(vy) if c in seen];assert saved['ids']==ids and len(saved['predictions'])==len(ids)
        correct=[0]*23;den=[0]*23
        for i,p in zip(ids,saved['predictions']):den[vy[i]]+=1;correct[vy[i]]+=int(p==vy[i])
        assert result['correct']==correct and result['denominator']==den and result['evaluation_started_epoch']>=sealed['epoch']
        def mean(cs):
            cs=[c for c in cs if den[c]>0];return sum(correct[c]/den[c] for c in cs)/len(cs) if cs else None
        computed={'macro':mean(seen),'head':mean(c for c in seen if counts[c]>=250),'tail':mean(c for c in seen if counts[c]<250),'tiny':mean(c for c in seen if counts[c]<25),'micro':sum(correct)/sum(den)}
        for key,value in computed.items():assert result[key]==value
        seq=[x for x in curves if x['arm']==arm];assert [x['update'] for x in seq]==list(range(128))
        assert [x['ids'] for x in journal if x['stage']==stage and x['arm']==arm and x['purpose']=='current_training']==expected.tolist()
        for event in journal:
            if event['stage']==stage and event['arm']==arm:
                if event['split']=='TRAIN':assert all(y[i] in STAGES[stage] for i in event['ids'])
                if event['split']=='VAL':assert event['purpose']=='isolated_audit' and event['all_stage_models_sealed'] and event['epoch']>=sealed['epoch']
        previous=pt(f'STATE_{stage-1}_{arm}.private.pt') if stage>1 else None
        binding=next(x for x in load('STATE_BINDINGS.json') if x['stage']==stage and x['arm']==arm)
        assert binding['previous_weights_restored'] and binding['anchor_from_own_previous_weights'] and not binding['old_TRAIN_access_allowed']
        assert binding['optimizer_steps_before']==(stage-1)*128 and binding['sigma_before']==(previous['sigma'] if previous else 1.)
        sigma=binding['sigma_before'];success=eligible=wrong_eligible=nonzero=positive=negative=0;all_wrong=mixed=all_correct=0
        for k,curve in enumerate(seq):
            item=raw_by[arm,k];labels=[y[i] for i in expected[k].tolist()];assert item['labels']==labels
            parsed=[lookup.get(norm(text),-1) if done else -1 for text,done in zip(item['text'],item['done'])];assert len(parsed)==64 and parsed==item['parsed']
            negatives=[-1]*64;task=[int(parsed[i]==labels[i//8] and item['done'][i]) for i in range(64)]
            for q,gold in enumerate(labels):
                slots=range(q*8,q*8+8);wrong=[parsed[i] for i in slots if parsed[i] in seen and parsed[i]!=gold]
                votes=Counter(wrong);mode=min(votes,key=lambda c:(-votes[c],c)) if votes else -1
                for i in slots:
                    if parsed[i] in seen and parsed[i]!=gold:negatives[i]=parsed[i]
                    elif parsed[i]==gold and mode>=0:negatives[i]=mode
            assert negatives==item['negatives'];can=[c>=0 for c in negatives]
            assert sum(can)==curve['eligible_teacher_responses'] and abs(sum(task)/64-curve['task_reward_mean'])<1e-7
            for q in range(8):
                n=sum(task[q*8:q*8+8]);all_wrong+=n==0;all_correct+=n==8;mixed+=0<n<8
                wrong_eligible+=n==0 and any(can[q*8:q*8+8])
            assert curve['sigma_before']==sigma;sigma=.999*sigma+.001*curve['total_reward_batch_std'];assert abs(curve['sigma_after']-sigma)<1e-12
            assert curve['nonzero_update']==(curve['parameter_L2']>0) and curve['unupdated_behavior_logp_difference']==0 and curve['teacher_and_actor_anchor_restored']
            if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']:assert curve['teacher_last_sync_step']==(((stage-1)*128+k)//10)*10
            assert curve['positive_teacher_rows']==(64 if arm=='POS_KD_TARS_RAPO' else sum(can) if arm=='CONTRAST_TARS_RAPO' else 0)
            assert curve['negative_teacher_rows']==(sum(can) if arm=='CONTRAST_TARS_RAPO' else 0)
            success+=sum(task);eligible+=sum(can);nonzero+=curve['nonzero_update'];positive+=curve['positive_teacher_rows'];negative+=curve['negative_teacher_rows']
        state=pt(f'STATE_{stage}_{arm}.private.pt');assert state['updates']==stage*128 and state['sigma']==sigma
        steps=[int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v];assert steps and set(steps)=={stage*128}
        if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']:assert state['teacher_sync_step']==sync and state['teacher'] is not None
        arms.append({'arm':arm,**computed,'old_macro':mean(old),'new_macro':mean(new),'old_first_task_head4':mean([1,2,3,4]),'TRAIN_correct_responses':success,'TRAIN_responses':8192,'all_wrong_groups':all_wrong,'all_correct_groups':all_correct,'mixed_groups':mixed,'eligible_teacher_responses':eligible,'all_wrong_groups_with_valid_teacher_pair':wrong_eligible,'actual_positive_teacher_TF_rows':positive,'actual_negative_teacher_TF_rows':negative,'nonzero_parameter_steps':nonzero,'optimizer_step':stage*128,'final_sigma':sigma})
        for c in seen:classes.append({'arm':arm,'class':c,'name':names[c],'correct':correct[c],'denominator':den[c],'TRAIN_count':counts[c],'role':'old' if c in old else 'new'})
    summary={'scope':'interim_one_sealed_stage_only_not_final_efficacy','stage':stage,'CPU_saved_only_verification_passed':True,'new_model_calls':0,'new_optimizer_updates':0,'arm_rows':4,'class_rows':len(classes),'new_stage_VAL_responses':len(ids)*4,'all_arms_sealed_before_stage_VAL':True,'query_tape_reward_eligibility_negative_ties_teacher_K10_optimizer_sigma_and_integer_metrics_independently_verified':True,'arms':arms,'all_nine_final_gates_await_complete_four_post_stages':True}
    write(r/f'INTERIM_STAGE_{stage}.json',summary)
    with (r/f'INTERIM_STAGE_{stage}_CLASSES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
    table=[f'# R128 stage{stage}（第{stage+1}任务）保存结果中间核验','','仅本stage，不判定完整四post平均/最终head/遗忘九门；不据中间reward/VAL改参。','','|arm|seenmacro %|tail %|oldmacro %|newmacro %|firsttask head4 %|tiny %|','|---|---:|---:|---:|---:|---:|---:|']
    for x in arms:table.append(f"|{x['arm']}|{100*x['macro']:.4f}|{100*x['tail']:.4f}|{100*x['old_macro']:.4f}|{100*x['new_macro']:.4f}|{100*x['old_first_task_head4']:.4f}|{100*x['tiny']:.4f}|")
    main=next(x for x in arms if x['arm']=='CONTRAST_TARS_RAPO');table+=['','主相对各控制增量pp：']
    for control in [x for x in arms if x['arm']!=main['arm']]:table.append(control['arm']+'：'+', '.join(f"{m} {100*(main[m]-control[m]):+.4f}" for m in ['macro','tail','old_macro','new_macro']))
    table+=['',f'4臂各128attempt/8192回答；连续Adam step{stage*128}、sigma接自己上一stage、K10最后同步{sync}、当前TRAIN同tape/seen-only VAL在全4臂seal后、整数类分母与EOS负例/众数平票均独立核对。',
      '完整类别标签是辅助监督；CE_AUX/POS_KD和主得到同标签，实际计算不同。组份额、正确TRAIN回答数、非零L2或teacher资格不等同梯度贡献/VAL效果。',
      'oldmacro不等firsttask head4；完整类表保留tiny与所有类分母。单order开发VAL组件不是独立患者证据。',
      '0新模型调用/生成/optimizer；私有预测/normalized rollout/ACCESS/索引/STATE仅远端CPU读取，不fetch/public。校准逐例correct/teacher原始logp/每类梯度未存，不补造。',
      '后续所有臂继续原冻结协议，23h原钟/累计10GB/hardguard不变；未运行full finalizer。']
    (r/f'INTERIM_STAGE_{stage}_ZH.md').write_text('\n'.join(table)+'\n')
