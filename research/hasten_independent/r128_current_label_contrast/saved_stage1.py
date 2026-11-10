"""CPU-only interim audit of the first fully sealed new task; no generation."""
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from .core import ARMS,STAGES,write


def run(cfg):
    import torch
    torch.set_num_threads(1);r=Path(cfg['output']);load=lambda n:json.loads((r/n).read_text());pt=lambda n:torch.load(r/n,map_location='cpu',weights_only=True)
    sealed=load('STAGE_1_ALL_MODELS_SEALED.json');assert sealed['arms']==ARMS
    vals=[x for x in load('EVALUATIONS.json') if x['stage']==1];assert {x['arm'] for x in vals}==set(ARMS)
    train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()))
    y=[int(x['original_label']) for x in train];vy=[int(x['original_label']) for x in val];counts=[y.count(c) for c in range(23)]
    names={int(x['original_label']):x['official_class_name'].replace('-',' ') for x in train}
    norm=lambda s:re.sub(r'[\s_\-]+','',s.lower().strip());lookup={norm(v):c for c,v in names.items()}
    curves=[x for x in load('TRAINING_CURVES.json') if x['stage']==1];raw=[x for x in load('ROLLOUT_RECORDS.private.json') if x['stage']==1]
    assert len(curves)==len(raw)==512
    raw_by={(x['arm'],x['update']):x for x in raw};assert len(raw_by)==512
    journal=load('ACCESS.private.json');current=[i for i,c in enumerate(y) if c in STAGES[1]]
    gen=torch.Generator().manual_seed(12474006);expected=torch.tensor(current)[torch.randint(len(current),(128,8),generator=gen)]
    assert torch.equal(pt('QUERY_TAPE_1.private.pt'),expected)
    classes=[];arms=[]
    for arm in ARMS:
        result=next(x for x in vals if x['arm']==arm);saved=pt(f'VAL_1_{arm}.private.pt')
        ids=[i for i,c in enumerate(vy) if c<10];assert saved['ids']==ids and len(saved['predictions'])==len(ids)==964
        correct=[0]*23;den=[0]*23
        for i,p in zip(ids,saved['predictions']):den[vy[i]]+=1;correct[vy[i]]+=int(p==vy[i])
        assert result['correct']==correct and result['denominator']==den and result['evaluation_started_epoch']>=sealed['epoch']
        def mean(cs):
            cs=[c for c in cs if den[c]>0];return sum(correct[c]/den[c] for c in cs)/len(cs) if cs else None
        computed={'macro':mean(range(10)),'head':mean(c for c in range(10) if counts[c]>=250),'tail':mean(c for c in range(10) if counts[c]<250),'tiny':mean(c for c in range(10) if counts[c]<25),'micro':sum(correct)/sum(den)}
        for key,value in computed.items():assert result[key]==value
        seq=[x for x in curves if x['arm']==arm];assert [x['update'] for x in seq]==list(range(128))
        assert [x['ids'] for x in journal if x['stage']==1 and x['arm']==arm and x['purpose']=='current_training']==expected.tolist()
        for event in journal:
            if event['stage']==1 and event['arm']==arm:
                if event['split']=='TRAIN':assert all(y[i] in STAGES[1] for i in event['ids'])
                if event['split']=='VAL':assert event['purpose']=='isolated_audit' and event['all_stage_models_sealed'] and event['epoch']>=sealed['epoch']
        sigma=1.;success=eligible=wrong_eligible=nonzero=positive=negative=0;all_wrong=mixed=all_correct=0
        for k,curve in enumerate(seq):
            item=raw_by[arm,k];labels=[y[i] for i in expected[k].tolist()];assert item['labels']==labels
            parsed=[lookup.get(norm(text),-1) if done else -1 for text,done in zip(item['text'],item['done'])];assert len(parsed)==64 and parsed==item['parsed']
            negatives=[-1]*64;task=[int(parsed[i]==labels[i//8] and item['done'][i]) for i in range(64)]
            for q,gold in enumerate(labels):
                slots=range(q*8,q*8+8);wrong=[parsed[i] for i in slots if 0<=parsed[i]<10 and parsed[i]!=gold]
                votes=Counter(wrong);mode=min(votes,key=lambda c:(-votes[c],c)) if votes else -1
                for i in slots:
                    if 0<=parsed[i]<10 and parsed[i]!=gold:negatives[i]=parsed[i]
                    elif parsed[i]==gold and mode>=0:negatives[i]=mode
            assert negatives==item['negatives'];can=[c>=0 for c in negatives]
            assert sum(can)==curve['eligible_teacher_responses'] and abs(sum(task)/64-curve['task_reward_mean'])<1e-7
            for q in range(8):
                n=sum(task[q*8:q*8+8]);all_wrong+=n==0;all_correct+=n==8;mixed+=0<n<8
                wrong_eligible+=n==0 and any(can[q*8:q*8+8])
            assert curve['sigma_before']==sigma;sigma=.999*sigma+.001*curve['total_reward_batch_std'];assert abs(curve['sigma_after']-sigma)<1e-12
            assert curve['nonzero_update']==(curve['parameter_L2']>0) and curve['unupdated_behavior_logp_difference']==0 and curve['teacher_and_actor_anchor_restored']
            if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']:assert curve['teacher_last_sync_step']==(k//10)*10
            assert curve['positive_teacher_rows']==(64 if arm=='POS_KD_TARS_RAPO' else sum(can) if arm=='CONTRAST_TARS_RAPO' else 0)
            assert curve['negative_teacher_rows']==(sum(can) if arm=='CONTRAST_TARS_RAPO' else 0)
            success+=sum(task);eligible+=sum(can);nonzero+=curve['nonzero_update'];positive+=curve['positive_teacher_rows'];negative+=curve['negative_teacher_rows']
        state=pt(f'STATE_1_{arm}.private.pt');assert state['updates']==128 and state['sigma']==sigma
        steps=[int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v];assert steps and set(steps)=={128}
        if arm in ['POS_KD_TARS_RAPO','CONTRAST_TARS_RAPO']:assert state['teacher_sync_step']==120 and state['teacher'] is not None
        arms.append({'arm':arm,**computed,'old5_macro':mean(range(5)),'new5_macro':mean(range(5,10)),'old_first_task_head4':mean([1,2,3,4]),'TRAIN_correct_responses':success,'TRAIN_responses':8192,'all_wrong_groups':all_wrong,'all_correct_groups':all_correct,'mixed_groups':mixed,'eligible_teacher_responses':eligible,'all_wrong_groups_with_valid_teacher_pair':wrong_eligible,'actual_positive_teacher_TF_rows':positive,'actual_negative_teacher_TF_rows':negative,'nonzero_parameter_steps':nonzero,'optimizer_step':128,'final_sigma':sigma})
        for c in range(10):classes.append({'arm':arm,'class':c,'name':names[c],'correct':correct[c],'denominator':den[c],'TRAIN_count':counts[c],'role':'old' if c<5 else 'new'})
    summary={'scope':'interim_first_new_task_only_not_final_efficacy','stage':1,'CPU_saved_only_verification_passed':True,'new_model_calls':0,'new_optimizer_updates':0,'arm_rows':4,'class_rows':40,'new_stage_VAL_responses':3856,'all_arms_sealed_before_stage_VAL':True,'query_tape_reward_eligibility_negative_ties_teacher_K10_optimizer_sigma_and_integer_metrics_independently_verified':True,'arms':arms,'all_nine_final_gates_await_complete_four_post_stages':True}
    write(r/'INTERIM_STAGE_1.json',summary)
    with (r/'INTERIM_STAGE_1_CLASSES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
    table=['# R128首个新增任务：保存结果中间核验','','仅第一个posttask，非最终R128结论；不改原四post平均/最终head/遗忘九门，不因中间reward/VAL调整训练。','','|arm|seenmacro %|tail %|old5macro %|new5macro %|oldhead4 %|','|---|---:|---:|---:|---:|---:|']
    for x in arms:table.append(f"|{x['arm']}|{100*x['macro']:.4f}|{100*x['tail']:.4f}|{100*x['old5_macro']:.4f}|{100*x['new5_macro']:.4f}|{100*x['old_first_task_head4']:.4f}|")
    main=next(x for x in arms if x['arm']=='CONTRAST_TARS_RAPO');table+=['','主相对各控制的macro/tail/old5/new5增量pp：']
    for control in [x for x in arms if x['arm']!=main['arm']]:table.append(control['arm']+'：'+', '.join(f"{m} {100*(main[m]-control[m]):+.4f}" for m in ['macro','tail','old5_macro','new5_macro']))
    table+=['','全部4臂各128真实参数步/8192 TRAIN回答；同current query tape、连续Adam step128、sigma逐式、POS_KD和主臂teacher最后同步120、负例资格/众数平票、所有964组件/类整数分母与seal先后均CPU独立核对。',
        'teacher正负上下文含完整当前类别答案，收益是辅助监督构建，不当纯RL必要性或临床证据。CE/KD同标签且实际计算不同，不能只比较正确TRAIN回答数/teacher行数宣称效果。',
        'old5 macro不等firsttask head4；当前全10类tail/tiny小分母完整披露，原firsttaskhead门要到最终task判定。组份额或teacher资格只说明可用机会，不是梯度/泛化效果。',
        '仅用远端保存的VAL预测、normalized rollout/done/parsed/negative与各stage权重optimizer状态核验，无新生成/模型前向/optimizer；私有逐例输出与索引不fetch或公开，全部CPU费用另记。',
        '主对BASE的macro/tail均未增益；对CE_AUX和POS_KD的macro/tail更高伴随new5macro更低，不能称全面获益。所有臂tiny为0；新增两个tiny类VAL各1，完整类表保留分母。',
        '本核验不独立重评分未保存的校准逐例correct、teacher原始logp或每类梯度；实际L2只证明参数确实更新，组份额/资格不证明teacher贡献。',
        '后续真实任务与全部四臂继续原冻结协议，原23h首次准备钟、累计10GB与独立hardguard不变。']
    (r/'INTERIM_STAGE_1_ZH.md').write_text('\n'.join(table)+'\n')
