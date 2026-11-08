"""Saved-only fixed-reference budget and final count verification."""
import csv
import json
from pathlib import Path


def run(cfg):
    root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    final=load('FINAL_RESULTS.json');curves=load('TRAINING_CURVES.json')
    weights=json.loads((Path(cfg['base_source'])/'TARS_WEIGHTS.json').read_text());n=weights['counts']
    arms=['KL_PROJECTED_TARS','KL_SHRINK_TARS','PLAIN_TARS'];alphas=[1.,.5,.25,.125,.0625,0.]
    summary={};errors=[];table=[]
    for arm in arms:
        rr=[r for r in curves if r['arm']==arm]
        assert len(rr)==256 and [r['update'] for r in rr]==list(range(256))
        assert all(r['unchanged_policy_logp_max_difference']==0 for r in rr)
        assert all(sum(r['responses_by_class'])==64 for r in rr)
        seal=load(arm+'_SEALED.json');assert seal['updates']==256
        if arm!='PLAIN_TARS':
            for r in rr:
                choice=next(i for i,v in enumerate(r['max_KL_trials']) if v<=.001)
                assert r['accepted_alpha']==alphas[choice]
                assert r['max_reference_KL_before']<=.001
                assert r['max_reference_KL_after']==r['max_KL_trials'][choice]<=.001
                assert max(r['class_reference_KL_after'])==r['max_reference_KL_after']
                assert r['fixed_reference_budget_verified']
            assert max(seal['final_reference_KL_by_class'])<=.001
        summary[arm]={'attempted_AdamW_updates':256,'nonzero_parameter_updates':sum(r['nonzero_parameter_update'] for r in rr),
                      'zero_alpha_steps':sum(r.get('accepted_alpha',1)==0 for r in rr),
                      'groups':sum(r['groups'] for r in rr),'constant_groups':sum(r['constant_groups'] for r in rr),
                      'responses':sum(sum(r['responses_by_class']) for r in rr),
                      'mean_reward':sum(r['reward_mean'] for r in rr)/256,
                      'final_reference_KL_by_class':seal['final_reference_KL_by_class'],
                      'reference_constraint_enforced':arm!='PLAIN_TARS'}
    for r in final['results']:
        assert len(r['correct'])==len(r['denominator'])==23 and sum(r['denominator'])==1698
        recalls=[]
        for c,(good,den) in enumerate(zip(r['correct'],r['denominator'])):
            assert isinstance(good,int) and 0<=good<=den and den>0
            recalls.append(good/den)
            table.append({'arm':r['name'],'class':c,'name':weights['class_names'][str(c)],'TRAIN_count':n[c],
                          'correct':good,'denominator':den,'recall_percent':100*good/den})
        for metric,classes in {'macro':list(range(23)),'head':[c for c in range(23) if n[c]>=250],
                               'tail':[c for c in range(23) if n[c]<250],'tiny':[c for c in range(23) if n[c]<25]}.items():
            errors.append(abs(r[metric]-sum(recalls[c] for c in classes)/len(classes)))
        errors.append(abs(r['micro']-sum(r['correct'])/1698))
    assert max(errors)<1e-7
    main,shrink,plain=final['results'];sft=final['historical_SFT']
    gates={'macro_over_SFT_ge_1pp':main['macro']-sft['macro']>=.01,'tail_over_SFT_ge_2pp':main['tail']-sft['tail']>=.02,
           'head_over_SFT_ge_minus1pp':main['head']-sft['head']>=-.01,
           'macro_over_KL_SHRINK_nonnegative':main['macro']>=shrink['macro'],'tail_over_KL_SHRINK_positive':main['tail']>shrink['tail'],
           'head_over_PLAIN_positive':main['head']>plain['head'],'tail_over_PLAIN_nonnegative':main['tail']>=plain['tail']}
    assert gates==final['gates'] and all(gates.values())==final['all_gates_pass']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    report=['# R116 固定参考分布约束 TARS-RL：最终诊断','','**冻结成功门：'+('通过' if all(gates.values()) else '未通过')+'。**','',
            '|方法|macro %|head %|tail %|tiny %|micro %|','|---|---:|---:|---:|---:|---:|']
    for r in [sft]+final['results']:
        report.append('|'+r['name']+'|'+'|'.join(f'{100*r[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report+=['','逐项门：'+'；'.join(k+'='+str(v) for k,v in gates.items())+'。','',
             '两个约束臂每步在相同48个固定TRAIN头类锚点上检查12类相对固定SFT的条件分布KL，预算.001nats；普通TARS控制同为256更新，但没有这部分保护反馈，前向成本更少。','',
             '完整词表KL只在参考贪心响应前缀处计算，不是全序列KL或临床/人群准确率保护证明；锚点见于原SFT训练，不是独立holdout。','',
             '实际非零参数更新：'+'；'.join(a+'='+str(summary[a]['nonzero_parameter_updates'])+'/256' for a in arms)+'。零接受步、奖励常量组与最终逐类参考KL见FINAL_VERIFICATION.json。','',
             '复用SFT/缓存，不重复计作新训练；本轮256更新是运行前确定的预算，R114/R115的512更新仅作历史背景，不声称同预算比较。','',
             '单开发seed74005、静态HK23、反复开发VAL1698、TEST0；全23类整数计数公开，不挑最好epoch/seed，不宣称完整持续学习或最终创新。图像/身份/特征/权重/逐例预测/私有路径不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    result={'passed':True,'rows':len(table),'max_metric_error':max(errors),'learning':summary,'gates':gates,
            'all_gates_pass':all(gates.values()),'scope':'saved_only','new_model_queries_updates':0}
    (root/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2));return result
