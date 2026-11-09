"""Independent saved count/allocation checks, without additional model queries."""
import csv
import json
from pathlib import Path


def run(cfg):
    root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    final=load('FINAL_RESULTS.json');curves=load('TRAINING_CURVES.json');allocation=load('ALLOCATION.json')
    names=['ALLOC_TARS_FULL_IS','ALLOC_TARS_PG_ONLY_IS','ALLOC_GRPO_FULL_IS','ALLOC_CE_FULL_IS']
    assert load('ALL_BRANCHES_SEALED.json')['arms']==names
    p,q,w,rho=[allocation[k] for k in ['natural_class_probability','query_class_probability','TARS_weights','importance_p_over_q']]
    Z=allocation['Z'];assert abs(sum(q)-1)<1e-12 and sum(allocation['realized_queries_by_class'])==2048
    assert max(abs(p[i]/q[i]-rho[i]) for i in range(23))<1e-12
    assert max(abs(rho[i]*w[i]-Z) for i in range(23))<1e-12
    info=json.loads((Path(cfg['base_source'])/'TARS_WEIGHTS.json').read_text());counts=info['counts']
    learning={};rows=[];errors=[]
    for arm in names:
        rr=[r for r in curves if r['arm']==arm]
        assert len(rr)==256 and [r['update'] for r in rr]==list(range(256))
        assert load(arm+'_SEALED.json')['updates']==256
        learning[arm]={'updates':256,'maximum_gradient_norm':max(r['gradient_norm'] for r in rr)}
        if arm!='ALLOC_CE_FULL_IS':
            assert all(r['unchanged_policy_logp_max_difference']==0 for r in rr)
            assert sum(r['groups'] for r in rr)==2048
            assert sum(sum(r['responses_by_class']) for r in rr)==16384
            for c in range(23):assert sum(r['responses_by_class'][c] for r in rr)==8*allocation['realized_queries_by_class'][c]
            learning[arm].update(constant_groups=sum(r['constant_groups'] for r in rr),groups=2048,
                                 responses=16384,mean_reward=sum(r['reward_mean'] for r in rr)/256,
                                 mean_KL=sum(r['KL'] for r in rr)/256,
                                 mean_weighted_KL=sum(r['importance_weighted_KL'] for r in rr)/256)
    for r in final['results']:
        assert len(r['correct'])==len(r['denominator'])==23 and sum(r['denominator'])==1698
        recall=[]
        for c,(good,den) in enumerate(zip(r['correct'],r['denominator'])):
            assert isinstance(good,int) and 0<=good<=den and den>0
            recall.append(good/den);rows.append({'arm':r['name'],'class':c,'name':info['class_names'][str(c)],
                'TRAIN_count':counts[c],'correct':good,'denominator':den,'recall_percent':100*good/den})
        for metric,classes in {'macro':list(range(23)),'head':[c for c in range(23) if counts[c]>=250],
                               'tail':[c for c in range(23) if counts[c]<250],'tiny':[c for c in range(23) if counts[c]<25]}.items():
            errors.append(abs(r[metric]-sum(recall[c] for c in classes)/len(classes)))
        errors.append(abs(r['micro']-sum(r['correct'])/1698))
    assert max(errors)<1e-7
    main,_,ordinary,ce=final['results'];sft=final['historical_SFT'];natural=final['reused_natural_TARS256']
    gates={'macro_over_SFT_ge_1pp':main['macro']-sft['macro']>=.01,'tail_over_SFT_ge_2pp':main['tail']-sft['tail']>=.02,
           'head_over_SFT_ge_minus1pp':main['head']-sft['head']>=-.01,
           'macro_over_ALLOC_GRPO_positive':main['macro']>ordinary['macro'],'tail_over_ALLOC_GRPO_positive':main['tail']>ordinary['tail'],
           'head_over_natural_TARS_nonnegative':main['head']>=natural['head'],'tail_over_natural_TARS_positive':main['tail']>natural['tail'],
           'macro_over_ALLOC_CE_nonnegative':main['macro']>=ce['macro'],'tail_over_ALLOC_CE_positive':main['tail']>ce['tail']}
    assert gates==final['gates'] and all(gates.values())==final['all_gates_pass']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=['# R117 TARS查询分配：最终诊断','','**冻结成功门：'+('通过' if all(gates.values()) else '未通过')+'。**','',
            '|方法|macro %|head %|tail %|tiny %|micro %|','|---|---:|---:|---:|---:|---:|']
    for r in [sft,natural]+final['results']:
        report.append('|'+r['name']+'|'+'|'.join(f'{100*r[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report+=['','逐项门：'+'；'.join(k+'='+str(v) for k,v in gates.items())+'。','',
             '按TARS质量q=p*w/Z分配查询，主臂对PG和KL同时做p/q修正；PG-only臂改变参考惩罚目标，普通RL移除TARS权重，CE控制查询数匹配但前向更少且无KL。','',
             '修正后每查询PG系数固定为Z，固定策略期望目标相同，不等于Adam/全局梯度裁剪后更新或最终准确率相同。数学等价不作新颖性或效果证明。','',
             '自然TARS256/SFT是已有固定锚点，非新独立重复。四个新臂共用同一查询tape；实际类暴露、常量组及全部类计数完整公开，无筛选/重采成功组。','',
             '单seed74005、静态HK23、反复开发VAL1698、TEST0，非独立患者/完整持续学习验证。保持旧失败与原始时钟，不按结果选最好epoch/seed或改门。图片/身份/特征/权重/逐例预测/私有路径不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    result={'passed':True,'rows':len(rows),'max_metric_error':max(errors),'allocation_verified':True,'learning':learning,
            'gates':gates,'all_gates_pass':all(gates.values()),'scope':'saved_only','new_model_queries_updates':0}
    (root/'FINAL_VERIFICATION.json').write_text(json.dumps(result,indent=2));return result
