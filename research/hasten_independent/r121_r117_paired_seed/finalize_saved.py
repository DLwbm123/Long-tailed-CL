"""Saved-only verification of one predeclared new-seed paired mechanism replication."""
import csv
import json
import math
from collections import Counter
from pathlib import Path


def run(cfg):
    import torch
    torch.set_num_threads(1)
    root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    final,curves,cost=load('FINAL_RESULTS.json'),load('TRAINING_CURVES.json'),load('COSTS.json')
    arms=['SEED_TARS_FULL_IS','SEED_GRPO_FULL_IS','SEED_CE_FULL_IS']
    assert load('GENERIC_INITIALIZATION.json')['all_task_adapter_B_zero']
    assert not load('GENERIC_INITIALIZATION.json')['old_task_checkpoints_loaded']
    assert load('NATIVE_ADMISSION.json')['generic_parameters_and_reference_restored']
    assert load('NATIVE_ADMISSION.json')['reused_frozen_visual_feature_max_difference']<=.05
    assert load('NATIVE_ADMISSION.json')['native_manual_max_logit_difference']<=.05
    initialization=load('GENERIC_INITIALIZATION.json')
    assert initialization['seed']==74006 and initialization['TRAIN_shuffle_seed']==74006
    assert initialization['sampler_seed']==11774006 and initialization['response_seed_base']==11481000
    assert initialization['first_adapter_seed_replay_verified'] and initialization['all_adapter_A_changed_from_constructor74005']
    reuse=load('FEATURE_REUSE.json')
    assert reuse=={'TRAIN_rows':6821,'VAL_rows':1698,'TRAIN_bytes':1788084224,'VAL_bytes':445120512,'new_feature_disk_bytes':0,'mode':'read_only_no_copy'}
    assert not (root/'TRAIN_FEATURES.private.bin').exists() and not (root/'VAL_FEATURES.private.bin').exists()
    expected={'engineering_optimizer_updates':4,'SFT_updates':854,'posttraining_optimizer_updates':768,
              'generation_responses':46415,'TRAIN_encoder_rows':4,'VAL_encoder_rows':0}
    assert cost['counts']==expected and cost['new_feature_disk_bytes']==0
    sft_rows=[r for r in curves if r['phase']=='SFT']
    assert len(sft_rows)==854 and [r['update'] for r in sft_rows]==list(range(854))
    assert sum(r['queries'] for r in sft_rows)==13642
    assert all(abs(r['lr']-2e-4*.5*(1+math.cos(math.pi*r['update']/854)))<1e-15 for r in sft_rows)
    assert load('SEED_SFT_SEALED.json')['updates']==854
    assert load('ALL_BRANCHES_SEALED.json')['arms']==['SEED_SFT']+arms
    assert load('ALL_BRANCHES_SEALED.json')['epoch']<=load('VAL_READ_STARTED.json')['epoch']
    info,allocation=load('TARS_WEIGHTS.json'),load('ALLOCATION.json');counts=info['counts']
    assert sum(counts)==6821 and len(counts)==23
    error=1-torch.tensor(info['SFT_TRAIN_correct'],dtype=torch.float64)/torch.tensor(counts,dtype=torch.float64)
    t=(torch.tensor(counts,dtype=torch.float64).mean()/torch.tensor(counts,dtype=torch.float64)*error)**.75
    w=(t/t.quantile(.5).clamp_min(1e-8)).clamp_min(1)
    assert torch.allclose(w,torch.tensor(info['weights'],dtype=torch.float64),atol=1e-12,rtol=1e-12)
    p=torch.tensor(counts,dtype=torch.float64)/6821;Z=float((p*w).sum());q=p*w/Z;rho=p/q
    for key,value in [('natural_class_probability',p),('query_class_probability',q),('importance_p_over_q',rho)]:
        assert torch.allclose(torch.tensor(allocation[key],dtype=torch.float64),value,atol=1e-12,rtol=1e-12)
    assert abs(allocation['Z']-Z)<1e-12 and torch.allclose(rho*w,torch.full_like(w,Z))
    tape=torch.load(root/'QUERY_TAPE.private.pt',map_location='cpu',weights_only=True)
    train_predictions=torch.load(root/'SEED_SFT_TRAIN_PREDICTIONS.private.pt',map_location='cpu',weights_only=True)
    assert len(train_predictions['labels'])==6821 and tape.shape==(256,8)
    labels=train_predictions['labels'];queries=allocation['realized_queries_by_class']
    assert torch.bincount(labels[tape].flatten(),minlength=23).tolist()==queries and sum(queries)==2048
    for c in range(23):
        assert int(((train_predictions['predictions']==c)&(labels==c)).sum())==info['SFT_TRAIN_correct'][c]
        assert len(set(tape[labels[tape]==c].tolist()))==allocation['unique_TRAIN_components_by_class'][c]
    learning={'SEED_SFT':{'optimizer_attempts':854,'actual_nonzero_parameter_updates':sum(r['parameter_delta_L2']>0 for r in sft_rows)}}
    for arm in arms:
        rr=[r for r in curves if r['arm']==arm]
        assert len(rr)==256 and [r['update'] for r in rr]==list(range(256))
        assert load(arm+'_SEALED.json')['common_initialization_reference_verified']
        learning[arm]={'optimizer_attempts':256,'actual_nonzero_parameter_updates':sum(r['parameter_delta_L2']>0 for r in rr),
                       'cumulative_parameter_path_L2':sum(r['parameter_delta_L2'] for r in rr)}
        if arm==arms[-1]:continue
        assert all(r['unchanged_policy_logp_max_difference']==0 for r in rr)
        classes=[]
        for c in range(23):
            wrong,right,mixed=[sum(r[k][c] for r in rr) for k in ['allwrong_groups_by_class','allright_groups_by_class','mixed_groups_by_class']]
            success=sum(r['correct_by_class'][c] for r in rr)
            assert wrong+right+mixed==queries[c] and sum(r['responses_by_class'][c] for r in rr)==8*queries[c]
            assert 8*right+mixed<=success<=8*right+7*mixed
            classes.append({'class':c,'queries':queries[c],'successful_responses':int(success),'allwrong_groups':wrong,'allright_groups':right,'mixed_groups':mixed})
        learning[arm]['classes']=classes
    assert len(curves)==1622
    rows=[];errors=[]
    assert [r['name'] for r in final['results']]==['SEED_SFT']+arms
    for result in final['results']:
        assert len(result['correct'])==len(result['denominator'])==23 and sum(result['denominator'])==1698
        recall=[]
        for c,(good,den) in enumerate(zip(result['correct'],result['denominator'])):
            assert isinstance(good,int) and 0<=good<=den and den>0
            recall.append(good/den);rows.append({'arm':result['name'],'class':c,'name':info['class_names'][str(c)],'TRAIN_count':counts[c],
                                                  'correct':good,'denominator':den,'recall_percent':100*good/den})
        for metric,ids in {'macro':list(range(23)),'head':[c for c in range(23) if counts[c]>=250],
                           'tail':[c for c in range(23) if counts[c]<250],'tiny':[c for c in range(23) if counts[c]<25]}.items():
            errors.append(abs(result[metric]-sum(recall[c] for c in ids)/len(ids)))
        errors.append(abs(result['micro']-sum(result['correct'])/1698))
    assert max(errors)<1e-7
    sft,main,ordinary,ce=final['results'];original=final['historical_original_SFT']
    gates={'macro_over_seedSFT_ge_1pp':main['macro']-sft['macro']>=.01,'tail_over_seedSFT_ge_2pp':main['tail']-sft['tail']>=.02,
           'head_over_seedSFT_ge_minus1pp':main['head']-sft['head']>=-.01,'macro_over_GRPO_positive':main['macro']>ordinary['macro'],
           'tail_over_GRPO_positive':main['tail']>ordinary['tail'],'macro_over_CE_positive':main['macro']>ce['macro'],
           'tail_over_CE_positive':main['tail']>ce['tail']}
    directional={base+'_'+metric+'_positive':main[metric]>row[metric] for base,row in [('SFT',sft),('GRPO',ordinary),('CE',ce)] for metric in ['macro','tail']}
    assert gates==final['gates'] and all(gates.values())==final['all_gates_pass']
    assert directional==final['directional_positive_signal'] and all(directional.values())==final['all_six_positive_directions_repeat']
    assert final['seed']==74006 and not final['independent_confirmation']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    predictions={r['name']:torch.load(root/(r['name']+'_PREDICTIONS.private.pt'),map_location='cpu',weights_only=True) for r in final['results']}
    predictions['ORIGINAL_224_SFT']=torch.load(Path(cfg['base_source'])/'SFT_PREDICTIONS.private.pt',map_location='cpu',weights_only=True)
    transitions={}
    for baseline in ['ORIGINAL_224_SFT','SEED_SFT']:
        reference=predictions[baseline];transitions[baseline]={}
        for result in final['results']:
            endpoint=predictions[result['name']];assert torch.equal(reference['labels'],endpoint['labels'])
            classes=[]
            for c in range(23):
                mask=endpoint['labels']==c;old,new=reference['predictions'][mask],endpoint['predictions'][mask]
                rescued=int(((old!=c)&(new==c)).sum());harmed=int(((old==c)&(new!=c)).sum())
                assert int((new==c).sum())==result['correct'][c] and int((new==c).sum()-(old==c).sum())==rescued-harmed
                pairs=Counter(zip(old.tolist(),new.tolist()))
                classes.append({'class':c,'denominator':int(mask.sum()),'rescued':rescued,'harmed':harmed,
                                'prediction_transitions':[{'before':a,'after':b,'count':n} for (a,b),n in sorted(pairs.items())]})
            transitions[baseline][result['name']]=classes
    (root/'PREDICTION_TRANSITIONS.json').write_text(json.dumps({'scope':'saved_only_aggregate','transitions':transitions},indent=2)+'\n')
    verified={'passed':True,'scope':'saved_only','new_model_queries_updates':0,'rows':len(rows),'max_metric_error':max(errors),
              'generic_newseed_and_readonly_feature_reuse_verified':True,'directional_positive_signal':directional,'all_six_positive_directions_repeat':all(directional.values()),'fresh_TARS_table_verified':True,'gates':gates,'all_gates_pass':all(gates.values()),'learning':learning}
    (root/'FINAL_VERIFICATION.json').write_text(json.dumps(verified,indent=2)+'\n')
    report=['# R121 单新seed配对机制复验','','**冻结联合门：'+('通过' if all(gates.values()) else '未通过')+'。**','',
            '|方法|macro %|head %|tail %|tiny %|micro %|','|---|---:|---:|---:|---:|---:|']
    for row in [original]+final['results']:report.append('|'+row['name']+'|'+'|'.join(f'{100*row[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report+=['','新seed74006共同SFT和主RL/普通RL/CE配对；七项诊断联合门与六项方向复现分开报告。不是重跑旧九门，也不撤销R117原头类失败。','',
             '原224特征TRAIN6821/VAL1698只读复用，无新增副本。恰一新seed、不选seed，仍复用开发VAL；非独立患者、完整持续学习或成功方法确认。CE缺同KL/计算的限制保持。','',
             '本次七诊断门：'+json.dumps(gates),'六项方向复现：'+json.dumps(directional)]
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    return verified
