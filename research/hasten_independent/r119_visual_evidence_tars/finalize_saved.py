"""CPU-only accounting, paired predictions and frozen-gate checks."""
import csv
import json
from collections import Counter
from pathlib import Path


def run(cfg):
    import torch
    torch.set_num_threads(1)
    root=Path(cfg['output']);load=lambda n:json.loads((root/n).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    final,curves,cost=load('FINAL_RESULTS.json'),load('TRAINING_CURVES.json'),load('COSTS.json')
    arms=['VISUAL_TARS_FULL_IS','VISUAL_GRPO_FULL_IS','VISUAL_CE_FULL_IS']
    assert load('ALL_BRANCHES_SEALED.json')['arms']==['VISUAL_SFT']+arms
    assert load('ALL_BRANCHES_SEALED.json')['VAL_encoder_rows']==0
    assert load('VAL_CACHE_STARTED.json')['epoch']>=load('ALL_BRANCHES_SEALED.json')['epoch']
    native=load('NATIVE_ADMISSION.json');assert native['passed'] and native['native_manual_max_logit_difference']<=.05
    assert native['actual_grid']==[1,32,32] and native['visual_tokens']==256
    expected={'engineering_optimizer_updates':3,'SFT_updates':0,'posttraining_optimizer_updates':768,
              'generation_responses':39594,'TRAIN_encoder_rows':1560,'VAL_encoder_rows':1698}
    assert cost['counts']==expected
    compact=torch.load(root/'COMPACT_INDEX.private.pt',map_location='cpu',weights_only=True)
    original_tape=torch.load(Path(cfg['allocation_source'])/'QUERY_TAPE.private.pt',map_location='cpu',weights_only=True)
    assert len(compact['global_TRAIN_rows'])==1556
    assert torch.equal(compact['global_TRAIN_rows'][compact['compact_query_tape']],original_tape)
    for split,n in [('TRAIN',1556),('VAL',1698)]:
        assert (root/(split+'_FEATURES.private.bin')).stat().st_size==n*256*2048*2
        assert load(split+'_CACHE_COMPLETE.json')['rows']==n
    allocation=load('ALLOCATION.json');queries=allocation['realized_queries_by_class']
    assert sum(queries)==2048
    assert max(abs(a*b-allocation['Z']) for a,b in zip(allocation['importance_p_over_q'],allocation['TARS_weights']))<1e-12
    learning={}
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
            wrong,right,mixed=[sum(r[k][c] for r in rr) for k in
                               ['allwrong_groups_by_class','allright_groups_by_class','mixed_groups_by_class']]
            success=sum(r['correct_by_class'][c] for r in rr)
            assert wrong+right+mixed==queries[c]
            assert sum(r['responses_by_class'][c] for r in rr)==8*queries[c]
            assert 8*right+mixed<=success<=8*right+7*mixed
            classes.append({'class':c,'queries':queries[c],'successful_responses':int(success),
                            'allwrong_groups':wrong,'allright_groups':right,'mixed_groups':mixed})
        learning[arm]['classes']=classes
    assert len(curves)==768
    info=json.loads((Path(cfg['base_source'])/'TARS_WEIGHTS.json').read_text());counts=info['counts'];rows=[];errors=[]
    assert [r['name'] for r in final['results']]==['VISUAL_SFT']+arms
    for result in final['results']:
        assert len(result['correct'])==len(result['denominator'])==23 and sum(result['denominator'])==1698
        recall=[]
        for c,(good,den) in enumerate(zip(result['correct'],result['denominator'])):
            assert isinstance(good,int) and 0<=good<=den and den>0
            recall.append(good/den);rows.append({'arm':result['name'],'class':c,'name':info['class_names'][str(c)],
                'TRAIN_count':counts[c],'correct':good,'denominator':den,'recall_percent':100*good/den})
        for metric,ids in {'macro':list(range(23)),'head':[c for c in range(23) if counts[c]>=250],
                           'tail':[c for c in range(23) if counts[c]<250],'tiny':[c for c in range(23) if counts[c]<25]}.items():
            errors.append(abs(result[metric]-sum(recall[c] for c in ids)/len(ids)))
        errors.append(abs(result['micro']-sum(result['correct'])/1698))
    assert max(errors)<1e-7
    visual,main,ordinary,ce=final['results'];original=final['historical_original_SFT']
    gates={'macro_over_visualSFT_ge_1pp':main['macro']-visual['macro']>=.01,
           'tail_over_visualSFT_ge_2pp':main['tail']-visual['tail']>=.02,
           'head_over_visualSFT_ge_minus1pp':main['head']-visual['head']>=-.01,
           'macro_over_GRPO_positive':main['macro']>ordinary['macro'],'tail_over_GRPO_positive':main['tail']>ordinary['tail'],
           'macro_over_CE_positive':main['macro']>ce['macro'],'tail_over_CE_positive':main['tail']>ce['tail'],
           'end_to_end_macro_over_original_ge_1pp':main['macro']-original['macro']>=.01,
           'end_to_end_tail_over_original_ge_2pp':main['tail']-original['tail']>=.02,
           'end_to_end_head_over_original_ge_minus1pp':main['head']-original['head']>=-.01}
    assert gates==final['gates'] and all(gates.values())==final['all_gates_pass']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    predictions={r['name']:torch.load(root/(r['name']+'_PREDICTIONS.private.pt'),map_location='cpu',weights_only=True) for r in final['results']}
    predictions['ORIGINAL_224_SFT']=torch.load(Path(cfg['base_source'])/'SFT_PREDICTIONS.private.pt',map_location='cpu',weights_only=True)
    transitions={}
    for baseline in ['ORIGINAL_224_SFT','VISUAL_SFT']:
        reference=predictions[baseline];transitions[baseline]={}
        for result in final['results']:
            endpoint=predictions[result['name']];assert torch.equal(reference['labels'],endpoint['labels'])
            classes=[]
            for c in range(23):
                mask=endpoint['labels']==c;old,new=reference['predictions'][mask],endpoint['predictions'][mask]
                rescued=int(((old!=c)&(new==c)).sum());harmed=int(((old==c)&(new!=c)).sum())
                assert int((new==c).sum())==result['correct'][c]
                assert int((new==c).sum()-(old==c).sum())==rescued-harmed
                pairs=Counter(zip(old.tolist(),new.tolist()))
                classes.append({'class':c,'denominator':int(mask.sum()),'rescued':rescued,'harmed':harmed,
                                'prediction_transitions':[{'before':a,'after':b,'count':n} for (a,b),n in sorted(pairs.items())]})
            transitions[baseline][result['name']]=classes
    (root/'PREDICTION_TRANSITIONS.json').write_text(json.dumps({'scope':'saved_only_aggregate','transitions':transitions},indent=2)+'\n')
    verified={'passed':True,'scope':'saved_only','new_model_queries_updates':0,'rows':len(rows),
              'max_metric_error':max(errors),'compact_mapping_verified':True,'VAL_encoding_after_seal_verified':True,
              'gates':gates,'all_gates_pass':all(gates.values()),'learning':learning}
    (root/'FINAL_VERIFICATION.json').write_text(json.dumps(verified,indent=2)+'\n')
    report=['# R119 固定448视觉证据与RL增量','','**冻结联合门：'+('通过' if all(gates.values()) else '未通过')+'。**','',
            '|方法|macro %|head %|tail %|tiny %|micro %|','|---|---:|---:|---:|---:|---:|']
    for row in [original]+final['results']:
        report.append('|'+row['name']+'|'+'|'.join(f'{100*row[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report+=['','分辨率本身的读出变化、相同448下RL增量及相对原224起点的端到端变化分别报告。无新增SFT，CE计算更少且无KL。','',
             '单开发seed/静态HK23/重复开发VAL，非独立患者、完整持续学习或成熟方法创新证明。未按VAL扫分辨率或选checkpoint。','',
             '全部原门：'+json.dumps(gates)]
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    return verified
