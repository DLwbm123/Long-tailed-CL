"""Independent CPU-only recomputation from sealed diagnostic records."""
import csv
import json
from pathlib import Path

from .core import METHODS,parse_box,write


def run(cfg):
    import torch
    torch.set_num_threads(1);root=Path(cfg['output']);load=lambda name:json.loads((root/name).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    final=load('FINAL_RESULTS.json');rows=load('MEASUREMENTS.private.json');selection=load('SELECTION.private.json')
    definitions=load('CLASS_DEFINITIONS.json');counts=definitions['counts'];train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()))
    labels=torch.tensor([int(r['original_label']) for r in train]);assert torch.bincount(labels,minlength=23).tolist()==counts
    generator=torch.Generator().manual_seed(12274006);expected=[]
    for c in range(23):
        pool=torch.where(labels==c)[0];ids=pool[torch.randperm(len(pool),generator=generator)[:min(8,len(pool))]]
        expected.extend({'row':int(row),'class':c,'split':'A' if k%2==0 else 'B'} for k,row in enumerate(ids))
    assert selection==expected and len(rows)==175 and len({r['global_TRAIN_row'] for r in rows})==175
    assert len({train[r['global_TRAIN_row']]['identity_component'] for r in rows})==175
    assert sum(r['split']=='A' for r in rows)==88 and sum(r['split']=='B' for r in rows)==87
    for rank,(item,row) in enumerate(zip(selection,rows)):
        assert (row['selection_rank'],row['global_TRAIN_row'],row['class'],row['split'])==(rank,item['row'],item['class'],item['split'])
        for method in METHODS[2:]:
            prop=row['results'][method]['proposal'];box,valid=parse_box(prop['text']);valid=valid and not prop['truncated']
            assert prop['valid']==valid and prop['box']==(box if valid else [0,0,1000,1000])
            assert prop['vision_encoder_rows']==2 and row['results'][method]['recognition']['vision_encoder_rows']==1
            if not valid:assert row['results'][method]['recognition']['prediction']==row['results']['ROOT']['recognition']['prediction']
    csv_rows=[];maximum_error=0.
    ids={'macro':list(range(23)),'head':[c for c,n in enumerate(counts) if n>=250],'tail':[c for c,n in enumerate(counts) if n<250],
         'tiny':[c for c,n in enumerate(counts) if n<25],'non_tiny':[c for c,n in enumerate(counts) if n>=25]}
    for split,results in [('OVERALL',final['results'])]+[(k,final['partitions'][k]) for k in ['A','B']]:
        chosen=rows if split=='OVERALL' else [r for r in rows if r['split']==split]
        assert [r['name'] for r in results]==METHODS
        den=[sum(r['class']==c for r in chosen) for c in range(23)];assert all(n>0 for n in den)
        for result in results:
            good=[sum(r['class']==c and r['results'][result['name']]['recognition']['prediction']==c for r in chosen) for c in range(23)]
            assert good==result['correct'] and den==result['denominator']
            metrics={key:sum(good[c]/den[c] for c in group)/len(group) for key,group in ids.items()};metrics['micro']=sum(good)/sum(den)
            maximum_error=max(maximum_error,max(abs(result[key]-value) for key,value in metrics.items()))
            for c in range(23):csv_rows.append({'split':split,'policy':result['name'],'class':c,'name':definitions['class_names'][str(c)],
                                               'original_TRAIN_count':counts[c],'correct':good[c],'denominator':den[c],'recall_percent':100*good[c]/den[c]})
    assert maximum_error<1e-12 and len(csv_rows)==276
    baseline,main=final['results'][2:];parts=final['partitions']
    gates={'macro_increment_ge_1pp':main['macro']-baseline['macro']>=.01,'tail_increment_ge_2pp':main['tail']-baseline['tail']>=.02,
           'head_increment_ge_minus1pp':main['head']-baseline['head']>=-.01,'non_tiny_increment_positive':main['non_tiny']>baseline['non_tiny'],
           'A_non_tiny_increment_positive':parts['A'][3]['non_tiny']>parts['A'][2]['non_tiny'],
           'B_non_tiny_increment_positive':parts['B'][3]['non_tiny']>parts['B'][2]['non_tiny'],
           'GLOBAL_valid_fraction_ge_90pct':baseline['valid_fraction']>=.9,'CENTER_valid_fraction_ge_90pct':main['valid_fraction']>=.9,
           'equal_visual_and_input_token_costs':main['vision_encoder_rows']==baseline['vision_encoder_rows'] and main['input_tokens']==baseline['input_tokens'],
           'CENTER_generated_tokens_not_larger':main['generated_tokens']<=baseline['generated_tokens']}
    assert gates==final['gates'] and all(gates.values())==final['all_gates_pass']
    cost=load('COSTS.json');assert cost['counts']=={'generation_responses':1062,'proposal_attempts':354,'proposal_responses':354,
            'recognition_attempts':708,'recognition_responses':708,'TRAIN_encoder_rows':1420,'VAL_encoder_rows':0,'optimizer_updates':0}
    native=load('NATIVE_ADMISSION.json');assert native['passed'] and native['recognition_native_manual_max_difference']<=.05
    assert load('MEASUREMENTS_SEALED.json')['instrument_and_generic_reference_immutable']
    assert load('INSTRUMENT_BINDING.json')['generic_proposer_B_zero']
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    proof={'passed':True,'scope':'saved_only_no_regeneration','new_model_queries_updates':0,'components':175,'A':88,'B':87,
           'class_rows':276,'max_metric_error':maximum_error,'selection_seed_and_components_verified':True,'fallback_consistency_verified':True,
           'gates':gates,'all_gates_pass':all(gates.values()),'counts_verified':True,'classifier_TRAIN_in_sample':True}
    write(root/'FINAL_VERIFICATION.json',proof)
    report=['# R122 冻结仪器的TRAIN条件信息测量','','**这是信息干预诊断，不是RL训练/continual/独立患者效果。**','',
            '|策略|macro %|head %|tail %|tiny %|micro %|','|---|---:|---:|---:|---:|---:|']
    for row in final['results']:report.append('|'+row['name']+'|'+'|'.join(f'{100*row[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report+=['','诊断十门：'+json.dumps(gates),'全部通过：'+str(all(gates.values())),
             '','GLOBAL_REPEAT与CENTER_DETAIL均为标签不可见的通用模型提议，再由同一已拟合TRAIN的静态SFT分类；所有参数冻结。',
             'ROOT/CENTER固定控制更便宜，只作描述；重复全图是信息消融，不是最强应用baseline。',
             '两个配对分支相同视觉调用数和生成上限不等于实际token/time相同；完整实际费用见FINAL_RESULTS/COSTS。',
             '175独立TRAIN组件分为88/87，但分类仪器已经拟合全部TRAIN，不宣称全链交叉拟合、患者泛化或RL必要性。',
             '即使诊断通过，也仅提示后续机制设计机会；失败不换prompt/ROI/seed救分，保留全部负门。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    return proof
