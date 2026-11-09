"""CPU-only recomputation from fixed source boxes and sealed joint readouts."""
import csv
import json
from pathlib import Path

from .core import METHODS, summarize, write


def run(cfg):
    root=Path(cfg['output']);load=lambda name:json.loads((root/name).read_text())
    assert load('CONTROL_EXIT.json')['exit_code']==0
    rows=load('MEASUREMENTS.private.json');old=json.loads((Path(cfg['previous_source'])/'MEASUREMENTS.private.json').read_text())
    assert len(rows)==len(old)==175 and len({r['global_TRAIN_row'] for r in rows})==175
    assert sum(r['split']=='A' for r in rows)==88 and sum(r['split']=='B' for r in rows)==87
    for rank,(r,o) in enumerate(zip(rows,old)):
        assert (r['selection_rank'],r['global_TRAIN_row'],r['class'],r['split'])==(rank,o['global_TRAIN_row'],o['class'],o['split'])
        assert r['boxes']=={m:o['results'][m]['proposal']['box'] for m in ['GLOBAL_REPEAT','CENTER_DETAIL']}
        assert set(r['results'])==set(METHODS)
        assert len({r['results'][m]['input_tokens'] for m in METHODS[1:]})==1
        for m in METHODS:assert r['results'][m]['vision_encoder_rows']==(1 if m=='NATIVE_ROOT' else 2)
        for before,after in [('GLOBAL_REPEAT','GLOBAL_BOX'),('CENTER_DETAIL','DETAIL_BOX')]:
            if not o['results'][before]['proposal']['valid']:assert r['results'][after]['prediction']==r['results']['DUPLICATE']['prediction']
    defs=load('CLASS_DEFINITIONS.json');final=load('FINAL_RESULTS.json');recomputed=summarize(rows,defs['counts'])
    assert recomputed==final
    groups={'macro':list(range(23)),'head':[c for c,n in enumerate(defs['counts']) if n>=250],
            'tail':[c for c,n in enumerate(defs['counts']) if n<250],
            'tiny':[c for c,n in enumerate(defs['counts']) if n<25],
            'non_tiny':[c for c,n in enumerate(defs['counts']) if n>=25]}
    for split,results in [('OVERALL',final['results'])]+[(s,final['partitions'][s]) for s in ['A','B']]:
        subset=rows if split=='OVERALL' else [r for r in rows if r['split']==split]
        den=[sum(r['class']==c for r in subset) for c in range(23)]
        for result in results:
            good=[sum(r['class']==c and r['results'][result['name']]['prediction']==c for r in subset) for c in range(23)]
            assert good==result['correct'] and den==result['denominator']
            for key,ids in groups.items():assert abs(result[key]-sum(good[c]/den[c] for c in ids)/len(ids))<1e-12
            assert abs(result['micro']-sum(good)/sum(den))<1e-12
    root_result,duplicate,fixed,global_box,detail=final['results']
    checks=[detail[k]-ref[k]>=t for ref in [duplicate,root_result] for k,t in [('macro',.01),('tail',.02),('head',-.01)]]
    checks += [detail[k]>ref[k] for ref in [fixed,global_box] for k in ['macro','tail']]
    checks += [final['partitions'][s][4]['non_tiny']>final['partitions'][s][1]['non_tiny'] for s in ['A','B']]
    checks += [len({(r['vision_encoder_rows'],r['input_tokens']) for r in final['results'][1:]})==1]
    assert checks==list(final['gates'].values())
    assert load('MEASUREMENTS_SEALED.json')['instrument_and_reference_immutable']
    native=load('NATIVE_ADMISSION.json');assert native['passed'] and native['same_shape_last_head_parity']<=.05 and native['single_image_hidden_parity']<=.05
    cost=load('COSTS.json');assert cost['counts']=={'generation_attempts':885,'generation_responses':885,'TRAIN_encoder_rows':1597,'VAL_encoder_rows':0,'optimizer_updates':0,'new_proposals':0}
    records=[]
    for split,results in [('OVERALL',final['results'])]+[(s,final['partitions'][s]) for s in ['A','B']]:
        for r in results:
            for c in range(23):records.append({'split':split,'readout':r['name'],'class':c,'name':defs['class_names'][str(c)],
                'correct':r['correct'][c],'denominator':r['denominator'][c],'recall_percent':100*r['correct'][c]/r['denominator'][c]})
    assert len(records)==345
    with (root/'CLASS_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    proof={'passed':True,'scope':'saved_only_no_regeneration','new_model_queries_updates':0,'components':175,'A':88,'B':87,
        'class_rows':345,'all_original_boxes_and_fallbacks_verified':True,'joint_native_HF_not_manual_cache':True,
        'metrics_and_all_13_gates_verified':True,'gates':final['gates'],'all_gates_pass':final['all_gates_pass'],
        'counts_verified':True,'classifier_TRAIN_in_sample':True}
    write(root/'FINAL_VERIFICATION.json',proof)
    write(root/'ALL_ATTEMPT_COSTS.json',{'successful_run':cost,'new_proposals_optimizer_VAL_TEST':0,
        'failure_accounting':'Normal path only; any failed attempt must be added before public completion.','original_clock_not_reset':True,
        'historical_costs':'R1221062responses1424encodingsincludingfailednative;R121SFTandpriorcostsremainhistorical'})
    report=['# R123 固定框联合观察诊断','', '**TRAIN-only冻结仪器诊断，非RL训练、非独立患者或continual结果。**','',
        '|读出|macro %|head %|tail %|tiny %|micro %|','|---|---:|---:|---:|---:|---:|']
    for r in final['results']:report.append('|'+r['name']+'|'+'|'.join(f'{100*r[k]:.4f}' for k in ['macro','head','tail','tiny','micro'])+'|')
    report+=['','十三门：'+json.dumps(final['gates']),'全部通过：'+str(final['all_gates_pass']),
        '','固定全部175个原R122组件和框（含回退），无新提议/ROI/seed/prompt搜索；原R122十门失败不变。',
        '四联合接口实际inputtokens/视觉编码预算相同，native单图更便宜；实际生成tokens和time在FINAL_RESULTS逐臂报告。',
        '权重和原始参考均不可变；分类仪器已拟合全部TRAIN，A/B不是crossfit或患者独立。',
        '正结果只说明联合证据机会，不证明非myopic价值、RL必要性或创新；负结果关闭本冻结仪器联合救援，不继续接口调参。']
    (root/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
    return proof
