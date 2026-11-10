"""Independent saved prediction scoring and exact menu/parameter decomposition."""
import csv
import json
from pathlib import Path
from .core import ARMS,READOUTS,STAGES,metrics,decompose,old_access,question,write


def run(cfg):
    import torch
    torch.set_num_threads(1);r=Path(cfg['output']);source=Path(cfg['saved_source']);load=lambda n:json.loads((r/n).read_text());pt=lambda p:torch.load(p,map_location='cpu',weights_only=True)
    assert load('CONTROL_EXIT.json')['exit_code']==0 and load('MEASUREMENTS_SEALED.json')['all_parameters_immutable']
    val=list(csv.DictReader((Path(cfg['manifest'])/'val.csv').open()));train=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()))
    labels=[int(x['original_label']) for x in val];ids=[i for i,y in enumerate(labels) if y in STAGES[0]];assert len(ids)==624
    ys=[labels[i] for i in ids];counts=[sum(int(x['original_label'])==c for x in train) for c in range(23)];names={int(x['original_label']):x['official_class_name'].replace('-',' ') for x in train}
    rows=load('READOUT_RESULTS.json');assert [x['readout'] for x in rows]==[x[0] for x in READOUTS]
    native=load('NATIVE_ADMISSION.json');assert native['passed'] and native['same_hidden_parity']<=.05 and native['same_shape_head_parity']<=.05 and native['first16_saved_prediction_parity']
    baseline=pt(source/'VAL_0_COMMON.private.pt');assert baseline['ids']==ids
    scored={};preds={};classes=[]
    for row,(name,arm,menu) in zip(rows,READOUTS):
        saved=pt(r/f'PRED_{name}.private.pt');assert saved['ids']==ids
        pred=saved['predictions'];computed=metrics(pred,ys,counts);assert all(row[k]==v for k,v in computed.items())
        assert row['weights_equal_loaded_before_and_after'] and row['no_trainable_parameters'] and row['model']==arm and row['stage_menu']==menu
        if name=='COMMON_INITIAL_MENU':assert pred==baseline['predictions']
        scored[name]=computed;preds[name]=pred
        for c in STAGES[0]:classes.append({'readout':name,'model':arm,'stage_menu':menu,'class':c,'name':names[c],'TRAIN_count':counts[c],'correct':computed['correct'][c],'denominator':computed['denominator'][c],'recall_percent':100*computed['recall'][c]})
    binding=load('PROMPT_BINDINGS.json');original=json.loads((source/'PROMPT_BINDINGS.json').read_text())
    for b in binding:
        assert b['stage_menu'] in [0,4] and b['question']==question(b['stage_menu'],names)==original[str(b['stage_menu'])]['question']
        assert b['input_tokens']==original[str(b['stage_menu'])]['input_tokens'] and b['visual_tokens']==64
    journal=load('ACCESS.private.json')
    for e in journal:old_access(e['ids'],labels);assert e['stage_menu'] in [0,4] and e['model'] in ['COMMON']+ARMS
    assert sum(len(e['ids'])*e['uses'] for e in journal)==4372
    assert sum(len(e['ids'])*e['uses'] for e in journal if e['kind']=='native_forward_parity')==4
    generated=[e for e in journal if e['kind']!='native_forward_parity'];assert all(e['uses']==1 for e in generated)
    assert sum(len(e['ids']) for e in generated)==4368
    for arm,menu in [('COMMON',0),('COMMON',4)]+[(a,4) for a in ARMS]:
        chunks=[e['ids'] for e in generated if e['model']==arm and e['stage_menu']==menu]
        assert chunks==[ids[lo:lo+16] for lo in range(0,624,16)]
    cost=load('COSTS.json');c=cost['counts']
    for k,value in {'generation_responses':4368,'generation_attempts':4368,'cached_feature_forward_rows':4372,'native_extra_cached_forward_rows':4,'TRAIN_cached_feature_rows':0,'TRAIN_encoder_rows':0,'VAL_encoder_rows':0,'optimizer_updates':0,'SFT_updates':0}.items():assert c[k]==value
    assert cost['new_weight_or_feature_copies']==0
    expected_input=sum(624*original[str(menu)]['input_tokens'] for _,_,menu in READOUTS);assert c['prefill_input_tokens']==expected_input
    assert c['generated_active_tokens']<=c['decoded_token_cells']<=4368*24 and c['decoder_forward_token_rows']==c['decoded_token_cells']-4368
    a=scored['COMMON_INITIAL_MENU'];b=scored['COMMON_FINAL_MENU'];attribution=[];class_attr=[];paired=[]
    for arm in ARMS:
        name=arm+'_FINAL_MENU';now=scored[name];old=pt(source/f'VAL_4_{arm}.private.pt');lookup=dict(zip(old['ids'],old['predictions']));assert all(i in lookup for i in ids)
        history=[lookup[i] for i in ids];h=metrics(history,ys,counts)
        record={'arm':arm,'saved_vs_matched_prediction_changes':sum(x!=y for x,y in zip(history,preds[name]))}
        for metric in ['first_task_head','old5_macro','micro']:
            record[metric]={'initial':a[metric],'fixed_common_final_menu':b[metric],'matched_final_policy':now[metric],'historical_final_policy':h[metric],**decompose(a[metric],b[metric],now[metric],h[metric])}
        attribution.append(record)
        for cls in STAGES[0]:class_attr.append({'arm':arm,'class':cls,'name':names[cls],**decompose(a['recall'][cls],b['recall'][cls],now['recall'][cls],h['recall'][cls])})
    main=next(x for x in attribution if x['arm']=='TARS_RAPO')
    for control in [a for a in ARMS if a!='TARS_RAPO']:
        rec={'control':control}
        other=next(x for x in attribution if x['arm']==control)
        for metric in ['first_task_head','old5_macro','micro']:
            diff=main[metric]['matched_final_policy']-other[metric]['matched_final_policy']
            pdiff=main[metric]['parameter_at_final_menu']-other[metric]['parameter_at_final_menu'];assert abs(diff-pdiff)<1e-12
            rec[metric]=diff
        paired.append(rec)
    assert len(classes)==35 and len(class_attr)==25
    def csvout(name,values):
        with (r/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    csvout('CLASS_RESULTS.csv',classes);csvout('CLASS_ATTRIBUTION.csv',class_attr)
    write(r/'ATTRIBUTION.json',{'scope':'retrospective_final_first_task_only','arms':attribution,'main_minus_controls_menu_effect_cancels':paired,'original_R125_gates_unchanged':True,'units':'recall_fraction; multiply100 for percentage points'})
    write(r/'FINAL_VERIFICATION.json',{'passed':True,'saved_only_no_new_model_calls':True,'readouts':7,'class_rows':35,'class_attribution_rows':25,'full624_baseline_prediction_parity':True,'all_prompt_access_same_batch_count_metrics_and_additive_identities_verified':True,'all_loaded_weights_immutable_receipts_verified':True,'counts':c})
    write(r/'ALL_ATTEMPT_COSTS.json',{'successful_run':cost,'controller_exit':load('CONTROL_EXIT.json'),'CPU_software':load('CPU_SOFTWARE_RUN_COSTS.json'),'historical_R125_and_cache_creation_costs_not_repeated_or_free':True,'first_preparation_and_4h_clock_not_reset':True,'scientific_optimizer_updates':0,'input_token_accounting':{'generation_prefill_input_tokens':expected_input,'native_extra_cached_forward_input_tokens':4*original['0']['input_tokens'],'all_prefill_and_native_input_tokens':expected_input+4*original['0']['input_tokens'],'decoder_forward_token_rows':c['decoder_forward_token_rows'],'all_transformer_input_token_rows':expected_input+4*original['0']['input_tokens']+c['decoder_forward_token_rows']}})
    report=['# R127：类别菜单与最终首task head参数变化归因','','回顾性冻结仪器诊断，不是RL训练/临床验证；原R125/R126结果和门不变。',
        '', '|arm|原始head %|共同权重最终菜单 %|最终匹配head %|菜单项 pp|参数项 pp|原读出残差 pp|', '|---|---:|---:|---:|---:|---:|---:|']
    for record in attribution:
        x=record['first_task_head'];report.append(f"|{record['arm']}|{100*x['initial']:.4f}|{100*x['fixed_common_final_menu']:.4f}|{100*x['matched_final_policy']:.4f}|{100*x['menu']:.4f}|{100*x['parameter_at_final_menu']:.4f}|{100*x['saved_vs_matched_readout_residual']:.4f}|")
    x=main['first_task_head'];report+=['','菜单项超过原1pp head容差：'+str(abs(x['menu'])>.01)+'。只描述归因，不能撤销原失败门。',
        '主模型匹配总变化pp='+str(100*x['matched_total'])+'；历史总变化pp='+str(100*x['historical_total'])+'。两者差由原全seen批次与本固定旧类批次的读出残差单列，不未经实验归因某种数值机制。']
    if x['matched_total']<0:report+=['有符号menu/total比='+str(x['menu']/x['matched_total'])+'（不裁剪；异号或大于1可表示两项抵消）。']
    report+=['','共同SFT的同624图像预测在原5类菜单下逐例完全等于R125保存结果；最终23类菜单也是R125原模板。全部learned模型只用最终23类菜单，没有给它们oracle旧5类限制菜单，不搜索新prompt或挑菜单部署。',
        '这是预先固定的条件分解：菜单项在共同初始权重下定义，参数项在最终菜单下定义，不是唯一分解；未测learned模型原5类菜单，不能估计或排除菜单与参数的交互。',
        '相对控制的共同基线菜单项在代数上抵消：主与控制的匹配读出差精确等于参数项差，因此共同菜单效应不能解释原同菜单相对保留增量。全部四个控制与各类分母均报。',
        '所有任务已到达后回顾性读取历史首任务模型；不能把最终schema当作首任务当时在线可得。首任务模型拟合TRAIN，这些VAL是组件级开发证据，非患者独立。',
        '只涵盖首任务5类，其中head4类、tail仅class0、无tiny；不能由此解释全23类tail/tiny失败或认定新保留机制。所有权重只读、0 optimizer/额外SFT。',
        '4368响应包含native复用32，额外cached-forward4，共4372缓存特征使用行，0新视觉encoder/0TRAIN图像特征访问。Input/generated active tokens、包括padding的decoder token cells及wall/CPU全部计费，原cache/模型和R125训练费用不当免费。']
    (r/'REPORT_ZH.md').write_text('\n'.join(report)+'\n')
