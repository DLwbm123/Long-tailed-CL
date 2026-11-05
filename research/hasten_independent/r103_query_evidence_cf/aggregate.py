"""Both complete fixed streams: all positive/negative metrics and costs."""
import json,csv,time
from pathlib import Path
from common import write
from evaluate import csvout,MODES

def run(root):
    root=Path(root);summaries={};lines=['# R103 查询条件证据选择与整链交叉拟合','','教师条件、全已到达历史可用的机制实验，非严格有限记忆 LT-CIL。固定seed74002，仅Ordered与分层order0。无其他顺序/种子、无超参搜索、无独立患者或显著性主张。旧F已转为反馈训练，VAL有历史开发使用，TEST0。','','|流|读出|mean macro|final macro|final tail|mean old-tail|有效old-tail阶段|','|---|---|---:|---:|---:|---:|---:|']
    for k in ['a','b']:
        r=root/k;s=json.loads((r/'RESULT_SUMMARY.json').read_text());assert json.loads((r/'VERIFICATION.json').read_text())['passed'];summaries[k]=s
        for mode in MODES:
            old=s['mean'][mode]['old_tail'];old='N/A' if old is None else f'{old:.6f}'
            lines.append(f"|{k}|{mode}|{s['mean'][mode]['macro']:.6f}|{s['final'][mode]['macro']:.6f}|{s['final'][mode]['tail']:.6f}|{old}|{s['mean_valid_stages'][mode]['old_tail']}|")
    for name in ['STAGE_METRICS.csv','CLASS_STAGE.csv','GROUP_STAGE.csv','RESCUE_HARM.csv','ROUTING_DIAGNOSTICS.csv','ORACLE_DIAGNOSTICS.csv']:
        rows=[]
        for k in ['a','b']:
            with (root/k/name).open() as f:rows += [dict(stream=k,**x) for x in csv.DictReader(f)]
        csvout(root/name,rows)
    reward=[];fallback=[];shift=[]
    for k in ['a','b']:
        for s in range(5):
            for arm in ['SUP_CF','RAW_EXACT_CF','GROUP_EXACT_CF','GRPO_CF','GRPO_IN']:
                obj=json.loads((root/k/f's{s}'/f'{arm}_TRAIN_DIAGNOSTICS.json').read_text());recs=obj['records'];row={'stream':k,'stage':s,'arm':arm,'updates':obj['updates']}
                for key in ['loss','ratio_clip_fraction','max_ratio_deviation','gradient_norm','gradient_clipped','sampled_zero_group_fraction','sampled_reward_std','analytic_old_tail_harm_estimate','KL','entropy']:
                    vals=[x[key] for x in recs if x[key] is not None];row[key]=sum(vals)/len(vals) if vals else None
                reward.append(row)
        fallback += [dict(stream=k,**x) for x in json.loads((root/k/'INNER_SUPPORT_FALLBACK.json').read_text())]
        shift += [dict(stream=k,**x) for x in json.loads((root/k/'FOLD_TO_FULL_SHIFT.json').read_text())]
    csvout(root/'REWARD_DIAGNOSTICS.csv',reward)
    # Vector-valued action distributions remain valid JSON lists in CSV cells.
    csvout(root/'FOLD_TO_FULL_SHIFT.csv',shift)
    if fallback:csvout(root/'INNER_SUPPORT_FALLBACK.csv',fallback)
    else:csvout(root/'INNER_SUPPORT_FALLBACK.csv',[{'stream':'all','trigger_count':0}])
    write(root/'RESULT_SUMMARY.json',{'streams':summaries,'primary':'GRPO_CF','formal_router_updates':25600,'formal_fusion_updates':30720,'single_train_seed':True,'TEST':0})
    lines += ['','## 预定配对差值（GRPO_CF−参照，pp）','','|流|参照|mean macro|final macro|final tail|','|---|---|---:|---:|---:|']
    for k,s in summaries.items():
        for ref in ['T','UNIFORM','SUP_CF','RAW_EXACT_CF','GROUP_EXACT_CF','GRPO_IN']:
            d=s['comparisons'][ref];lines.append(f"|{k}|{ref}|{d['mean']['macro']:+.6f}|{d['final']['macro']:+.6f}|{d['final']['tail']:+.6f}|")
    lines += ['','六视图始终全部编码；选择教师也不节省编码计算。固定融合UNIFORM为概率平均；学习臂为确定性argmax动作，不混合、不采样部署。候选Oracle仅封存后诊断，不能当作可部署方法。','', '归因按以下六项读取完整表：候选互补性（CENTER/MULTI与ORACLE）、固定融合解释量（UNIFORM）、学习选择对T/UNIFORM、GRPO对SUP与两种EXACT、CF对IN旧尾损害、OOF到FULL输入/动作变化。候选有效不等于GRPO有效；胜SUP而不胜GROUP_EXACT不支持采样必要性。CF与IN同时含上游拟合规模和置信度差异，不能只归因于泄漏。','', 'Ordered前两阶段无tail，null；旧尾有效均值分母另列。tiny按原始身份绑定，CLASS_STAGE提供correct/count，重复阶段不当独立病例。F是训练反馈，未做旧F独立评估。','', '全体50router/60fusion/20新折teacher及10复用FULL教师封存后统一VAL，共90stage读出。新增更新/编码、复用历史教师成本和失败费用分别记账；worker子项不能重复相加。GPUkernel时间未采样。准备/文档/SSH/发布CPU非0未全采样。','', '本轮完成后停止，monitor OFF，不启动后继。私有身份/图像/逐例预测/特征/权重/优化器/原始日志仅留授权data盘。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    costs={k:json.loads((root/k/'COSTS.json').read_text()) for k in ['a','b']};write(root/'COSTS.json',{'workers':costs,'historical_FULL_teacher_costs_separate':json.loads((root/'REUSED_TEACHER_COSTS.json').read_text()),'original_elapsed_at_aggregation':time.time()-json.loads((root/'RUN_CONFIG.private.json').read_text())['start'],'wait4_costs':'CONTROL_EXIT.json, written after aggregation; native/subitems already included','new_followup_experiments':0})
    write(root/'VERIFICATION.json',{'passed':True,'streams':2,'router_endpoints':50,'fusion_endpoints':60,'new_fold_teachers':20,'reused_FULL_teachers':10,'router_updates':25600,'fusion_updates':30720,'stage_readouts':90,'all_stream_verifications_passed':True,'TEST':0})
