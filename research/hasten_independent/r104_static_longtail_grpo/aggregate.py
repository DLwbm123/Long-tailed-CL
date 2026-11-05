"""One static task; complete paired metrics and costs, no continual averages."""
import json,csv,time,shutil
from pathlib import Path
from common import write
from evaluate import MODES,csvout

def run(root):
    root=Path(root);r=root/'a';s=json.loads((r/'RESULT_SUMMARY.json').read_text());v=json.loads((r/'VERIFICATION.json').read_text());assert v['passed']
    for name in ['STAGE_METRICS.csv','CLASS_STAGE.csv','GROUP_STAGE.csv','RESCUE_HARM.csv','ROUTING_DIAGNOSTICS.csv','ORACLE_DIAGNOSTICS.csv','FOLD_COUNTS.csv']:
        shutil.copyfile(r/name,root/name)
    reward=[]
    for arm in MODES[4:]:
        obj=json.loads((r/'s0'/f'{arm}_TRAIN_DIAGNOSTICS.json').read_text());row={'arm':arm,'updates':obj['updates']}
        for key in ['loss','ratio_clip_fraction','max_ratio_deviation','gradient_norm','gradient_clipped','sampled_zero_group_fraction','sampled_reward_std','analytic_old_tail_harm_estimate','KL','entropy']:
            values=[x[key] for x in obj['records'] if x[key] is not None];row[key]=sum(values)/len(values) if values else None
        reward.append(row)
    csvout(root/'REWARD_DIAGNOSTICS.csv',reward)
    for source,target in [('INNER_SUPPORT_FALLBACK.json','INNER_SUPPORT_FALLBACK.csv'),('FOLD_TO_FULL_SHIFT.json','FOLD_TO_FULL_SHIFT.csv')]:
        rows=json.loads((r/source).read_text());csvout(root/target,rows if rows else [{'trigger_count':0}])
    write(root/'RESULT_SUMMARY.json',s);write(root/'VERIFICATION.json',dict(v,static_tasks=1,stage_readouts=9,teacher_optimizer=560,router_optimizer=2560,fusion_optimizer=3072))
    lines=['# R104 静态长尾分类：现有 GRPO 路线','','HK23类从开始共同可见，TRAIN6821/VAL1698，单seed74002；不含持续学习到达、回放、遗忘或跨阶段carry。FULL和两折教师均从通用BiomedCLIP重新训练，未复用HK监督权重。通用冻结前缀只读复用。','','|读出|micro %|macro %|tail %|','|---|---:|---:|---:|']
    for mode in MODES:
        x=s['final'][mode];lines.append(f"|{mode}|{x['micro']:.6f}|{x['macro']:.6f}|{x['tail']:.6f}|")
    lines+=['','|GRPO_CF−参照|micro pp|macro pp|tail pp|','|---|---:|---:|---:|']
    for ref in ['T','UNIFORM','SUP_CF','RAW_EXACT_CF','GROUP_EXACT_CF','GRPO_IN']:
        x=s['comparisons'][ref]['final'];lines.append(f"|{ref}|{x['micro']:+.6f}|{x['macro']:+.6f}|{x['tail']:+.6f}|")
    lines+=['','主要问题是GRPO能否同时超过强教师、监督选择和两种精确优化；固定融合或候选互补性不等于GRPO收益。正负均完整报告，不选epoch/动作/类别。单seed开发结果不能支持显著性或创新成立。','','与R103保持动作T/CENTER/MULTI、硬正确性增量奖励、G8、真实old policy和4次更新、KL.01、学习率和512更新不变。静态任务所有age=0/new=1，不存在old-tail，原old-tail harm项自然为0，没有改成全tail harm。旧尾评估N/A不是0；GROUP_STAGE中的new仅为实现常量，不作持续学习含义。','','原始tiny身份及correct/count见CLASS_STAGE；tail按TRAIN<250固定。所有23类教师训练/信息暴露轨迹改变，因此与R103绝对差不是持续学习因素的单一因果估计。组件不是已验证患者身份，VAL已用于历轮开发，TEST0。','','三教师、六fusion、五router端点全部seal后一次VAL，9读出。teacher560/fusion3072/router2560更新；32768组/262144动作，工程另计。全部六视图始终编码，不声称选择节省计算。REWARD_DIAGNOSTICS和FOLD_TO_FULL_SHIFT给出零组、clip、梯度和迁移；oracle仅事后诊断。','','worker计时含native/profile/fit/eval，子项不能重复加；原24h包含准备/排队/修复/分析/发布，未重置。GPU kernel时间未采样，准备/SSH/文档/发布CPU非0未完整采样。最终wait4、原钟和存储见CONTROL_EXIT。','','完整交付后停止，monitor OFF，不开新seed/算法；私有图像、身份、逐例输出、特征、权重、优化器和原始日志不公开。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write(root/'COSTS.json',{'worker':json.loads((r/'COSTS.json').read_text()),'historical_supervised_teacher_reuse':False,'generic_pretrained_prefix_reused_readonly':True,'original_elapsed_at_aggregation':time.time()-json.loads((root/'RUN_CONFIG.private.json').read_text())['start'],'wait4_costs':'CONTROL_EXIT.json appended by controller; includes worker subitems'})
