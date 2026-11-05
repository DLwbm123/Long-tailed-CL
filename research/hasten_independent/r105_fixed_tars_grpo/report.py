"""Publishable complete paired results; no selection or followup training."""
from pathlib import Path
import json,csv,shutil,time
from common import write
from router import ARMS

def run(root):
    root=Path(root);r=root/'a';s=json.loads((r/'RESULT_SUMMARY.json').read_text());v=json.loads((r/'VERIFICATION.json').read_text());assert v['passed'];metrics=s['final'];contrasts=[]
    for a in ARMS:
        refs=['T','CENTER','MULTI','UNIFORM']
        if a.startswith('TARS_'):refs+=[a.removeprefix('TARS_')]
        if a=='TARS_GRPO':refs+=['TARS_SUP','TARS_RAW_EXACT','TARS_GROUP_EXACT','SUP','RAW_EXACT','GROUP_EXACT']
        for ref in dict.fromkeys(refs):contrasts.append(dict(arm=a,reference=ref,**{k:metrics[a][k]-metrics[ref][k] for k in ['micro','macro','tail']}))
    with (root/'PAIRED_COMPARISONS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=['arm','reference','micro','macro','tail'],lineterminator='\n');w.writeheader();w.writerows(contrasts)
    lines=['# R105 固定教师与候选：普通与TARS匹配优化','','HK23静态分类，单seed74002；同一冻结R104 FULL教师、CENTER/MULTI候选用于TRAIN反馈与VAL部署，无半数据→FULL构建器切换。所有上游均不更新，训练反馈仍是上游拟合内TRAIN，不称全链交叉拟合或独立奖励holdout。VAL已历轮开发使用，TEST0。','','|读出|micro %|macro %|tail %|','|---|---:|---:|---:|']
    for mode,m in metrics.items():lines.append(f"|{mode}|{m['micro']:.6f}|{m['macro']:.6f}|{m['tail']:.6f}|")
    lines+=['','|TARS_GRPO−参照|micro pp|macro pp|tail pp|','|---|---:|---:|---:|']
    for d in contrasts:
        if d['arm']=='TARS_GRPO':lines.append(f"|{d['reference']}|{d['micro']:+.6f}|{d['macro']:+.6f}|{d['tail']:+.6f}|")
    lines+=['','TARS对同初始化、同自然查询序列的普通GRPO给出加权效果；与TARS_SUP/RAW_EXACT/GROUP_EXACT比较随机策略优化增量。固定CENTER/UNIFORM正信号不能归GRPO。没有全局常数放大或权重错配控制，因此即使TARS正，也不能独立分离类别分配、总梯度尺度和相对KL减弱贡献。所有主负结果保留，不挑臂/epoch/类别或调参救援。','','权重由固定教师TRAIN逐类错误率与原TRAIN计数一次计算，gamma.75，中点median，w=max(1,t/median)，只乘每查询数据项，不乘KL.01。所有臂512更新，总4096；32768真实G8组/262144动作，old/actions/advantages固定4步后刷新，保持原clip.8–1.2，constant groups不重采。自然查询相对R104类别均匀查询是一项明示变化，跨run绝对差不能归TARS。','','所有8最终router封存后一次六视图VAL，12固定读出。micro/macro/tail、全部23类correct/count及原tiny身份、救回损害、路由、oracle诊断完整公开；oracle不能部署。tiny分母很小，单seed非患者独立或显著性证据。所有六视图总是编码，无选择节省计算主张。','','COSTS记录本轮新增，R104教师/融合历史成本另列并不重复加；新native/profile/fit/eval已含worker。原FIRST24h包含准备/排队/修复/分析交付未重置，GPU kernel时间未采样，准备/SSH/发布CPU非0未全采样。CONTROL_EXIT在汇总后写入最终wait4费用。','','完成公开交付后停止，monitor OFF，不自动开新seed/算法；图像/身份/逐例features、policy traces、预测、权重、优化器、原始日志保持私有。']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    for name in ['RESULT_SUMMARY.json','VERIFICATION.json','STAGE_METRICS.csv','CLASS_STAGE.csv','GROUP_STAGE.csv','RESCUE_HARM.csv','ROUTING_DIAGNOSTICS.csv','ORACLE_DIAGNOSTICS.csv','TARS_WEIGHTS.json','TRAIN_CLASS_WEIGHTS.csv','QUERY_EXPOSURE.json','FROZEN_BINDING.json','BUDGET.json','COSTS.json']:shutil.copyfile(r/name,root/name)
    diag=[]
    for arm in ARMS:
        p=json.loads((r/f'{arm}_TRAIN_DIAGNOSTICS.json').read_text());row={'arm':arm,'updates':512,'strict_restore':p['strict_restore']}
        for key in ['loss','gradient_norm','gradient_clipped','ratio_clip_fraction','KL','entropy','structural_zero_group_fraction','sampled_zero_group_fraction','mean_query_weight']:
            x=[v[key] for v in p['records'] if v[key] is not None];row[key]=sum(x)/len(x) if x else None
        diag.append(row)
    with (root/'REWARD_DIAGNOSTICS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(diag[0]),lineterminator='\n');w.writeheader();w.writerows(diag)
