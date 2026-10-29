"""Complete class-stage reporting from sealed counts; no model scoring."""
import json,os,csv,io
from pathlib import Path
from fractions import Fraction
from common import write
from decision import ARMS
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);assert json.loads((root/'results/VERIFICATION.json').read_text())['passed'];agg=json.loads((root/'HK/aggregate.json').read_text());tail=[c for c,n in enumerate(agg['train_counts']) if n<250]
# Historical original SFT counts are copied verbatim from the already verified handoff evidence.
history=json.loads((root/'HISTORICAL_SFT_COUNTS.json').read_text());summaries={};classrows=[];byarm={}
for arm in (*ARMS,'frozen_initial','historical_stagewise_sft'):
 metrics=[v['metrics'][arm] for v in agg['results']] if arm!='historical_stagewise_sft' else history
 values=[];byarm[arm]=metrics
 for stage,m in enumerate(metrics):
  per=[Fraction(c,n) for c,n in zip(m['correct'],m['counts'])];values.append([100*sum(per)/len(per),100*sum(per[c] for c in tail if c<len(per))/sum(c<len(per) for c in tail)])
  prev=0 if stage==0 else sum(map(len,cfg['domains']['HK']['stages'][:stage]));groups={}
  for name,ii in [('old_head',[c for c in range(prev) if c not in tail]),('old_tail',[c for c in range(prev) if c in tail]),('new_head',[c for c in range(prev,len(per)) if c not in tail]),('new_tail',[c for c in range(prev,len(per)) if c in tail])]:groups[name]=None if not ii else float(100*sum(per[c] for c in ii)/len(ii))
  for c,(correct,count) in enumerate(zip(m['correct'],m['counts'])):classrows.append(dict(arm=arm,stage=stage,class_index=c,TRAIN_count=agg['train_counts'][c],VAL_count=count,correct=correct,accuracy=correct/count,tail=c in tail,old=c<prev))
  values[-1].append(groups)
 summaries[arm]=dict(avg=float(sum(v[0] for v in values)/6),final=float(values[-1][0]),tail=float(values[-1][1]),stages=[dict(stage=i,macro=float(v[0]),tail=float(v[1]),groups=v[2]) for i,v in enumerate(values)])
primary='protected_controller';comparisons={}
for control in (*ARMS,'frozen_initial','historical_stagewise_sft'):
 if control==primary:continue
 comparisons[control]={k:summaries[primary][k]-summaries[control][k] for k in ['avg','final','tail']}
 for row in classrows:
  if row['arm']==primary:
   old=byarm[control][row['stage']];c=row['class_index'];assert row['VAL_count']==old['counts'][c]
contexts=json.loads((root/'HK/CONTEXT_RECORDS.json').read_text());diagnostics={}
for arm in ARMS:
 rows=[r for r in contexts if r['arm']==arm];diagnostics[arm]=dict(all_eight_negative_contexts=sum(v['all_candidate_returns_negative'] for v in rows),negative_retained_returns=sum(v['retained_absolute_return'] is not None and v['retained_absolute_return']<0 for v in rows),positive_advantage_negative_return=sum(v['retained_advantage'] is not None and v['retained_advantage']>0 and v['retained_absolute_return']<0 for v in rows),actual_updates=sum(v['actual_updates'] for v in rows),committed_updates=sum(v['committed_updates'] for v in rows))
gate=dict(primary=primary,original_SFT_final_tail_pass=all(comparisons['historical_stagewise_sft'][k]>0 for k in ['final','tail']),allocation_increment_final_tail=all(comparisons['balanced_short'][k]>0 for k in ['final','tail']),compute_control_final_tail=all(comparisons['balanced_long'][k]>0 for k in ['final','tail']),single_development_seed=True,no_independent_patient_or_clinical_confirmation=True)
write(root/'results/ANALYSIS.json',dict(summary=summaries,comparisons=comparisons,diagnostics=diagnostics,gate=gate));write(root/'results/ALL_CLASS_COUNTS.json',classrows)
lines=['# R80 长尾训练分配 GRPO：完整结果','', '主候选为 protected_controller；视觉模型由监督学习更新，GRPO 只更新训练分配控制器。单 HK74002 开发 seed，不构成独立患者或临床确认。','','| 分支 | 阶段平均BA | 最终BA | 最终尾类BA |','|---|---:|---:|---:|']
for arm,v in summaries.items():lines.append(f"| {arm} | {v['avg']:.6f} | {v['final']:.6f} | {v['tail']:.6f} |")
lines+=['','原SFT最终/尾类保留门：'+str(gate['original_SFT_final_tail_pass'])+'；相同保留更新量分配增量门：'+str(gate['allocation_increment_final_tail'])+'；额外监督计算控制最终/尾类门：'+str(gate['compute_control_final_tail']), '', '奖励查询集来自重复使用的 TRAIN，并非独立验证。所有36端点封存后统一读取VAL；完整逐类分母、旧头/旧尾/新头/新尾、绝对负回报、实际与保留更新量均保存。组内正优势不代表相对均衡控制有正收益。固定保留序号，无reward赢家筛选。', '', '五分支各1296次实际视觉更新，总6480次/25920micro；保留1872次，控制器576次。balanced_long使用同一共享数据流上的额外监督更新，重复batch曝光已计入，不能解释成额外独立样本。主要视觉工作量匹配不等于GPU秒或总耗时严格相同，实测费用以CONTROL_EXIT及各阶段wall记录为准。', '', '历史原SFT各stage从独立SFT开始，仅为强历史参照；本实验从stage0 SFT初始化后真实carry，因果对照为本轮balanced_short/long。所有极小类保留，不改变历史门，不把最佳消融事后换成主候选。', '', '原始支持features、模型/控制器/优化器权重和逐例响应仅私有保留；公开只含自有代码、协议和安全聚合。']
(root/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summaries,gate=gate),ensure_ascii=False))
