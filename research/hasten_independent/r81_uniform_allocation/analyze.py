"""Only sealed aggregate counts; learned-minus-uniform is fixed before this outcome."""
import os,json
from pathlib import Path
from fractions import Fraction as F
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);assert json.loads((r/'results/VERIFICATION.json').read_text())['passed'];agg=json.loads((r/'HK/aggregate.json').read_text());prior=json.loads((r/'R80_MATCHED_COUNTS.json').read_text());tail=[i for i,n in enumerate(agg['train_counts']) if n<250];metrics={'uniform_controller':[v['metrics']['uniform_controller'] for v in agg['results']]}
for arm in sorted(set(v['arm'] for v in prior)):metrics[arm]=[dict(correct=[v['correct'] for v in prior if v['arm']==arm and v['stage']==stage],counts=[v['VAL_count'] for v in prior if v['arm']==arm and v['stage']==stage]) for stage in range(6)]
summary={};classrows=[];contrib=[]
for arm,stages in metrics.items():
 vv=[]
 for stage,m in enumerate(stages):
  p=[F(k,n) for k,n in zip(m['correct'],m['counts'])];vv.append([100*sum(p)/len(p),100*sum(p[i] for i in tail if i<len(p))/sum(i<len(p) for i in tail)])
  for c,(k,n) in enumerate(zip(m['correct'],m['counts'])):classrows.append(dict(arm=arm,stage=stage,class_index=c,correct=k,VAL_count=n,TRAIN_count=agg['train_counts'][c],tail=c in tail))
 groups=[]
 for stage,m in enumerate(stages):
  old=0 if stage==0 else sum(map(len,cfg['domains']['HK']['stages'][:stage]));seen=len(m['counts']);row={}
  for name,ii in [('old_head',[c for c in range(old) if c not in tail]),('old_tail',[c for c in range(old) if c in tail]),('new_head',[c for c in range(old,seen) if c not in tail]),('new_tail',[c for c in range(old,seen) if c in tail])]:row[name]=None if not ii else float(100*sum(F(m['correct'][c],m['counts'][c]) for c in ii)/len(ii))
  groups.append(row)
 summary[arm]=dict(groups=groups,avg=float(sum(v[0] for v in vv)/6),final=float(vv[-1][0]),tail=float(vv[-1][1]),stage_macro_tail=[[float(x) for x in v] for v in vv])
for stage in range(6):
 a=metrics['protected_controller'][stage];b=metrics['uniform_controller'][stage];n=len(a['counts']);T=sum(c<n for c in tail);assert a['counts']==b['counts']
 for c,(ka,kb,den) in enumerate(zip(a['correct'],b['correct'],a['counts'])):contrib.append(dict(stage=stage,class_index=c,VAL_count=den,learned_correct=ka,uniform_correct=kb,delta_correct=ka-kb,macro_contribution_pp=100*(ka-kb)/den/n,tail_contribution_pp=100*(ka-kb)/den/T if c in tail else 0))
delta={k:summary['protected_controller'][k]-summary['uniform_controller'][k] for k in ['avg','final','tail']};gate=dict(conditional_learned_final_tail_increment=delta['final']>0 and delta['tail']>0,does_not_reverse_R80_original_gate_failure=True,does_not_reverse_R80_shuffled_superiority=True,not_independent_confirmation=True)
write(r/'results/ANALYSIS.json',dict(summary=summary,learned_minus_uniform_pp=delta,interpretation=gate));write(r/'results/ALL_CLASS_COUNTS.json',classrows);write(r/'results/LEARNED_CLASS_CONTRIBUTIONS.json',contrib)
ctx=json.loads((r/'HK/CONTEXT_RECORDS.json').read_text());safe=[{k:v[k] for k in ['stage','context','retained_branch','absolute_returns','advantages','all_exposure','committed_exposure','actual_updates','committed_updates','controller_updates','reward_used_for_commit_selection']} for v in ctx];write(r/'results/REWARD_AND_EXPOSURE_CURVES.json',safe)
lines=['# R81 固定均匀类别动作：GRPO必要归因控制','','仅新增一条不学习的均匀合法动作控制器，R80主候选为事前固定封存参照。沿用同seed/common/action-uniform/fill流、视觉目标/更新量/真实跨stage carry/动作唯一性和完整rollout代价，不是独立确认。','','| 分支 | 平均BA | 最终BA | 尾类BA |','|---|---:|---:|---:|']
for arm,v in summary.items():lines.append(f"| {arm} | {v['avg']:.6f} | {v['final']:.6f} | {v['tail']:.6f} |")
lines+=['','学习控制器减固定均匀控制(pp)：'+json.dumps(delta),'','最终/尾类条件增量：'+str(gate['conditional_learned_final_tail_increment'])+'。无论该比较如何，R80原SFT联合门失败、打乱奖励D优于C的结果保持；不能据补充对照改写原主门或宣称普遍GRPO收益。','', '1296实际视觉update/144保留/5184micro/0controlleroptimizer，48context/432分支，6head全部封存后6新VAL/0TEST。原SFT及R80参照只用封存计数，无重新评分。所有极小类/完整分母保留，Q为重复TRAIN反馈并非独立holdout，不保证独立患者。','', 'SOURCE/PROTOCOL明确继承原目标、两阶段LR、动作名额不对称和资源硬门；无调参或VAL筛选。费用以CONTROL_EXIT和cleanup证据为准，结案不新增模型评分/优化。只公开自有代码与安全聚合，不公开图像/身份/逐例概率/特征/权重/私有配置路径。']
(r/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summary,delta=delta,gate=gate)))
