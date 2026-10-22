"""Frozen descriptive/paired analysis; reads sealed outputs and never trains."""
import os,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ['RESULT_ROOT']);agg=json.loads((ROOT/'aggregate.json').read_text());rows=agg['results']
assert len(rows)==36
seeds=[74510,74511,74512];arms=['H-full','H-no-tree','H-shuffle','H-no-proj'];classes=[c for c in range(13) if c!=1]
def mean(v):return float(np.mean(v))
def read(seed,arm,stage):return torch.load(ROOT/f'trajectories/{seed}/{arm}/stage{stage}_responses.private.pt',map_location='cpu',weights_only=False)
summary={}
for arm in arms:
 rr=[r for r in rows if r['arm']==arm];last=[r for r in rr if r['stage']==2]
 summary[arm]=dict(stage_BA=[mean([r['metrics']['final']['BA'] for r in rr if r['stage']==k]) for k in range(3)],stage_average_BA=mean([r['metrics']['final']['BA'] for r in rr]),final_BA=mean([r['metrics']['final']['BA'] for r in last]),hyper_final_BA=mean([r['metrics']['hyper']['BA'] for r in last]),prototype_final_BA=mean([r['metrics']['prototype']['BA'] for r in last]),old_final_BA=mean([r['metrics']['final']['old_BA'] for r in last]),new_final_BA=mean([r['metrics']['final']['new_BA'] for r in last]),old_candidates_BA=mean([r['metrics']['final']['old_candidates_BA'] for r in last]),final_recall=[None if c==1 else mean([r['metrics']['final']['recall'][c] for r in last]) for c in range(13)])
comparisons={}
for arm in arms[1:]:
 differences=[];paired=[];final_seed=[];avg_seed=[]
 for seed in seeds:
  a=read(seed,'H-full',2);b=read(seed,arm,2);assert a['ids']==b['ids'] and torch.equal(a['labels'],b['labels'])
  y=a['labels'].numpy();aa=(a['scores']['final'].argmax(1)==a['labels']).numpy();bb=(b['scores']['final'].argmax(1)==b['labels']).numpy();d=aa.astype(float)-bb.astype(float);differences.append(d)
  per=[float(d[y==c].mean()) for c in classes];final_seed.append(mean(per))
  paired.append(dict(seed=seed,rescued=int((aa&~bb).sum()),harmed=int((~aa&bb).sum()),per_class=[dict(class_id=c,count=int((y==c).sum()),rescued=int((aa&~bb&(y==c)).sum()),harmed=int((~aa&bb&(y==c)).sum())) for c in classes]))
  ra=[r for r in rows if r['seed']==seed and r['arm']=='H-full'];rb=[r for r in rows if r['seed']==seed and r['arm']==arm]
  avg_seed.append(mean([r['metrics']['final']['BA'] for r in ra])-mean([r['metrics']['final']['BA'] for r in rb]))
 d=np.mean(differences,axis=0);rng=np.random.default_rng(86425);boot=np.zeros(5000)
 for c in classes:
  vals=d[y==c];boot+=vals[rng.integers(0,len(vals),size=(5000,len(vals)))].mean(1)/len(classes)
 ci=np.quantile(boot,[.025,.975]).tolist();recalls={c:float(d[y==c].mean()) for c in classes};point=mean(final_seed)
 leave=[];undefined=[]
 for i in range(len(y)):
  c=int(y[i]);n=int((y==c).sum())
  if n<=1:undefined.append(c);continue
  leave.append(point+((d[y==c].sum()-d[i])/(n-1)-recalls[c])/12)
 protection=all(v>=(-1e-12 if c in [2,12] else -.01-1e-12) for c,v in recalls.items())
 qualified=point>=.01 and mean(avg_seed)>=.01 and sum(v>0 for v in final_seed)>=2 and ci[0]>0 and protection
 comparisons[arm]=dict(final_difference=point,stage_average_difference=mean(avg_seed),seed_final_differences=final_seed,conditional_95_interval=ci,protection_pass=protection,per_class_recall_difference=recalls,paired=paired,eligible_leave_one_group_range=[min(leave),max(leave)],leave_one_group_undefined_classes=sorted(set(undefined)),qualified_exploratory_signal=qualified)
report=dict(summary=summary,comparisons=comparisons,singleton_class_protection='UNKNOWN',patient_independence='UNKNOWN',independent_confirmation=False,bootstrap='5000 class-stratified known-group draws; paired across seeds; fixed trained models; no multiplicity correction',method_vs_standard_baseline='NOT_TESTED',stage_trainings=36,updates=agg['updates'])
(ROOT/'ANALYSIS.json').write_text(json.dumps(report,indent=2))
lines=['# HASTEN独立路线 r01 结果','', '本轮为支持侧固定划分的短增量探索；不是原方法实验，不是独立确认，也不是原论文数值复现。','', '|实验臂|阶段0 BA%|阶段1 BA%|阶段2 BA%|阶段平均BA%|纯双曲最终BA%|原型最终BA%|','|---|---:|---:|---:|---:|---:|---:|']
for arm,s in summary.items():lines.append('|'+arm+'|'+'|'.join(f'{100*v:.4f}' for v in s['stage_BA']+[s['stage_average_BA'],s['hyper_final_BA'],s['prototype_final_BA']])+'|')
lines+=['','主指标是预定官方融合读出。纯分支不能事后替换主表。以下差值均为H-full减匹配消融。','', '|对照|最终差pp|阶段均差pp|条件95%区间pp|逐类保护|探索性门|','|---|---:|---:|---|---|---|']
for arm,s in comparisons.items():lines.append(f"|{arm}|{100*s['final_difference']:.4f}|{100*s['stage_average_difference']:.4f}|{[round(100*v,4) for v in s['conditional_95_interval']]}|{s['protection_pass']}|{s['qualified_exploratory_signal']}|")
lines+=['','痔类无留组样本，保护不可估计；回肠、UC1–2级各只有1个留组样本，删除其唯一组后该类召回不可定义，因此不能宣称任意删组均稳健。跨种子仍为同一批样本。bootstrap条件于固定模型，不包含重新训练及患者关系不确定性。','', '四臂只回答内部语义/投影贡献；未训练标准无机制基线，不能据此声称完整方法优于标准baseline。','', '逐种子救回/损害、全类召回与保护失败见ANALYSIS.json；几何、实际投影和恢复验收见aggregate.json。工程检查通过不代表科学效果通过。']
(ROOT/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(stage_trainings=36,updates=agg['updates'],signals={k:v['qualified_exploratory_signal'] for k,v in comparisons.items()})))
