"""Saved-only pilot/full-trial summary; stage/class/tiny/cost reports without private traces."""
import csv,json
from pathlib import Path
from common import write
from constants import ARMS,SEEDS

def csvout(p,rows):
 if rows:
  with Path(p).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def run(root):
 r=Path(root);lines=['# R108-CGMR 完整结案','','集合条件有限记忆修补；两流、三seed、六臂仅在A反馈迁移门通过后执行。主方法GRPO-like有限候选全反馈目标，不是标准GSPO，也不预设RL必要性。'];gates={};coverage=[];candidate=[];transfer=[];qcounts=[];behavior=[];costs={}
 for k in ['a','b']:
  p=r/k;costs[k]=json.loads((p/'COSTS.json').read_text())
  if not (p/'PILOT_GATE.json').exists():continue
  gates[k]=json.loads((p/'PILOT_GATE.json').read_text())
  for x in json.loads((p/'pilot/ROOTS.private.json').read_text()):
   if 'candidate_count' not in x:transfer.append(dict(stream=k,root=x['root'],status='SKIPPED',candidates=0,chosen_fit=None,audit_delta_R=None,audit_delta_U=None,audit_H_selected=None,old_scarce_evaluable=False));continue
   transfer.append(dict(stream=k,root=x['root'],status='COMPLETE',candidates=x['candidate_count'],chosen_fit=x['chosen_fit'],audit_delta_R=x['audit_delta_R'],audit_delta_U=x['audit_delta_U'],audit_H_selected=x['audit_H_selected'],old_scarce_evaluable=x['old_scarce_evaluable']));c=x['coverage'];n=x['candidate_count'];p0=[1.] if n==1 else [.5]+[.5/(n-1)]*(n-1);import math
   behavior.append(dict(stream=k,phase='A_prior_no_actor',seed=74002,stage=0,arm='INITIAL_PRIOR',root=x['root'],noop_probability=p0[0],KL_vs_prior=0.,TV_vs_prior=0.,candidate_entropy=-sum(v*math.log(v) for v in p0)))
   coverage.append(dict(stream=k,phase='A',seed=74002,stage=0,arm='NO_ACTOR',root=x['root'],actual_B=x['actual_B'],maximum_age=x['maximum_age'],**c))
   for j,a in enumerate(x['proposal_audit']):candidate.append(dict(stream=k,phase='A',seed=74002,stage=0,arm='NO_ACTOR',root=x['root'],target=j,**a))
   for ci,br in enumerate(x['raw_query_counts']):
    for role,hs in br.items():
     for h,score in enumerate(hs):
      for cl,v in score['classes'].items():qcounts.append(dict(stream=k,root=x['root'],candidate=ci,query_role=role,future=h+1,shadow_class=cl,correct=v['correct'],count=v['count']))
 if gates:
  write(r/'PILOT_GATE.json',{'passed':len(gates)==2 and all(x['passed'] for x in gates.values()),'streams':gates,'A_actor_optimizer':0,'VAL_TEST_before_B':0,'no_threshold_changes':True});lines+=['','## A反馈迁移门','','|流|可评价／非平凡根|旧稀缺可评价根|mean ΔR audit|mean ΔU audit|mean H audit|通过|','|---|---:|---:|---:|---:|---:|---|']
  for k,g in gates.items():
   f=lambda x:'N/A' if g[x] is None else f'{g[x]:+.8f}';lines.append(f"|{k}|{g['evaluable_roots']}/{g['nontrivial_evaluable_roots']}|{g['old_scarce_evaluable_roots']}|{f('mean_audit_delta_R')}|{f('mean_audit_delta_U')}|{f('mean_audit_H_selected')}|{g['passed']}|")
  lines+=['','门要求每流非平凡>=16、旧稀缺>=8、meanΔR>=.005、meanΔU>=0、meanH<=.005。0.005是U/R分数，不等于0.5BA百分点。所有可评估no-op和并列根均计入均值，不选有利子集；缺失根、query分母与视图缺失完整保留。共享来源数据的根不是独立患者重复。']
 passed=len(gates)==2 and all(x['passed'] for x in gates.values());summaries={};classrows=[];stage=[];group=[];harm=[];tiny=[]
 if passed and (r/'GLOBAL_LEARNING_SEAL.json').exists():
  lines+=['','## B六臂三seed两流','','|流|seed|方法|mean macro|final macro|final tail|final old-tail|','|---|---:|---|---:|---:|---:|---:|'];audit=[]
  for k in ['a','b']:
   summaries[k]={}
   for seed in SEEDS:
    p=r/k/f'seed{seed}';assert json.loads((p/'VERIFICATION.json').read_text())['passed'];s=json.loads((p/'RESULT_SUMMARY.json').read_text());summaries[k][str(seed)]=s
    for arm in ARMS:
     f=s['final'][arm];lines.append(f"|{k}|{seed}|{arm}|{s['mean'][arm]['macro']:.6f}|{f['macro']:.6f}|{f['tail']:.6f}|{f['old_tail']:.6f}|")
    for n,rr in [('CLASS_STAGE.csv',classrows),('STAGE_METRICS.csv',stage),('GROUP_STAGE.csv',group),('RESCUE_HARM.csv',harm)]:rr +=[dict(stream=k,seed=seed,**x) for x in csv.DictReader((p/n).open())]
    for t in range(5):
     for x in json.loads((p/f's{t}/META.private.json').read_text()):
      if 'coverage' not in x:continue
      coverage.append(dict(stream=k,phase='B',seed=seed,stage=t,arm=x['arm'],root=x['root'],actual_B=x['actual_B'],maximum_age=x['maximum_age'],**x['coverage']))
      for j,a in enumerate(x['proposal_audit']):candidate.append(dict(stream=k,phase='B',seed=seed,stage=t,arm=x['arm'],root=x['root'],target=j,**a))
      bh={z:v for z,v in x['behavior'].items() if z not in ['probability','draw32_counts']};behavior.append(dict(stream=k,phase='B_shadow',seed=seed,stage=t,arm=x['arm'],root=x['root'],**bh))
   for x in json.loads((r/k/'MEMORY_AND_ACCESS_AUDIT.private.json').read_text()):
    z={n:v for n,v in x.items() if n not in ['behavior','selected_edit','per_class_parent_coverage']};audit.append(dict(stream=k,**z));bh={n:v for n,v in x['behavior'].items() if n not in ['probability','draw32_counts']};behavior.append(dict(stream=k,phase='B_formal',seed=x['seed'],stage=x['stage'],arm=x['arm'],root=-1,**bh))
  write(r/'RESULT_SUMMARY.json',{'streams':summaries,'primary':'CGMR_MAIN','TEST':0});csvout(r/'MEMORY_ACCESS.csv',audit)
  for n,rr in [('CLASS_STAGE.csv',classrows),('STAGE_METRICS.csv',stage),('GROUP_STAGE.csv',group),('RESCUE_HARM.csv',harm)]:csvout(r/n,rr)
  tiny=[x for x in classrows if x['tiny']=='True'];csvout(r/'TINY_CLASS_STAGE.csv',tiny);contrib=[]
  for k in ['a','b']:
   for seed in SEEDS:
    for t in range(5):
     rr=[x for x in classrows if x['stream']==k and x['seed']==seed and int(x['stage'])==t];by={(x['arm'],x['class']):x for x in rr};main=[x for x in rr if x['arm']=='CGMR_MAIN'];C=len(main)
     for ref in ARMS[:-1]:
      d=lambda a:sum(float(x['recall'])-float(by[ref,x['class']]['recall']) for x in a)/C;ti=d([x for x in main if x['tiny']=='True']);nt=d([x for x in main if x['tiny']=='False']);full=d(main);assert abs(full-ti-nt)<1e-10;contrib.append(dict(stream=k,seed=seed,stage=t,reference=ref,full_macro_delta_pp=full,non_tiny_original_denominator_contribution_pp=nt,tiny_original_denominator_contribution_pp=ti))
  csvout(r/'TINY_CONTRIBUTION.csv',contrib);flags={}
  for k,ss in summaries.items():
   avg=lambda ph,arm,metric:sum(s[ph][arm][metric] for s in ss.values())/3
   delta=lambda ph,ref,metric:avg(ph,'CGMR_MAIN',metric)-avg(ph,ref,metric)
   learning=all(delta(ph,'BASE_ZERO','macro')>=.5 and delta(ph,'RANDOM_REPAIR','macro')>0 and delta(ph,'REWARD_SHUFFLED','macro')>0 for ph in ['mean','final']) and sum(s['mean']['CGMR_MAIN']['macro']>s['mean']['BASE_ZERO']['macro'] for s in ss.values())>=2 and all(s['final']['CGMR_MAIN']['macro']-s['final']['BASE_ZERO']['macro']>=-1 for s in ss.values())
   risk=delta('final','BASE_ZERO','tail')>=-.5 and delta('final','BASE_ZERO','old_tail')>=-.5
   for x in tiny:
    if x['stream']==k and x['arm']=='CGMR_MAIN':
     b=next(y for y in tiny if y['stream']==k and y['seed']==x['seed'] and y['stage']==x['stage'] and y['class']==x['class'] and y['arm']=='BASE_ZERO');risk=risk and int(x['correct'])>=int(b['correct'])
   flags[k]={'learning_increment':learning,'risk_and_all_stage_tiny_protection':risk,'BALANCED_competitiveness':all(delta(ph,'BALANCED','macro')>=-.5 for ph in ['mean','final']),'initial_RL_necessity_support':all(delta(ph,'SUP_GREEDY','macro')>0 for ph in ['mean','final']) and delta('mean','SUP_GREEDY','macro')>=.3}
  write(r/'INTERPRETATION_GATES.json',{'per_stream':flags,'all_require_both_streams':True,'not_significance':True});lines+=['','四项判定分开，见INTERPRETATION_GATES.json；单项通过不替代总体证据。未通过不换seed/主候选/奖励或读TEST救援。']
 elif gates:
  lines+=['','## 结论','','A反馈排序跨查询迁移未通过或证据不足，按预定门结案；B未启动，actor0、VAL0、TEST0。这不是RL算法优劣的直接测量，而是当前局部候选与训练反馈未满足值得进入完整学习比较的前置条件。']
 else:lines+=['','原生完整预算未准入；未启动A/B科学运行。工程费用与失败/预算收据保留。']
 for n,rr in [('QUERY_COVERAGE.csv',coverage),('CANDIDATE_AUDIT.csv',candidate),('FEEDBACK_TRANSFER.csv',transfer),('A_QUERY_CLASS_COUNTS.csv',qcounts),('POLICY_BEHAVIOR.csv',behavior)]:csvout(r/n,rr)
 write(r/'COSTS.json',{'workers':costs,'wait4_and_controller':'CONTROL_EXIT.json','native_profile_included_in_workers_once':True,'full_A_B_classifier_upper':631344,'GPU_kernel_seconds':None,'prep_import_SSH_documents_publication_CPU':'nonzero incomplete sampling'})
 write(r/'VERIFICATION.json',{'passed':True,'A_streams':len(gates),'B_executed':bool(summaries),'classifier_endpoints':180 if summaries else 0,'TEST':0,'all_science_complete_before_aggregate':True,'new_report_forward_optimizer':0})
 lines+=['','完整定义与必要细化见USER_EXECUTION_SPEC_ZH/PROTOCOL/FEATURE_SCHEMA/SEEDS；不与独立路线混同，不更改R107负结果。所有影子支持来自自身合法上一记忆与当前TRAIN，query角色跨阶段不变，不恢复旧视图。审计查询条件于历史记忆筛选，不是整套算法的独立临床测试。单数据集、多轮开发VAL、三seed不等于独立患者样本。no-op先验和有限编辑不构成风险保证。','','所有学习器原始通用前端起点，参数/优化器与实际回放字节分开记账；每臂回放528568B以内。重复native特征/合成标签只作最大形状工程检查，正式权重重新初始化。完整原钟不重置，native/profile计入worker费用不重复加；无GPUkernel效率主张。私有身份/像素/逐例特征概率/权重/optim/rawlog不公开。监测OFF；本轮交付后停止，不自动追加实验。'];(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
