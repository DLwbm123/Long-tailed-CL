"""Prespecified saved-only within-state diagnostics, private risks excluded publicly."""
import os,json
from pathlib import Path
from common import write
from diagnostic_stats import kendall,selection
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);assert json.loads((r/'results/VERIFICATION.json').read_text())['passed'];records=json.loads((r/'HK/RECORDS.private.json').read_text());diagnostics=[];actionrows=[]
for v in records:
 bs=v['branches'];dup=bs[-1];base=bs[0];noise=dup['parameter_difference']['L2'];risknoise=max(abs(a-b) for a,b in zip(dup['query']['full']['brier'],base['query']['full']['brier']));rr=[b for b in bs if not b['duplicate']];tau=kendall([b['stats']['Qa']['utility'] for b in rr],[b['stats']['Qb']['utility'] for b in rr]);diag=dict(stage=v['stage'],fold=v['fold'],repeat=v['repeat'],legal_actions=v['legal_actions'],duplicate_parameter_L2=noise,duplicate_max_Brier_difference=risknoise,rank_tau_b=tau,forward_selection=selection(bs),reverse_selection=selection(bs,'Qb','Qa'));diagnostics.append(diag)
 for b in bs:
  resp=max(abs(a-z) for a,z in zip(b['query']['full']['brier'],base['query']['full']['brier']));actionrows.append(dict(stage=v['stage'],fold=v['fold'],repeat=v['repeat'],action=b['action'],duplicate=b['duplicate'],exposure=b['exposure'],exposure_minus_action0=b['exposure_minus_action0'],parameter_difference=b['parameter_difference'],parameter_distinguishable=b['parameter_difference']['L2']>max(10*noise,1e-7),risk_distinguishable=resp>max(10*risknoise,1e-6),macro_stats=b['stats']))
stage=[]
for s in range(6):
 vals=[v for v in diagnostics if v['stage']==s];stage.append(dict(stage=s,BA_gain=sum(v['forward_selection']['BA']['gain_vs_uniform'] for v in vals)/6,tail_gain=sum(v['forward_selection']['tail_BA']['gain_vs_uniform'] for v in vals)/6))
strata=sum(sum(v['rank_tau_b'] is not None and v['rank_tau_b']>0 for v in diagnostics if v['stage']==s and v['fold']==f)>=2 for s in range(6) for f in [0,1]);BA=sum(v['BA_gain'] for v in stage)/6;tail=sum(v['tail_gain'] for v in stage)/6;gate=BA>0 and tail>0 and sum(v['BA_gain']>0 and v['tail_gain']>0 for v in stage)>=5 and strata>=9
summary=dict(blocks=36,stage_results=stage,mean_QAselect_QB_BA_gain_vs_uniform=BA,mean_QAselect_QB_tail_gain_vs_uniform=tail,positive_rank_strata=strata,necessary_diagnostic_pattern_supported=gate,not_independent_confirmation=True,unsupported_tail_fold1_class1=True,new_model_scores=0,new_optimizer_steps=0)
write(r/'results/ANALYSIS.json',summary);write(r/'results/BLOCK_DIAGNOSTICS.json',diagnostics);write(r/'results/ACTION_RESPONSES.json',actionrows)
(r/'RESULTS_ZH.md').write_text('# R82 TRAIN-only 固定状态动作—奖励诊断\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n完整Q保留所有类；fold1 class1不能双查询检验。所有端点固定、随机流为条件重复，非新seed/独立患者/VAL确认。原R80/R81负结果保持；详细正负诊断、费用与覆盖解释须在公开交付前补全。\n');print(json.dumps(summary))
