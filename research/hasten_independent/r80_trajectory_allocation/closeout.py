"""Saved-only scientific interpretation and safe release aggregates."""
import json,os,math,statistics,io,csv
from pathlib import Path
from fractions import Fraction
import torch
from common import write
from decision import ARMS
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);assert json.loads((r/'results/VERIFICATION.json').read_text())['passed'];d=r/'HK';analysis=json.loads((r/'results/ANALYSIS.json').read_text());counts=json.loads((r/'results/ALL_CLASS_COUNTS.json').read_text());ctx=json.loads((d/'CONTEXT_RECORDS.json').read_text());records=json.loads((d/'TRAIN_RECORDS.json').read_text());agg=json.loads((d/'aggregate.json').read_text());hist=json.loads((r/'HISTORICAL_SFT_COUNTS.json').read_text());tail=[c for c,n in enumerate(agg['train_counts']) if n<250]
metrics={a:[v['metrics'][a] for v in agg['results']] for a in (*ARMS,'frozen_initial')};metrics['historical_stagewise_sft']=hist
contributions=[]
for control in metrics:
 if control=='protected_controller':continue
 for stage in range(6):
  a=metrics['protected_controller'][stage];b=metrics[control][stage];C=len(a['counts']);T=sum(c<C for c in tail)
  for c,(ka,kb,n) in enumerate(zip(a['correct'],b['correct'],a['counts'])):
   contributions.append(dict(control=control,stage=stage,class_index=c,TRAIN_count=agg['train_counts'][c],VAL_count=n,primary_correct=ka,control_correct=kb,delta_correct=ka-kb,macro_contribution_pp=100*(ka-kb)/n/C,tail_contribution_pp=100*(ka-kb)/n/T if c in tail else 0.,tail=c in tail))
write(r/'results/CLASS_CONTRIBUTIONS.json',contributions)
exposure=[];reward_rows=[];policy={}
for arm in ARMS:
 rows=[v for v in ctx if v['arm']==arm];sums=[0]*23;alls=[0]*23;action_n=tail_action=balanced_action=0;stds=[];kls=[];clips=[]
 for v in rows:
  for c,x in enumerate(v['committed_exposure']):sums[c]+=x
  for c,x in enumerate(v['all_exposure']):alls[c]+=x
  for br in v['branches'][:8]:
   for a in br['actions']:
    action_n+=1;tail_action+=int(a>0 and a-1 in tail);balanced_action+=int(a==0)
  stds.append(statistics.pstdev(v['absolute_returns']));kls.extend(x['controller_KL'] for x in v['policy_stats']);clips.extend(x['clipped_fraction'] for x in v['policy_stats'])
  reward_rows.append({k:v[k] for k in ['arm','stage','context','retained_branch','utilities','damage','absolute_returns','advantages','assigned_returns','retained_absolute_return','retained_advantage','all_candidate_returns_negative','actual_updates','committed_updates','all_exposure','committed_exposure','policy_stats','reward_used_for_commit_selection']})
 for c in range(23):exposure.append(dict(arm=arm,class_index=c,TRAIN_count=agg['train_counts'][c],all_branch_exposures=alls[c],retained_exposures=sums[c],tail=c in tail))
 policy[arm]=dict(retained_tail_exposure_fraction=sum(sums[c] for c in tail)/sum(sums),all_branch_tail_exposure_fraction=sum(alls[c] for c in tail)/sum(alls),decision_entries=action_n,sampled_decisions=(action_n if arm in ['macro_controller','protected_controller','shuffled_controller'] else 0),priority_tail_decisions=tail_action,balanced_decisions=balanced_action,mean_reward_population_std=statistics.mean(stds),zero_reward_std_contexts=sum(x==0 for x in stds),mean_controller_KL=None if not kls else statistics.mean(kls),mean_clipped_fraction=None if not clips else statistics.mean(clips))
 if arm in ['macro_controller','protected_controller','shuffled_controller']:
  first=torch.load(d/arm/'74002/before0.private.pt',map_location='cpu',weights_only=False);last=torch.load(d/arm/'74002/head5.private.pt',map_location='cpu',weights_only=False);policy[arm]['controller_parameter_delta_L2']=math.sqrt(sum(float((last['controller_state'][k]-v).double().square().sum()) for k,v in first['controller'].items()))
actions={a:{(v['stage'],v['context']):[x for br in v['branches'][:8] for x in br['actions']] for v in ctx if v['arm']==a} for a in ['macro_controller','protected_controller','shuffled_controller']};matches={}
for a,b in [('protected_controller','macro_controller'),('protected_controller','shuffled_controller')]:
 pairs=[(x,y) for key,aa in actions[a].items() for x,y in zip(aa,actions[b][key])];matches[a+'_vs_'+b]=dict(same=sum(x==y for x,y in pairs),total=len(pairs),fraction=sum(x==y for x,y in pairs)/len(pairs))
write(r/'results/EXPOSURE_BY_CLASS.json',exposure);write(r/'results/REWARD_AND_EXPOSURE_CURVES.json',reward_rows);write(r/'results/POLICY_DIAGNOSTICS.json',dict(per_arm=policy,action_matches=matches,interpretation='Action agreement is under coupled random numbers, not independent replication or proof of a cause. Policy KL is to uniform legal actions, not visual drift.'))
exit=json.loads((r/'CONTROL_EXIT.json').read_text());cleanup=json.loads((d/'REGENERABLE_CACHE_CLEANUP.json').read_text());cost=dict(**exit,cleanup=cleanup,per_arm_stage_wall_seconds=[dict(arm=v['arm'],stage=v['stage'],wall_seconds=v['wall_seconds']) for v in records],hardware_roles={'GPU0':'visual learner','GPU1':'frozen prefix'},CPU_cores=4,visual_work_matching='same numbers of inner updates and full-S/Q passes; controller/anchor/cache overhead measured separately; not equal GPU seconds',first_arm_pays_cold_prefix_cost=True,actual_controller_updates=576)
write(r/'results/COST.json',cost)
summary=[]
for control in ['balanced_short','macro_controller','shuffled_controller','balanced_long','historical_stagewise_sft']:
 rr=[v for v in contributions if v['stage']==5 and v['control']==control and v['tail']];summary.append(dict(control=control,tail_delta_pp=sum(v['tail_contribution_pp'] for v in rr),classes_with_gain=sum(v['delta_correct']>0 for v in rr),classes_with_loss=sum(v['delta_correct']<0 for v in rr),all_tail_classes=rr))
write(r/'results/TAIL_ATTRIBUTION.json',summary)
print(json.dumps(dict(comparisons=analysis['comparisons'],diagnostics=analysis['diagnostics'],policy=policy,action_matches=matches,tail_attribution=summary),ensure_ascii=False))
