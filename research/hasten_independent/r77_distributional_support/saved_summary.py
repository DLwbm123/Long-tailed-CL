from pathlib import Path
import json,os
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);out=root/'results';records=json.loads((root/'HK/TRAIN_RECORDS.json').read_text());a=json.loads((root/'HK/aggregate.json').read_text());d={}
for arm in ['prototype_grpo','distribution_grpo','prototype_exact','distribution_exact']:
 rr=[v for v in records if v['arm']==arm];ss={k:sum(v['statistics'][k] for v in rr) for k in rr[0]['statistics']};N=sum(v['query_rows'] for v in rr);v=dict(mean_TRAIN_KL=ss['KL_sum']/N,query_mean_weight=ss['mean_weight_sum']/N,query_rows=N,statistics=ss)
 if ss['groups']:v.update(zero_group_fraction=ss['zero_advantage_groups']/ss['groups'],reward_fraction=ss['reward_sum']/ss['actions'],clip_fraction=ss['clipped_actions']/ss['actions'])
 d[arm]=v
decomp=[]
for row in a['results']:
 m=row['metrics'];tail=[i for i,c in enumerate(a['train_counts'][:row['seen']]) if c<250]
 for i,n in enumerate(m['distribution_grpo']['counts']):
  names=['sft_reference','distribution_sft_reference','prototype_grpo','distribution_grpo','prototype_exact','distribution_exact'];v={k:m[k]['correct'][i] for k in names};decomp.append(dict(stage=row['stage'],class_index=i,TRAIN_count=a['train_counts'][i],VAL_count=n,is_tail=i in tail,correct=v,raw_readout_delta_pp=100*(v['distribution_sft_reference']-v['sft_reference'])/n,RL_delta_vs_distributionSFT_pp=100*(v['distribution_grpo']-v['distribution_sft_reference'])/n,tail_contribution_RL_pp=100*(v['distribution_grpo']-v['distribution_sft_reference'])/n/len(tail) if i in tail else None))
co=json.loads((out/'COMPARISONS.json').read_text())['HK']['DistributionSFT']['mean_delta_pp'];f=[v for v in decomp if v['stage']==5];assert abs(sum(v['RL_delta_vs_distributionSFT_pp'] for v in f)/len(f)-co[1])<1e-10;assert abs(sum(v['tail_contribution_RL_pp'] for v in f if v['is_tail'])-co[2])<1e-10
for name,value in [('TRAINING_DIAGNOSTICS_SUMMARY.json',d),('CLASS_STAGE_DECOMPOSITION.json',decomp)]:(out/name).write_text(json.dumps(value,indent=2)+'\n')
print('Saved arithmetic identities passed, zero new model scoring or optimization.')
