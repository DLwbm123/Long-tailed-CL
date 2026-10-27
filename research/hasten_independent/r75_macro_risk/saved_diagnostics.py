"""Saved aggregate arithmetic only, no model/image scoring."""
from pathlib import Path
import json,os
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);out=root/'results';recs=json.loads((root/'HK/TRAIN_RECORDS.json').read_text());a=json.loads((root/'HK/aggregate.json').read_text());groups={}
for arm in ['tars_grpo','macro_grpo','tars_exact','macro_exact']:
 rr=[v for v in recs if v['arm']==arm];ss={k:sum(v['statistics'][k] for v in rr) for k in rr[0]['statistics']};N=sum(v['query_rows'] for v in rr);d=dict(mean_TRAIN_KL=ss['KL_sum']/N,query_mean_weight=ss['mean_weight_sum']/N,query_rows=N,statistics=ss)
 if ss['groups']:d.update(zero_group_fraction=ss['zero_advantage_groups']/ss['groups'],reward_fraction=ss['reward_sum']/ss['actions'],clip_fraction=ss['clipped_actions']/ss['actions'])
 groups[arm]=d
last=a['results'][-1];counts=last['metrics']['macro_grpo']['counts'];tail=[i for i,n in enumerate(a['train_counts']) if n<250];delta=[]
for i,n in enumerate(counts):
 m=last['metrics']['macro_grpo']['correct'][i];b=last['metrics']['sft_reference']['correct'][i];t=last['metrics']['tars_grpo']['correct'][i]
 delta.append(dict(class_index=i,TRAIN_count=a['train_counts'][i],VAL_count=n,is_tail=i in tail,macro_correct=m,SFT_correct=b,TARS_correct=t,delta_correct_vs_SFT=m-b,delta_class_BA_pp=100*(m-b)/n,tail_BA_contribution_pp=100*(m-b)/n/len(tail) if i in tail else None))
comp=json.loads((out/'COMPARISONS.json').read_text())['HK']['SFT']['mean_delta_pp'];assert abs(sum(v['delta_class_BA_pp'] for v in delta)/len(delta)-comp[1])<1e-10;assert abs(sum(v['tail_BA_contribution_pp'] for v in delta if v['is_tail'])-comp[2])<1e-10
for name,value in [('TRAINING_DIAGNOSTICS_SUMMARY.json',groups),('FINAL_CLASS_DECOMPOSITION.json',delta)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
print('Saved diagnostics identities passed; 0 new models/images/optimization.')
