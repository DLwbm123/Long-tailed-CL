import os,json
from pathlib import Path
r=Path(os.environ['RUN_ROOT']);z=json.loads((r/'RESULTS.json').read_text());eligible=[];summary={}
for d,v in z.items():
 assert v['optimizer_steps']==v['VAL_reads']==v['TEST_reads']==0 and v['passed'] and len(v['summary'])==3
 assert all(c['disjoint_TRAIN_rows'] and c['parameters_unchanged'] and sum(c['counts'])==64 for c in v['cells'])
 assert all(any(c in cell['represented_tail_classes'] for cell in v['cells'] if cell['seed']==seed) for c in v['initial_tail_classes'] for seed in [74002,74003,74004])
 summary[d]={k:sum(vv[k] for vv in v['summary'])/3 for k in ['macro_cos_original','macro_cos_candidate','tail_cos_original','tail_cos_candidate','macro_cos_delta','tail_cos_delta']}
 if d=='NIH':eligible=[p['seed'] for p in v['summary'] if p['macro_cos_candidate']>0 and p['tail_cos_candidate']>0 and p['macro_cos_delta']>=.01 and p['tail_cos_delta']>=.01]
other=all(summary[d][k]>=-.01 for d in ['ISIC','HK'] for k in ['macro_cos_delta','tail_cos_delta'])
res=dict(passed=len(eligible)>=2 and other,NIH_eligible_seeds=eligible,other_domains_direction_noharm=other,summary=summary,not_AdamW_or_accuracy_guarantee=True)
(r/'FEASIBILITY.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
