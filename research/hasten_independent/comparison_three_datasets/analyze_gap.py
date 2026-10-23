import json,statistics as st
from pathlib import Path
p=Path(__file__).parent;h=p.parent/'r12_three_datasets';a=json.loads((h/'results/ANALYSIS.json').read_text());t=json.loads((h/'results/TAIL_TRAJECTORIES.json').read_text());old=json.loads((p/'results/RESULTS.json').read_text());nihold=json.loads((p/'results/NIH_OLD_STAGE_METRICS.json').read_text());out={}
for ds in ['NIH','ISIC','HK']:
 rr=json.loads((h/('NIH_REUSED_RESULTS.json' if ds=='NIH' else f'results/{ds}_aggregate.json')).read_text())['results'];first=st.mean(v['metrics']['final']['BA']*100 for v in rr if v['stage']==0)
 oldfirst=nihold[0]['metrics']['df_matched_joint']['macro']*100 if ds=='NIH' else next(v for v in old if v['dataset']==ds and v['method']=='cf_group')['stages'][0]['BA']
 x=[v for v in t if v['dataset']==ds];last=max(v['stage'] for v in x);entry=[v for v in x if v['stage']==v['arrival']];fin=[v for v in x if v['stage']==last]
 macro=lambda z,key:100*st.mean(v[key]/v['eval_count'] for v in z)
 m=a[ds]['summary'];out[ds]=dict(initial_old=oldfirst,initial_hasten=first,initial_gap=first-oldfirst,final_prototype=m['prototype']['final_BA']['mean'],final_hyper=m['hyper']['final_BA']['mean'],final_fused=m['final']['final_BA']['mean'],hyper_minus_prototype=m['hyper']['final_BA']['mean']-m['prototype']['final_BA']['mean'],fused_minus_hyper=m['final']['final_BA']['mean']-m['hyper']['final_BA']['mean'],allclass_arrival=macro(entry,'final_correct'),allclass_final=macro(fin,'final_correct'),allclass_fixed_arrival_candidates_final=macro(fin,'final_fixed_correct'))
(p/'results/GAP_DECOMPOSITION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
