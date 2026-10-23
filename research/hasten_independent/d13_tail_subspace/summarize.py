import json,statistics as st
from pathlib import Path
root=Path(__file__).parent;p=root/'results'
def read(n):return json.loads((p/n).read_text())
t=read('TRAJECTORIES.json');b=read('BRANCH_TRANSITIONS.json');q=read('COVERAGE.json');d=read('DRIFT.json');s=read('SPECTRA.json')
u=[v for v in t if v['arm']=='H-unit'];tail=sorted({v['class_index'] for v in u if v['tail']});rows=[]
for c in tail:
 r=[v for v in u if v['class_index']==c];first=[v for v in r if v['stage']==v['arrival']];last=[v for v in r if v['stage']==5]
 rows.append(dict(class_index=c,name=r[0]['name'],train_count=r[0]['train_count'],arrival=r[0]['arrival'],first=sum(v['final_correct'] for v in first),last=sum(v['final_correct'] for v in last),last_fixed=sum(v['final_arrival_candidates_correct'] for v in last),denominator=45))
last=[v for v in b if v['arm']=='H-unit' and v['tail'] and v['stage']==5]
branches={k:sum(v[k] for v in last) for k in ['prototype_correct','hyper_correct','final_correct','prototype_to_fusion_rescue','prototype_to_fusion_harm','hyper_to_fusion_rescue','hyper_to_fusion_harm','both_wrong']}
summary=dict(tail_trajectories=rows,first=sum(v['first'] for v in rows),last=sum(v['last'] for v in rows),last_fixed=sum(v['last_fixed'] for v in rows),denominator=315,final_tail_branches=branches,coverage={},drift={},decision='DEFER_DISCRIMINATIVE_PROJECTION_TRAINING',next='Complete fixed R12 three-dataset assessment; focus follow-up diagnostics on initial recognition and candidate competition; no validation-fitted fusion or rank search.')
for istail in [False,True]:
 r=[v['q'] for v in q if v['tail']==istail and v['nearest_train_mean'] and v['q'] is not None]
 summary['coverage']['tail' if istail else 'non_tail']=dict(median=st.median(r),mean=st.mean(r),n=len(r))
 r=[v for v in d if v['tail']==istail];summary['drift']['tail' if istail else 'non_tail']=dict(mapping_mean_RMS=st.mean(v['mapping_RMS'] for v in r),adapter_mean_RMS=st.mean(v['adapter_RMS'] for v in r),adapter_greater_count=sum(v['adapter_RMS']>v['mapping_RMS'] for v in r),n=len(r))
summary['spectrum']=dict(protected_ranks=sorted({v['protected_rank'] for v in s}),top_squared_fraction_range=[min(v['top_squared_fraction'] for v in s),max(v['top_squared_fraction'] for v in s)],linear95_rank_range=[min(v['linear_95_rank'] for v in s),max(v['linear_95_rank'] for v in s)])
assert summary['first']==45 and summary['last']==38 and summary['last_fixed']==44
assert branches['prototype_correct']+branches['prototype_to_fusion_rescue']-branches['prototype_to_fusion_harm']==38
assert branches['hyper_correct']+branches['hyper_to_fusion_rescue']-branches['hyper_to_fusion_harm']==38
(p/'ANALYSIS.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
