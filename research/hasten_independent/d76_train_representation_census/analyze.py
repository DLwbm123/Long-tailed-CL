from pathlib import Path
import os,json,statistics
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);assert json.loads((r/'results/VERIFICATION.json').read_text())['passed'];rows=json.loads((r/'HK/CLASS_SUMMARIES.json').read_text());result=[]
for row in rows:
 v=row['classes'];tails=[c for c in v if c['count']<250]
 a=dict(stage=row['stage'],model=row['model'],seen=row['seen'],TRAIN_rows=sum(c['count'] for c in v),class_equal_mean_embedding_norm=statistics.mean(c['mean_embedding_norm'] for c in v),class_equal_within_trace=statistics.mean(c['unbiased_within_trace'] for c in v),class_equal_nearest_centroid_cosine=statistics.mean(c['nearest_other_centroid_cosine'] for c in v))
 for mode in ['raw','natural']:
  a[mode]=dict(TRAIN_crossfold_macro_BA=100*statistics.mean(c[mode]['correct']/c['count'] for c in v),TRAIN_crossfold_tail_BA=100*statistics.mean(c[mode]['correct']/c['count'] for c in tails),class_equal_endpoint_p=statistics.mean(c[mode]['correct_probability_mean'] for c in v),class_equal_hypothetical_h=statistics.mean(c[mode]['endpoint_hypothetical_snapshot_h_mean'] for c in v))
 result.append(a)
for name,value in [('FULL_CLASS_SUMMARIES.json',rows),('SUMMARY.json',result),('CENSUS_RECORDS.json',json.loads((r/'HK/CENSUS_RECORDS.json').read_text()))]:(r/'results'/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(result,indent=2))
