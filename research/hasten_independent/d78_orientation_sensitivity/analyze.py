from pathlib import Path
import os,json,statistics
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);assert json.loads((r/'results/VERIFICATION.json').read_text())['passed'];rows=json.loads((r/'HK/CLASS_SUMMARIES.json').read_text());result=[]
for row in rows:
 a=dict(stage=row['stage'],model=row['model'],seen=row['seen'],modes={})
 for mode,classes in row['modes'].items():
  v=dict(TRAIN_rows=sum(c['count'] for c in classes),class_equal_paired_embedding_cosine=statistics.mean(c['paired_embedding_cosine_mean'] for c in classes),class_equal_prediction_disagreement_pct=100*statistics.mean(c['prediction_disagreements']/c['count'] for c in classes),original_only_correct_rows=sum(c['original_only_correct'] for c in classes),flipped_only_correct_rows=sum(c['flipped_only_correct'] for c in classes))
  for group,cc in [('all',classes),('tail',[c for c in classes if c['count']<250])]:
   v[group]=dict(classes=len(cc),original_macro_BA=100*statistics.mean(c['original_correct']/c['count'] for c in cc),flipped_macro_BA=100*statistics.mean(c['flipped_correct']/c['count'] for c in cc),original_true_probability_mean=statistics.mean(c['original_true_probability_mean'] for c in cc),flipped_true_probability_mean=statistics.mean(c['flipped_true_probability_mean'] for c in cc))
  a['modes'][mode]=v
 result.append(a)
for name,value in [('FULL_CLASS_SUMMARIES.json',rows),('SUMMARY.json',result),('CENSUS_RECORDS.json',json.loads((r/'HK/CENSUS_RECORDS.json').read_text()))]:(r/'results'/name).write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(saved_only=True,records=len(result),new_model_scores=0)))
