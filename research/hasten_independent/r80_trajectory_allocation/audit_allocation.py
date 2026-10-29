"""Aggregate actual priority-slot influence from saved TRAIN allocation counts."""
import json,os,time,resource,statistics
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);start=time.time();rows=json.loads((r/'HK/CONTEXT_RECORDS.json').read_text());counts=cfg['domains']['HK']['train_counts_ordered'];summary={}
for arm in ['macro_controller','protected_controller','shuffled_controller']:
 groups={'tail':[],'head':[]};availability={'tail':[],'head':[]};distinct=[]
 for v in rows:
  if v['arm']!=arm:continue
  distinct.append(len({tuple(a for q in br['allocation'] for a in q['controlled_counts']) for br in v['branches'][:8]}))
  for br in v['branches'][:8]:
   for a,q in zip(br['actions'],br['allocation']):
    if a:
     key='tail' if counts[a-1]<250 else 'head';groups[key].append(q['priority_allocated']);availability[key].append(q['priority_available']);assert 0<q['priority_available'] and q['priority_allocated']<=min(64,q['priority_available'])
 summary[arm]=dict(priority={k:dict(decisions=len(v),mean_actual_slots=statistics.mean(v),mean_available_unique=statistics.mean(availability[k]),slot64_fully_used=sum(x==64 for x in v)) for k,v in groups.items()},mean_distinct_controlled_class_count_vectors_among_8=statistics.mean(distinct))
write(r/'results/ALLOCATION_INFLUENCE.json',dict(summary=summary,scope='Saved TRAIN class-count aggregates only; no query scores or model inference',wall_seconds=time.time()-start,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,new_model_scores=0,new_optimizer_steps=0));print(json.dumps(summary))
