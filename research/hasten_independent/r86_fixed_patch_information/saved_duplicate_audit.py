"""Compare only already saved predictions/logits; no classifier or encoder forward."""
import os,json,time,resource
from pathlib import Path
import torch
start=time.time();cpu=time.process_time();torch.set_num_threads(4)
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);old=Path(cfg['feature_source_run']);rows=[]
for stage in range(6):
 for fold in range(2):
  name=f'scores_s{stage}f{fold}.private.pt';a=torch.load(r/name,map_location='cpu',weights_only=False);b=torch.load(old/name,map_location='cpu',weights_only=False)
  assert a['ids']==b['ids'] and a['Q']==b['Q'] and torch.equal(a['labels'],b['labels'])
  x=a['arms']['duplicate'];y=b['arms']['balanced'];rows.append(dict(stage=stage,fold=fold,max_abs_saved_logit_difference=float((x['logits']-y['logits']).abs().max()),prediction_differences=int((x['prediction']!=y['prediction']).sum())))
out=dict(saved_only=True,new_model_scores=0,new_optimizer_steps=0,new_VAL_TEST_reads=0,cells=rows,max_abs_saved_logit_difference=max(v['max_abs_saved_logit_difference'] for v in rows),prediction_differences=sum(v['prediction_differences'] for v in rows),CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(r/'DUPLICATE_SAVED_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
