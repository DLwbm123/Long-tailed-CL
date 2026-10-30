"""Supplemental source-index binding, only saved tensors; no forward calls."""
import os,time,resource,json,fcntl
from pathlib import Path
start=time.time();cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20));os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]))
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);s=Path(cfg['feature_source_run'])
lock=Path(cfg['shared_memory_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert time.time()<json.loads(Path(cfg['clock']).read_text())['deadline_unix']-90
for stage in range(6):
 feature=torch.load(s/f'stage{stage}.private.pt',map_location='cpu',weights_only=False)
 for fold in [0,1]:
  head=torch.load(r/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
  scores=torch.load(r/f'scores_s{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
  original=torch.load(s/f'scores_s{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
  Q=head['Q'];assert head['source_feature_ids']==feature['ids'];assert scores['ids']==[feature['ids'][i] for i in Q]==original['ids']
  assert torch.equal(scores['labels'],feature['labels'][Q]) and torch.equal(scores['labels'],original['labels'])
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<1_000_000_000
v=dict(passed=True,cells=12,source_feature_ids_and_query_labels_bound=True,new_forward_calls=0,new_optimizer_steps=0,
 CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(r/'SUPPLEMENTAL_VERIFICATION.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
