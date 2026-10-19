import os,json,time,resource,traceback
from pathlib import Path
import numpy as np
import torch
from solver import selfcheck
from evaluation import evaluate
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
resource.setrlimit(resource.RLIMIT_FSIZE,(512*2**20,512*2**20));torch.set_num_threads(4)
def write(directory,name,obj):
 s=json.dumps(obj,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)));f=directory/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(s);t.replace(f)
def guarded(action,directory):
 directory.mkdir(exist_ok=True,parents=True);start=time.monotonic();code=1
 try:action();code=0
 except BaseException as error:
  write(directory,'FAILURE.json',dict(error=type(error).__name__,phase=cfg['mode'],no_silent_retry=True));(directory/'FAILURE.private.txt').write_text(traceback.format_exc())
 finally:
  u=resource.getrusage(resource.RUSAGE_SELF);write(directory,'WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0));write(directory,'EXIT.json',dict(exit_code=code))
 return code

def preflight():
 result=selfcheck();ty=torch.arange(6).repeat_interleave(4);groups=['s'+str(i//2) for i in range(24)];complete=[a+'_'+v for a in ['G','P','C'] for v in ['Linear','Poly2','RBF']]
 for perfect in [True,False]:
  scores={}
  for st in range(3):
   k=st+4
   for a in ['G','P','C']:
    for variant in ['ridge','CE','S','B','Borda','Hodge','Vote','Linear','Poly2','RBF']:
     name=a+'_'+variant;target=ty%k if perfect and variant=='RBF' else (ty+1)%k;scores[name,st]=torch.nn.functional.one_hot(target,k).double()
  got=evaluate(scores,ty,groups,complete);assert len(got['comparisons'])==3 and sum(len(v['vs_ridge']) for v in got['comparisons'].values())==15
  assert all(bool(v['pass_all'])==perfect for v in got['gates'].values());write(out,'SYNTHETIC_'+str(perfect)+'.json',got)
 # Missing numerical controls must remain unresolved instead of crashing or fabricating a gate.
 missing=[n for n in complete if n!='G_Linear'];partial_scores={key:v for key,v in scores.items() if key[0]!='G_Linear'}
 unresolved=evaluate(partial_scores,ty,groups,missing);assert unresolved['decision']=='UNRESOLVED_OPTIMIZATION_NO_EXTENSION'
 injected=out/'injected'
 def fail():write(injected,'checkpoint.json',dict(saved=True));raise RuntimeError('injected synthetic failure')
 assert guarded(fail,injected)==1 and json.loads((injected/'EXIT.json').read_text())['exit_code']==1 and (injected/'WORKER_RESOURCE_RECEIPT.json').exists() and json.loads((injected/'checkpoint.json').read_text())['saved']
 result.update(full_nine_arm_aggregate_and_15_comparisons=True,JSON_nine_arm_gate_roundtrip=True,failure_receipt=True);write(out,'PRECHECK.json',result)
def action():
 if cfg['mode']=='PREFLIGHT':preflight()
 else:
  import pipeline
  pipeline.main(cfg)
raise SystemExit(guarded(action,out))
