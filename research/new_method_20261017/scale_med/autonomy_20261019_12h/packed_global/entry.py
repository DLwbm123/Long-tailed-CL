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
 result=selfcheck();ty=torch.arange(6).repeat_interleave(4);groups=['s'+str(i//2) for i in range(24)];complete=[a+'_'+v for a in ['G','P','C'] for v in ['Full','Scale']]
 for perfect in [True,False]:
  scores={}
  for st in range(3):
   k=st+4
   for name in complete+[a+'_ridge' for a in ['G','P','C']]:
    target=ty%k if perfect and name=='C_Full' else (ty+1)%k;scores[name,st]=torch.nn.functional.one_hot(target,k).double()
  got=evaluate(scores,ty,groups,complete);assert len(got['comparisons'])==1 and len(got['comparisons']['C_Full']['vs_ridge'])==6
  assert got['gates']['C_Full']['pass_all']==perfect;write(out,'SYNTHETIC_'+str(perfect)+'.json',got)
 assert evaluate(scores,ty,groups,complete[:-1])['decision']=='UNRESOLVED_OPTIMIZATION_NO_EXTENSION'
 # Full and scale-only deployment map must match reconstructed frozen extras.
 from solver import mapper,PERM
 torch.manual_seed(74158);g=torch.randn(9,1536,dtype=torch.float64);h=torch.randn(9,768,dtype=torch.float64)
 for arm in ['G','P','C']:
  d={'G':1535,'P':1375,'C':1318}[arm];n=1536+(128 if arm!='G' else 0);t=dict(arm=arm,d=d,g_norm=(g.mean(0),torch.tensor(2.)),out_norm=(torch.zeros(n,dtype=torch.float64),torch.tensor(3.)))
  if arm!='G':t.update(h_norm=(h.mean(0),torch.tensor(4.)),U=torch.randn(768,128,dtype=torch.float64))
  if arm=='C':t['V']=torch.randn(d,128,dtype=torch.float64)
  full=mapper(g,h,t);idx=torch.cat([torch.arange(d),torch.arange(1536,n)]);t.update(scale_only=True,out_norm=(t['out_norm'][0][idx].clone(),t['out_norm'][1]));scale=mapper(g,h,t);assert torch.equal(scale[:,:-1],full[:,idx])
 injected=out/'injected'
 def fail():write(injected,'checkpoint.json',dict(saved=True));raise RuntimeError('injected synthetic failure')
 assert guarded(fail,injected)==1 and (injected/'WORKER_RESOURCE_RECEIPT.json').exists() and (injected/'checkpoint.json').exists()
 result.update(full_aggregate_six_comparisons_JSON=True,deploy_mapper_matches_scale_control=True,failure_receipt=True);write(out,'PRECHECK.json',result)
def action():
 if cfg['mode']=='PREFLIGHT':preflight()
 else:
  import pipeline
  pipeline.main(cfg)
raise SystemExit(guarded(action,out))
