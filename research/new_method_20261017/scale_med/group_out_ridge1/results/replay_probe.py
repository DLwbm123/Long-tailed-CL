import os,json,time,resource
from pathlib import Path
import torch
from train_base import setup,raw
from train import load_training,schedule
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);start=time.monotonic();torch.set_num_threads(4)
pix,y,groups,cached=load_training(cfg);ids=schedule(y,groups)[0][0];m,p,bk,ck,heads,pw=setup(cfg);result={};vals={}
with torch.no_grad():
 for size in [8,32]:
  legacy=[];manual=[]
  for ii in ids.split(size):
   x=pix[ii].cuda();legacy.append(torch.nn.functional.normalize(m(x).cpu().double(),dim=1));o=raw(m,pw,x);manual.append(torch.nn.functional.normalize(torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1).cpu())
  a,b=torch.cat(legacy),torch.cat(manual);vals[size]=a;result[str(size)]=dict(legacy_vs_cache_max=float((a-cached[ids]).abs().max()),manual_vs_legacy_max=float((a-b).abs().max()),manual_vs_cache_max=float((b-cached[ids]).abs().max()),legacy_cache_row_error_quantiles=torch.quantile((a-cached[ids]).norm(dim=1),torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist())
result['batch8_vs32_max']=float((vals[8]-vals[32]).abs().max());result['CPU_seconds']=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime;result['wall_seconds']=time.monotonic()-start;result['GPU_peak']=torch.cuda.max_memory_allocated();(out/'aggregate.json').write_text(json.dumps(result,indent=2));(out/'EXIT.json').write_text('{"exit_code":0}')
