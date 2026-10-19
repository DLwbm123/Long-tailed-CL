"""Qualified original CPU reader, explicitly staged by the sealed parent."""
import os,json,sys,time,resource,traceback
from pathlib import Path
import torch
from canonical_reader import require_canonical_reader
from space import check_space
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());stage=int(os.environ['READ_STAGE']);out=Path(cfg['output'])/('pixels_'+str(stage));out.mkdir(parents=True,exist_ok=True)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
code=1;start=time.monotonic()
try:
 torch.set_num_threads(2);signature=require_canonical_reader()
 if stage==-1:
  torch.save(torch.arange(12,dtype=torch.float32).reshape(3,4),out/'bridge.private.pt');write('PROBE.json',dict(pass_all=True,torch=torch.__version__,signature=signature));write('EXIT.json',dict(exit_code=0,CPU_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,wall_seconds=time.monotonic()-start));raise SystemExit(0)
 bc=json.loads(Path(cfg['benchmark_config']).read_text());ref=Path(cfg['reference']).resolve();testroot=Path(bc['output']).resolve();legacy=Path(bc['legacy_source']);sys.path.insert(0,str(legacy/'tools'))
 from run_prototype_single import Images
 from run_medical_v2 import transform
 if stage==3:
  assert (Path(cfg['output'])/'MODEL_SEAL.json').exists(),'test before model seal'
  rr=json.loads((testroot/'ISIC.test.rows.private.json').read_text());image_root=Path(bc['datasets']['ISIC']['images']).resolve()
 else:
  obj=torch.load(ref/'inputs'/f'{stage}.private.pt',weights_only=False);rr=obj['rows'];image_root=Path(bc['datasets']['ISIC']['development_images']).resolve();del obj
 allowed={(image_root/r['relative_path']).resolve() for r in rr}
 def guard(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(image_root):assert p in allowed,'unlisted image'
   if p.is_relative_to(ref/'inputs'):raise AssertionError('reader bundle reopened')
 sys.addaudithook(guard);ds=Images(rr,str(image_root),transform(False),74002);chunks=[]
 for first in range(0,len(rr),64):
  count=min(64,len(rr)-first);check_space(cfg['window_root'],reserve=count*3*224*224*4*3+1048576)
  x=torch.stack([ds[i][0] for i in range(first,first+count)]);assert x.shape==(count,3,224,224) and torch.isfinite(x).all()
  fn=out/f'{first:05d}.private.pt';torch.save(x,fn);chunks.append(dict(first=first,rows=count,file=fn.name));assert fn.stat().st_size<512*2**20
 write('INDEX.json',dict(stage=stage,rows=len(rr),chunks=chunks,signature=signature,reader_only=True));code=0
except SystemExit as e:
 code=int(e.code)
except BaseException as e:
 write('FAILURE.json',dict(error=type(e).__name__));(out/'FAILURE.private.txt').write_text(traceback.format_exc())
finally:
 u=resource.getrusage(resource.RUSAGE_SELF);write('EXIT.json',dict(exit_code=code,CPU_seconds=u.ru_utime+u.ru_stime,wall_seconds=time.monotonic()-start,peak_RSS_bytes=u.ru_maxrss*1024))
raise SystemExit(code)
