import sys,os,json
from pathlib import Path

def main(cfg):
 ref=Path(cfg['reference']).resolve();matched=Path(cfg['matched']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());testroot=Path(bc['output']).resolve();raw=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']];old=Path(cfg['previous']).resolve();prior=Path(cfg['inherited_predictions']).resolve()
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();assert not any(p.is_relative_to(v) for v in raw),'raw image forbidden';assert not p.is_relative_to(testroot) and p!=prior,'test forbidden';assert not p.is_relative_to(old),'old learned E/I/O checkpoints forbidden'
  if p.is_relative_to(ref/'inputs'):assert p==ref/'inputs/0.private.pt','future support forbidden'
  if p.is_relative_to(matched):assert p.is_relative_to(matched/'pixels_0'),'future/test/matched predictions forbidden'
 sys.addaudithook(guard)
 for bad in [ref/'inputs/1.private.pt',matched/'pixels_3/INDEX.json',prior,old/'train/O.encoder.private.pt']:
  try:guard('open',(str(bad),'rb'))
  except AssertionError:pass
  else:raise AssertionError('guard selfcheck')
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('planned injected failure')
 if cfg['mode']=='SYNTHETIC':
  from protection import synthetic
  from evaluation import synthetic as ev
  from train import write
  write(Path(cfg['output'])/'aggregate.json',dict(pass_all=True,**synthetic(),**ev()));return
 if cfg['mode'] in ['PREFLIGHT','TRAIN']:
  from train import run
  run(cfg)
  if cfg['mode']=='PREFLIGHT':
   from evaluation import synthetic
   from train import write
   write(Path(cfg['output'])/'EVALUATION_PREFLIGHT.json',synthetic())
  return
 from pipeline import run
 run(cfg)
