import json,sys,os
from pathlib import Path

def main(cfg):
 access={'stage':0,'sealed_test':False};ref=Path(cfg['reference']).resolve();matched=Path(cfg['matched']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());testroot=Path(bc['output']).resolve();prior=Path(cfg['inherited_predictions']).resolve();raw=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve()
  assert not any(p.is_relative_to(r) for r in raw),'raw decoding forbidden'
  if p.is_relative_to(ref/'inputs'):assert access['stage']<3 and p==ref/'inputs'/f"{access['stage']}.private.pt",'retired/future support forbidden'
  if p.is_relative_to(matched) and p!=prior:assert p.is_relative_to(matched/f"pixels_{access['stage']}"),'retired/future pixels forbidden'
  if p==prior or p.is_relative_to(testroot):assert access['sealed_test'],'test before complete seal'
 sys.addaudithook(guard)
 for bad in [ref/'inputs/1.private.pt',prior,matched/'pixels_3/INDEX.json']:
  try:guard('open',(str(bad),'rb'))
  except AssertionError:pass
  else:raise AssertionError('guard selfcheck failed')
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('synthetic injected failure')
 if cfg['mode']=='GRAD_PROBE':
  from grad_probe import run
  run(cfg);return
 from train import run as train,write
 if cfg['mode']=='PREFLIGHT':
  train(cfg);from pipeline import synthetic;a=synthetic();write(Path(cfg['output'])/'EVALUATION_PREFLIGHT.json',a);return
 if cfg['mode']=='TRAIN':train(cfg);return
 from pipeline import run
 run(cfg,access)
