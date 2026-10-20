import os,sys,json
from pathlib import Path

def main(cfg):
 ref=Path(cfg['reference']).resolve();matched=Path(cfg['matched']).resolve();previous=Path(cfg['previous']).resolve();bc=json.loads(Path(cfg['benchmark_config']).read_text());testroot=Path(bc['output']).resolve();raw=[Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images','development_images']]
 allowed={previous/'fit/aggregate.json',previous/'train/ENCODER_SEAL.json'}|{previous/'train'/f'{arm}.encoder.private.pt' for arm in ['E','I','O']}|{previous/'fit'/f'{arm}.0.state.private.pt' for arm in ['E','I','O']}
 def guard(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  p=Path(os.fsdecode(args[0])).resolve();assert not any(p.is_relative_to(r) for r in raw),'raw image forbidden';assert not p.is_relative_to(testroot),'test forbidden'
  if p.is_relative_to(ref/'inputs'):assert p==ref/'inputs/0.private.pt','future/retired stage forbidden'
  if p.is_relative_to(matched):assert p.is_relative_to(matched/'pixels_0'),'future/test/old predictions forbidden'
  if p.is_relative_to(previous):assert p in allowed,'unapproved previous asset'
 sys.addaudithook(guard)
 for bad in [ref/'inputs/1.private.pt',matched/'pixels_3/INDEX.json',previous/'fit/predictions.private.pt']:
  try:guard('open',(str(bad),'rb'))
  except AssertionError:pass
  else:raise AssertionError('guard failure')
 if cfg['mode']=='INJECT_FAILURE':raise RuntimeError('synthetic injected failure')
 from audit import synthetic,run,write
 if cfg['mode']=='PREFLIGHT':write(Path(cfg['output'])/'aggregate.json',dict(pass_all=True,**synthetic()));return
 run(cfg)
