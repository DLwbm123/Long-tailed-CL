import os,json,time,subprocess,traceback
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
try:
 for script,label in [('software','SOFTWARE'),('preflight','NATIVE')]:
  write(r/'STATUS.json',{'phase':label});m=__import__(script);result=m.run(cfg);write(r/(label+'_RECEIPT.json'),result)
 write(r/'STATUS.json',{'phase':'TRAIN'})
 import train
 base,tf=train.train(cfg);train.evaluate(cfg,base,tf)
 import verify,analyze
 verify.run(cfg);analyze.run(cfg);write(r/'STATUS.json',{'phase':'COMPLETE'})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'STATUS.json',{'phase':'FAILED'});raise
