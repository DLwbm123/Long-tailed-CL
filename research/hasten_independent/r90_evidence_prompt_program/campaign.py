import json,os,time,traceback
from pathlib import Path
import torch
import counts
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
try:
 import software,preflight
 write(r/'STATUS.json',{'phase':'SOFTWARE'})
 if (r/'SOFTWARE_CPU_PROCESS.private.json').exists():
  receipt=json.loads((r/'SOFTWARE_CPU_PROCESS.private.json').read_text());assert receipt['passed'] and receipt['core']['passed'];counts.C['toy_controller_optimizer_steps']=receipt['core']['toy_controller_optimizer_steps'];write(r/'SOFTWARE_REUSED.private.json',{'passed':True,'new_toy_steps':0,'external_CPU_cost':receipt})
 else:software.run(cfg)
 write(r/'STATUS.json',{'phase':'NATIVE'});native,base,tf=preflight.run(cfg)
 if not native['fixed_plan_admitted']:
  write(r/'ENGINEERING_REJECTED.json',{'reason':'FIXED_PLAN_DOES_NOT_FIT_ORIGINAL_CLOCK','native':native,'counts':dict(counts.C)});write(r/'STATUS.json',{'phase':'NOT_ADMITTED_ENGINEERING','counts':dict(counts.C)})
 else:
  counts.MODE='TRAIN';import experiment,evaluate,verify,analyze
  dictionary=experiment.train(cfg,base,tf);counts.MODE='VAL';evaluate.run(cfg,base,tf,dictionary);verify.run(cfg);analyze.run(cfg);write(r/'STATUS.json',{'phase':'COMPLETE'})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'STATUS.json',{'phase':'FAILED','counts':dict(counts.C)});raise
