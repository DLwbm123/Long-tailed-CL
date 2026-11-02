import os,json,traceback,time
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
try:
 write(r/'STATUS.json',{'phase':'NATIVE_EQUIVALENCE','formal_optimizer':0,'VAL_TEST':0})
 import preflight
 base,tf,tok,images,engineering=preflight.native(cfg)
 write(r/'STATUS.json',{'phase':'BUDGET','formal_optimizer':0,'VAL_TEST':0})
 if not preflight.budget(cfg,(base,tf,tok),images,engineering):write(r/'STATUS.json',{'phase':'NOT_ADMITTED_BUDGET','formal_optimizer':0,'VAL_TEST':0})
 else:
  assert not (r/'FORMAL_STARTED.private.json').exists();write(r/'FORMAL_STARTED.private.json',{'original_start':cfg['start'],'original_deadline':cfg['deadline'],'formal_optimizer_before':0,'unchanged_epochs_controls':True});write(r/'STATUS.json',{'phase':'TRAIN','formal_optimizer':0,'VAL_TEST':0})
  import train
  base,tf=train.train(cfg,(base,tf,tok));train.evaluate(cfg,base,tf)
  import verify,analyze
  verify.run(cfg);analyze.run(cfg);write(r/'STATUS.json',{'phase':'COMPLETE','counts':train.COUNTS})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc())
 try:write(r/'ENGINEERING_FAILURE.private.json',{'completed_scope_counts':preflight.ENGINEERING,'active_scope':preflight.ACTIVE,'inflight_forward_may_need_audit':True})
 except BaseException:pass
 try:
  import train
  write(r/'FAILURE_COUNTS.private.json',train.COUNTS)
 except BaseException:pass
 write(r/'STATUS.json',{'phase':'FAILED'});raise
