import os,json,traceback
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
try:
 import check
 if (r/'SOFTWARE.json').exists():assert json.loads((r/'SOFTWARE.json').read_text())['passed']
 else:check.run(r/'SOFTWARE.json')
 import preflight
 write(r/'STATUS.json',{'phase':'NATIVE_AND_BUDGET','formal_optimizer':0});passed,prepared=preflight.native_budget(cfg)
 if not passed:write(r/'STATUS.json',{'phase':'NOT_ADMITTED_BUDGET','formal_optimizer':0})
 else:
  assert not (r/'FORMAL_STARTED.private.json').exists();write(r/'FORMAL_STARTED.private.json',{'original_start':cfg['start'],'deadline':cfg['deadline'],'formal_optimizer_before':0})
  import train
  base,tf=train.train(cfg,prepared);train.evaluate(cfg,base,tf)
  import verify,analyze
  verify.run(cfg);analyze.run(cfg);write(r/'STATUS.json',{'phase':'COMPLETE','counts':train.COUNTS})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc())
 try:
  import train,preflight
  write(r/'FAILURE_COUNTS.private.json',{'formal':train.COUNTS,'engineering':preflight.ENGINEERING})
 except BaseException:pass
 write(r/'STATUS.json',{'phase':'FAILED'});raise
