import os,json,traceback
from pathlib import Path
from common import write
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output'])
try:
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 import check
 check.run(r/'SOFTWARE.json');write(r/'STATUS.json',{'phase':'NATIVE_BUDGET','formal_optimizer':0,'VAL_TEST':0})
 import preflight
 passed,prepared=preflight.native_budget(c)
 if not passed:write(r/'STATUS.json',{'phase':'NOT_ADMITTED_BUDGET','formal_optimizer':0,'VAL_TEST':0})
 else:
  assert not (r/'FORMAL_STARTED.private.json').exists();write(r/'FORMAL_STARTED.private.json',{'original_start':c['start'],'deadline':c['deadline'],'formal_optimizer_before':0,'strong_SFT_frozen':True})
  import train
  base,tf=train.train(c,prepared);train.evaluate(c,base,tf)
  import verify,analyze
  verify.run(c);analyze.run(c);a=json.loads((r/'ANALYSIS.json').read_text());write(r/'STATUS.json',{'phase':a['conclusion'],'counts':train.COUNTS})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc())
 try:
  import train,preflight
  write(r/'FAILURE_COUNTS.private.json',{'formal':train.COUNTS,'engineering':preflight.ENGINEERING})
 except BaseException:pass
 write(r/'STATUS.json',{'phase':'FAILED'});raise
