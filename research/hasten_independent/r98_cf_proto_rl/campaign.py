"""One immutable-clock worker, conditional only on correctness and full budget."""
import os,json,time,traceback,resource
from pathlib import Path
from common import write
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);start=time.time();cpu=time.process_time()
try:
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 import check
 assert json.loads((r/'PRECHECK.json').read_text())['passed'], 'NEW_MECHANISM_SOFTWARE_NOT_PASSED'
 import data_contract
 meta=data_contract.metadata(c)
 import preflight
 if not preflight.native_budget(c,meta):write(r/'STATUS.json',{'phase':'BUDGET_NOT_ADMITTED','formal_optimizer_steps':0,'F_VAL_TEST':0})
 else:
  assert not (r/'FORMAL_STARTED.private.json').exists();write(r/'FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'source_commit':c['source_commit'],'formal_steps_before':0,'main_arm':'CF','five_arms_fixed':True})
  import train,evaluate,analyze,verify
  train.run(c,meta);evaluate.run(c,meta,train.COUNTS);a=analyze.run(c);verify.run(c);write(r/'STATUS.json',{'phase':a['conclusion'],'counts':train.COUNTS});write(r/'WORKER_COMPLETE.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'counts':train.COUNTS})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc())
 try:
  import train,preflight
  write(r/'FAILURE_COUNTS.private.json',{'formal':train.COUNTS,'engineering':preflight.ENGINEERING})
 except BaseException:pass
 write(r/'STATUS.json',{'phase':'FAILED'});raise
finally:
 try:
  import train,preflight
  costs={'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-start,'worker_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_and_readout_counts':train.COUNTS,'engineering_prefix_in_worker':preflight.ENGINEERING,'native_profile_costs_included_do_not_double_add':True,'teacher_history_cost':'Borrowed six original R67 SFT heads and R97 frozen feature assets; historical compute not included in new worker; not claimed free lifetime cost','external_preparation_SSH_document_publication_CPU':'nonzero not fully sampled'};write(r/'COSTS.json',costs)
 except BaseException:pass
