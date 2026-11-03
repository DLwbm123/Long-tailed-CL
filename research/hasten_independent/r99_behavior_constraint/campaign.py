import os,json,time,traceback,resource
from pathlib import Path
from common import write
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);start=time.time();cpu=time.process_time()
try:
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert json.loads((r/'PRECHECK.json').read_text())['passed']
 import data_contract,calibration,preflight,train,evaluate,analyze,verify
 meta=data_contract.metadata(c);write(r/'STATUS.json',{'phase':'CALIBRATION','formal_optimizer':0,'F_VAL_TEST':0});kappas=calibration.run(c,meta)
 write(r/'STATUS.json',{'phase':'NATIVE_BUDGET','formal_optimizer':0,'F_VAL_TEST':0})
 if not preflight.native_budget(c,meta,kappas):write(r/'STATUS.json',{'phase':'BUDGET_NOT_ADMITTED','formal_optimizer':0,'F_VAL_TEST':0})
 else:
  assert not (r/'FORMAL_STARTED.private.json').exists();write(r/'FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'source_commit':c['source_commit'],'main_arm':'CF_BHV','nine_arms_fixed':True,'kappas_fixed':kappas,'formal_optimizer_before':0,'borrowed_warmup_steps_not_new':3072});train.run(c,meta,kappas);evaluate.run(c,meta,train.COUNTS);a=analyze.run(c);verify.run(c);write(r/'STATUS.json',{'phase':a['conclusion'],'counts':train.COUNTS});write(r/'WORKER_COMPLETE.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'counts':train.COUNTS})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc())
 try:write(r/'FAILURE_COUNTS.private.json',{'formal':train.COUNTS,'engineering':preflight.ENGINEERING})
 except BaseException:pass
 write(r/'STATUS.json',{'phase':'FAILED'});raise
finally:
 try:
  write(r/'COSTS.json',{'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-start,'worker_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_readout_and_diagnostic_counts':train.COUNTS,'engineering_in_worker':preflight.ENGINEERING,'calibration_in_worker':json.loads((r/'CALIBRATION.json').read_text()) if (r/'CALIBRATION.json').exists() else None,'do_not_double_add_worker_subitems':True,'borrowed_assets':'R67 frozen teachers/R97 cached features/R98 six SUP0 checkpoints including3072 historical warmup updates; not new training or free lifetime cost','external_software_CPU_separate':True,'external_preparation_SSH_document_publication_CPU':'nonzero not fully sampled'})
 except BaseException:pass
