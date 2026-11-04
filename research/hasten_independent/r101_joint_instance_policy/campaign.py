"""One independent frozen campaign; only software/native/resources can block fit."""
import os,json,time,traceback,resource
from pathlib import Path
from common import write
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);start=time.time();cpu=time.process_time()
try:
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(r/'NATIVE_PROCESS.private.json',{'worker_cmdline':Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),'parent_cmdline':Path(f'/proc/{os.getppid()}/cmdline').read_bytes().replace(b'\0',b' ').decode()})
    import data_contract,preflight,train,evaluate,analyze,verify
    meta=data_contract.metadata(c);write(r/'STATUS.json',{'phase':'NATIVE_BUDGET','formal_optimizer':0,'F_VAL_TEST':0})
    if preflight.run(c,meta):
        assert not (r/'FORMAL_STARTED.private.json').exists()
        write(r/'FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'source_commit':c['source_commit'],'arms':list(train.ARMS),'supervised_steps':12288,'policy_steps':9216,'no_scientific_gain_precondition':True,'main':'PROTO_GRPO'})
        train.run(c,meta);evaluate.run(c,meta,train.COUNTS);a=analyze.run(c);verify.run(c)
        write(r/'STATUS.json',{'phase':a['conclusion'],'counts':train.COUNTS});write(r/'WORKER_COMPLETE.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'counts':train.COUNTS})
    else:write(r/'STATUS.json',{'phase':'BUDGET_NOT_ADMITTED','formal_optimizer':0,'F_VAL_TEST':0})
except BaseException:
    (r/'FAILURE.private.txt').write_text(traceback.format_exc())
    try:write(r/'FAILURE_COUNTS.private.json',{'formal':train.COUNTS,'engineering':preflight.ENGINEERING})
    except BaseException:pass
    write(r/'STATUS.json',{'phase':'FAILED'});raise
finally:
    try:write(r/'COSTS.json',{'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-start,'worker_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'counts':train.COUNTS,'engineering_included_in_worker':preflight.ENGINEERING,'native_profile_included_no_double_add':True,'borrowed_teacher_feature_costs_historical_not_free':True,'preparation_SSH_document_publication_CPU':'nonzero not fully sampled'})
    except BaseException:pass
