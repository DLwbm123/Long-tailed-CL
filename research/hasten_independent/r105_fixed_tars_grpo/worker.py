"""One bounded fixed-producer experiment: eight matched policy arms then sealed VAL."""
import os,sys,json,time,traceback
from pathlib import Path
import torch
from common import write,guard
c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);sys.path.append(c['frozen_runtime']);wall=time.time();cpu=time.process_time()
counts={'teacher_optimizer':0,'fusion_optimizer':0,'router_optimizer':0,'sampled_groups':0,'sampled_actions':0,'engineering_router_optimizer':0,'ENGINEERING_generic_view_rows':0,'ENGINEERING_teacher_view_rows':0,'VAL_generic_view_rows':0,'VAL_teacher_view_rows':0}
try:
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from binding import bind,unchanged
    from train import train
    import preflight
    assert json.loads((Path(c['root_output'])/'SOFTWARE_CHECK.json').read_text())['status']=='PASS'
    rows,y,head,table,weights=bind(c);write(r/'STATUS.json',{'phase':'NATIVE_FULL_PLAN_PROFILE','counts':counts})
    passed,store=preflight.run(c,rows,y,head,table,weights,counts);assert passed,'FULL_PLAN_NOT_ADMITTED';unchanged(c)
    write(r/'FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'scope':'8 policies, fixed R104 FULL teacher and both candidates; natural queries; no upstream refit'})
    write(r/'STATUS.json',{'phase':'TRAIN_ROUTERS','counts':counts});train(c,table,y.cuda(),weights,counts);unchanged(c)
    write(r/'LEARNING_SEAL.json',{'router_endpoints':8,'router_updates':4096,'frozen_teacher_endpoints':1,'frozen_fusion_endpoints':2,'counts':counts,'VAL_TEST_images':0})
    write(r/'STATUS.json',{'phase':'WAIT_GLOBAL_SEAL','counts':counts})
    while not (Path(c['root_output'])/'GLOBAL_LEARNING_SEAL.json').exists():guard(c);time.sleep(2)
    import evaluate
    evaluate.run(c,rows,y,store,counts);unchanged(c)
    write(r/'STATUS.json',{'phase':'COMPLETE','counts':counts});write(r/'WORKER_COMPLETE.json',{'counts':counts,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall})
except BaseException:
    (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'FAILURE_COUNTS.json',counts);write(r/'STATUS.json',{'phase':'FAILED_PRESERVED','counts':counts});raise
finally:
    write(r/'COSTS.json',{'counts':counts,'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-wall,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'native_and_profile_included_once_in_worker':True,'historical_teacher_fusion_fit':'R104 immutable fixed producers, separate historical cost, no retraining or efficiency attribution','preparation_import_binding_SSH_document_publication_CPU':'nonzero not fully sampled'})
