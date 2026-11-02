import json,os,time,traceback
from pathlib import Path
import torch
import counts
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output'])
try:
 import software,preflight
 boundary=os.environ.get('R90_SEALED_BOUNDARY')=='1'
 if boundary:
  audit=json.loads((r/'COUNTER_FAILURE_BOUNDARY_AUDIT.private.json').read_text());assert audit['boundary_audit_passed'];counts.C.update(audit['counts'])
  assert counts.C['formal_visual_optimizer_steps']==1230 and counts.C['formal_controller_optimizer_steps']==counts.C['VAL_rows']==counts.C['TEST_rows']==0
 elif (r/'ENGINEERING_RETRY.private.json').exists():
  prior=json.loads((r/'ENGINEERING_RETRY.private.json').read_text())['previous_counts'];assert prior['formal_visual_optimizer_steps']==prior['formal_controller_optimizer_steps']==prior['TRAIN_encoder_rows']==prior['VAL_rows']==0
  counts.C.update(prior)
 write(r/'STATUS.json',{'phase':'SOFTWARE'})
 if (r/'SOFTWARE_CPU_PROCESS.private.json').exists():
  receipt=json.loads((r/'SOFTWARE_CPU_PROCESS.private.json').read_text());assert receipt['passed'] and receipt['core']['passed'];counts.C['toy_controller_optimizer_steps']=receipt['core']['toy_controller_optimizer_steps'];write(r/'SOFTWARE_REUSED.private.json',{'passed':True,'new_toy_steps':0,'external_CPU_cost':receipt})
 else:software.run(cfg)
 if boundary:
  from program import make_base
  import subprocess
  assert not (r/'BOUNDARY_RESTORED.private.json').exists() and not (r/'active_context.private.pt').exists()
  base,tf=make_base(cfg);native={'fixed_plan_admitted':True}
  own=[s for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if s.split(',')[0].strip()==str(os.getpid())]
  assert len(own)==1 and own[0].split(',')[1].strip()==json.loads((r/'ADMISSION.private.json').read_text())['physical_GPU_uuid'] and os.environ['CUDA_VISIBLE_DEVICES']=='1'
  write(r/'BOUNDARY_GPU_BINDING.private.json',{'own_PID':os.getpid(),'physical_GPU':1,'GPU_UUID':own[0].split(',')[1].strip(),'new_engineering_steps':0})
 else:
  write(r/'STATUS.json',{'phase':'NATIVE'});native,base,tf=preflight.run(cfg)
 if not native['fixed_plan_admitted']:
  write(r/'ENGINEERING_REJECTED.json',{'reason':'FIXED_PLAN_DOES_NOT_FIT_ORIGINAL_CLOCK','native':native,'counts':dict(counts.C)});write(r/'STATUS.json',{'phase':'NOT_ADMITTED_ENGINEERING','counts':dict(counts.C)})
 else:
  counts.MODE='TRAIN';import experiment,evaluate,verify,analyze
  dictionary=experiment.train(cfg,base,tf,sealed_boundary=boundary);counts.MODE='VAL';evaluate.run(cfg,base,tf,dictionary);verify.run(cfg);analyze.run(cfg);write(r/'STATUS.json',{'phase':'COMPLETE'})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'STATUS.json',{'phase':'FAILED','counts':dict(counts.C)});raise
