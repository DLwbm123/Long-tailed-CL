import os,json,time,traceback,torch
from pathlib import Path
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);counter={k:0 for k in ['builder_optimizer_steps','builder_episode_backward','actor_optimizer_steps','precheck_contexts','joint_reward_calls','engineering_real_encoder_rows','VAL_prefix_encoder_rows','VAL_tail_encoder_rows','VAL_classifier_score_rows','TEST_reads']};torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
try:
 import software,native,budget,train_build,train_actor,evaluate,verify
 write(r/'STATUS.json',{'phase':'SOFTWARE','counts':counter});s=json.loads((r/'SOFTWARE_EXTERNAL.private.json').read_text()) if (r/'SOFTWARE_EXTERNAL.private.json').exists() else software.run();assert s['passed'];write(r/'SOFTWARE.json',s);budget.initial(cfg)
 write(r/'STATUS.json',{'phase':'NATIVE','counts':counter});native.run(cfg,counter)
 write(r/'STATUS.json',{'phase':'BUDGET','counts':counter});b=budget.run(cfg)
 if not b['passed']:write(r/'STATUS.json',{'phase':'NOT_ADMITTED_BUDGET','counts':counter})
 else:
  train_build.run(cfg,counter);gate=train_build.precheck(cfg,counter)
  if not gate['passed']:write(r/'STATUS.json',{'phase':'COMPLETE_PRECHECK_NEGATIVE','counts':counter,'VAL_TEST':0})
  else:
   train_actor.run(cfg,counter);evaluate.run(cfg,counter);verify.run(cfg,counter);evaluate.analyze(cfg);write(r/'STATUS.json',{'phase':'COMPLETE','counts':counter})
except BaseException:
 (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'STATUS.json',{'phase':'FAILED','counts':counter});raise
