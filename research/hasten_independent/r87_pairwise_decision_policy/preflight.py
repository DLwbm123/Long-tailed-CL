import os,json,time,torch,subprocess
from pathlib import Path
from source import load_stage
from decision import inputs,outcomes,Policy,joint
from common import write,guard
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);t=time.time();torch.set_num_threads(4);guard(cfg)
v=load_stage(cfg,0,0);z=v['q']['arms']['NCM']['logits'][:2].double();ctx=inputs(z,v['proto'],v['support'],v['ages']);o=outcomes(ctx,v['q']['labels'][:2],v['eligible'],v['protected']);torch.manual_seed(5);p=Policy().distribution(ctx);assert torch.isfinite(joint(p)).all() and abs(float(joint(p).sum())-1)<1e-12
assert torch.equal(z,ctx['scores']);ps=subprocess.check_output(['ps','-ww','-o','pid,ppid,stat,args','-p',str(os.getpid())+','+str(os.getppid())],text=True);assert not any(s in ps.lower() for s in json.loads(os.environ.get('FORBIDDEN_PROCESS_TOKENS_JSON','[]')));(r/'NATIVE_ps.private.txt').write_text(ps)
write(r/'NATIVE_CHECK.json',dict(passed=True,kind='saved_TRAIN_logit_CPU_decoder_engineering_not_pixel_GPU_native',real_saved_TRAIN_score_rows=2,source_ids_labels_optimizersteps_bound=True,zero_action_identity=True,finite_legal_policy=True,encoder_classifier_forward_rows=0,visual_optimizer_steps=0,controller_optimizer_steps=0,VAL_TEST_reads=0,wall_seconds=time.time()-t))
