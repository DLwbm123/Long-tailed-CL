"""True S/B new-loss tests and fixed-second-pass full nine-arm admission."""
import torch,time,copy,os,json,subprocess,importlib.util,io
from pathlib import Path
from common import write,guard
from data_contract import load_stage,require_seal
from calibration import load_reference
from train import make_episode,optimizer,flat_grad
from episodes import noise
from losses import components,ARMS
ENGINEERING={'optimizer_steps':0,'image_encoder_rows':0,'episodes':0,'joint_rollouts':0,'active_class_actions':0,'policy_class_prototype_builds':0,'raw_score_rows':0,'replacement_utility_contexts':0,'factual_utility_contexts':0,'utility_contexts':0,'diagnostic_forward':0,'diagnostic_backward':0,'diagnostic_compute_counts':{},'legacy_ACT_equivalence_counts':{},'F_VAL_TEST':0}

def native_budget(c,meta,kappas):
 start=time.time();cpu=time.process_time();r=Path(c['output']);profiles=[];equivalence=[]
 gpu_names={v.split(',')[0].strip():v.split(',',1)[1].strip() for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,name','--format=csv,noheader'],text=True).splitlines()};assert gpu_names[str(c['physical_gpu'])]==gpu_names['0'],'REUSED_ENCODER_DEVICE_ARCHITECTURE_MISMATCH'
 try:require_seal(c)
 except AssertionError:pass
 else:raise AssertionError('PRESEAL_LOADER_NOT_BLOCKED')
 spec=importlib.util.spec_from_file_location('frozen_r98_losses',str(Path(c['init_source'])/'losses.py'));legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
 for s in [0,5]:
  guard(c);d=load_stage(c,s,meta[1],meta[2]);ref=load_reference(c,s);ep=make_episode(d,s,0,False,kappas[s]);per={}
  for obj in ['SC','CF','PATH']:
   m=copy.deepcopy(ref).requires_grad_(True)
   with torch.no_grad():m.net[-1].bias.add_(.01)
   mm=copy.deepcopy(m);eps=noise(ep,s,0,0);new=components(obj+'_ACT',m,ref,ep,eps,ENGINEERING)[0];old=legacy.loss_for_arm(obj,mm,ref,ep,eps,ENGINEERING['legacy_ACT_equivalence_counts'])[0];gn=flat_grad(new,m);go=flat_grad(old,mm);de=float((new-old).abs());dg=float((gn-go).abs().max());assert de<=2e-6 and dg<=2e-5,('ACT_EQUIVALENCE',s,obj,de,dg);equivalence.append({'stage':s,'objective':obj,'loss_maxabs':de,'gradient_maxabs':dg})
  for arm in ARMS:
   times=[];diags=[]
   for repeat in range(2):
    m=copy.deepcopy(ref).requires_grad_(True);opt=optimizer(m);torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True)
    for half in [False,True]:
     ep=make_episode(d,s,7,half,kappas[s]);v,*_=components(arm,m,ref,ep,noise(ep,s,7,int(half)),ENGINEERING);(.5*v).backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();ENGINEERING['optimizer_steps']+=1;torch.cuda.synchronize();times.append(time.time()-t)
    ep=make_episode(d,s,0,False,kappas[s]);torch.cuda.synchronize();t=time.time();_,ce,rl,hold,_=components(arm,m,ref,ep,noise(ep,s,0,0),ENGINEERING['diagnostic_compute_counts']);flat_grad(ce,m,True);flat_grad(rl,m,True);gh=flat_grad(hold,m);torch.cuda.synchronize();diags.append(time.time()-t);ENGINEERING['diagnostic_forward']+=1;ENGINEERING['diagnostic_backward']+=3
    if arm.endswith('_BHV'):assert gh.norm()>0
   per[arm]={'two_episode_step_seconds':times,'three_component_diagnostic_seconds':diags}
  buf=io.BytesIO();torch.save(m.state_dict(),buf);buf.seek(0);mm=copy.deepcopy(m);mm.load_state_dict(torch.load(buf,weights_only=False),strict=True);assert torch.equal(m(ep['state']),mm(ep['state']));profiles.append({'stage':s,'C':d['C'],'S_rows':len(d['S']),'B_rows':len(d['B']),'arms':per})
 own=[v for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if v.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==c['GPU_UUID'] and os.environ['CUDA_VISIBLE_DEVICES']==str(c['physical_gpu']);peak=torch.cuda.max_memory_reserved();assert peak<12*2**30
 step={a:max(v['arms'][a]['two_episode_step_seconds'][1] for v in profiles) for a in ARMS};diag=max(v['arms'][a]['three_component_diagnostic_seconds'][1] for v in profiles for a in ARMS);previous=json.loads((Path(c['init_source'])/'BUDGET.json').read_text());perrow=previous['fixed_second_pass_worst_frozen_perrow_seconds'];training=1536*sum(step.values());diagnostics=162*diag;readouts=132*max(step.values())+10188*perrow;projection=2*(training+diagnostics+readouts)+7200;remaining=c['deadline']-time.time();passed=projection<remaining
 write(r/'NATIVE.json',{'passed':True,'ACT_equivalence_to_R98':equivalence,'ACT_equivalence_forward_and_backward_each':12,'profile_cache_S_B_rows_accessed':8056,'behavior_constraint_gradients_nonzero_finite':True,'strict_policy_restore':True,'own_physical_GPU_UUID':True,'teacher_no_training_parameters':'Only shared Policy parameters ever enter optimizer; frozen teacher features reused, no image encoder constructed during fit or this native check','GPU_peak_reserved_bytes':peak,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'engineering_prefix_in_worker':ENGINEERING,'formal_optimizer':0,'F_VAL_TEST':0})
 write(r/'BUDGET.json',{'passed':passed,'profiles':profiles,'fixed_second_pass_worst_seconds':step,'fixed_second_pass_worst_three_gradient_diagnostic_seconds':diag,'borrowed_encoder_forward_perrow_seconds':perrow,'encoder_timing_scope':'R98 same verified Biomed/LoRA/FP32/preprocess path on same GPU architecture; no unnecessary re-encoding; safety2 plus7200s reserve','formal_optimizer_steps':13824,'borrowed_warmup_updates_not_new':3072,'final_endpoints':54,'middle_endpoints':54,'diagnostic_contexts':162,'VAL_encoder_rows_upper':10188,'safety_factor':2,'IO_analysis_delivery_reserve_seconds':7200,'projection_remaining_seconds':projection,'remaining_original_clock_seconds':remaining,'calibration_already_charged_to_original_clock':True,'no_arm_or_step_reduction':True});return passed
