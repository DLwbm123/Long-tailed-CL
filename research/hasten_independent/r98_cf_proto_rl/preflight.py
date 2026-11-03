"""Native cache identity and full-plan admission; fixed second-pass worst profile."""
import copy,time,torch,subprocess,os,json
from pathlib import Path
from common import write,guard
from data_contract import load_stage,require_seal
from episodes import episode,noise
from prototype_policy import Policy
from losses import loss_for_arm,ARMS
from train import optimizer,flat_grad
ENGINEERING={'optimizer_steps':0,'image_encoder_rows':0,'cached_feature_rows':0,'episodes':0,'joint_rollouts':0,'active_class_actions':0,'prototype_builds':0,'raw_score_rows':0,'counterfactual_utility_calls':0,'utility_calls':0,'diagnostic_forward':0,'diagnostic_backward':0,'VAL_TEST':0}

def native_budget(c,meta):
 start=time.time();cpu=time.process_time();r=Path(c['output']);rows,y,roles=meta
 from prototype_policy import build_state,build_prototypes
 torch.manual_seed(983);zz=[torch.nn.functional.normalize(torch.randn(k,512),dim=-1) for k in [2,3,4,1]];ep=build_state(zz,[4,6,8,2],[0,0,0,0],0);a=torch.randn(8,4,4);pp=build_prototypes(a,ep);perm=[2,0,3,1];sw=build_state([zz[k] for k in perm],[[4,6,8,2][k] for k in perm],[0]*4,0);assert torch.allclose(build_prototypes(a[:,perm],sw),pp[:,perm],atol=2e-5);sw=build_state([z.flip(0) for z in zz],[4,6,8,2],[0]*4,0);assert torch.allclose(build_prototypes(a,sw),pp,atol=2e-5)
 try:require_seal(c)
 except AssertionError:pass
 else:raise AssertionError('PRESEAL_F_ACCESS_NOT_BLOCKED')
 from encoder import make_base,load_teacher,visual
 from data import loader
 base,tf,_=make_base(c,'cuda');assert not any(p.requires_grad for p in base.parameters());profiles=[];perrow=[];maxerr=0.;distinct=set()
 for s in [0,5]:
  guard(c);h=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);load_teacher(base,h);n=h['seen'];legal=[i for i in roles['S']+roles['B'] if int(y[i])<n];chosen=legal[:3] if s==0 else legal[-3:];distinct.update(chosen);pack=torch.load(Path(c['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False);x,j=next(iter(loader(c,rows,chosen,tf)))
  with torch.no_grad():z=visual(base,x.cuda()).cpu();ENGINEERING['image_encoder_rows']+=len(j)
  err=float((z-pack['z'][j]).abs().max());maxerr=max(maxerr,err);assert err<2e-5,('CACHE_NATIVE_BINDING',s,err)
  for batch in [1,3]:
   times=[]
   for repeat in range(2):
    torch.cuda.synchronize();t=time.time()
    with torch.no_grad():visual(base,x[:batch].cuda())
    torch.cuda.synchronize();times.append((time.time()-t)/batch);ENGINEERING['image_encoder_rows']+=batch
   perrow.append(times[1])
  d=load_stage(c,s,y,roles);ENGINEERING['cached_feature_rows']+=len(d['S'])+len(d['B']);scale={}
  for arm in ['WARM',*ARMS]:
   times=[];diags=[]
   for repeat in range(2):
    torch.manual_seed(981000+s);m=Policy().cuda();ref=copy.deepcopy(m).requires_grad_(False);opt=optimizer(m,1e-4);torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True)
    for half in [False,True]:
     ep=episode(d,s,7,half);v,_,_,_=loss_for_arm(arm,m,None if arm=='WARM' else ref,ep,noise(ep,s,7,int(half)),ENGINEERING);(.5*v).backward()
    assert any(v.grad is not None and v.grad.norm()>0 for v in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step();ENGINEERING['optimizer_steps']+=1;torch.cuda.synchronize();times.append(time.time()-t)
    if arm!='WARM':
     ep=episode(d,s,0);torch.cuda.synchronize();t=time.time();_,ce,aux,_=loss_for_arm(arm,m,ref,ep,noise(ep,s,0,0));flat_grad(ce,m,True);flat_grad(aux,m);torch.cuda.synchronize();diags.append(time.time()-t);ENGINEERING['diagnostic_forward']+=1;ENGINEERING['diagnostic_backward']+=2
   scale[arm]={'two_episode_step_seconds':times,'second_pass_diagnostic_seconds':diags[-1] if diags else None}
  profiles.append({'stage':s,'C':n,'B_rows':len(d['B']),'S_rows':len(d['S']),'arms':scale})
 own=[v for v in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if v.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==c['GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']==str(c['physical_gpu']);peak=torch.cuda.max_memory_reserved();assert peak<12*2**30
 step={arm:max(z['arms'][arm]['two_episode_step_seconds'][1] for z in profiles) for arm in ['WARM',*ARMS]};diag=max(z['arms'][arm]['second_pass_diagnostic_seconds'] for z in profiles for arm in ARMS);training=3072*step['WARM']+1536*sum(step[a] for a in ARMS);diagnostics=90*diag+6*32*2*diag;VAL=10188*max(perrow);other=84*max(step.values());projected=2*(training+diagnostics+VAL+other)+7200;remaining=c['deadline']-time.time();passed=projected<remaining
 write(r/'NATIVE.json',{'passed':True,'cache_native_maxabs':maxerr,'distinct_TRAIN_images':len(distinct),'all_teacher_frozen':True,'own_physical_GPU_UUID':True,'peak_GPU_reserved_bytes':peak,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'engineering':ENGINEERING,'formal_optimizer_steps':0,'F_VAL_TEST':0})
 write(r/'BUDGET.json',{'passed':passed,'profiles':profiles,'fixed_second_pass_worst_step_seconds':step,'fixed_second_pass_worst_frozen_perrow_seconds':max(perrow),'formal_optimizer_steps':10752,'final_endpoints':36,'midpoint_endpoints':30,'warmup_steps':3072,'post_steps':7680,'full_diagnostic_seconds_upper':diagnostics,'VAL_encoder_rows_upper':10188,'safety_factor':2,'IO_analysis_publication_reserve_seconds':7200,'projected_seconds':projected,'remaining_original_clock_seconds':remaining,'engineering_prefix_counts':ENGINEERING,'no_truncation_or_arm_removal':True})
 del base,m,opt,ref,d;torch.cuda.empty_cache();return passed
