import time,copy,math,os,subprocess,torch
from pathlib import Path
from strong import StrongAdapter,load_teacher
from model import make_base,visual,unit
from losses import Loss
from compute import fast_group
from data import train_rows,loader
from common import write,guard
ENGINEERING={'image_rows':0,'text_rows':0,'optimizer_steps':0,'formal_optimizer':0,'VAL_TEST':0}
EC={'real_image_encoder_rows':0,'text_encoder_rows':0,'cached_visual_rows':0,'cached_text_rows':0}

def native_budget(c):
 r=Path(c['output']);cpu=time.process_time();start=time.time();base,tf,tok=make_base(c,'cuda');rows,y=train_rows(c);ids=[]
 for cl in [0,1,2]:
  chosen=[];components=set()
  for i,yy in enumerate(y.tolist()):
   if yy==cl and rows[i]['identity_component'] not in components:chosen.append(i);components.add(rows[i]['identity_component'])
   if len(chosen)==2:break
  assert len(chosen)==2;ids+=chosen
 images=torch.cat([x for x,j in loader(c,rows,ids,tf)]);profiles=[];checks=[];last=None
 for stage in [0,5]:
  guard(c);head=torch.load(Path(c['strong_heads'])/f'head{stage}.private.pt',map_location='cpu',weights_only=False);load_teacher(base,head);C=head['seen'];birth=[s for s,cl in enumerate(c['stages'][:stage+1]) for _ in cl];torch.manual_seed(c['seed']);m=StrongAdapter().cuda()
  for _ in range(stage+1):m.add()
  rawcounts=torch.tensor(c['train_counts'][:C],device='cuda',dtype=torch.float);by=torch.arange(C,device='cuda').repeat_interleave(rawcounts.long());bank=unit(torch.randn(len(by),512,device='cuda'));center=torch.stack([unit(bank[by==k].mean(0)) for k in range(C)]);m.context(center,rawcounts,rawcounts,birth,stage);N=64;idx=torch.arange(N)%len(images);xx=images[idx];yy=y[ids][idx];batches=[(xx[k:k+4],yy[k:k+4],torch.arange(k,k+4)) for k in range(0,N,4)];cache={b:(unit(torch.randn(N,512)),torch.ones(N,dtype=torch.bool)) for b in range(stage)};tokens=tok(['A photo of a '+s for s in c['names'][:C]]).cuda()
  with torch.no_grad():z=visual(base,images.cuda());teacher=(20*z@head['prototypes'].cuda().T).cpu()[idx];ENGINEERING['image_rows']+=len(images);probe=torch.randn(N,C,device='cuda');assert torch.equal(m.combine(probe,teacher.cuda()),teacher.cuda())
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=c['lr'],weight_decay=c['weight_decay']);ot={};sup=[];rl=[];gen=torch.Generator(device='cuda').manual_seed(74002);stats={};opened=False
  for rep in range(2):
   torch.cuda.synchronize();t=time.time();opt.zero_grad(set_to_none=True);fn=Loss(teacher,rawcounts);fast_group(base,m,batches,bank,by,tokens,birth,cache,ot,EC,lossfn=fn)
   assert all(p.grad is None and not p.requires_grad for p in base.parameters());assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
   if rep==1:
    assert m.branches[-1].vprompt.grad.norm()>0 and m.branches[-1].tprompt.grad.norm()>0 and m.policy.net[-1].weight.grad.norm()>0;opened=True
   opt.step();ENGINEERING['optimizer_steps']+=1;torch.cuda.synchronize();sup.append(time.time()-t)
   fn=Loss(teacher,rawcounts,'RL',teacher+rawcounts.cpu().log()[None],torch.linspace(1,2,C,device='cuda'),gen,stats);torch.cuda.synchronize();t=time.time()
   for update in range(2):
    fn.update=update;opt.zero_grad(set_to_none=True);loss,_=fast_group(base,m,batches,bank,by,tokens,birth,cache,ot,EC,lossfn=fn);assert math.isfinite(loss)
    if rep==0 and update==0:assert m.branches[-1].vprompt.grad.norm()>0 and m.branches[-1].tprompt.grad.norm()>0 and m.policy.net[-1].weight.grad.norm()>0
    opt.step();ENGINEERING['optimizer_steps']+=1
   torch.cuda.synchronize();rl.append(time.time()-t);assert torch.cuda.max_memory_reserved()<12*2**30
  state=copy.deepcopy(m.state_dict());m.load_state_dict(state,strict=True);assert all(torch.isfinite(x).all() for x in state.values());checks.append({'stage':stage,'strong_delta_parameters':49152,'zero_gain_teacher_logits_exact':True,'prompt_and_policy_gradient_after_gate_opens':opened,'strict_restore_finite':True});profiles.append({'stage':stage,'classes':C,'support_rows':len(by),'SUP_step_seconds':sup,'two_GRPO_updates_seconds':rl});del opt,cache,bank,state,fn;last=m;torch.cuda.empty_cache()
 forward=[]
 with torch.no_grad():
  for B in [1,4]:
   times=[]
   for repeat in range(2):
    torch.cuda.synchronize();t=time.time();visual(base,images[:B].cuda(),last.branches[-1]);torch.cuda.synchronize();times.append(time.time()-t);ENGINEERING['image_rows']+=B
   forward.append({'rows':B,'seconds':times})
 own=[s for s in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if s.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==c['GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']==str(c['physical_gpu']);write(r/'NATIVE.json',{'passed':True,'checks':checks,'base_and_LoRA_frozen':True,'own_GPU_binding':True,'physical_gpu':c['physical_gpu'],'distinct_TRAIN_images':6,'fixture_image_repeats_and_synthetic_support_for_cost_only':True,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start})
 foldrows=[[1961,1955],[2020,2013],[2045,2036],[2427,2416],[2767,2756],[3416,3405]];groups=sum(sum(math.ceil(n/64) for n in v) for v in foldrows);assert groups==462;sft=groups*10;rgroups=groups*2;ns=[sum(v) for v in foldrows];teacher=sum(ns);old=sum(s*n for s,n in enumerate(ns));observation=sum(ns);evalrows=1698*sum(1+2*(s+1) for s in range(6));perrow=max(z['seconds'][1]/z['rows'] for z in forward);other=teacher+old+observation+evalrows;projection=2*(sft*max(z['SUP_step_seconds'][1] for z in profiles)+rgroups*max(z['two_GRPO_updates_seconds'][1] for z in profiles)+other*perrow)+7200;remaining=c['deadline']-time.time();passed=projection<remaining-180;ENGINEERING['image_rows']+=EC['real_image_encoder_rows'];ENGINEERING['text_rows']=EC['text_encoder_rows'];write(r/'BUDGET.json',{'passed':passed,'full_bank_profiles':profiles,'fixed_second_pass_worst':True,'frozen_forward_profiles':forward,'safety_factor':2,'IO_save_analysis_reserve_seconds':7200,'SUP_steps':sft,'GRPO_optimizer_batch_rollouts':rgroups,'GRPO_steps':rgroups*2,'total_steps':sft+rgroups*2,'teacher_TRAIN_encoder_rows':teacher,'old_branch_cache_rows':old,'SUP_observation_rows':observation,'positive_path_VAL_encoder_upper':evalrows,'nonstep_encoder_upper':other,'projected_seconds':projection,'remaining_seconds':remaining,'engineering_counts':ENGINEERING,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'formal_optimizer':0,'VAL_TEST':0});del m,last;torch.cuda.empty_cache();return passed,(base,tf,tok)
