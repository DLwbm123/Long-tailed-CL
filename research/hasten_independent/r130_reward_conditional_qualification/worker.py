"""Independent five-arm single-transition qualification; no historical TRAIN access."""
import os,json,time,copy,importlib.util,sys,contextlib
from pathlib import Path
import torch
from .core import ARMS,write,fixtures,coefficients,outcome

def dependency(path):
 spec=importlib.util.spec_from_file_location('q128_dependency',Path(path)/'__init__.py',submodule_search_locations=[str(path)])
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 return __import__(spec.name+'.worker',fromlist=['Run'])
P=dependency(json.loads(Path(os.environ['Q_CONFIG']).read_text())['parent_source'])

class Run(P.Run):
 def __init__(self,cfg):
  super().__init__({**cfg,'contrast_kappa':.2,'contrast_delta':.2,'CE_aux_coefficient':1.,'positive_KD_coefficient':1.})
  self.tf={};self.tf_kind='unclassified';self.phase='native';self.probe=None;self.menu=[]
  original=self.e.engine.token_logps
  def counted(f,t,m):
   r=self.tf.setdefault(self.phase+':'+self.tf_kind,{'rows':0,'input_token_cells':0,'active_response_tokens':0})
   r['rows']+=len(t);r['input_token_cells']+=len(t)*(self.e.engine.prompt_ids.shape[1]+t.shape[1]);r['active_response_tokens']+=int(m.sum())
   return original(f,t,m)
  self.e.engine.token_logps=counted
  self.counts.update({'probe_responses':0,'menu_responses':0,'qualification_CE_queries':0,'probe_optimizer_attempts':0})
 @contextlib.contextmanager
 def kind(self,k):
  old=self.tf_kind;self.tf_kind=k
  try:yield
  finally:self.tf_kind=old
 def same(self,a,b):return all(torch.equal(x,u) and torch.equal(y,v) for (x,y),(u,v) in zip(a,b))
 def score_candidates(self,f,classes,kind,profile_length=None):
  values=[]
  with self.kind(kind):
   for row in range(len(f)):
    chunks=[]
    for lo in range(0,len(classes),2):
     cs=classes[lo:lo+2];t,m=self.e.engine.label_tokens([self.e.names[c] for c in cs])
     if profile_length:
      t=t[:,:1].expand(len(cs),profile_length).contiguous();t[:,-1]=self.e.engine.eos;m=torch.ones_like(t,dtype=torch.bool)
     lp=self.e.engine.token_logps(f[row:row+1].expand(len(cs),-1,-1),t,m);chunks.append((lp*m).sum(-1))
    values.append(torch.cat(chunks))
  return torch.stack(values)
 def backward_candidates(self,f,classes,coef,kind,profile_length=None):
  with self.kind(kind):
   for row in range(len(f)):
    for lo in range(0,len(classes),2):
     cs=classes[lo:lo+2];t,m=self.e.engine.label_tokens([self.e.names[c] for c in cs])
     if profile_length:
      t=t[:,:1].expand(len(cs),profile_length).contiguous();t[:,-1]=self.e.engine.eos;m=torch.ones_like(t,dtype=torch.bool)
     lp=self.e.engine.token_logps(f[row:row+1].expand(len(cs),-1,-1),t,m)
     (coef[row,lo:lo+len(cs)]*(lp*m).sum(-1)).sum().backward()
 def rollout(self,features,labels,sigma,seed,kind):
  f=features.repeat_interleave(8,0);gen=torch.Generator(device='cuda').manual_seed(seed)
  t,m,sampled,text,done=self.generate(f,kind,gen,True);target=[self.base.normalize(self.e.names[int(c)]) for c in labels for _ in range(8)]
  b=torch.tensor([float(x==y and bool(d)) for x,y,d in zip(text,target,done)],device='cuda').reshape(len(labels),8)
  old=[];ref=[]
  with torch.no_grad(),self.kind(kind+'_behavior'):
   for lo in range(0,len(f),2):old.append(self.e.engine.token_logps(f[lo:lo+2],t[lo:lo+2],m[lo:lo+2]))
  with torch.no_grad(),self.e.engine.reference(),self.kind(kind+'_anchor'):
   for lo in range(0,len(f),2):ref.append(self.e.engine.token_logps(f[lo:lo+2],t[lo:lo+2],m[lo:lo+2]))
  old=torch.cat(old).detach();ref=torch.cat(ref).detach();a,h,r,drift,ns=P.rewards_and_advantages(b,old,ref,m,True,sigma)
  ac=(b-b.mean(-1,keepdim=True))/(ns+1e-6);ar=.5*(h-h.mean(-1,keepdim=True))/(ns+1e-6)
  assert torch.allclose(a,ac+ar,atol=2e-6)
  return {'f':f,'t':t,'m':m,'sampled':sampled,'text':text,'done':done,'b':b,'h':h,'r':r,'drift':drift,'old':old,'ref':ref,'a':a,'ac':ac,'ar':ar,'sigma_before':float(sigma),'sigma_after':ns,'labels':labels}
 def backward_rollout(self,z,component,weights=None):
  count=len(z['f']);total=0.;ratio_error=0.
  with self.kind('gradient_'+component):
   for lo in range(0,count,2):
    sl=slice(lo,lo+2);lp=self.e.engine.token_logps(z['f'][sl],z['t'][sl],z['m'][sl]);mask=z['m'][sl];length=mask.sum(-1).clamp_min(1)
    ratio_error=max(ratio_error,float(((lp.detach()-z['old'][sl]).abs()*mask).max()))
    if component=='KL':
     d=z['ref'][sl]-lp;assert float(d.detach().abs().max())<80;loss=.04*((torch.expm1(d)-d)*mask).sum(-1)/length
    else:
     adv=z[component].flatten()[sl];ratio=(lp-z['old'][sl]).exp();pg=torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None]);loss=-(pg*mask).sum(-1)/length
     if weights is not None:loss=loss*weights[sl]
    v=loss.sum()/count;assert torch.isfinite(v);v.backward();total+=float(v.detach())
  assert ratio_error==0.,'UNUPDATED_BEHAVIOR_RATIO'
  return total
 def gradient(self):return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten().float() for p in self.e.engine.parameters]).detach()
 def update(self,f,labels,opt,sigma,seed,kind):
  z=self.rollout(f,labels,sigma,seed,kind);opt.zero_grad(set_to_none=True);loss=0.
  if self.arm!='CE_OLD_COND':loss+=self.backward_rollout(z,'a' if self.arm=='RAPO' else 'ac')
  else:
   with self.kind('qualification_gold_CE'):loss+=self.ce_accumulate(f,labels,torch.ones(23,device='cuda'))
   if kind=='post_RL_responses':self.counts['qualification_CE_queries']+=len(labels)
   else:self.counts['engineering_teacherforced_queries']+=len(labels)
  loss+=self.backward_rollout(z,'KL');semantic=0.
  if self.arm in ARMS[2:]:
   classes=list(range(5 if self.stage==0 else 10));old_count=2 if self.stage==0 else 5
   with torch.no_grad(),self.e.engine.reference():target=self.score_candidates(f,classes,'semantic_anchor')
   with torch.no_grad():scores=self.score_candidates(f,classes,'semantic_actor_eval');sem,coef=coefficients(scores,target,old_count,self.arm!='ACC_FULL_KL')
   semantic=float(sem);loss+=semantic;self.backward_candidates(f,classes,coef,'semantic_actor_backward')
  norm,delta=self.step(opt)
  correct=z['b'].sum(-1).long();groups={}
  for c in sorted(set(labels.tolist())):
   s=correct[labels.to('cuda')==c];groups[str(c)]={'queries':len(s),'all_wrong':int((s==0).sum()),'mixed':int(((s>0)&(s<8)).sum()),'all_correct':int((s==8).sum()),'correct_responses':int(s.sum())}
  record={'arm':self.arm,'update':getattr(self,'current_update',-1),'sigma_before':sigma,'sigma_after':z['sigma_after'],'total_reward_batch_std':float(z['r'].std(unbiased=False)),'correct_responses':int(z['b'].sum()),'retention_mean':float(z['h'].mean()),'loss':loss,'semantic_loss':semantic,'gradient_norm':norm,'parameter_L2':delta,'nonzero_update':delta>0,'sampled_vs_TF_logp_max_difference':float(((z['sampled']-z['old']).abs()*z['m']).max()),'unupdated_behavior_ratio_log_difference':0.,'class_groups':groups}
  private={k:P.cpu_tree(v) for k,v in z.items() if k not in ['f']};self.checkpoint(private,('ROLLOUT_%s_%03d.private.pt'%(self.arm,self.current_update)) if kind=='post_RL_responses' else 'NATIVE_'+self.arm+'.private.pt')
  return record,z['sigma_after']
 def native(self):
  if self.cfg.get('reuse_native_proof'):
   old=Path(self.cfg['reuse_native_proof']);proof=json.loads(old.read_text());prior=json.loads((old.parent/'CONFIG.private.json').read_text());assert proof['passed'] and prior['source_commit']==self.cfg['reuse_native_source_sha']
   assert all(self.cfg[k]==v for k,v in prior.items() if k!='source_commit')
   assert self.same(self.e.engine.snapshot(),self.prefix) and all(torch.equal(a.ref_a.detach().cpu(),x) and torch.equal(a.ref_b.detach().cpu(),y) for a,(x,y) in zip(self.e.engine.adapters,self.prefix))
   proof.update(reused_previous_successful_native=True,native_model_calls_this_attempt=0,original_native_runtime=prior['source_commit'],remaining_at_recovery=self.cfg['deadline']-time.time());assert proof['forecast_seconds']<self.cfg['deadline']-time.time();write(self.out/'NATIVE_ADMISSION.json',proof);return
  self.status('NATIVE_R130');write(self.out/'SOFTWARE_CHECK.json',fixtures());ids=[i for i,c in enumerate(self.y) if c in range(5)][:16];first=ids[:2];expected=self.e.engine.prompt_ids.clone()
  self.access.check('TRAIN',first,0,'NATIVE','native_images');features,raw=self.e.engine.encode(self.e.images(self.e.train,first));self.counts['TRAIN_encoder_rows']+=2;assert torch.equal(expected,self.e.engine.prompt_ids)
  with torch.no_grad():
   full=self.e.engine.model(**raw,use_cache=False,output_hidden_states=True);self.counts['TRAIN_encoder_rows']+=2;prompt=self.e.engine.prompt_ids.expand(2,-1);pos,_=self.e.engine.positions(prompt,torch.ones_like(prompt));hidden=self.e.engine.model.model(inputs_embeds=self.e.engine.embeddings(prompt,features),position_ids=pos,attention_mask=torch.ones_like(prompt),use_cache=False,return_dict=True).last_hidden_state
   head=self.e.engine.model.lm_head(hidden[:,-1:])[:,0].float();hdiff=float((hidden[:,-1:]-full.hidden_states[-1][:,-1:]).abs().max());pdiff=float((head-self.e.engine.model.lm_head(full.hidden_states[-1][:,-1:])[:,0].float()).abs().max());rawdiff=float((head-full.logits[:,-1].float()).abs().max())
  assert hdiff<=.05 and pdiff<=.05
  # Longer old R128 profile uses only first-task words and neutral padding.
  self.bind_prompt(0,profiling=True);records=[];self.current_update=-1
  for arm in ARMS:
   self.arm=arm;self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();opt=self.opt();start=time.time();self.counts['engineering_optimizer_attempts']+=1
   rec,_=self.update(self.get('TRAIN',first,'native_training'),self.labels(first),opt,1.,124000,'engineering_responses');torch.cuda.synchronize();rec['seconds']=time.time()-start;records.append(rec);assert rec['nonzero_update'];del opt
  self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();f=self.get('TRAIN',first,'native_candidate_profile');classes=[i%5 for i in range(10)];length=max(len(self.e.engine.tokenizer.encode(self.e.names[c],add_special_tokens=False))+1 for c in range(10));assert length<=24
  opt=self.opt();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.time()
  with torch.no_grad(),self.e.engine.reference():target=self.score_candidates(f,classes,'native_semantic_anchor',length)
  with torch.no_grad():scores=self.score_candidates(f,classes,'native_semantic_eval',length);_,coef=coefficients(scores,target.roll(1,-1),5,False)
  self.backward_candidates(f,classes,coef,'native_semantic_backward',length);torch.cuda.synchronize();profile_seconds=time.time()-start;profile_norm=float(torch.nn.utils.clip_grad_norm_(self.e.engine.parameters,1.));assert profile_norm>0;opt.zero_grad(set_to_none=True)
  start_cond=time.time()
  with torch.no_grad():_,condcoef=coefficients(scores,target.roll(1,-1),5,True)
  self.backward_candidates(f,classes,condcoef,'native_conditional_backward',length);torch.cuda.synchronize();conditional_norm=float(torch.nn.utils.clip_grad_norm_(self.e.engine.parameters,1.));assert conditional_norm>0;profile_seconds+=time.time()-start_cond;opt.zero_grad(set_to_none=True);del opt
  start=time.time();self.generate(self.get('TRAIN',ids,'native_greedy_profile'),'engineering_responses');torch.cuda.synchronize();greedy=time.time()-start
  # Include doubled 5->10 candidate envelope beyond the measured short native paths.
  measured=sum(r['seconds'] for r in records)*4*32
  extra_sem=profile_seconds*4*32*3
  probe_envelope=max(r['seconds'] for r in records)*4*6
  forecast=1.5*(measured+extra_sem+probe_envelope+(2496+5784)/16*greedy)+1800
  self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();assert self.same(self.e.engine.snapshot(),self.prefix)
  proof={'passed':forecast<self.cfg['deadline']-time.time(),'matched_hidden_difference':hdiff,'matched_head_difference':pdiff,'raw_full_head_difference':rawdiff,'native_paths':records,'all_five_nonzero_native_updates_discarded':True,'prefix_and_reference_restored':True,'semantic_profile_seconds':profile_seconds,'semantic_profile_gradient_norm':profile_norm,'conditional_profile_gradient_norm':conditional_norm,'shifted_target_profile_engineering_only':True,'greedy16_seconds':greedy,'profile_prompt':self.profile_prompt,'profile_max_answer_length_CPU_only':length,'forecast_seconds':forecast,'remaining_seconds':self.cfg['deadline']-time.time(),'native_responses':96,'TRAIN_encodings':4,'new_VAL_encodings':0,'native_optimizer_attempts':5}
  write(self.out/'NATIVE_ADMISSION.json',proof);assert proof['passed'],'R130_COMPLETE_SCOPE_NOT_TIME_ADMITTED';self.bind_prompt(0)
 def tape(self):
  current=[i for i,c in enumerate(self.y) if c in range(5,10)];g=torch.Generator().manual_seed(12474006);return torch.tensor(current)[torch.randint(len(current),(32,8),generator=g)]
 def diagnostic(self):
  self.phase='diagnostic';self.status('FIXED_CURRENT_GRADIENT_PROBE')
  state=torch.load(Path(self.cfg['prefix_source'])/'STATE_1_RAPO.private.pt',map_location='cpu',weights_only=True);assert state['updates']==128
  self.stage=1;self.bind_prompt(1);self.arm='PROBE';self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();self.e.engine.restore(state['weights']);ids=self.tape()[0].tolist();f=self.get('TRAIN',ids,'fixed_current_gradient_probe');labels=self.labels(ids)
  z=self.rollout(f,labels,state['sigma'],13081000,'probe_responses');opt=self.opt();opt.load_state_dict(state['optimizer']);vectors={};losses={}
  for component in ['ac','ar','KL']:
   opt.zero_grad(set_to_none=True);losses[component]=self.backward_rollout(z,component);vectors[component]=self.gradient()
  calibration=json.loads((Path(self.cfg['prefix_source'])/'CALIBRATION_1_TARS_RAPO.json').read_text());w=torch.ones(23,device='cuda')
  for c,v in zip(calibration['classes'],calibration['used_weights']):w[c]=v
  opt.zero_grad(set_to_none=True);losses['TARS_PG']=self.backward_rollout(z,'a',w[labels.to('cuda')].repeat_interleave(8));vectors['TARS_PG']=self.gradient()
  norms={k:float(v.norm()) for k,v in vectors.items()};cos={a+'__'+b:float(torch.dot(vectors[a],vectors[b])/(vectors[a].norm()*vectors[b].norm()).clamp_min(1e-30)) for i,a in enumerate(vectors) for b in list(vectors)[i+1:]}
  opt.zero_grad(set_to_none=True);self.backward_rollout(z,'a');self.backward_rollout(z,'KL');before=self.e.engine.snapshot();pre=float(self.gradient().norm());self.counts['probe_optimizer_attempts']+=1;norm,delta=self.step(opt);postclip=float(self.gradient().norm());assert delta>0
  self.checkpoint({k:P.cpu_tree(v) for k,v in z.items() if k!='f'},'PROBE_ROLLOUT.private.pt')
  self.e.engine.restore(before);assert self.same(self.e.engine.snapshot(),state['weights']);assert all(torch.equal(a.ref_a.detach().cpu(),x) and torch.equal(a.ref_b.detach().cpu(),y) for a,(x,y) in zip(self.e.engine.adapters,self.prefix))
  self.checkpoint({k:P.cpu_tree(v) for k,v in z.items() if k!='f'},'PROBE_ROLLOUT.private.pt')
  self.probe={'queries':8,'responses':64,'saved_RAPO_stage1_Adam_step':128,'sigma_before':state['sigma'],'sigma_after_one_hypothetical_stat_update':z['sigma_after'],'gradient_norms':norms,'gradient_cosines':cos,'component_losses':losses,'combined_preclip_norm':pre,'combined_postclip_norm':postclip,'counterfactual_actual_L2':delta,'counterfactual_discarded_actor_anchor_restored':True,'TARS_table_other_arm_diagnostic_only_not_RAPO_training':True,'correct_responses':int(z['b'].sum()),'mixed_queries':int(((z['b'].sum(-1)>0)&(z['b'].sum(-1)<8)).sum())};write(self.out/'GRADIENT_PROBE.json',self.probe);del opt,z,vectors
  # Four same-batch readouts; old menu is diagnostic only and cannot enter training.
  ids=[i for i,c in enumerate(self.vy) if c in range(5)]
  for parameter,weights in [(0,self.prefix),(1,state['weights'])]:
   self.e.engine.restore(weights)
   for menu in [0,1]:
    self.bind_prompt(menu);self.stage=menu;self.arm='MENU_DIAGNOSTIC';good=[0]*23;den=[0]*23;preds=[];lookup={self.base.normalize(n):c for c,n in self.e.names.items()}
    for lo in range(0,len(ids),16):
     self.status('MENU_PARAMETER_2X2',parameter=parameter,menu=menu,rows_done=lo);chunk=ids[lo:lo+16];_,_,_,text,done=self.generate(self.get('VAL',chunk,'isolated_audit',True),'menu_responses');pred=[lookup.get(t,-1) if bool(d) else -1 for t,d in zip(text,done)];preds.extend(pred)
     for i,p in zip(chunk,pred):den[self.vy[i]]+=1;good[self.vy[i]]+=int(p==self.vy[i])
    self.checkpoint({'ids':ids,'predictions':preds},f'MENU_{parameter}_{menu}.private.pt');self.menu.append({'parameter':parameter,'menu':menu,'correct':good,'denominator':den,'old_macro':sum(good[c]/den[c] for c in range(5))/5})
  d={(r['parameter'],r['menu']):r['old_macro'] for r in self.menu};write(self.out/'MENU_INTERACTION.json',{'readouts':self.menu,'interaction':d[1,1]-d[1,0]-d[0,1]+d[0,0],'not_unique_causal_fraction_or_deployment_prompt':True});self.stage=1;self.bind_prompt(1)
 def science(self):
  self.diagnostic();self.phase='qualification';self.stage=1;self.bind_prompt(1);tape=self.tape();torch.save(tape,self.out/'QUERY_TAPE.private.pt');states={}
  # Prefix is sealed and immutable; its audit never enters the five update rules.
  write(self.out/'PREFIX_AUDIT_SEAL.json',{'epoch':time.time(),'weights_immutable':True});self.e.engine.restore(self.prefix);self.arm='PREFIX_CURRENT_MENU';self.evaluations.append(self.evaluate(self.arm,True));self.save_logs()
  for arm in ARMS:
   self.arm=arm;self.e.engine.restore(self.prefix);self.e.engine.freeze_reference();opt=self.opt();sigma=1.
   self.bindings.append({'arm':arm,'same_prefix_exact':self.same(self.e.engine.snapshot(),self.prefix),'initial_optimizer_empty':not opt.state,'initial_sigma':sigma,'anchor_same_initial_prefix':True})
   for k in range(32):
    self.current_update=k;self.status('SINGLE_TRANSITION_QUALIFICATION',update=k,total=32);ids=tape[k].tolist();self.counts['post_optimizer_attempts']+=1
    rec,sigma=self.update(self.get('TRAIN',ids),self.labels(ids),opt,sigma,12481000+k,'post_RL_responses');self.counts['post_completed_updates']+=1;self.curves.append({'phase':'POST','stage':1,**rec})
    if k%4==0:self.save_logs()
   state={'weights':self.e.engine.snapshot(),'optimizer':P.cpu_tree(opt.state_dict()),'sigma':sigma,'updates':32};states[arm]=state;self.checkpoint(state,'STATE_1_'+arm+'.private.pt');del opt
  write(self.out/'STAGE_1_ALL_MODELS_SEALED.json',{'epoch':time.time(),'arms':ARMS,'updates_each':32})
  for arm in ARMS:self.arm=arm;self.e.engine.restore(states[arm]['weights']);self.evaluations.append(self.evaluate(arm,True));self.save_logs()
  result=outcome(self.evaluations,self.e.class_counts.tolist());result['all_arms_32_nonzero_steps']=all(sum(r['arm']==a and r['nonzero_update'] for r in self.curves)==32 for a in ARMS);result['qualification_passed']=result['performance_qualification_passed'] and result['all_arms_32_nonzero_steps'];write(self.out/'FINAL_RESULTS.json',result);self.status('COMPLETE_READY_FOR_SAVED_VERIFICATION')
 def save_logs(self):
  super().save_logs()
  if hasattr(self,'tf'):write(self.out/'TF_COSTS.json',self.tf)

def run(cfg):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
 begin=time.time();cpu=time.process_time();r=None
 try:r=Run(cfg);r.native();r.science()
 finally:
  if r is not None:r.save_logs();write(r.out/'COSTS.json',{'counts':r.counts,'TF':r.tf,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'new_feature_model_copy_bytes':0,'no_old_TRAIN':True})
  else:write(Path(cfg['output'])/'LOAD_FAILURE_COSTS.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-begin,'model_load_failure_before_native':True})
