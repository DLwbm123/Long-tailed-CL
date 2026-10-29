"""Sequential, bounded, real optimizer trajectories; all heads sealed before VAL."""
import os,json,time,copy,random,traceback,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import write,guard
from data import train_rows,get_cache,Tokens,OrderSampler,DataLoader,store
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
from decision import *

def cpu_copy(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_copy(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu_copy(v) for v in x)
 return copy.deepcopy(x)

def encode(model,data,indices,n,cfg):
 sampler=OrderSampler(len(data));sampler.order=list(indices);loader=DataLoader(data,batch_size=256,sampler=sampler);zs=[];ys=[]
 with torch.no_grad():
  for tx,bx,yy,ix in loader:
   guard(cfg)
   for tt,y in zip(tx.split(64),yy.split(64)):zs.append(model(tt.cuda()).cpu());ys.append(y)
 z=torch.cat(zs);y=torch.cat(ys);counts=torch.bincount(y,minlength=n);assert bool((counts>0).all())
 sums=torch.zeros(n,512,dtype=torch.double);sums.index_add_(0,y,z.double());proto=F.normalize((sums/counts[:,None]).float(),dim=1)
 return dict(prototypes=proto,features=z,labels=y,counts=counts)

def evaluate(model,data,indices,proto,n,cfg):
 sampler=OrderSampler(len(data));sampler.order=list(indices);dl=DataLoader(data,batch_size=256,sampler=sampler);losses=[];margins=[];ces=[];ys=[];correct=[]
 with torch.no_grad():
  for tx,bx,yy,ix in dl:
   guard(cfg)
   for tt,y in zip(tx.split(64),yy.split(64)):
    scores=20*model(tt.cuda())@proto.cuda().T;p=scores.softmax(1);yg=y.cuda();target=F.one_hot(yg,n).float();losses.append((.5*(p-target).square().sum(1)).cpu());ces.append(F.cross_entropy(scores,yg,reduction='none').cpu());v=p.clone();v[torch.arange(len(y),device='cuda'),yg]=-1;margins.append((p[torch.arange(len(y),device='cuda'),yg]-v.max(1).values).cpu());ys.append(y);correct.append((scores.argmax(1)==yg).cpu())
 y=torch.cat(ys);den=torch.bincount(y,minlength=n);assert bool((den>0).all())
 def mean(values):
  sums=torch.zeros(n);sums.index_add_(0,y,torch.cat(values));return sums/den
 cc=torch.bincount(y[torch.cat(correct)],minlength=n)
 return dict(brier=mean(losses),margin=mean(margins),ce=mean(ces),correct=cc,counts=den)

def inner_step(model,opt,data,indices,proto,nc,cfg):
 assert len(indices)==len(set(indices))==256;assert len(proto)==len(nc)
 tokens=store().batch(data,indices,True);yy=data.labels[indices];base=data.base[indices];opt.zero_grad(set_to_none=True);loss_sum=0.
 for tx,bx,by in zip(tokens.split(64),base.split(64),yy.split(64)):
  guard(cfg);z=model(tx.cuda());logits=20*z@proto.cuda().T+nc.cuda().float().log()[None,:];loss=(F.cross_entropy(logits,by.cuda(),reduction='none')+.01*(z-bx.cuda()).square().sum(1)).sum()/256
  assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach())
 params=[p for p in model.parameters() if p.requires_grad];assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step();assert torch.cuda.max_memory_reserved(0)<12*2**30
 return loss_sum

def main(cfg):
 root=Path(cfg['output']);dc=cfg['domains']['HK'];out=Path(dc['output']);out.mkdir(exist_ok=True);rows,all_labels=train_rows(dc);bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False)
 reference=Path(cfg['matched_reference_run'])/'HK/74002';initial=torch.load(reference/'head0.private.pt',map_location='cpu',weights_only=False);assert initial['seen']==13
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def model_new(delta):
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(74002);m=NativeTail(bundle).cuda().eval();adapt_native_tail(m);load_delta(m,delta)
  return m
 model=model_new(initial['delta']);frozen={k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad};params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==49152
 records=[];contexts=[];head_count=0;actual_total=0;committed_total=0;curves=[];fold_receipts={}
 birth=torch.tensor([next(s for s,v in enumerate(dc['stages']) if c in v) for c in range(23)])
 for arm in ARMS:
  load_delta(model,initial['delta']);opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);torch.manual_seed(74002);controller=Controller();copt=torch.optim.AdamW(controller.parameters(),lr=.0003,weight_decay=0.);last_exposure=torch.zeros(23);previous_n=0;paths=[];old_folds={};margin_history={};arm_actual=arm_commit=0;controller_updates=0
  uniform_initial=cpu_copy(controller.state_dict())
  for stage in range(6):
   stage_started=time.time();write(root/'STATUS.json',dict(phase='TRAIN',arm=arm,stage=stage,context=0,time=time.time()));paths.extend(get_cache(cfg,dc,stage=stage));data=Tokens(paths);n=sum(map(len,dc['stages'][:stage+1]));nc=torch.bincount(data.labels,minlength=n);assert nc.tolist()==dc['train_counts_ordered'][:n]
   folds=assign_folds(data.labels.tolist(),[rows[i]['identity_component'] for i in data.ids]);mapping=dict(zip(data.ids,folds));assert all(mapping[i]==f for i,f in old_folds.items());old_folds=mapping;parts=[[i for i,f in enumerate(folds) if f==b] for b in [0,1]];assert all(bool((torch.bincount(data.labels[ii],minlength=n)>0).all()) for ii in parts)
   fold_receipts[str(stage)]=dict(fold_counts=[torch.bincount(data.labels[ii],minlength=n).tolist() for ii in parts],train_counts=nc.tolist(),no_patient_independence_claim=True)
   before_folder=out/arm/'74002';before_folder.mkdir(parents=True,exist_ok=True);torch.save(dict(delta=delta_state(model),optimizer=cpu_copy(opt.state_dict()),controller=cpu_copy(controller.state_dict()),controller_optimizer=cpu_copy(copt.state_dict())),before_folder/f'before{stage}.private.pt')
   prev_delta=delta_state(model);prev_model=model_new(prev_delta) if previous_n else None;stage_controller_before=cpu_copy(controller.state_dict());stage_model_before=delta_state(model);stage_actual=stage_commit=0;anchor_cache={}
   for context in range(CONTEXTS):
    guard(cfg);b=context%2;S=parts[1-b];Q=parts[b];assert not set(S)&set(Q);write(root/'STATUS.json',dict(phase='TRAIN',arm=arm,stage=stage,context=context,time=time.time(),actual_inner_updates=actual_total))
    opt.param_groups[0]['lr']=.001 if context<4 else .0005
    pre=encode(model,data,S,n,cfg);start_eval=evaluate(model,data,Q,pre['prototypes'],n,cfg);base_delta=delta_state(model);base_opt=cpu_copy(opt.state_dict())
    if previous_n and b not in anchor_cache:
     oldS=[i for i in S if int(data.labels[i])<previous_n];oldQ=[i for i in Q if int(data.labels[i])<previous_n];op=encode(prev_model,data,oldS,previous_n,cfg)
     # Evaluate old probabilities padded with zero on newly arrived classes; Brier is unchanged.
     anchor_cache[b]=evaluate(prev_model,data,oldQ,op['prototypes'],previous_n,cfg)['brier']
    anchor=start_eval['brier'].clone()
    if previous_n:anchor[:previous_n]=anchor_cache[b]
    protected=(torch.arange(n)<previous_n)|(nc<250)
    state=torch.stack([nc.float().log()/math.log(1000),pre['counts'].float().log()/math.log(1000),(stage-birth[:n]).float()/5,(birth[:n]==stage).float(),start_eval['brier'],start_eval['margin'],torch.zeros(n),last_exposure[:n]],1)
    if b in margin_history:
     old=margin_history[b];state[:len(old),6]=start_eval['margin'][:len(old)]-old
    margin_history[b]=start_eval['margin'].clone()
    gbase=74002+10000*stage+100*context;common=[]
    for h in range(3):
     gen=torch.Generator().manual_seed(gbase+h);common.append([S[i] for i in torch.randperm(len(S),generator=gen).tolist()[:(192 if h<2 else 256)]])
    endpoints=[];trajectory=[];states=[];masks=[];actions=[];oldlp=[];commit_index=(stage*CONTEXTS+context)%GROUP;chosen_delta=chosen_opt=None;chosen_exposure=None;all_exposure=torch.zeros(n,dtype=torch.long);point_logs=[]
    for branch in range(9):
     if arm!='balanced_long' or branch==0:load_delta(model,base_delta);opt.load_state_dict(cpu_copy(base_opt))
     branch_proto=pre if branch==0 else encode(model,data,S,n,cfg)
     exposure=torch.zeros(n,dtype=torch.long);acts=[];realized=[];losses=[]
     for h in range(3):
      if h<2:
       mask=legal_actions(data.labels,S,common[h]);ss=state.clone();ss[:,7]=exposure.float()/max(1,int(exposure.sum())) if h else state[:,7]
       if arm in SAMPLED and branch<GROUP:
        with torch.no_grad():lp=controller(ss,mask).log_softmax(0);uniform=float(torch.rand((),generator=torch.Generator().manual_seed(gbase+700000+branch*10+h)));cdf=lp.exp().cumsum(0);cdf=cdf/cdf[-1];a=int(torch.searchsorted(cdf,torch.tensor(uniform)).clamp_max(n));assert bool(mask[a])
        states.append(ss);masks.append(mask);actions.append(a);oldlp.append(lp[a]);assert torch.allclose(lp.exp()[mask],torch.full_like(lp.exp()[mask],1/int(mask.sum())),atol=1e-7,rtol=0)
       else:a=0
       fillgen=torch.Generator().manual_seed(gbase+800000+h);ii=allocation(data.labels,S,common[h],a-1,fillgen);acts.append(a);realized.append(dict(priority_available=int(sum(int(data.labels[i])==a-1 and i not in set(common[h]) for i in S)) if a else 0,priority_allocated=int(sum(int(data.labels[i])==a-1 for i in ii[192:])) if a else 0,controlled_counts=torch.bincount(data.labels[ii[192:]],minlength=n).tolist()))
      else:ii=common[h]
      assert not set(ii)&set(Q);losses.append(inner_step(model,opt,data,ii,branch_proto['prototypes'],nc,cfg));exposure+=torch.bincount(data.labels[ii],minlength=n);actual_total+=1;stage_actual+=1;arm_actual+=1
     post=encode(model,data,S,n,cfg);ev=evaluate(model,data,Q,post['prototypes'],n,cfg);endpoints.append(ev['brier']);trajectory.append(exposure);all_exposure+=exposure
     point_logs.append(dict(branch=branch,actions=acts,exposure_counts=exposure.tolist(),allocation=realized,class_brier=ev['brier'].tolist(),query_correct=ev['correct'].tolist(),inner_loss=losses))
     if branch==commit_index and arm!='balanced_long':chosen_delta=delta_state(model);chosen_opt=cpu_copy(opt.state_dict());chosen_exposure=exposure.clone()
    if arm=='balanced_long':chosen_exposure=all_exposure;committed_here=27
    else:load_delta(model,chosen_delta);opt.load_state_dict(chosen_opt);committed_here=3
    stage_commit+=committed_here;arm_commit+=committed_here;committed_total+=committed_here;last_exposure[:n]=chosen_exposure.float()/int(chosen_exposure.sum())
    penalty=arm!='macro_controller';utilities=torch.stack([utility(v,anchor,protected,penalty)[0] for v in endpoints]);damages=torch.stack([utility(v,anchor,protected,True)[1] for v in endpoints]);rewards=utilities[:8]-utilities[8];adv=(rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-8);policy_stats=[]
    if arm in LEARNED:
     assigned=rewards.roll(1) if arm=='shuffled_controller' else rewards;aa=(assigned-assigned.mean())/(assigned.std(unbiased=False)+1e-8)
     for _ in range(4):
      copt.zero_grad(set_to_none=True);loss,stats=grpo_loss(controller,states,masks,actions,torch.stack(oldlp).reshape(8,2),aa);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in controller.parameters());copt.step();controller_updates+=1;policy_stats.append(stats)
    selected=8 if arm=='balanced_long' else commit_index
    contexts.append(dict(arm=arm,stage=stage,context=context,query_fold=b,query_counts=start_eval['counts'].tolist(),support_counts=pre['counts'].tolist(),pre_brier=start_eval['brier'].tolist(),anchor_brier=anchor.tolist(),protected_classes=protected.tolist(),branches=point_logs,utilities=utilities.tolist(),damage=damages.tolist(),absolute_returns=rewards.tolist(),all_candidate_returns_negative=bool((rewards<0).all()),advantages=adv.tolist(),retained_branch=selected,retained_absolute_return=None if arm=='balanced_long' else float(rewards[commit_index]),retained_advantage=None if arm=='balanced_long' else float(adv[commit_index]),actual_updates=27,committed_updates=committed_here,all_exposure=all_exposure.tolist(),committed_exposure=chosen_exposure.tolist(),unique_batch=True,no_query_in_inner_or_support=True,assigned_returns=(rewards.roll(1) if arm=='shuffled_controller' else rewards).tolist(),policy_stats=policy_stats,controller_updates=controller_updates,lr=opt.param_groups[0]['lr'],reward_used_for_commit_selection=False))
    write(out/'CONTEXT_RECORDS.json',contexts)
    # TRAIN feedback curves are not independent validation or efficacy gates.
    curves.append(dict(arm=arm,stage=stage,context=context,fold=b,counts=start_eval['counts'].tolist(),pre_correct=start_eval['correct'].tolist(),retained_correct=point_logs[selected]['query_correct'],pre_macro_brier=float(start_eval['brier'].mean()),retained_macro_brier=float(endpoints[selected].mean())))
   full=encode(model,data,list(range(len(data))),n,cfg);folder=out/arm/'74002';folder.mkdir(parents=True,exist_ok=True);head=dict(delta=delta_state(model),prototypes=full['prototypes'],features=full['features'],labels=full['labels'],support_counts=full['counts'],train_counts=nc,seen=n,stage=stage,seed=74002,arm=arm,raw_inference=True,stage0_SFT_initialization_only=True,carry_previous_delta=stage>0,optimizer_state=cpu_copy(opt.state_dict()),controller_state=cpu_copy(controller.state_dict()),controller_optimizer=cpu_copy(copt.state_dict()),last_exposure=last_exposure.clone(),previous_n=previous_n);torch.save(head,folder/f'head{stage}.private.pt')
   restored=model_new(head['delta']);tx=store().batch(data,[0],True).cuda()
   with torch.no_grad():assert torch.equal(model(tx),restored(tx))
   del restored
   assert all(torch.equal(p.detach().cpu(),frozen[k]) for k,p in model.named_parameters() if not p.requires_grad)
   assert all(torch.equal(v,uniform_initial[k]) for k,v in controller.state_dict().items()) and not copt.state and controller_updates==0
   records.append(dict(arm=arm,stage=stage,seen=n,train_counts=nc.tolist(),actual_optimizer_steps=stage_actual,committed_optimizer_steps=stage_commit,micro_backwards=4*stage_actual,contexts=8,controller_updates=controller_updates,LoRA_parameters=49152,original_parameters_frozen=True,strict_restore_exact=True,carry_previous_delta=stage>0,carry_visual_optimizer=stage>0,previous_stage_SFT_not_loaded=True,VAL_reads=0,wall_seconds=time.time()-stage_started));write(out/'TRAIN_RECORDS.json',records);write(out/'TRAIN_CURVES.json',curves);write(out/'FOLD_COUNTS.json',fold_receipts)
   head_count+=1;previous_n=n;del prev_model,head,data;torch.cuda.empty_cache()
  assert arm_actual==1296 and arm_commit==(1296 if arm=='balanced_long' else 144)
 assert (actual_total,committed_total,head_count)==(1296,144,6)
 write(out/'ALL_HEADS_SEALED.json',dict(records=6,trained_heads=6,zero_update_heads=0,actual_optimizer_steps=1296,committed_optimizer_steps=144,micro_backwards=5184,VAL_reads=0,time=time.time()))
 val=Tokens(get_cache(cfg,dc,val=True));results=[]
 for stage in range(6):
  item=dict(seed=74002,stage=stage,seen=sum(map(len,dc['stages'][:stage+1])),metrics={})
  for arm in ARMS:
   folder=out/arm/'74002';h=torch.load(folder/f'head{stage}.private.pt',map_location='cpu',weights_only=False);load_delta(model,h['delta']);sampler=OrderSampler(len(val));dl=DataLoader(val,batch_size=64,sampler=sampler);scores=[];labels=[];ids=[]
   with torch.no_grad():
    for tx,bx,yy,ix in dl:
     guard(cfg);score=(20*model(tx.cuda())@h['prototypes'].cuda().T).cpu();mask=yy<h['seen'];scores.append(score[mask]);labels.append(yy[mask]);ids.extend(val.ids[i] for i in ix[mask].tolist())
   score=torch.cat(scores);yy=torch.cat(labels);den=torch.bincount(yy,minlength=h['seen']);cc=torch.bincount(yy[score.argmax(1)==yy],minlength=h['seen']);item['metrics'][arm]=dict(correct=cc.tolist(),counts=den.tolist());torch.save(dict(scores=score,labels=yy,ids=ids),folder/f'response{stage}.private.pt')
  results.append(item);write(out/'aggregate.json',dict(dataset='HK',results=results,train_counts=dc['train_counts_ordered'],no_TEST=True,stagewise_SFT_restart=False,online_RL=False,controller_GRPO_visual_supervised=False,uniform_actor_no_learning=True))
 engine=store();evidence=dict(encoded_rows=engine.encoded,reference_max_abs=engine.max_abs,retained_prefix_bytes=engine.ram_bytes,temporary_swap_bytes=engine.spill_bytes,remote_swap_bytes=engine.remote_spill_bytes,scientific_artifacts_preserved=True);engine.index.clear();engine.disk_index.clear();engine.ram_bytes=0
 for folder in [engine.spill_root,engine.remote_spill_root]:
  for p in folder.glob('prefix_*.npy'):p.unlink()
 write(out/'REGENERABLE_CACHE_CLEANUP.json',evidence);write(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except BaseException:(Path(cfg['output'])/'FAILURE.private.txt').write_text(traceback.format_exc());raise
