"""Matched stagewise post-training of saved R67 SFT CLIP heads, four complete arms."""
import os,json,time,math,random,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from common import write,guard
from data import train_rows,get_cache,Tokens,OrderSampler,DataLoader,store
from folds import assign_folds
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
from decision import ARMS,tailness,objective,policy_logits
from support import make_support,support_scores
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());rc=json.loads(Path(cfg['root_config']).read_text());root=Path(rc['output']);out=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def status(**kw):write(out/'STATUS.json',dict(time=time.time(),**kw))
def main():
 rows,y=train_rows(cfg);counts=torch.bincount(y,minlength=len(cfg['order']));bundle=torch.load(root/'cache/0/tail.private.pt',map_location='cpu',weights_only=False);records=[];totals={a:dict(steps=0,micro=0) for a in ARMS}
 for seed in cfg['seeds']:
  paths=[];old_folds={}
  for stage,classes in enumerate(cfg['stages']):
   status(phase='ARRIVED_STAGE_CACHE',seed=seed,stage=stage);paths.extend(get_cache(rc,cfg,stage=stage));data=Tokens(paths);n=sum(map(len,cfg['stages'][:stage+1]));N=len(data);nc=counts[:n].cuda();assert torch.bincount(data.labels,minlength=n).tolist()==nc.tolist()
   folds=assign_folds(data.labels.tolist(),[rows[i]['identity_component'] for i in data.ids]);mapping=dict(zip(data.ids,folds));assert all(mapping[i]==f for i,f in old_folds.items());old_folds=mapping;indices=[[i for i,f in enumerate(folds) if f==b] for b in [0,1]]
   sampler=OrderSampler(N);dl=DataLoader(data,batch_size=256,sampler=sampler)
   reference=Path(rc['matched_reference_run'])/cfg['dataset']/str(seed);head=torch.load(reference/f'head{stage}.private.pt',map_location='cpu',weights_only=False);assert head['seen']==n and head['train_counts'].tolist()==nc.tolist()
   saved=json.loads((reference/f'curves{stage}.json').read_text())[-1];assert saved['epoch']==10 and saved['query_counts']==nc.tolist();weights=tailness(nc.cpu(),torch.tensor(saved['crossfit_query_correct'])).cuda()
   write(out/str(seed)/f'TAILNESS{stage}.json',dict(counts=nc.tolist(),SFT_correct=saved['crossfit_query_correct'],weights=weights.tolist(),gamma=.75,fixed_before_posttraining=True,source='R67 saved SFT TRAIN crossfold predictions; no VAL or TEST'))
   def new_model():
    with torch.random.fork_rng(devices=[0]):m=NativeTail(bundle).cuda();adapt_native_tail(m);m.eval();load_delta(m,head['delta'])
    return m
   ref=new_model()
   @torch.no_grad()
   def encode(model,ii):
    sampler.order=ii;sums=torch.zeros(n,512,device='cuda',dtype=torch.double);cc=torch.zeros(n,device='cuda',dtype=torch.long);features=[];support_labels=[]
    for tokens,base,yy,ix in dl:
     for tx,by in zip(tokens.split(64),yy.split(64)):
      guard(cfg);by=by.cuda();z=model(tx.cuda());features.append(z);support_labels.append(by);sums.index_add_(0,by,z.double());cc+=torch.bincount(by,minlength=n)
    assert bool((cc>0).all()) and int(cc.sum())==len(ii)
    return dict(prototypes=F.normalize((sums/cc[:,None]).float(),dim=1).detach(),features=torch.cat(features).detach(),labels=torch.cat(support_labels).detach(),counts=cc.detach())
   refproto=[encode(ref,indices[1-b]) for b in [0,1]]
   merged=make_support(torch.cat([v['features'] for v in refproto]),torch.cat([v['labels'] for v in refproto]),n)
   assert torch.allclose(merged['prototypes'],head['prototypes'].cuda(),atol=2e-5,rtol=0)
   torch.save({k:v.cpu() for k,v in merged.items()},out/str(seed)/f'SFT_SUPPORT{stage}.private.pt')
   for arm in ARMS:
    random.seed(seed+10000*stage);np.random.seed(seed+10000*stage);torch.manual_seed(seed+10000*stage);torch.cuda.manual_seed_all(seed+10000*stage)
    folder=out/arm/str(seed);folder.mkdir(parents=True,exist_ok=True);assert not (folder/f'head{stage}.private.pt').exists(),'PARTIAL_POSTTRAIN_REQUIRES_AUDIT'
    model=new_model();params=[p for p in model.parameters() if p.requires_grad];original={k:p.detach().cpu().clone() for k,p in model.named_parameters() if not p.requires_grad};old=new_model();opt=torch.optim.AdamW(params,lr=.001,weight_decay=.001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,2);pg=torch.Generator(device='cuda').manual_seed(seed+10000*stage);rg=torch.Generator(device='cuda').manual_seed(seed+10000*stage+670000);curves=[];steps=0;micro=0;queryrows=0;supportrows=0;stats=dict(groups=0,zero_advantage_groups=0,reward_sum=0,actions=0,clipped_actions=0,KL_sum=0.,mean_weight_sum=0.)
    @torch.no_grad()
    def observe(epoch,obj):
     cc=torch.zeros(n,device='cuda',dtype=torch.long);den=torch.zeros_like(cc)
     for b in [0,1]:
      proto=encode(model,indices[1-b]);sampler.order=indices[b]
      for tokens,base,yy,ix in dl:
       for tx,by in zip(tokens.split(64),yy.split(64)):
        by=by.cuda();pred=(support_scores(model(tx.cuda()),proto,arm)).argmax(1);cc+=torch.bincount(by[pred==by],minlength=n);den+=torch.bincount(by,minlength=n)
     assert torch.equal(den,nc);a=cc.double()/den;curves.append(dict(epoch=epoch,objective_mean=obj,optimizer_steps=steps,micro_backwards=micro,query_counts=nc.tolist(),crossfit_query_correct=cc.tolist(),crossfit_query_macro_BA=float(a.mean()),crossfit_query_tail_BA=float(a[nc<250].mean()),is_independent_holdout=False));write(folder/f'curves{stage}.json',curves)
    observe(0,None)
    for epoch in range(2):
     total=0.
     for b in [0,1]:
      # A frozen rollout snapshot per fold permits nontrivial importance ratios after updates.
      load_delta(old,delta_state(model));proto=encode(model,indices[1-b]);supportrows+=len(indices[1-b]);sampler.order=[indices[b][i] for i in torch.randperm(len(indices[b]),device='cuda',generator=pg).cpu().tolist()]
      for bi,(tokens,base,yy,ix) in enumerate(dl):
       if bi%32==0:guard(cfg);status(phase='POSTTRAIN',seed=seed,stage=stage,arm=arm,epoch=epoch,query_fold=b,batch=bi,total_optimizer_steps=sum(v['steps'] for v in totals.values()))
       opt.zero_grad(set_to_none=True);B=len(yy)
       for tx,bx,by in zip(tokens.split(64),base.split(64),yy.split(64)):
        tx=tx.cuda();by=by.cuda();z=model(tx);logits=policy_logits(support_scores(z,proto,arm),nc,arm)
        with torch.no_grad():old_logits=policy_logits(support_scores(old(tx),proto,arm),nc,arm);ref_logits=policy_logits(support_scores(ref(tx),refproto[b],arm),nc,arm)
        risk,st=objective(logits,old_logits,ref_logits,by,weights,arm,rg);loss=(risk+.01*(z-bx.cuda()).square().sum(1)).sum()/B;assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*B;micro+=1;totals[arm]['micro']+=1;queryrows+=len(by)
        for k,v in st.items():stats[k]+=v
       assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);opt.step();steps+=1;totals[arm]['steps']+=1;assert torch.cuda.max_memory_reserved(0)<12*2**30
     sched.step();observe(epoch+1,total/N)
    assert steps==2*sum(math.ceil(len(ii)/256) for ii in indices) and micro==2*sum(math.ceil(len(ii)/64) for ii in indices) and queryrows==supportrows==2*N
    assert all(torch.equal(p.detach().cpu(),original[k]) for k,p in model.named_parameters() if not p.requires_grad)
    proto=encode(model,list(range(N)));h=dict(inference_mode=('empirical_logmeanexp_20cos' if arm.startswith('distribution_') else 'normalized_centroid_20cos'),delta=delta_state(model),prototypes=proto['prototypes'].cpu(),features=proto['features'].cpu(),labels=proto['labels'].cpu(),support_counts=proto['counts'].cpu(),support_mode=('distribution' if arm.startswith('distribution_') else 'prototype'),train_counts=nc.cpu(),seen=n,seed=seed,stage=stage,arm=arm,raw_inference=True,matched_SFT_stage_initialization=True);torch.save(h,folder/f'head{stage}.private.pt');tx=dl.token_at(0).cuda();load_delta(old,h['delta'])
    with torch.no_grad():assert torch.equal(model(tx),old(tx))
    records.append(dict(seed=seed,stage=stage,arm=arm,seen=n,train_counts=nc.tolist(),fold_rows=list(map(len,indices)),optimizer_steps=steps,micro_backwards=micro,epochs=2,curve_records=3,query_rows=queryrows,support_rows=supportrows,LoRA_trainable_parameters=sum(p.numel() for p in params),original_parameters_frozen=True,strict_restore_exact=True,old_fold_assignment_exact=True,opposite_fold_support=True,raw_inference=True,VAL_reads=0,tailness_frozen=True,support_mode=('distribution' if arm.startswith('distribution_') else 'prototype'),support_features_detached=True,statistics=stats));write(out/'TRAIN_RECORDS.json',records);del model,old,opt,params,original;torch.cuda.empty_cache()
   del ref,refproto,head,data,dl
  write(out/str(seed)/'TRAIN_SEAL.json',dict(branches=4*len(cfg['stages']),VAL_reads=0))
 assert len(records)==len(cfg['seeds'])*len(ARMS)*len(cfg['stages'])
 for v in totals.values():assert v['steps']==cfg['expected_steps']//5
 write(out/'ALL_HEADS_SEALED.json',dict(records=len(records),optimizer_steps=sum(v['steps'] for v in totals.values()),micro_backwards=sum(v['micro'] for v in totals.values()),VAL_reads=0))
 val=Tokens(get_cache(rc,cfg,val=True));dl=DataLoader(val,batch_size=64,shuffle=False);results=[]
 with torch.no_grad():
  for seed in cfg['seeds']:
   for stage in range(len(cfg['stages'])):
    item=dict(seed=seed,stage=stage,seen=sum(map(len,cfg['stages'][:stage+1])),metrics={})
    for arm in (*ARMS,'sft_reference','distribution_sft_reference'):
     h=torch.load((Path(rc['matched_reference_run'])/cfg['dataset']/str(seed) if arm in ['sft_reference','distribution_sft_reference'] else out/arm/str(seed))/f'head{stage}.private.pt',map_location='cpu',weights_only=False);m=NativeTail(bundle).cuda();adapt_native_tail(m);m.eval();load_delta(m,h['delta']);proto={k:v.cuda() for k,v in h.items() if k in ['prototypes','features','labels','support_counts']};scores=[];labels=[];ids=[]
     if arm=='distribution_sft_reference':proto={k:v.cuda() for k,v in torch.load(out/str(seed)/f'SFT_SUPPORT{stage}.private.pt',map_location='cpu',weights_only=False).items()}
     for tx,base,yy,ix in dl:
      guard(cfg);score=support_scores(m(tx.cuda()),proto,arm);mask=yy<h['seen'];scores.append(score.cpu()[mask]);labels.append(yy[mask]);ids.extend(val.ids[i] for i in ix[mask].tolist())
     score=torch.cat(scores);yy=torch.cat(labels);assert torch.isfinite(score).all();pr=score.argmax(1);nn=torch.bincount(yy,minlength=h['seen']);cc=torch.bincount(yy[pr==yy],minlength=h['seen']);item['metrics'][arm]=dict(correct=cc.tolist(),counts=nn.tolist());destination=out/arm/str(seed);destination.mkdir(parents=True,exist_ok=True);torch.save(dict(scores=score,labels=yy,ids=ids),destination/f'response{stage}.private.pt');del m,h
    results.append(item)
 write(out/'aggregate.json',dict(dataset=cfg['dataset'],results=results,train_counts=counts.tolist(),totals=totals,no_TEST=True,stagewise_posttraining_not_online_RL=True));engine=store();evidence=dict(backend='bounded_lossless_FP32_RAM',encoded_rows=engine.encoded,reference_max_abs=engine.max_abs,retained_prefix_bytes=engine.ram_bytes,all_requested_seeds_TRAIN_and_VAL_complete=True,completed_seeds=cfg['seeds'],scientific_artifacts_preserved=True);evidence['temporary_swap_bytes']=engine.spill_bytes;evidence['remote_swap_bytes']=engine.remote_spill_bytes;engine.index.clear();engine.ram_bytes=0;engine.disk_index.clear()
 for folder in [engine.spill_root,engine.remote_spill_root]:
  for swap_file in folder.glob('prefix_*.npy'):swap_file.unlink()
 engine.spill_bytes=0;engine.remote_spill_bytes=0;write(out/'REGENERABLE_CACHE_CLEANUP.json',evidence);status(phase='COMPLETE')
if __name__=='__main__':
 try:main()
 except BaseException:(out/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED');raise
