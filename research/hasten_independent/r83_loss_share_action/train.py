"""Fixed R81 states, enumerated first action, complete TRAIN-only endpoints."""
import os,json,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from common import write,guard
from data import train_rows,get_cache,Tokens,OrderSampler,DataLoader,store
from vision import NativeTail,adapt_native_tail,delta_state,load_delta
from decision import allocation,legal_actions
from folds import assign_folds
from learner import cpu_copy,encode,inner_step
from loss_share import weights,step as share_step

def evaluate_slices(model,data,Q,proto,n,qa_ids,qb_ids,cfg):
 sampler=OrderSampler(len(data));sampler.order=Q;sets=[None,set(qa_ids),set(qb_ids)];sums=torch.zeros(3,n,dtype=torch.float64);correct=torch.zeros(3,n,dtype=torch.long);counts=torch.zeros(3,n,dtype=torch.long)
 with torch.no_grad():
  for tx,bx,yy,ix in DataLoader(data,batch_size=256,sampler=sampler):
   guard(cfg)
   for tt,y,ii in zip(tx.split(64),yy.split(64),ix.split(64)):
    scores=20*model(tt.cuda())@proto.cuda().T;p=scores.softmax(1);loss=(.5*(p-F.one_hot(y.cuda(),n)).square().sum(1)).double().cpu();ok=(scores.argmax(1).cpu()==y);ids=[data.ids[i] for i in ii.tolist()]
    for k,ss in enumerate(sets):
     mask=torch.ones(len(y),dtype=torch.bool) if ss is None else torch.tensor([i in ss for i in ids]);z=y[mask];counts[k]+=torch.bincount(z,minlength=n);correct[k]+=torch.bincount(z[ok[mask]],minlength=n);sums[k].index_add_(0,z,loss[mask])
 return {tag:dict(brier=(sums[k]/counts[k].clamp_min(1)).tolist(),correct=correct[k].tolist(),counts=counts[k].tolist()) for k,tag in enumerate(['full','Qa','Qb'])}

def parameters_diff(a,b):
 z=torch.cat([(a[k]-b[k]).flatten() for k in a]);return dict(L2=float(z.double().norm()),max_abs=float(z.abs().max()),changed_parameters=int(torch.count_nonzero(z)))

def main(cfg):
 r=Path(cfg['output']);dc=cfg['domains']['HK'];src=Path(cfg['source_run'])/'HK/uniform_controller/74002';feas=json.loads((r/'FEASIBILITY.private.json').read_text());rows,labels=train_rows(dc);folds=assign_folds(labels.tolist(),[x['identity_component'] for x in rows]);bundle=torch.load(r/'cache/0/tail.private.pt',map_location='cpu',weights_only=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def new_model(delta):
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(74002);m=NativeTail(bundle).cuda().eval();adapt_native_tail(m);load_delta(m,delta)
  return m
 model=None;paths=[];records=[];actual=micro=0;new_scores=0
 for stage in range(6):
  paths.extend(get_cache(cfg,dc,stage=stage));data=Tokens(paths);n=sum(map(len,dc['stages'][:stage+1]));assert set(data.labels.tolist())==set(range(n));nc=torch.bincount(data.labels,minlength=n);assert nc.tolist()==dc['train_counts_ordered'][:n]
  state=torch.load(src/f'head{stage}.private.pt',map_location='cpu',weights_only=False);assert state['seen']==n;start_delta=cpu_copy(state['delta']);start_opt=cpu_copy(state['optimizer_state']);assert start_opt['param_groups'][0]['lr']==.0005
  if model is None:model=new_model(start_delta)
  else:load_delta(model,start_delta)
  params=[p for p in model.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==49152;frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad};opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.001);opt.load_state_dict(cpu_copy(start_opt))
  oldn=0 if stage==0 else sum(map(len,dc['stages'][:stage]));oldmodel=None
  if stage:
   prev=torch.load(src/f'head{stage-1}.private.pt',map_location='cpu',weights_only=False);assert prev['seen']==oldn;oldmodel=new_model(prev['delta']);del prev
  tail=(nc<250).tolist();protected=[c<oldn or tail[c] for c in range(n)]
  for b in [0,1]:
   split=next(x for x in feas['blocks'] if x['stage']==stage and x['fold']==b);S=[i for i,idx in enumerate(data.ids) if folds[idx]==1-b];Q=[i for i,idx in enumerate(data.ids) if folds[idx]==b];assert not set(S)&set(Q)
   load_delta(model,start_delta);pre=encode(model,data,S,n,cfg);initial=evaluate_slices(model,data,Q,pre['prototypes'],n,split['Qa_ids'],split['Qb_ids'],cfg);new_scores+=len(Q)
   oldeval=None
   if oldn:
    oldS=[i for i in S if int(data.labels[i])<oldn];oldQ=[i for i in Q if int(data.labels[i])<oldn];op=encode(oldmodel,data,oldS,oldn,cfg);oldeval=evaluate_slices(oldmodel,data,oldQ,op['prototypes'],oldn,split['Qa_ids'],split['Qb_ids'],cfg);new_scores+=len(oldQ)
   anchors={tag:[(oldeval[tag]['brier'][c] if c<oldn else initial[tag]['brier'][c]) for c in range(n)] for tag in initial}
   for rep in range(3):
    guard(cfg);seed=74002+2000000+10000*stage+100*b+10*rep;common=[]
    for h in range(3):common.append([S[i] for i in torch.randperm(len(S),generator=torch.Generator().manual_seed(seed+h)).tolist()[:192 if h<2 else 256]])
    fixed_first=allocation(data.labels,S,common[0],-1,torch.Generator().manual_seed(seed+800000));assert set(data.labels[fixed_first].tolist())==set(range(n));legal=list(range(n+1));actions=legal+[0,-1];block=[];blockid=f's{stage}_f{b}_r{rep}';folder=r/'HK/endpoints'/blockid;folder.mkdir(parents=True,exist_ok=False);baseline_delta=None;baseline_expo=None
    write(r/'STATUS.json',dict(phase='DIAGNOSTIC',stage=stage,fold=b,repeat=rep,actual_optimizer_steps=actual,block=blockid))
    for j,a in enumerate(actions):
     load_delta(model,start_delta);opt.load_state_dict(cpu_copy(start_opt));assert all(int(v['step'])==24*(stage+1) for v in opt.state.values());exposure=torch.zeros(n,dtype=torch.long);losses=[];alloc=[]
     for h in range(3):
      aa=a if h==0 else 0;ii=allocation(data.labels,S,common[h],-1,torch.Generator().manual_seed(seed+800000+h)) if h<2 else common[h];assert len(ii)==len(set(ii))==256 and not set(ii)&set(Q);exposure+=torch.bincount(data.labels[ii],minlength=n);alloc.append(dict(action=aa if h<2 else None,counts=torch.bincount(data.labels[ii],minlength=n).tolist(),CE_class_mass=[float(weights(data.labels[ii],n,aa)[data.labels[ii]==c].sum()) for c in range(n)] if h==0 else (torch.bincount(data.labels[ii],minlength=n).double()/256).tolist()));losses.append(share_step(model,opt,data,ii,pre['prototypes'],nc,cfg,aa));actual+=1;micro+=4
     assert all(torch.equal(v.detach().cpu(),frozen[k]) and v.grad is None for k,v in model.named_parameters() if not v.requires_grad)
     end=delta_state(model);post=encode(model,data,S,n,cfg);ev=evaluate_slices(model,data,Q,post['prototypes'],n,split['Qa_ids'],split['Qb_ids'],cfg);new_scores+=len(Q)
     if j==0:baseline_delta=cpu_copy(end);baseline_expo=exposure.clone()
     stats={}
     for tag,m in ev.items():
      covered=[c for c,z in enumerate(m['counts']) if z];pp=[c for c in covered if protected[c]];tt=[c for c in covered if tail[c]];damage=sum(max(0,m['brier'][c]-anchors[tag][c]) for c in pp)/len(pp);u=-sum(m['brier'][c] for c in covered)/len(covered)-damage
      stats[tag]=dict(utility=u,damage=damage,BA=100*sum(m['correct'][c]/m['counts'][c] for c in covered)/len(covered),tail_BA=100*sum(m['correct'][c]/m['counts'][c] for c in tt)/len(tt))
     filename=f'b{j:02d}.private.pt';torch.save(dict(delta=end,optimizer_state=cpu_copy(opt.state_dict()),prototypes=post['prototypes'],features=post['features'],labels=post['labels'],stage=stage,fold=b,repeat=rep,action=a,duplicate=j==len(actions)-2,fixed_mean=(j==len(actions)-1)),folder/filename)
     block.append(dict(action=a,duplicate=j==len(actions)-2,fixed_mean=(j==len(actions)-1),endpoint=blockid+'/'+filename,exposure=exposure.tolist(),exposure_minus_action0=(exposure-baseline_expo).tolist(),parameter_difference=parameters_diff(end,baseline_delta),losses=losses,allocation=alloc,query=ev,stats=stats,optimizer_steps=24*(stage+1)+3,actual_updates=3,micro_backwards=12))
     write(r/'LIVE_PROGRESS.json',dict(block=blockid,completed_branches=j+1,total_branches=len(actions),actual_updates=actual,micro_backwards=micro,new_TRAIN_query_rows_scored=new_scores));assert all(torch.cuda.max_memory_reserved(i)<12*2**30 for i in [0,1])
    rec=dict(stage=stage,fold=b,repeat=rep,seed=seed,seen=n,old_seen=oldn,tail=tail,protected=protected,source_optimizer_step=24*(stage+1),saved_lr=.0005,legal_actions=legal,source_delta=start_delta,source_optimizer_state=start_opt,initial=initial,anchors=anchors,branches=block)
    # Private exact source states allow saved-only restore checks without reopening any future head.
    torch.save(dict(delta=rec.pop('source_delta'),optimizer_state=rec.pop('source_optimizer_state')),folder/'START.private.pt');write(folder/'BLOCK.private.json',rec);records.append(rec);write(r/'HK/RECORDS.private.json',records)
  del oldmodel,state;torch.cuda.empty_cache()
 assert len(records)==36 and actual==2268 and micro==actual*4;write(r/'HK/ALL_ENDPOINTS_SEALED.json',dict(blocks=36,branches=sum(len(v['branches']) for v in records),fixed_mean_controls=36,actual_updates=actual,micro_backwards=micro,controller_updates=0,VAL_reads=0,TEST_reads=0,new_TRAIN_query_rows_scored=new_scores))
 engine=store();write(r/'HK/REGENERABLE_CACHE_CLEANUP.json',dict(encoded_rows=engine.encoded,reference_max_abs=engine.max_abs,pre_clear_RAM_bytes=engine.ram_bytes,root_spill_bytes=engine.spill_bytes,remote_spill_bytes=engine.remote_spill_bytes,scientific_artifacts_preserved=True));engine.index.clear();engine.disk_index.clear();engine.ram_bytes=0
 for folder in [engine.spill_root,engine.remote_spill_root]:
  for p in folder.glob('prefix_*.npy'):p.unlink()
 write(r/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
 try:main(cfg)
 except BaseException:(Path(cfg['output'])/'FAILURE.private.txt').write_text(traceback.format_exc());raise
