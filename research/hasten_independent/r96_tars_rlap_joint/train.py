import os,json,time,copy,math,resource,csv
from pathlib import Path
import torch
import torch.nn.functional as F
from model import Adapter,make_base,visual,textual,unit
from data import train_rows,loader
from common import write,guard
from compute import fast_group
ARMS=['SFT','JOINT']
COUNTS={'real_image_encoder_rows':0,'text_encoder_rows':0,'formal_optimizer_steps':0,'formal_visual_optimizer_steps':0,'VAL_rows':0,'TEST_rows':0,'cached_visual_rows':0,'cached_text_rows':0}
def safe(cfg):
 guard(cfg);assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED_LIMIT'
def frozen_cache(cfg,base,m,rows,ids,tf,stage,arm):
 out={}
 with torch.no_grad():
  for b,branch in enumerate(m.branches[:-1]):
   assert all(not p.requires_grad for p in branch.parameters())
   values=torch.zeros(len(rows),512);filled=torch.zeros(len(rows),dtype=torch.bool)
   for x,j in loader(cfg,rows,ids,tf):
    safe(cfg);values[j]=visual(base,x.cuda(),branch).cpu();filled[j]=True;COUNTS['real_image_encoder_rows']+=len(j)
   out[b]=(values,filled)
 torch.save(out,Path(cfg['output'])/f'frozen_s{stage}_{arm}.private.pt');return out

def encode(cfg,base,m,rows,ids,tf,births,y,cache=None):
 out=[];lab=[];ordered=[]
 with torch.no_grad():
  for b in range(len(m.branches)):
   ix=[i for i in ids if births[int(y[i])]==b]
   if cache is not None and b<len(m.branches)-1:
    j=torch.tensor(ix);values,filled=cache[b];assert bool(filled[j].all());out.append(values[j].cuda());lab.append(y[j].cuda());ordered.extend(ix);COUNTS['cached_visual_rows']+=len(j)
   else:
    for x,j in loader(cfg,rows,ix,tf):
     safe(cfg);out.append(visual(base,x.cuda(),m.branches[b]));lab.append(y[j].cuda());ordered.extend(j.tolist());COUNTS['real_image_encoder_rows']+=len(j)
 return torch.cat(out),torch.cat(lab),ordered

def texts(base,m,tokens,births,train):
 z=[]
 for b in range(len(m.branches)):
  ix=[i for i,t in enumerate(births) if t==b]
  if not ix:continue
  with torch.set_grad_enabled(train and b==len(m.branches)-1):a=textual(base,tokens[ix],m.branches[b])
  z.append(a);COUNTS['text_encoder_rows']+=len(ix)
 return torch.cat(z)
def batch_visual(base,m,x,train):
 z=[]
 for b,branch in enumerate(m.branches):
  with torch.set_grad_enabled(train and b==len(m.branches)-1):a=visual(base,x,branch)
  z.append(a);COUNTS['real_image_encoder_rows']+=len(x)
 return z

def save_endpoint(cfg,m,opt,stage,arm,bank,labels,ids,text,births):
 r=Path(cfg['output']);name=f's{stage}_{arm}.private.pt';torch.save({'stage':stage,'arm':arm,'model':m.state_dict(),'optimizer':opt.state_dict(),'bank':bank.cpu(),'bank_labels':labels.cpu(),'bank_ids':ids,'text':text.cpu(),'births':births,'reference_uniform':m.reference_uniform,'counts':dict(COUNTS)},r/name)
 return name

def posttrain(cfg,base,tf,rows,y,anchor,trainids,memids,stage,tokens,births,cache,old_text,epochcost):
 from joint import Group,tailness
 r=Path(cfg['output']);n=len(births);anchor._births=births;ref=torch.zeros(len(rows),n);correct=torch.zeros(n,dtype=torch.long);counts=torch.zeros(n,dtype=torch.long)
 with torch.no_grad():
  bank,by,bankids=encode(cfg,base,anchor,rows,memids,tf,births,y,cache);tt=texts(base,anchor,tokens,births,False);proto,_,_=anchor.prototypes(bank,by,n)
  for x,j in loader(cfg,rows,trainids,tf):
   safe(cfg);vs=[]
   for b,branch in enumerate(anchor.branches):
    if b<len(anchor.branches)-1:
     vv,filled=cache[b];assert bool(filled[j].all());vs.append(vv[j].cuda());COUNTS['cached_visual_rows']+=len(j)
    else:vs.append(visual(base,x.cuda(),branch));COUNTS['real_image_encoder_rows']+=len(j)
   ss=anchor.logits(vs,tt,proto,births)[0].cpu();ref[j]=ss;yy=y[j];counts+=torch.bincount(yy,minlength=n);correct+=torch.bincount(yy[ss.argmax(1)==yy],minlength=n)
 assert bool((counts>0).all());raw=torch.tensor(cfg['train_counts'][:n],dtype=torch.double);weights=tailness(raw,(correct.double()/counts)*raw).cuda();torch.save({'reference':ref,'legal_ids':trainids,'observed_counts':counts,'correct':correct,'raw_counts':raw,'weights':weights.cpu()},r/f'TARS_s{stage}.private.pt')
 m=copy.deepcopy(anchor);m.arm='JOINT';opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['rl_lr'],weight_decay=cfg['weight_decay']);gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+90000+stage);stats={};old_text={k:v.detach() for k,v in old_text.items()}
 for epoch in range(cfg['rl_epochs']):
  es=time.time();ec=time.process_time();bank,by,bankids=encode(cfg,base,m,rows,memids,tf,births,y,cache);seen=0;accum=0;batches=[];running=0.
  for x,j in loader(cfg,rows,trainids,tf,shuffle=True,seed=cfg['seed']+80000+stage*100+epoch):
   batches.append((x,y[j],j));accum+=len(j)
   if accum==cfg['batch_size'] or seen+accum==len(trainids):
    group=Group(ref,weights,gen,stats)
    for update in range(cfg['rl_updates']):
     safe(cfg);group.update=update;opt.zero_grad(set_to_none=True);value,_=fast_group(base,m,batches,bank,by,tokens,births,cache,old_text,COUNTS,lossfn=group);opt.step();opt.zero_grad(set_to_none=True);COUNTS['formal_optimizer_steps']+=1;COUNTS['formal_visual_optimizer_steps']+=1;COUNTS['GRPO_optimizer_steps']=COUNTS.get('GRPO_optimizer_steps',0)+1;running+=value*accum/cfg['rl_updates'];write(r/'ACTIVE_COUNTS.private.json',{'stage':stage,'arm':'JOINT','epoch':epoch,'seen':seen+accum,'update':update,'counts':dict(COUNTS)})
    seen+=accum;accum=0;batches=[]
  assert seen==len(trainids) and not batches;epochcost.append({'stage':stage,'arm':'JOINT','epoch':epoch,'rows':seen,'loss':running/seen,'CPU_seconds':time.process_time()-ec,'wall_seconds':time.time()-es});write(r/'EPOCH_COST.json',epochcost);write(r/f'RL_STATS_s{stage}.json',stats)
 bank,by,bankids=encode(cfg,base,m,rows,memids,tf,births,y,cache);text=texts(base,m,tokens,births,False).detach();name=save_endpoint(cfg,m,opt,stage,'JOINT',bank,by,bankids,text,births);del m,opt;return name

def train(cfg,prepared=None):
 start=time.time();cpu=time.process_time();r=Path(cfg['output']);base,tf,tokenize=make_base(cfg,'cuda') if prepared is None else prepared;rows,y=train_rows(cfg);models={};opts={};memory={};names=[];births=[];snapshots=[];epochcost=[];base_means=[];memorymeans=[]
 for arm in ['SFT']:
  torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);models[arm]=Adapter(arm).cuda()
 for stage,classes in enumerate(cfg['stages']):
  safe(cfg);write(r/'STATUS.json',{'phase':'TRAIN','stage':stage,'counts':COUNTS});new=[i for i,c in enumerate(y.tolist()) if c in classes]
  # Selection is label-available TRAIN-only, shared between all arms, never future data.
  vv=[];ii=[]
  with torch.no_grad():
   for x,j in loader(cfg,rows,new,tf):
    vv.append(visual(base,x.cuda()).cpu());ii.extend(j.tolist());COUNTS['real_image_encoder_rows']+=len(j);safe(cfg)
  vv=torch.cat(vv);ii=torch.tensor(ii)
  for c in classes:
   z=vv[y[ii]==c];ix=ii[y[ii]==c];mu=unit(z.mean(0));rank=torch.argsort(((z-mu)**2).sum(1),stable=True);memory[c]=ix[rank[:min(20,len(ix))]].tolist();base_means.append(mu);memorymeans.append(unit(z[rank[:min(20,len(ix))]].mean(0)))
  names.extend(cfg['names'][c] for c in classes);births.extend([stage]*len(classes));n=len(names);tokens=tokenize(['A photo of a '+z for z in names]).cuda();memids=[i for c in range(n) for i in memory[c]];trainids=new+[i for c in range(n) if c not in classes for i in memory[c]]
  write(r/f'memory_s{stage}.private.json',{'stage':stage,'classes':memory,'training_rows':len(trainids),'distinct_training_rows':len(set(trainids))});assert len(trainids)==len(set(trainids));torch.save({'memory':torch.stack(memorymeans),'all_TRAIN':torch.stack(base_means),'ids':copy.deepcopy(memory),'COUNTS':dict(COUNTS)},r/f'NCM_s{stage}.private.pt')
  for arm in ['SFT']:
   m=models[arm];torch.manual_seed(cfg['seed']+stage);torch.cuda.manual_seed_all(cfg['seed']+stage);m.add()
   # Carry shared AdamW moments; new task-specific parameters start with fresh moments.
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
   if arm in opts:
    old=opts[arm]
    for p in opt.param_groups[0]['params']:
     if p in old.state:opt.state[p]=old.state[p]
   opts[arm]=opt
   cache=frozen_cache(cfg,base,m,rows,trainids,tf,stage,arm);old_text={}
   for epoch in range(cfg['epochs']):
    es=time.time();ec=time.process_time();safe(cfg);bank,by,bankids=encode(cfg,base,m,rows,memids,tf,births,y,cache);seen=0;running=0.;batches=[];accum=0
    for x,j in loader(cfg,rows,trainids,tf,shuffle=True,seed=cfg['seed']+1000*stage+epoch):
     batches.append((x,y[j],j));accum+=len(j)
     if accum==cfg['batch_size'] or seen+accum==len(trainids):
      safe(cfg);opt.zero_grad(set_to_none=True);loss,_=fast_group(base,m,batches,bank,by,tokens,births,cache,old_text,COUNTS);opt.step();opt.zero_grad(set_to_none=True);COUNTS['formal_optimizer_steps']+=1;COUNTS['formal_visual_optimizer_steps']+=1;seen+=accum;running+=loss*accum;accum=0;batches=[]
      write(r/'ACTIVE_COUNTS.private.json',{'stage':stage,'arm':arm,'epoch':epoch,'seen':seen,'counts':dict(COUNTS)})
    assert seen==len(trainids) and not batches;epochcost.append({'stage':stage,'arm':arm,'epoch':epoch,'rows':seen,'loss':running/seen,'CPU_seconds':time.process_time()-ec,'wall_seconds':time.time()-es});write(r/'EPOCH_COST.json',epochcost);write(r/'STATUS.json',{'phase':'TRAIN','stage':stage,'arm':arm,'epoch_complete':epoch+1,'counts':COUNTS})
   bank,by,bankids=encode(cfg,base,m,rows,memids,tf,births,y,cache);text=texts(base,m,tokens,births,False).detach();snapshots.append(save_endpoint(cfg,m,opt,stage,arm,bank,by,bankids,text,births));safe(cfg)
   snapshots.append(posttrain(cfg,base,tf,rows,y,m,trainids,memids,stage,tokens,births,cache,old_text,epochcost))
  # Explicitly discard no scientific endpoint or optimizer state.
 write(r/'ALL_HEADS_SEALED.json',{'snapshots':snapshots,'count':len(snapshots),'VAL_reads':0,'TEST_reads':0,'counts':COUNTS});assert len(snapshots)==12 and COUNTS['formal_optimizer_steps']==7294
 write(r/'TRAIN_COMPLETE.json',{'counts':COUNTS,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 return base,tf

def evaluate(cfg,base,tf):
 r=Path(cfg['output']);seal=json.loads((r/'ALL_HEADS_SEALED.json').read_text());assert seal['count']==12
 with open(Path(cfg['manifest'])/'val.csv') as f:rows=list(csv.DictReader(f))
 assert all(z['split']=='val' for z in rows);y=torch.tensor([cfg['order'].index(int(z['original_label'])) for z in rows]);records=[]
 for stage,classes in enumerate(cfg['stages']):
  n=sum(map(len,cfg['stages'][:stage+1]));ids=[i for i,c in enumerate(y.tolist()) if c<n];cnt=torch.bincount(y[ids],minlength=n);scores={}
  for arm in ARMS:
   safe(cfg);z=torch.load(r/f's{stage}_{arm}.private.pt',map_location='cpu',weights_only=False);m=Adapter(arm)
   for _ in range(stage+1):m.add()
   m.load_state_dict(z['model']);m.reference_uniform=z['reference_uniform'];m.cuda().eval();bank=z['bank'].cuda();by=z['bank_labels'].cuda();text=z['text'].cuda();births=z['births'];vv=[];outids=[]
   with torch.no_grad():
    proto,w,_=m.prototypes(bank,by,n)
    for x,j in loader(cfg,rows,ids,tf):
     safe(cfg);v=batch_visual(base,m,x.cuda(),False);pred=m.logits(v,text,proto,births)[0];vv.append(pred.cpu());outids.extend(j.tolist());COUNTS['VAL_rows']+=len(j)
   scores[arm]=torch.cat(vv);assert outids==ids;del m;torch.cuda.empty_cache()
  ncm=torch.load(r/f'NCM_s{stage}.private.pt',weights_only=False,map_location='cpu');fp=[]
  with torch.no_grad():
   for x,j in loader(cfg,rows,ids,tf):
    fp.append(visual(base,x.cuda()).cpu());COUNTS['real_image_encoder_rows']+=len(j);COUNTS['VAL_rows']+=len(j);safe(cfg)
  fp=torch.cat(fp);scores['NCM']=fp@ncm['memory'].T;scores['NCM_ALL']=fp@ncm['all_TRAIN'].T
  torch.save({'stage':stage,'ids':[rows[i]['sample_id'] for i in ids],'labels':y[ids],'scores':scores},r/f'VAL_scores_s{stage}.private.pt')
  metrics={}
  for arm,s in scores.items():
   cor=torch.bincount(y[ids][s.argmax(1)==y[ids]],minlength=n);rec=cor.double()/cnt.clamp_min(1);available=cnt>0;tail=torch.tensor(cfg['train_counts'][:n])<250;old=sum(map(len,cfg['stages'][:stage]));groups={}
   for name,mask in [('old_head',(torch.arange(n)<old)&~tail),('old_tail',(torch.arange(n)<old)&tail),('new_head',(torch.arange(n)>=old)&~tail),('new_tail',(torch.arange(n)>=old)&tail)]:
    eligible=mask&available;groups[name]=None if not eligible.any() else float(rec[eligible].mean()*100)
   metrics[arm]={'counts':cnt.tolist(),'correct':cor.tolist(),'recall':(rec*100).tolist(),'macro':float(rec[available].mean()*100),'tail':float(rec[available&tail].mean()*100),'micro':float(cor.sum()/cnt.sum()*100),'groups':groups}
  records.append({'stage':stage,'classes':n,'arms':metrics});write(r/'STAGE_METRICS.json',records);write(r/'STATUS.json',{'phase':'EVALUATE','stage_complete':stage+1,'counts':COUNTS})
 write(r/'EVAL_COMPLETE.json',{'stages':6,'counts':COUNTS,'all_heads_sealed_before_VAL':True})
