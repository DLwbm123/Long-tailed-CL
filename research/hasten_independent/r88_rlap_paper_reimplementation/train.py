import os,json,time,copy,math,resource,csv
from pathlib import Path
import torch
import torch.nn.functional as F
from model import Adapter,make_base,visual,textual,unit
from data import train_rows,loader
from common import write,guard
ARMS=['FULL','CE_WEIGHT','UNIFORM']
COUNTS={'real_image_encoder_rows':0,'text_encoder_rows':0,'formal_optimizer_steps':0,'formal_visual_optimizer_steps':0,'VAL_rows':0,'TEST_rows':0}
def safe(cfg):
 guard(cfg);assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED_LIMIT'
def encode(cfg,base,m,rows,ids,tf,births,y):
 out=[];lab=[];ordered=[]
 with torch.no_grad():
  for b in range(len(m.branches)):
   ix=[i for i in ids if births[int(y[i])]==b]
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

def train(cfg):
 start=time.time();cpu=time.process_time();r=Path(cfg['output']);base,tf,tokenize=make_base(cfg,'cuda');rows,y=train_rows(cfg);models={};opts={};memory={};names=[];births=[];snapshots=[];epochcost=[];base_means=[];memorymeans=[]
 for arm in ARMS:
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
  for arm in ARMS:
   m=models[arm];torch.manual_seed(cfg['seed']+stage);torch.cuda.manual_seed_all(cfg['seed']+stage);m.add()
   # Carry shared AdamW moments; new task-specific parameters start with fresh moments.
   opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay'])
   if arm in opts:
    old=opts[arm]
    for p in opt.param_groups[0]['params']:
     if p in old.state:opt.state[p]=old.state[p]
   opts[arm]=opt
   for epoch in range(cfg['epochs']):
    es=time.time();ec=time.process_time();safe(cfg);bank,by,bankids=encode(cfg,base,m,rows,memids,tf,births,y);accum=0;opt.zero_grad(set_to_none=True);running=0.;batches=list(range(len(trainids)));seen=0
    for x,j in loader(cfg,rows,trainids,tf,shuffle=True,seed=cfg['seed']+1000*stage+epoch):
     safe(cfg);x=x.cuda();yy=y[j].cuda();proto,w,rl=m.prototypes(bank,by,n);text=texts(base,m,tokens,births,True);vs=batch_visual(base,m,x,True);scores,d,balance=m.logits(vs,text,proto,births);ce=(-(1+d.gather(1,yy[:,None]).squeeze(1))*scores.log_softmax(1).gather(1,yy[:,None]).squeeze(1)).mean();vactive=vs[-1];tt=text[yy];sim=vactive@tt.T/m.log_temp.exp().clamp(.01,1);target=torch.arange(len(x),device=x.device);clip=.5*(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target));loss=ce+clip+.01*balance+(.1*rl if arm=='FULL' else 0);assert torch.isfinite(loss)
     # Last effective batch is scaled by its own actual row count.
     group_start=(seen//cfg['batch_size'])*cfg['batch_size'];den=min(cfg['batch_size'],len(trainids)-group_start);(loss*len(x)/den).backward();accum+=len(x);seen+=len(x);running+=float(loss.detach())*len(x)
     if accum==cfg['batch_size'] or seen==len(trainids):
      opt.step();opt.zero_grad(set_to_none=True);accum=0;COUNTS['formal_optimizer_steps']+=1;COUNTS['formal_visual_optimizer_steps']+=1
    assert seen==len(trainids);epochcost.append({'stage':stage,'arm':arm,'epoch':epoch,'rows':seen,'loss':running/seen,'CPU_seconds':time.process_time()-ec,'wall_seconds':time.time()-es});write(r/'EPOCH_COST.json',epochcost);write(r/'STATUS.json',{'phase':'TRAIN','stage':stage,'arm':arm,'epoch_complete':epoch+1,'counts':COUNTS})
   bank,by,bankids=encode(cfg,base,m,rows,memids,tf,births,y);text=texts(base,m,tokens,births,False).detach();snapshots.append(save_endpoint(cfg,m,opt,stage,arm,bank,by,bankids,text,births));m.update_reference();safe(cfg)
  # Explicitly discard no scientific endpoint or optimizer state.
 write(r/'ALL_HEADS_SEALED.json',{'snapshots':snapshots,'count':len(snapshots),'VAL_reads':0,'TEST_reads':0,'counts':COUNTS});assert len(snapshots)==18
 write(r/'TRAIN_COMPLETE.json',{'counts':COUNTS,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()})
 return base,tf

def evaluate(cfg,base,tf):
 r=Path(cfg['output']);seal=json.loads((r/'ALL_HEADS_SEALED.json').read_text());assert seal['count']==18
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
