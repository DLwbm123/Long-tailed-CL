import json,time,math,copy,resource,csv
from pathlib import Path
import torch
import torch.nn.functional as F
from model import make_base,visual,textual,unit
from strong import StrongAdapter,load_teacher
from losses import Loss,tailness
from compute import fast_group,text_table
from data import train_rows,loader
from folds import assign_folds
from common import write,save,guard
COUNTS={'real_image_encoder_rows':0,'teacher_encoder_rows':0,'text_encoder_rows':0,'cached_visual_rows':0,'cached_text_rows':0,'classifier_rows':0,'formal_optimizer_steps':0,'SUP_optimizer_steps':0,'GRPO_optimizer_steps':0,'VAL_rows':0,'TEST_rows':0}

def safe(c):guard(c);assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED_LIMIT'
def head(c,s):
 h=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);n=sum(map(len,c['stages'][:s+1]));assert h['seen']==n and h['train_counts'].tolist()==c['train_counts'][:n] and h['prototypes'].shape==(n,512);return h

def meanproto(z,y,n):
 sums=torch.zeros(n,512,device=z.device,dtype=torch.double);sums.index_add_(0,y,z.double());count=torch.bincount(y,minlength=n);assert (count>0).all();return unit((sums/count[:,None]).float()),count

def prepare_teachers(c,base,tf,rows,y):
 r=Path(c['output']);records=[]
 for s in range(6):
  safe(c);h=head(c,s);load_teacher(base,h);n=h['seen'];ids=[i for i,v in enumerate(y.tolist()) if v<n];z=torch.zeros(len(rows),512);filled=torch.zeros(len(rows),dtype=torch.bool)
  with torch.no_grad():
   for x,j in loader(c,rows,ids,tf):safe(c);z[j]=visual(base,x.cuda()).cpu();filled[j]=True;COUNTS['real_image_encoder_rows']+=len(j);COUNTS['teacher_encoder_rows']+=len(j)
  p,den=meanproto(z[ids],y[ids],n);err=float((p-h['prototypes']).abs().max());assert err<=2e-5,('STRONG_PROTO_BINDING',s,err);assert torch.equal(den,h['train_counts'])
  fold=assign_folds(y[ids].tolist(),[rows[i]['identity_component'] for i in ids]);indices=[[ids[i] for i,f in enumerate(fold) if f==b] for b in [0,1]];proto=[];support=[];scores=torch.zeros(len(rows),n);correct=torch.zeros(n,dtype=torch.long)
  for b in [0,1]:
   pp,nn=meanproto(z[indices[1-b]],y[indices[1-b]],n);proto.append(pp);support.append(nn)
   with torch.no_grad():ss=(20*z[indices[b]].cuda()@pp.cuda().T).cpu()
   scores[indices[b]]=ss;yy=y[indices[b]];correct+=torch.bincount(yy[ss.argmax(1)==yy],minlength=n);COUNTS['classifier_rows']+=len(yy)
  birth=[b for b,cl in enumerate(c['stages'][:s+1]) for _ in cl];pack={'stage':s,'ids':ids,'folds':indices,'z':z,'filled':filled,'labels':y,'teacher_scores':scores,'fold_protos':proto,'fold_support':support,'full_proto':h['prototypes'],'raw_counts':den,'teacher_correct':correct,'births':birth};save(r/f'teacher_s{s}.private.pt',pack);records.append({'stage':s,'TRAIN_rows':len(ids),'fold_rows':list(map(len,indices)),'strong_proto_maxabs':err,'formal_optimizer':0,'VAL_TEST':0});write(r/'TEACHER_PREPARATION.json',{'stages':records,'counts':COUNTS});write(r/'STATUS.json',{'phase':'STRONG_TEACHER_CACHE','stage_complete':s+1,'counts':COUNTS})
 write(r/'TEACHER_CACHES_SEALED.json',{'stages':6,'all_prototype_binding_passed':True,'counts':COUNTS,'VAL_TEST':0})

def old_cache(c,base,m,rows,pack,tf):
 cache={}
 with torch.no_grad():
  for b,branch in enumerate(m.branches[:-1]):
   assert all(not x.requires_grad for x in branch.parameters());z=torch.zeros(len(rows),512);mask=torch.zeros(len(rows),dtype=torch.bool)
   for x,j in loader(c,rows,pack['ids'],tf):safe(c);z[j]=visual(base,x.cuda(),branch).cpu();mask[j]=True;COUNTS['real_image_encoder_rows']+=len(j)
   cache[b]=(z,mask)
 save(Path(c['output'])/f"old_visual_s{pack['stage']}.private.pt",cache);return cache

def fulltext(base,m,tokens,birth,ot):
 miss=sum(b<len(m.branches)-1 and b not in ot for b in birth);tt=text_table(base,m,tokens,birth,ot);new=sum(b==len(m.branches)-1 for b in birth);COUNTS['text_encoder_rows']+=new+miss;COUNTS['cached_text_rows']+=len(birth)-new-miss;return tt

def vbatch(base,m,x,j,cache):
 out=[]
 for b,branch in enumerate(m.branches):
  if b<len(m.branches)-1:
   z,mask=cache[b];assert mask[j].all();out.append(z[j].cuda());COUNTS['cached_visual_rows']+=len(j)
  else:out.append(visual(base,x.cuda(),branch));COUNTS['real_image_encoder_rows']+=len(j)
 return out

@torch.no_grad()
def observe(c,base,m,rows,y,pack,tf,tokens,cache,ot):
 birth=pack['births'];n=len(birth);m._births=birth;tt=fulltext(base,m,tokens,birth,ot);scores=torch.zeros(len(rows),n);cc=torch.zeros(n,dtype=torch.long)
 for b in [0,1]:
  support=pack['folds'][1-b];bank=pack['z'][support].cuda();by=y[support].cuda();p,_,_=m.prototypes(bank,by,n);m.context(pack['fold_protos'][b].cuda(),pack['raw_counts'].cuda(),pack['fold_support'][b].cuda(),birth,pack['stage'])
  for x,j in loader(c,rows,pack['folds'][b],tf):
   safe(c);raw=m.logits(vbatch(base,m,x,j,cache),tt,p,birth)[0];ss=m.combine(raw,pack['teacher_scores'][j].cuda()).cpu();scores[j]=ss;yy=y[j];cc+=torch.bincount(yy[ss.argmax(1)==yy],minlength=n);COUNTS['classifier_rows']+=len(j)
 ids=pack['ids'];tp=pack['teacher_scores'][ids].argmax(1)==y[ids];sp=scores[ids].argmax(1)==y[ids];harm={str(k):int(((y[ids]==k)&tp&~sp).sum()) for k in [1,2,12]};a=cc.double()/pack['raw_counts'];ta=pack['teacher_correct'].double()/pack['raw_counts'];tail=pack['raw_counts']<250
 return {'scores':scores,'correct':cc,'counts':pack['raw_counts'],'teacher_correct':pack['teacher_correct'],'SUP_macro':float(a.mean()*100),'SUP_tail':float(a[tail].mean()*100),'teacher_macro':float(ta.mean()*100),'teacher_tail':float(ta[tail].mean()*100),'tiny_harm':harm}

def checkpoint(c,m,opt,s,phase,pack,tokens,base,ot):
 with torch.no_grad():tt=fulltext(base,m,tokens,pack['births'],ot).cpu()
 save(Path(c['output'])/f's{s}_{phase}.private.pt',{'stage':s,'phase':phase,'model':m.state_dict(),'optimizer':opt.state_dict(),'text':tt,'births':pack['births'],'branch_count':s+1,'counts':dict(COUNTS)})

def fit_stage(c,base,m,opt,phase,stage,rows,y,pack,tf,tokens,cache,ot,reference=None,weights=None):
 r=Path(c['output']);epochs=c['epochs'] if phase=='SUP' else c['rl_epochs'];schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs);gen=torch.Generator(device='cuda').manual_seed(c['seed']+670000+stage);stats={};cost=[];n=len(pack['births']);raw=pack['raw_counts'].float().cuda()
 for ep in range(epochs):
  start=time.time();cpu=time.process_time();seen=0;total=0.
  for b in [0,1]:
   support=pack['folds'][1-b];bank=pack['z'][support].cuda();by=y[support].cuda();m.context(pack['fold_protos'][b].cuda(),raw,pack['fold_support'][b].cuda(),pack['births'],stage);batches=[];acc=0;foldseen=0;ids=pack['folds'][b]
   for x,j in loader(c,rows,ids,tf,shuffle=True,seed=c['seed']+10000*stage+100*ep+b+(800000 if phase=='JOINT' else 0)):
    batches.append((x,y[j],j));acc+=len(j)
    if acc==c['batch_size'] or foldseen+acc==len(ids):
     lossfn=Loss(pack['teacher_scores'],raw,'SUP' if phase=='SUP' else 'RL',reference,weights,gen,stats);updates=1 if phase=='SUP' else c['rl_updates']
     for update in range(updates):
      safe(c);lossfn.update=update;opt.zero_grad(set_to_none=True);loss,_=fast_group(base,m,batches,bank,by,tokens,pack['births'],cache,ot,COUNTS,lossfn=lossfn);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());opt.step();opt.zero_grad(set_to_none=True);COUNTS['formal_optimizer_steps']+=1;COUNTS['SUP_optimizer_steps' if phase=='SUP' else 'GRPO_optimizer_steps']+=1;COUNTS['classifier_rows']+=acc;total+=loss*acc/updates;write(r/'ACTIVE_COUNTS.private.json',{'phase':phase,'stage':stage,'epoch':ep,'fold':b,'seen':seen+foldseen+acc,'update':update,'counts':dict(COUNTS)})
     foldseen+=acc;acc=0;batches=[]
   assert foldseen==len(ids) and not batches;seen+=foldseen
  assert seen==len(pack['ids']);schedule.step();cost.append({'stage':stage,'phase':phase,'epoch':ep,'rows':seen,'loss':total/seen,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start});write(r/f'EPOCH_COST_s{stage}_{phase}.json',cost);write(r/'STATUS.json',{'phase':phase,'stage':stage,'epoch_complete':ep+1,'counts':COUNTS})
 if phase=='JOINT':write(r/f'GRPO_STATS_s{stage}.json',stats)

def train(c,prepared):
 start=time.time();cpu=time.process_time();r=Path(c['output']);base,tf,tok=prepared;rows,y=train_rows(c);prepare_teachers(c,base,tf,rows,y);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);m=StrongAdapter().cuda();sup=[];checks=[]
 for stage in range(6):
  safe(c);h=head(c,stage);load_teacher(base,h);pack=torch.load(r/f'teacher_s{stage}.private.pt',map_location='cpu',weights_only=False);torch.manual_seed(c['seed']+stage);torch.cuda.manual_seed_all(c['seed']+stage);m.add();m.zero_gate();cache=old_cache(c,base,m,rows,pack,tf);tokens=tok(['A photo of a '+s for s in c['names'][:h['seen']]]).cuda();ot={};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=c['lr'],weight_decay=c['weight_decay']);fit_stage(c,base,m,opt,'SUP',stage,rows,y,pack,tf,tokens,cache,ot);obs=observe(c,base,m,rows,y,pack,tf,tokens,cache,ot);save(r/f'SUP_TRAIN_s{stage}.private.pt',obs);checkpoint(c,m,opt,stage,'SUP',pack,tokens,base,ot);checks.append({k:v for k,v in obs.items() if k in ['SUP_macro','SUP_tail','teacher_macro','teacher_tail','tiny_harm']});sup.append(f's{stage}_SUP.private.pt');del opt,cache,pack;torch.cuda.empty_cache()
 assert COUNTS['SUP_optimizer_steps']==4620;write(r/'SUPERVISED_STATES_SEALED.json',{'snapshots':sup,'count':6,'counts':dict(COUNTS),'VAL_TEST':0});dm=[x['SUP_macro']-x['teacher_macro'] for x in checks];dt=[x['SUP_tail']-x['teacher_tail'] for x in checks];both=sum(a>=-1e-8 and b>=-1e-8 for a,b in zip(dm,dt));passed=sum(dm)/6>=-1e-8 and sum(dt)/6>=-1e-8 and both>=4 and all(sum(x['tiny_harm'][str(k)] for x in checks)==0 for k in [1,2,12]);write(r/'SUPERVISION_PRECHECK.json',{'passed':passed,'mean_macro_delta_pp':sum(dm)/6,'mean_tail_delta_pp':sum(dt)/6,'stages_both_nonnegative':both,'stage_observations':checks,'VAL_TEST':0})
 snapshots=sup[:]
 del m
 if passed:
  for stage in range(6):
   safe(c);h=head(c,stage);load_teacher(base,h);pack=torch.load(r/f'teacher_s{stage}.private.pt',map_location='cpu',weights_only=False);saved=torch.load(r/f's{stage}_SUP.private.pt',map_location='cpu',weights_only=False);m=StrongAdapter().cuda()
   for _ in range(stage+1):m.add()
   m.load_state_dict(saved['model'],strict=True);cache=torch.load(r/f'old_visual_s{stage}.private.pt',map_location='cpu',weights_only=False);obs=torch.load(r/f'SUP_TRAIN_s{stage}.private.pt',map_location='cpu',weights_only=False);w=tailness(pack['raw_counts'],obs['correct']).cuda();ref=obs['scores']+pack['raw_counts'].float().log()[None];save(r/f'TARS_s{stage}.private.pt',{'weights':w.cpu(),'counts':obs['counts'],'correct':obs['correct'],'fixed_before_RL':True});tokens=tok(['A photo of a '+s for s in c['names'][:h['seen']]]).cuda();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=c['rl_lr'],weight_decay=c['weight_decay']);ot={};fit_stage(c,base,m,opt,'JOINT',stage,rows,y,pack,tf,tokens,cache,ot,ref,w);checkpoint(c,m,opt,stage,'JOINT',pack,tokens,base,ot);snapshots.append(f's{stage}_JOINT.private.pt');del m,opt,cache,pack,saved;torch.cuda.empty_cache()
  assert COUNTS['GRPO_optimizer_steps']==1848 and COUNTS['formal_optimizer_steps']==6468
 write(r/'ALL_STATES_SEALED.json',{'snapshots':snapshots,'count':len(snapshots),'RL_trained':passed,'VAL_TEST':0,'counts':dict(COUNTS)});write(r/'TRAIN_COMPLETE.json',{'counts':COUNTS,'RL_trained':passed,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()});return base,tf

def evaluate(c,base,tf):
 r=Path(c['output']);seal=json.loads((r/'ALL_STATES_SEALED.json').read_text());modes=['TEACHER','SUP']+(['JOINT'] if seal['RL_trained'] else []);assert seal['count']==(12 if seal['RL_trained'] else 6);rows=list(csv.DictReader((Path(c['manifest'])/'val.csv').open()));assert all(x['split']=='val' for x in rows);y=torch.tensor([c['order'].index(int(x['original_label'])) for x in rows]);records=[]
 for stage in range(6):
  safe(c);h=head(c,stage);load_teacher(base,h);n=h['seen'];ids=[i for i,v in enumerate(y.tolist()) if v<n];teacher=[];ss={};allids=[];pack=torch.load(r/f'teacher_s{stage}.private.pt',map_location='cpu',weights_only=False);trainy=pack['labels'];trainids=pack['ids']
  with torch.no_grad():
   for x,j in loader(c,rows,ids,tf):safe(c);teacher.append(20*visual(base,x.cuda())@h['prototypes'].cuda().T);COUNTS['real_image_encoder_rows']+=len(j);COUNTS['VAL_rows']+=len(j);allids+=j.tolist()
   ss['TEACHER']=torch.cat(teacher).cpu();COUNTS['classifier_rows']+=len(ids);assert allids==ids
  for mode in modes[1:]:
   z=torch.load(r/f's{stage}_{mode}.private.pt',map_location='cpu',weights_only=False);m=StrongAdapter().cuda()
   for _ in range(stage+1):m.add()
   m.load_state_dict(z['model'],strict=True);birth=pack['births'];m.context(h['prototypes'].cuda(),h['train_counts'].cuda(),h['train_counts'].cuda(),birth,stage);vv=[];at=0
   with torch.no_grad():
    proto,_,_=m.prototypes(pack['z'][trainids].cuda(),trainy[trainids].cuda(),n);tt=z['text'].cuda()
    for x,j in loader(c,rows,ids,tf):
     safe(c);vs=[visual(base,x.cuda(),b) for b in m.branches];raw=m.logits(vs,tt,proto,birth)[0];vv.append(m.combine(raw,ss['TEACHER'][at:at+len(j)].cuda()).cpu());at+=len(j);COUNTS['real_image_encoder_rows']+=len(j)*len(m.branches);COUNTS['VAL_rows']+=len(j);COUNTS['classifier_rows']+=len(j)
   assert at==len(ids);ss[mode]=torch.cat(vv);del m,z;torch.cuda.empty_cache()
  save(r/f'VAL_scores_s{stage}.private.pt',{'ids':[rows[i]['sample_id'] for i in ids],'labels':y[ids],'scores':ss});cnt=torch.bincount(y[ids],minlength=n);metrics={};old=sum(map(len,c['stages'][:stage]));tail=torch.tensor(c['train_counts'][:n])<250
  for mode,score in ss.items():
   assert score.shape==(len(ids),n) and torch.isfinite(score).all();cor=torch.bincount(y[ids][score.argmax(1)==y[ids]],minlength=n);rec=cor.double()/cnt.clamp_min(1);available=cnt>0;groups={}
   for age,mask in [('old',torch.arange(n)<old),('new',torch.arange(n)>=old)]:
    for freq,fmask in [('head',~tail),('tail',tail)]:
     use=available&mask&fmask;groups[age+'_'+freq]=None if not use.any() else float(rec[use].mean()*100)
   metrics[mode]={'counts':cnt.tolist(),'correct':cor.tolist(),'recall':(rec*100).tolist(),'micro':float(cor.double().sum()/cnt.sum()*100),'macro':float(rec[available].mean()*100),'tail':float(rec[available&tail].mean()*100),'groups':groups}
  records.append({'stage':stage,'classes':n,'arms':metrics});write(r/'STAGE_METRICS.json',records);write(r/'STATUS.json',{'phase':'EVALUATE','stage_complete':stage+1,'counts':COUNTS})
 write(r/'EVAL_COMPLETE.json',{'stages':6,'modes':modes,'counts':COUNTS,'all_states_sealed_before_VAL':True})
