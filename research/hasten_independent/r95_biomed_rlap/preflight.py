import time,torch,copy,math,json,subprocess,os
from pathlib import Path
from model import Adapter,make_base,visual,textual,unit
from compute import fast_group,slow_group
from data import train_rows,loader
from common import write,guard

ENGINEERING={'image_rows':0,'text_rows':0,'optimizer_steps':0,'formal_optimizer':0,'VAL_TEST':0}
ACTIVE='not_started'

def new_model(cfg,arm,bcount):
 torch.manual_seed(cfg['seed']);m=Adapter(arm).cuda()
 for s in range(bcount):torch.manual_seed(cfg['seed']+s);m.add()
 return m

def native(cfg):
 root=Path(cfg['output']);cp=time.process_time();wt=time.time();base,tf,tok=make_base(cfg,'cuda');rows,y=train_rows(cfg);chosen=[]
 for c in [0,1,2]:
  got=[];comp=set()
  for i,yy in enumerate(y.tolist()):
   if yy==c and rows[i]['identity_component'] not in comp:got.append(i);comp.add(rows[i]['identity_component'])
   if len(got)==2:break
  assert len(got)==2;chosen.extend(got)
 images=torch.cat([x for x,j in loader(cfg,rows,chosen,tf)]);labels=y[chosen];images7=torch.cat([images,images[:1]]);labels7=torch.cat([labels,labels[:1]]);perm=[2,0,5,1,4,3,6];batches=[(images7[perm[k:k+2]],labels7[perm[k:k+2]],torch.tensor(perm[k:k+2])) for k in range(0,7,2)];tokens=tok(['A photo of a '+cfg['names'][i] for i in range(3)]).cuda();births=[0,0,1];checks=[];engineering=ENGINEERING
 for arm in ['FULL','CE_WEIGHT','UNIFORM']:
  global ACTIVE
  ACTIVE='native equivalence '+arm
  guard(cfg);m=new_model(cfg,arm,2)
  if arm!='UNIFORM':
   with torch.no_grad():m.policy.net[-1].weight.normal_(0,.01);m.reference.net[-1].weight.normal_(0,.01)
  m.reference_uniform=False;fast=copy.deepcopy(m);cache={0:(torch.zeros(7,512),torch.ones(7,dtype=torch.bool))}
  with torch.no_grad():
   for k in range(0,7,2):cache[0][0][k:k+2]=visual(base,images7[k:k+2].cuda(),m.branches[0]).cpu();engineering['image_rows']+=len(images7[k:k+2])
   bank=torch.cat([visual(base,images[2*c:2*c+2].cuda(),m.branches[births[c]]) for c in range(3)]);engineering['image_rows']+=6
  by=labels.cuda();o1=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay']);o2=torch.optim.AdamW([p for p in fast.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay']);o1.zero_grad();o2.zero_grad();l1,s1=slow_group(base,m,batches,bank,by,tokens,births);engineering['image_rows']+=14;engineering['text_rows']+=12;l2,s2=fast_group(base,fast,batches,bank,by,tokens,births,cache,{});engineering['image_rows']+=7;engineering['text_rows']+=3
  err=float((s1-s2).abs().max());lossdiff=abs(l1-l2);assert err<=2e-5 and lossdiff<=2e-5,(arm,err,lossdiff);ge=0.;gn=0.
  named=dict(fast.named_parameters())
  for name,p in m.named_parameters():
   q=named[name];assert (p.grad is None)==(q.grad is None),name
   if p.grad is not None:
    assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all();ee=float((p.grad-q.grad).abs().max());nn=ee/(1+float(p.grad.abs().max()));ge=max(ge,ee);gn=max(gn,nn);assert nn<=2e-4,(arm,name,ee,nn)
  assert all(p.grad is None and not p.requires_grad for p in base.parameters());assert fast.branches[-1].vprompt.grad.norm()>0 and fast.branches[-1].tprompt.grad.norm()>0
  o1.step();o2.step();engineering['optimizer_steps']+=2;pe=max(float((p-named[n]).abs().max())/(1+float(p.abs().max())) for n,p in m.named_parameters());assert pe<=2e-4,(arm,pe);checks.append({'arm':arm,'scores_maxabs':err,'loss_error':lossdiff,'gradient_maxabs':ge,'gradient_normalized_error':gn,'AdamW_parameter_normalized_error':pe})
  del m,fast,o1,o2,bank,cache,named;torch.cuda.empty_cache()
 assert all(p.grad is None and not p.requires_grad for p in base.parameters());reserved=torch.cuda.max_memory_reserved();assert reserved<12*2**30;own=[l for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if l.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==cfg['GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
 write(root/'NATIVE_EQUIVALENCE.json',{'passed':True,'checks':checks,'counts':engineering,'distinct_TRAIN_rows':6,'microbatch':2,'last_group_rows':7,'cache_deterministic_preprocess':True,'all_base_frozen':True,'GPU_reserved_bytes':reserved,'own_GPU1_binding':True,'CPU_seconds':time.process_time()-cp,'wall_seconds':time.time()-wt})
 return base,tf,tok,images,engineering

def budget(cfg,prepared,images,engineering):
 root=Path(cfg['output']);base,tf,tok=prepared;records=[]
 for C,births in [(13,[0]*13),(23,[0]*13+[s for s in range(1,6) for _ in range(2)])]:
  guard(cfg);m=new_model(cfg,'FULL',max(births)+1);N=16;xx=images[torch.arange(N)%len(images)];yy=torch.arange(N)%C;batches=[(xx[k:k+2],yy[k:k+2],torch.arange(k,k+2)) for k in range(0,N,2)];bank=unit(torch.randn(C*20,512,device='cuda'));by=torch.arange(C,device='cuda').repeat_interleave(20);tokens=tok(['A photo of a '+cfg['names'][i] for i in range(C)]).cuda();cache={b:(unit(torch.randn(N,512)),torch.ones(N,dtype=torch.bool)) for b in range(len(m.branches)-1)};ot={};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg['weight_decay']);times=[]
  for repeat in range(2):
   torch.cuda.synchronize();start=time.time();opt.zero_grad();fast_group(base,m,batches,bank,by,tokens,births,cache,ot);opt.step();torch.cuda.synchronize();times.append(time.time()-start);engineering['image_rows']+=16;engineering['text_rows']+=sum(k==max(births) for k in births)+(sum(k<max(births) for k in births) if repeat==0 else 0);engineering['optimizer_steps']+=1;assert torch.cuda.max_memory_reserved()<12*2**30
  records.append({'classes':C,'branches':len(m.branches),'step_seconds':times});last=m
 forward=[]
 with torch.no_grad():
  for B in [1,2]:
   tt=[]
   for repeat in range(2):
    torch.cuda.synchronize();start=time.time();visual(base,images[:B].cuda(),last.branches[0]);torch.cuda.synchronize();tt.append(time.time()-start);engineering['image_rows']+=B
   forward.append({'rows':B,'seconds':tt})
 step=max(r['step_seconds'][1] for r in records);perrow=max(r['seconds'][1]/r['rows'] for r in forward);seen=0;memory=0;trainlens=[];memorynew=[];cache_rows=0
 for s,cc in enumerate(cfg['stages']):
  new=sum(cfg['train_counts'][c] for c in cc);nr=new+memory;trainlens.append(nr);cache_rows+=3*s*nr;mn=sum(min(20,cfg['train_counts'][c]) for c in cc);memorynew.append(mn);memory+=mn
 steps=3*cfg['epochs']*sum(math.ceil(n/cfg['batch_size']) for n in trainlens);assert steps==31260;bank_rows=3*(cfg['epochs']+1)*sum(memorynew);arrival=sum(cfg['train_counts']);eval_upper=1698*sum(3*(s+1)+1 for s in range(6));forwardrows=cache_rows+bank_rows+arrival+eval_upper;projection=steps*step*2+forwardrows*perrow*2+7200;remaining=cfg['deadline']-time.time();passed=projection<remaining-180
 write(root/'BUDGET.json',{'passed':passed,'complete_step_profiles':records,'frozen_forward_profiles':forward,'fixed_second_pass_max_step_seconds':step,'frozen_seconds_per_row_upper':perrow,'formal_steps':steps,'stage_TRAIN_rows':trainlens,'old_cache_rows':cache_rows,'current_bank_rows':bank_rows,'arrival_rows':arrival,'VAL_encoder_rows_upper':eval_upper,'nonstep_forward_rows_upper':forwardrows,'safety_factor':2,'IO_save_analysis_reserve_seconds':7200,'projected_seconds':projection,'remaining_seconds':remaining,'GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'engineering_counts':engineering,'formal_optimizer':0,'VAL_TEST':0})
 del m,opt,cache,bank,ot,last;torch.cuda.empty_cache();return passed
