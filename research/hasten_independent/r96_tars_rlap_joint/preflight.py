import copy,time,math,os,subprocess,torch
from pathlib import Path
from model import Adapter,make_base,visual,unit
from compute import fast_group
from joint import Group
from data import train_rows,loader
from common import write,guard
ENGINEERING={'image_rows':0,'text_rows':0,'optimizer_steps':0,'formal_optimizer':0,'VAL_TEST':0}

def model(cfg,C):
 torch.manual_seed(cfg['seed']);m=Adapter('JOINT').cuda();births=[0]*13 if C==13 else [0]*13+[s for s in range(1,6) for _ in range(2)]
 for s in range(max(births)+1):torch.manual_seed(cfg['seed']+s);m.add()
 return m,births

def fixture(cfg,base,tok,images,C):
 m,births=model(cfg,C);xx=images[torch.arange(16)%len(images)];yy=torch.arange(16)%C;batches=[(xx[k:k+2],yy[k:k+2],torch.arange(k,k+2)) for k in range(0,16,2)];bank=unit(torch.randn(C*20,512,device='cuda'));by=torch.arange(C,device='cuda').repeat_interleave(20);tokens=tok(['A photo of a '+n for n in cfg['names'][:C]]).cuda();cache={b:(unit(torch.randn(16,512)),torch.ones(16,dtype=torch.bool)) for b in range(len(m.branches)-1)};return m,births,batches,bank,by,tokens,cache

def native_budget(cfg):
 r=Path(cfg['output']);start=time.time();cpu=time.process_time();base,tf,tok=make_base(cfg,'cuda');rows,y=train_rows(cfg);ids=[]
 for c in [0,1,2]:
  got=[];comp=set()
  for i,yy in enumerate(y.tolist()):
   if yy==c and rows[i]['identity_component'] not in comp:got.append(i);comp.add(rows[i]['identity_component'])
   if len(got)==2:break
  assert len(got)==2;ids+=got
 images=torch.cat([x for x,j in loader(cfg,rows,ids,tf)]);profiles=[];native=None
 for C in [13,23]:
  m,births,batches,bank,by,tokens,cache=fixture(cfg,base,tok,images,C);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['rl_lr'],weight_decay=cfg['weight_decay']);ot={};supervised=[];rl=[];stats={};reference=torch.zeros(16,C);weights=torch.linspace(1,2,C,device='cuda');gen=torch.Generator(device='cuda').manual_seed(74002)
  for repeat in range(2):
   guard(cfg);torch.cuda.synchronize();a=time.time();opt.zero_grad(set_to_none=True);fast_group(base,m,batches,bank,by,tokens,births,cache,ot,ENGINEERING_COUNTER);opt.step();torch.cuda.synchronize();supervised.append(time.time()-a);ENGINEERING['optimizer_steps']+=1
   group=Group(reference,weights,gen,stats);torch.cuda.synchronize();a=time.time()
   for update in range(2):
    group.update=update;opt.zero_grad(set_to_none=True);v,_=fast_group(base,m,batches,bank,by,tokens,births,cache,ot,ENGINEERING_COUNTER,lossfn=group);assert math.isfinite(v)
    if C==13 and repeat==0 and update==0:
     assert m.branches[-1].vprompt.grad.norm()>0 and m.branches[-1].tprompt.grad.norm()>0;assert m.policy.net[-1].weight.grad.norm()>0;assert all(p.grad is None and not p.requires_grad for p in base.parameters());native={'prompt_and_prototype_policy_nonzero_gradient':True,'base_frozen':True,'real_distinct_TRAIN_rows':6,'synthetic_bank_labels_for_engineering':True}
    opt.step();ENGINEERING['optimizer_steps']+=1
   torch.cuda.synchronize();rl.append(time.time()-a)
  saved=copy.deepcopy(m.state_dict());m.load_state_dict(saved,strict=True);assert all(torch.isfinite(x).all() for x in saved.values());profiles.append({'classes':C,'supervised_step_seconds':supervised,'two_GRPO_updates_seconds':rl});del opt,bank,cache,group,saved;torch.cuda.empty_cache()
 fw=[]
 with torch.no_grad():
  for B in [1,2]:
   tt=[]
   for rep in range(2):
    torch.cuda.synchronize();a=time.time();visual(base,images[:B].cuda(),m.branches[-1]);torch.cuda.synchronize();tt.append(time.time()-a);ENGINEERING_COUNTER['real_image_encoder_rows']+=B
   fw.append({'rows':B,'seconds':tt})
 own=[l for l in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines() if l.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==cfg['GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']=='1';assert torch.cuda.max_memory_reserved()<12*2**30
 native.update(passed=True,strict_restore=True,own_GPU1=True,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start);write(r/'NATIVE.json',native)
 memory=0;lens=[];cache_rows=0;banknew=0
 for s,cc in enumerate(cfg['stages']):
  nr=sum(cfg['train_counts'][c] for c in cc)+memory;lens.append(nr);cache_rows+=s*nr;mn=sum(min(20,cfg['train_counts'][c]) for c in cc);memory+=mn;banknew+=mn
 updates=sum(math.ceil(n/16) for n in lens);sft=updates*10;groups=updates*2;assert sft+groups*2==7294
 # Includes 10SFT+2RL banks, final banks, anchor score construction and extra IO margin.
 forward=cache_rows+20*banknew+sum(lens)+sum(cfg['train_counts'])+1698*sum(2*(s+1)+1 for s in range(6));perrow=max(t['seconds'][1]/t['rows'] for t in fw);projection=2*(sft*max(t['supervised_step_seconds'][1] for t in profiles)+groups*max(t['two_GRPO_updates_seconds'][1] for t in profiles)+forward*perrow)+7200;remaining=cfg['deadline']-time.time();passed=projection<remaining-180
 ENGINEERING.update(image_rows=ENGINEERING_COUNTER['real_image_encoder_rows'],text_rows=ENGINEERING_COUNTER['text_encoder_rows']);write(r/'BUDGET.json',{'passed':passed,'profiles':profiles,'frozen_forward':fw,'safety_factor':2,'reserve_seconds':7200,'projected_seconds':projection,'remaining_seconds':remaining,'supervised_steps':sft,'GRPO_groups':groups,'GRPO_steps':groups*2,'nonstep_encoder_upper':forward,'engineering_counts':ENGINEERING,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()});del m;torch.cuda.empty_cache();return passed,(base,tf,tok)
ENGINEERING_COUNTER={'text_encoder_rows':0,'cached_text_rows':0,'real_image_encoder_rows':0,'cached_visual_rows':0}
