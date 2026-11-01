import json,time,resource,os,subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import counts
from program import Dictionary,PROGRAMS,make_base,unit
from data import train_rows,loader
from common import write,guard

def run(cfg):
 start=time.time();cpu=time.process_time();base,tf=make_base(cfg);rows,y=train_rows(cfg);roles=json.loads(Path(cfg['output'],'ROLES.private.json').read_text());ids=[]
 for c in [0,1]:ids.append(next(i for i in roles['S'] if int(y[i])==c))
 images=next(iter(loader(cfg,rows,ids,tf)))[0].cuda();torch.manual_seed(cfg['seed']);dictionary=Dictionary().cuda();before=counts.C['engineering_encoder_rows']
 with torch.no_grad():
  z=dictionary(base,images,(0,0));counts.encoder(2);reference=unit(base.encode_image(images));parity=float((z-reference).abs().max());assert parity<=2e-5
 torch.cuda.synchronize();fstart=time.time();banks=[]
 with torch.no_grad():
  for program in PROGRAMS:banks.append(dictionary(base,images,program).detach())
 torch.cuda.synchronize();forward_wall=time.time()-fstart;baseline=banks[0]@banks[0].T;effect=max(float(((bank@bank.T)-baseline).abs().max()) for bank in banks);assert effect>1e-7
 opt=torch.optim.AdamW(dictionary.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);opt.zero_grad();torch.cuda.synchronize();gstart=time.time()
 for k,program in enumerate(PROGRAMS[1:],1):
  z=dictionary(base,images,program);loss=F.cross_entropy(20*z@banks[k].T,torch.arange(2,device='cuda'))/16;assert torch.isfinite(loss);loss.backward()
 torch.cuda.synchronize();cold_gradient_wall=time.time()-gstart;first_gradient=dictionary.atoms.grad.detach().clone();opt.zero_grad();torch.cuda.synchronize();warm_start=time.time()
 for k,program in enumerate(PROGRAMS[1:],1):
  z=dictionary(base,images,program);loss=F.cross_entropy(20*z@banks[k].T,torch.arange(2,device='cuda'))/16;loss.backward()
 torch.cuda.synchronize();gradient_wall=time.time()-warm_start;gradient_repeat_error=float((dictionary.atoms.grad-first_gradient).abs().max());assert torch.allclose(dictionary.atoms.grad,first_gradient,atol=2e-6,rtol=5e-4)
 assert dictionary.atoms.grad is not None and torch.isfinite(dictionary.atoms.grad).all() and dictionary.atoms.grad.abs().sum()>0;assert all(p.grad is None and not p.requires_grad for p in base.parameters());opt.step();counts.C['engineering_visual_optimizer_steps']+=1
 state={k:v.detach().cpu().clone() for k,v in dictionary.state_dict().items()};rest=Dictionary();rest.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in rest.state_dict().items())
 expected=json.loads(Path(cfg['output'],'ADMISSION.private.json').read_text())['physical_GPU_uuid'];nvidia=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True);own=[s for s in nvidia.splitlines() if s.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and own[0].split(',')[1].strip()==expected and os.environ['CUDA_VISIBLE_DEVICES']=='1';assert torch.cuda.max_memory_reserved()<12*2**30
 # Fixed-plan admission from timed actual encodes; no accuracy pilot or parameter reduction.
 projected=2*(gradient_wall/30*293250+forward_wall/32*(150000+27168))+6000+3600;remaining=json.loads(Path(cfg['clock']).read_text())['deadline_unix']-time.time();admitted=projected<remaining-90
 rec={'passed':True,'fixed_plan_admitted':admitted,'projected_remaining_seconds':projected,'remaining_seconds':remaining,'two_TRAIN_reference_rows':2,'encoder_exposures':counts.C['engineering_encoder_rows']-before,'engineering_visual_optimizer_steps':1,'formal_visual_optimizer_steps':0,'VAL_TEST_reads':0,'NOOP_parity':parity,'nonNOOP_cosine_change':effect,'core_forward32_rows_wall_seconds':forward_wall,'first_gradient30_rows_wall_seconds':cold_gradient_wall,'core_steady_gradient30_rows_wall_seconds':gradient_wall,'gradient_repeat_error':gradient_repeat_error,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'physical_GPU':1,'own_PID':os.getpid(),'own_GPU_UUID':expected,'ps':subprocess.check_output(['ps','-ww','-o','pid,ppid,stat,args','-p',str(os.getpid())],text=True)};write(Path(cfg['output'])/'NATIVE_CHECK.json',rec);return rec,base,tf
