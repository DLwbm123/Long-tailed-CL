import torch,time,resource,json,subprocess
from pathlib import Path
from model import Adapter,make_base,visual,textual,unit
from data import train_rows,loader
from common import write

def run(cfg,device='cuda'):
 start=time.time();cpu=time.process_time();base,tf,tok=make_base(cfg,device);rows,y=train_rows(cfg);ix=[]
 for i,c in enumerate(y.tolist()):
  if c in cfg['stages'][0] and all(int(y[j])!=c for j in ix):ix.append(i)
  if len(ix)==2:break
 im=next(iter(loader(cfg,rows,ix,tf)))[0].to(device);torch.manual_seed(cfg['seed']);m=Adapter('FULL').to(device);m.add();nn=len(set(y[ix].tolist()));assert nn==2
 tokens=tok(['A photo of a '+cfg['names'][int(c)] for c in y[ix]]).to(device)
 with torch.no_grad():
  z=visual(base,im);direct=unit(base.encode_image(im));err=float((z-direct).abs().max());assert err<=2e-5,err
 bank=visual(base,im,m.branches[0]).detach();pp,ww,rl=m.prototypes(bank,torch.arange(2,device=device),2);text=textual(base,tokens,m.branches[0]);v=visual(base,im,m.branches[0]);scores,d,balance=m.logits([v],text,pp,[0,0]);assert torch.isfinite(scores).all();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=cfg['lr']);opt.zero_grad();loss=torch.nn.functional.cross_entropy(scores,torch.arange(2,device=device))+.1*rl+.01*balance;loss.backward();assert all(p.grad is None and not p.requires_grad for p in base.parameters());assert m.branches[0].vprompt.grad is not None and m.branches[0].vprompt.grad.abs().sum()>0;assert m.branches[0].tprompt.grad is not None and m.branches[0].tprompt.grad.abs().sum()>0;opt.step()
 state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};rest=Adapter('FULL');rest.add();rest.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in rest.state_dict().items())
 rec={'passed':True,'device':device,'real_TRAIN_reference_rows':2,'frozen_wrapper_parity':err,'engineering_visual_optimizer_steps':1,'formal_optimizer_steps':0,'VAL_TEST_reads':0,'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'physical_GPU':cfg['physical_gpu'] if device=='cuda' else None,'GPU_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else 0}
 if device=='cuda':
  assert rec['GPU_reserved_bytes']<12*2**30;assert __import__('os').environ['CUDA_VISIBLE_DEVICES']==str(cfg['physical_gpu']);rec['live_ps']=subprocess.check_output(['ps','-ww','-o','pid,ppid,stat,args','-p',str(__import__('os').getpid())],text=True);rec['nvidia']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)
 if device=='cuda':
  ownpid=str(__import__('os').getpid());own=[v for v in rec['nvidia'].splitlines() if v.split(',')[0].strip()==ownpid];expected=json.loads(Path(cfg['output'],'ADMISSION.private.json').read_text())['physical_GPU_uuid'];assert len(own)==1 and own[0].split(',')[1].strip()==expected,'PHYSICAL_GPU_BINDING'
 write(Path(cfg['output'])/('NATIVE_CHECK.json' if device=='cuda' else 'CPU_ENGINEERING_CHECK.json'),rec);return rec
