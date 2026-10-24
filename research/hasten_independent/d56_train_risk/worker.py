import os,json,csv,time,math,traceback
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
class Map(nn.Module):
 def __init__(self):super().__init__();self.a=nn.Linear(512,64);self.b=nn.Linear(64,512)
 def forward(self,x):return F.normalize(x+self.b(F.gelu(self.a(x))),dim=1)
def main():
 source=Path(cfg['source_root']);results=[];proof=[]
 with torch.inference_mode():
  for ds in ['NIH','ISIC','HK']:
   conf=json.loads((source/ds/'config.private.json').read_text());assert json.loads((source/ds/'STATUS.json').read_text())['phase']=='COMPLETE';order=conf['order'];stages=conf['stages'];n=len(order)
   with open(Path(conf['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
   assert all(v['split']=='train' for v in rows);y=torch.tensor([order.index(int(v['original_label'])) for v in rows]);truth=torch.bincount(y,minlength=n)
   for seed in [74002,74003,74004]:
    with torch.random.fork_rng(devices=[0]):m=Map().cuda()
    p=source/ds/str(seed);s=torch.load(p/'MAP.private.pt',map_location='cuda',weights_only=False);m.load_state_dict(s['state'],strict=True);m.eval();R=s['R'];h=torch.load(p/f'head{len(stages)-1}.private.pt',map_location='cuda',weights_only=False);assert h['seen']==n
    count=torch.zeros(n,dtype=torch.long,device='cuda');cm={k:torch.zeros(n,n,dtype=torch.long,device='cuda') for k in h['weights']};marg={k:torch.zeros(n,dtype=torch.double,device='cuda') for k in h['weights']}
    for stage,classes in enumerate(stages):
     origin=json.loads((root/'FIRST_ACTUAL.private.json').read_text());assert time.time()<origin['deadline_unix']-30
     (root/'STATUS.json').write_text(json.dumps(dict(phase='TRAIN_RISK',dataset=ds,seed=seed,stage=stage)))
     b=torch.load(Path(conf['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);ids=[i for i,v in enumerate(y.tolist()) if v in classes];assert ids==b['ids'];x=F.normalize(b['x'],dim=1) if ds=='NIH' else b['x'];assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);yy=y[ids]
     for start in range(0,len(x),512):
      u=m(x[start:start+512].cuda());z=F.relu(u@R)*math.sqrt(2/2048);z=torch.cat([z,torch.ones(len(z),1,device='cuda')],1).double();target=yy[start:start+512].cuda();count+=torch.bincount(target,minlength=n)
      for k,W in h['weights'].items():
       logits=z@W;assert torch.isfinite(logits).all();pred=logits.argmax(1);cm[k]+=torch.bincount(target*n+pred,minlength=n*n).reshape(n,n);true=logits[torch.arange(len(target),device='cuda'),target].clone();logits[torch.arange(len(target),device='cuda'),target]=-float('inf');gap=true-logits.max(1).values;marg[k].scatter_add_(0,target,gap)
      assert torch.cuda.max_memory_reserved()<8*2**30
    assert torch.equal(count.cpu(),truth)
    for k in h['weights']:
     matrix=cm[k].cpu();correct=matrix.diag();results.append(dict(dataset=ds,seed=seed,arm=k,counts=count.tolist(),correct=correct.tolist(),mean_margin=(marg[k]/count).tolist(),confusion=matrix.tolist(),initial_classes=len(stages[0]),macro_BA=float((correct.double()/truth).mean()*100),tail_BA=float((correct.double()/truth)[truth<250].mean()*100)))
    proof.append(dict(dataset=ds,seed=seed,all_TRAIN_rows=int(count.sum()),exact_ids_and_counts=True));del m,R,s,h;torch.cuda.empty_cache()
 assert len(results)==18 and len(proof)==9
 (root/'RESULTS.json').write_text(json.dumps(dict(readouts=results,verification=proof,optimizer_steps=0,VAL_reads=0,TEST_reads=0,full_TRAIN_resubstitution=True,independent_confirmation=False),indent=2));(root/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE')))
try:main()
except BaseException:(root/'FAILURE.private.txt').write_text(traceback.format_exc());raise
