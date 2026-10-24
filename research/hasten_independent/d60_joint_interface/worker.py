"""Full TRAIN geometry/semantic/objective accounting, zero model training."""
import os,json,csv,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False

def write(n,z):(root/n).write_text(json.dumps(z,indent=2,allow_nan=False))
def guard():
 assert time.time()<json.loads((root/'FIRST_ACTUAL.private.json').read_text())['deadline_unix']-25
 assert torch.cuda.max_memory_reserved()<8*2**30

def main():
 # Exact duplicate-text identity, synthetic scores only.
 gen=torch.Generator().manual_seed(123);lab=torch.tensor([0]*5+[1]*3+[2]*2);score=torch.randn(10,3,generator=gen,dtype=torch.double,requires_grad=True);m=torch.bincount(lab);paired=F.cross_entropy(score[:,lab],torch.arange(10));prior=F.cross_entropy(score+m.double().log(),lab);gp=torch.autograd.grad(paired,score,retain_graph=True)[0];gc=torch.autograd.grad(prior,score)[0];assert torch.allclose(gp,gc,atol=1e-12) and torch.allclose(paired-prior,m[lab].double().log().mean(),atol=1e-12)
 records=[]
 with torch.no_grad():
  for ds in ['NIH','ISIC','HK']:
   dc=cfg['domains'][ds];order=dc['order'];stages=dc['stages'];C=len(order)
   with open(Path(dc['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
   assert all(r['split']=='train' for r in rows);yall=torch.tensor([order.index(int(r['original_label'])) for r in rows]);x=None;y=None;oldtext=None
   for stage,classes in enumerate(stages):
    guard();b=torch.load(Path(dc['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False);ids=[i for i,c in enumerate(yall.tolist()) if c in classes];assert b['ids']==ids;xx=F.normalize(b['x'],dim=1) if ds=='NIH' else b['x'];bank=F.normalize(b['text'],dim=1) if ds=='NIH' else b['text'];n=sum(len(v) for v in stages[:stage+1]);prompts=[dc['leaf_template'].format(z) for z in dc['ordered_names'][:n]];t=bank[[b['prompts'].index(v) for v in prompts]].cuda();assert torch.allclose(t.norm(dim=1),torch.ones(n,device='cuda'),atol=2e-6);prefix_exact=oldtext is None or torch.equal(t[:len(oldtext)],oldtext);prefix_delta=0. if oldtext is None else float((t[:len(oldtext)]-oldtext).abs().max());assert oldtext is None or torch.allclose(t[:len(oldtext)],oldtext,atol=2e-6,rtol=0);oldtext=t.clone();x=xx.cuda() if x is None else torch.cat([x,xx.cuda()]);y=yall[ids].cuda() if y is None else torch.cat([y,yall[ids].cuda()]);nc=torch.bincount(y,minlength=n);assert nc.tolist()==dc['train_counts_ordered'][:n] and int(nc.min())>1;assert torch.allclose(x.norm(dim=1),torch.ones(len(x),device='cuda'),atol=2e-6);sums=torch.stack([x[y==c].sum(0) for c in range(n)]);mu=F.normalize(sums,dim=1);gram=mu@mu.T;tg=t@t.T;correct={k:torch.zeros(n,dtype=torch.long,device='cuda') for k in ['visual_LOO','semantic']};marg={k:torch.zeros(n,dtype=torch.double,device='cuda') for k in correct};within=torch.zeros(n,dtype=torch.double,device='cuda')
    for ii in torch.arange(len(x),device='cuda').split(1024):
     a=x[ii];yy=y[ii];vis=a@mu.T;loo=F.normalize(sums[yy]-a,dim=1);vis[torch.arange(len(ii),device='cuda'),yy]=(a*loo).sum(1);sem=a@t.T
     for name,z in [('visual_LOO',vis),('semantic',sem)]:
      pred=z.argmax(1);correct[name]+=torch.bincount(yy[pred==yy],minlength=n);true=z[torch.arange(len(ii),device='cuda'),yy].clone();z[torch.arange(len(ii),device='cuda'),yy]=-torch.inf;marg[name].scatter_add_(0,yy,(true-z.max(1).values).double())
     within.scatter_add_(0,yy,(a-mu[yy]).square().sum(1).double())
    tail=nc<250;metrics={k:dict(correct=v.tolist(),macro_BA=float((v.double()/nc).mean()),tail_BA=float((v.double()[tail]/nc[tail]).mean()),class_mean_margin=(marg[k]/nc).tolist()) for k,v in correct.items()};eig=torch.linalg.eigvalsh(tg.double());rec=dict(dataset=ds,stage=stage,seen=n,TRAIN_rows=len(x),counts=nc.tolist(),exact_ids=True,text_prefix_exact=prefix_exact,text_prefix_max_absolute_delta=prefix_delta,geometry=metrics,visual_equalclass_within_squared=float((within/nc).mean()),visual_between_center_scatter=float((mu-mu.mean(0)).square().sum(1).mean()),visual_text_center_cosine=(mu*t).sum(1).tolist(),semantic_max_other_cosine=tg.masked_fill(torch.eye(n,device='cuda',dtype=torch.bool),-torch.inf).max(1).values.tolist(),visual_semantic_Gram_relative_gap=float((gram-tg).norm()/gram.norm()),semantic_Gram_numerical_rank=int((eig>eig.max()*1e-6).sum()),initial_natural_batch64_expected_multiplicity=(1+63*(nc.double()-1)/(len(x)-1)).tolist() if stage==0 else None);records.append(rec);write('RESULTS.json',dict(records=records,optimizer_steps=0,new_optimized_classifier_fit=False,class_mean_statistical_estimation=True,VAL_reads=0,TEST_reads=0,synthetic_duplicate_CE_gradient_identity=True,geometry_not_separability_ceiling=True));write('STATUS.json',dict(phase='AUDIT',dataset=ds,stage=stage,time=time.time()))
 assert len(records)==15;write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:(root/'FAILURE.private.txt').write_text(traceback.format_exc());raise
