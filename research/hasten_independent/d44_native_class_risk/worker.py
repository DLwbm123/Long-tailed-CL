"""Fixed disjoint TRAIN probes of native contrast risk reduction; no optimizer."""
import os,json,time,traceback,csv
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ds=cfg['dataset'];ref=Path(cfg['reference_source'])
ns=dict(__name__='native_risk_probe',__file__=cfg['base_runner']);exec(compile(Path(cfg['base_runner']).read_text(),'<runner-definitions>','exec'),ns)
def write(n,z):(out/n).write_text(json.dumps(z,indent=2,allow_nan=False))
def class_mean(loss,y,classes=None):
 classes=y.unique() if classes is None else classes
 return torch.stack([loss[y==c].mean() for c in classes]).mean()
def main():
 G=ns['GEO'];UP=ns['UP'];stages=ns['STAGES'];order=ns['ORDER'];names=ns['NAMES'];template=ns['LEAF_TEMPLATE'];n=len(stages[0])
 with open(Path(cfg['manifest'])/'train.csv') as f:train=list(csv.DictReader(f))
 labels=torch.tensor([order.index(int(r['original_label'])) for r in train]);assert all(r['split']=='train' for r in train)
 bank=torch.load(Path(cfg['cache_root'] if ds=='NIH' else cfg['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
 assert bank['ids']==[i for i,y in enumerate(labels.tolist()) if y in stages[0]]
 x=bank['x'];x=F.normalize(x,dim=1) if ds=='NIH' else x;y=labels[bank['ids']];counts=torch.bincount(y,minlength=n);tail=torch.where(counts<250)[0];assert len(tail)>0
 b=[]
 for c in range(n):
  ids=torch.where(y==c)[0];b.extend(ids[:min(4,len(ids)//2)].tolist())
 mask=torch.ones(len(x),dtype=torch.bool);mask[b]=False;pool=torch.where(mask)[0];gen=torch.Generator().manual_seed(20261025+100*['NIH','ISIC','HK'].index(ds));pool=pool[torch.randperm(len(pool),generator=gen)];batches=list(pool[:len(pool)//64*64].split(64));chosen=[];selection={}
 for c in tail.tolist():
  k=next(i for i,ids in enumerate(batches) if (y[ids]==c).any());selection[c]=k
  if k not in chosen:chosen.append(k)
 assert not set(b)&set(torch.cat([batches[k] for k in chosen]).tolist())
 assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6)
 records=[]
 for seed in [74002,74003,74004]:
  state=torch.load(ref/f'trajectories/{seed}/H-unit/stage0.private.pt',map_location='cuda',weights_only=False)
  learner=ns['restore_new'](ns['make_args'](seed,'H-unit'),state,bank['text'],bank['prompts'],bank['logit_scale']);net=learner._network;net.eval();hm=learner.hierarchy_manager
  assert torch.allclose(net.model.text_bank.norm(dim=1),torch.ones(len(net.model.text_bank),device='cuda'),atol=2e-6)
  saved={k:v.detach().clone() for k,v in net.state_dict().items()};hproto=hm.all_prototypes.clone();hhas=hm.has_proto.clone();hmutable=hm.mutable_nodes.clone()
  named=[(k,p) for k,p in net.named_parameters() if p.requires_grad];params=[p for k,p in named]
  def forward(ids):
   xx=x[ids].cuda();yy=y[ids].cuda();text=net.model.encode_text(net.tokenizer([template.format(v) for v in names[:n]]).cuda());im=net.trans_hyper(net.Image_encode(xx));tx=net.trans_hyper(net.Text_encode(text));curv=net.get_curv();scale=net.logit_scale.exp();z=-scale*G.pairwise_dist(im,tx[yy],curv);target=torch.arange(len(yy),device='cuda');loss=.5*(F.cross_entropy(z,target,reduction='none')+F.cross_entropy(z.T,target,reduction='none'));return loss,yy,xx,im,tx,curv
  def vector(loss):
   grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);chunks=[]
   for p,g in zip(params,grads):
    if g is None:g=torch.zeros_like(p)
    assert torch.isfinite(g).all();chunks.append(g.detach().double().flatten())
   z=torch.cat(chunks);assert z.norm()>1e-20;return z
  lb,yb,*_=forward(b);gb=vector(class_mean(lb,yb));gt=vector(class_mean(lb,yb,tail.cuda()));del lb,yb
  for k in chosen:
   write('STATUS.json',dict(phase='CLASS_RISK_DIRECTION',seed=seed,batch=k,time=time.time()))
   li,ya,xx,im,tx,curv=forward(batches[k]);original=li.mean();candidate=class_mean(li,ya)
   anchor=.1*UP.compute_stable_entailment_loss(tx[ya],im,curv,min_radius=.1);protos=hm.get_all_prototypes(tx,torch.arange(n,device='cuda'),curv);hier=.5*sum(hm.compute_hierarchy_loss(protos,curv,parent_child_margin=.2).values())
   with torch.no_grad():targetmap=G.exp_map0(net.Image_encode(xx),curv=curv)
   reg=.1*(im-targetmap.detach()).square().sum(-1).mean();geometry=anchor+hier+reg
   go=vector(original+geometry);gc=vector(candidate+geometry)
   def cos(a,b):return float(F.cosine_similarity(a,b,dim=0))
   result=dict(seed=seed,stage=0,conditional_batch_index=k,represented_tail_classes=[c for c in tail.tolist() if (ya==c).any()],counts=torch.bincount(ya,minlength=n).tolist(),reference_counts=torch.bincount(y[b],minlength=n).tolist(),disjoint_TRAIN_rows=True,original_contrast=float(original.detach()),class_reduced_contrast=float(candidate.detach()),macro_cos_original=cos(gb,go),macro_cos_candidate=cos(gb,gc),tail_cos_original=cos(gt,go),tail_cos_candidate=cos(gt,gc),parameters_unchanged=True)
   result.update(macro_cos_delta=result['macro_cos_candidate']-result['macro_cos_original'],tail_cos_delta=result['tail_cos_candidate']-result['tail_cos_original']);records.append(result)
   hm.all_prototypes.copy_(hproto);hm.has_proto.copy_(hhas);hm.mutable_nodes.copy_(hmutable)
   assert all(torch.equal(v,net.state_dict()[key]) for key,v in saved.items());assert all(p.grad is None for p in params)
   del li,ya,xx,im,tx,anchor,hier,reg,geometry,original,candidate,go,gc
  assert torch.cuda.max_memory_reserved()<8*2**30
  del learner,net,hm,state,saved,gb,gt;torch.cuda.empty_cache()
 summary=[]
 for seed in [74002,74003,74004]:
  rr=[z for z in records if z['seed']==seed];summary.append(dict(seed=seed,cells=len(rr),**{key:sum(z[key] for z in rr)/len(rr) for key in ['macro_cos_original','macro_cos_candidate','tail_cos_original','tail_cos_candidate','macro_cos_delta','tail_cos_delta']}))
 write('RESULTS.json',dict(dataset=ds,cells=records,summary=summary,initial_tail_classes=tail.tolist(),tail_batch_coverage={c:selection[c] for c in tail.tolist()},optimizer_steps=0,VAL_reads=0,TEST_reads=0,not_actual_training_trajectory=True,passed=len(summary)==3));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
