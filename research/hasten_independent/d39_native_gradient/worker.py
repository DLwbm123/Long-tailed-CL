"""Fixed TRAIN probes of original loss gradients; no optimizer or validation."""
import os,json,time,traceback,csv
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ds=cfg['dataset'];ref=Path(cfg['reference_source'])
ns=dict(__name__='sealed_gradient_probe',__file__=cfg['base_runner']);exec(compile(Path(cfg['base_runner']).read_text(),'<original-runner-definitions>','exec'),ns)
def write(n,z):(out/n).write_text(json.dumps(z,indent=2,allow_nan=False))
def main():
 G=ns['GEO'];UP=ns['UP'];stages=ns['STAGES'];order=ns['ORDER'];names=ns['NAMES'];template=ns['LEAF_TEMPLATE']
 with open(Path(cfg['manifest'])/'train.csv') as f:train=list(csv.DictReader(f))
 labels=torch.tensor([order.index(int(r['original_label'])) for r in train]);assert all(r['split']=='train' for r in train)
 records=[]
 for stage in [0,len(stages)-1]:
  n=sum(map(len,stages[:stage+1]));old=sum(map(len,stages[:stage]));bank=torch.load(Path(cfg['cache_root'] if ds=='NIH' else cfg['shared_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False)
  expected=[i for i,y in enumerate(labels.tolist()) if y in stages[stage]];assert bank['ids']==expected
  gen=torch.Generator().manual_seed(20261024+100*['NIH','ISIC','HK'].index(ds)+stage);pick=torch.randperm(len(bank['ids']),generator=gen)[:min(64,len(expected))]
  real=bank['x'][pick];real=F.normalize(real,dim=1) if ds=='NIH' else real;yr=labels[torch.tensor(bank['ids'])[pick]]
  assert torch.allclose(real.norm(dim=1),torch.ones(len(real)),atol=2e-6)
  for seed in [74002,74003,74004]:
   write('STATUS.json',dict(phase='NATIVE_GRADIENT_PROBE',seed=seed,stage=stage,time=time.time()))
   state=torch.load(ref/f'trajectories/{seed}/H-unit/stage{stage}.private.pt',map_location='cuda',weights_only=False)
   learner=ns['restore_new'](ns['make_args'](seed,'H-unit'),state,bank['text'],bank['prompts'],bank['logit_scale']);net=learner._network;net.eval();hm=learner.hierarchy_manager
   saved={k:v.detach().clone() for k,v in net.state_dict().items()};hproto=hm.all_prototypes.clone();hhas=hm.has_proto.clone();hmutable=hm.mutable_nodes.clone()
   replay=[];gy=torch.Generator().manual_seed(20261024+100*['NIH','ISIC','HK'].index(ds)+stage)
   for c in range(old):
    cov=state['cov'][c].double().cpu();mean=state['prototype'][c].double().cpu();replay.append((torch.randn(1,512,generator=gy,dtype=torch.double)@torch.linalg.cholesky(cov).T+mean).float())
   x=torch.cat([real,*replay]).cuda() if old else real.cuda();y=torch.cat([yr,torch.arange(old)]).cuda() if old else yr.cuda()
   text=net.model.encode_text(net.tokenizer([template.format(v) for v in names[:n]]).cuda());im=net.trans_hyper(net.Image_encode(x));tx=net.trans_hyper(net.Text_encode(text));curv=net.get_curv();scale=net.logit_scale.exp()
   pair=-G.pairwise_dist(im,tx[y],curv);target=torch.arange(len(y),device='cuda');contrast=.5*(F.cross_entropy(scale*pair,target)+F.cross_entropy(scale*pair.T,target))
   anchor=.1*UP.compute_stable_entailment_loss(tx[y],im,curv,min_radius=.1)
   protos=hm.get_all_prototypes(tx,torch.arange(n,device='cuda'),curv);hier=.5*sum(hm.compute_hierarchy_loss(protos,curv,parent_child_margin=.2).values())
   with torch.no_grad():targetmap=G.exp_map0(net.Image_encode(x),curv=curv)
   reg=.1*(im-targetmap.detach()).square().sum(-1).mean();geometry=anchor+hier+reg
   named=[(k,p) for k,p in net.named_parameters() if p.requires_grad];params=[p for k,p in named]
   grads={key:torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) for key,loss in [('contrast',contrast),('geometry',geometry)]}
   vectors={};projected={}
   for key,gg in grads.items():
    chunks=[];pc=[]
    for (name,p),g in zip(named,gg):
     g=torch.zeros_like(p) if g is None else g;assert torch.isfinite(g).all();chunks.append(g.detach().double().flatten())
     if stage>0 and name=='learned_lorentz_exp.exp_map.0.weight':g=net.learned_lorentz_exp.project_gradient(g)
     pc.append(g.detach().double().flatten())
    vectors[key]=torch.cat(chunks);projected[key]=torch.cat(pc)
   def describe(v):
    a,b=v['contrast'],v['geometry'];assert a.norm()>0 and b.norm()>0
    return dict(contrast_norm=float(a.norm()),geometry_norm=float(b.norm()),cosine=float(F.cosine_similarity(a,b,dim=0)),geometry_to_contrast_ratio=float(b.norm()/a.norm()))
   hm.all_prototypes.copy_(hproto);hm.has_proto.copy_(hhas);hm.mutable_nodes.copy_(hmutable)
   assert all(torch.equal(v,net.state_dict()[k]) for k,v in saved.items()),'NETWORK_STATE_CHANGED';assert torch.equal(hm.all_prototypes,hproto) and torch.equal(hm.has_proto,hhas) and torch.equal(hm.mutable_nodes,hmutable)
   assert torch.cuda.max_memory_reserved()<8*2**30
   records.append(dict(seed=seed,stage=stage,real_rows=len(real),replay_rows=old,classes=n,counts=torch.bincount(y,minlength=n).tolist(),losses=dict(contrast=float(contrast.detach()),weighted_anchor=float(anchor.detach()),weighted_hierarchy=float(hier.detach()),weighted_mapreg=float(reg.detach())),before_mapping_projection=describe(vectors),after_mapping_projection=describe(projected),network_and_hierarchy_unchanged=True,not_actual_training_trajectory=True))
   del learner,net,hm,state,saved,grads,vectors,projected;torch.cuda.empty_cache()
 write('RESULTS.json',dict(dataset=ds,cells=records,passed=len(records)==6,optimizer_steps=0,VAL_reads=0,TEST_reads=0));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
