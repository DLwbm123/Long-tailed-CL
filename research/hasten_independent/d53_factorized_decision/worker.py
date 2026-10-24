"""Fresh fit-only native/flat/factorized decision pilot; no VAL or trained checkpoint."""
import os,json,csv,time,traceback,math
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ds=cfg['dataset']
ns=dict(__name__='factorized_decision_pilot',__file__=cfg['base_runner']);exec(compile(Path(cfg['base_runner']).read_text(),'<runner-definitions>','exec'),ns)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def balanced(v,y):return torch.stack([v[y==c].mean() for c in y.unique()]).mean()
def main():
 INC=ns['INC'];n=len(ns['STAGES'][0]);order=ns['ORDER'];names=ns['NAMES'];template=ns['LEAF_TEMPLATE']
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 assert all(r['split']=='train' for r in rows);labels=torch.tensor([order.index(int(r['original_label'])) for r in rows]);bank=torch.load(Path(cfg['cache_root'] if ds=='NIH' else cfg['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
 assert bank['ids']==[i for i,y in enumerate(labels.tolist()) if y<n];x=bank['x'];text=bank['text'];x=F.normalize(x,dim=1) if ds=='NIH' else x;text=F.normalize(text,dim=1) if ds=='NIH' else text
 assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);assert torch.allclose(text.norm(dim=1),torch.ones(len(text)),atol=2e-6)
 y=labels[bank['ids']];counts=torch.bincount(y,minlength=n);tail=torch.where(counts<250)[0].tolist();rawtypes=ns['TYPE'][:n];types=sorted(set(rawtypes));groups=torch.tensor([types.index(t) for t in rawtypes],device='cuda');ng=len(types);sizes=torch.bincount(groups,minlength=ng);prior=sizes.float().log()
 original_update=INC.hyper_net.update_task;records=[];steps_total=0
 def posterior(net,raw,z):
  g=net.logit_scale.exp()*(F.normalize(raw,dim=1)@F.normalize(net.coarse_prototypes,dim=1).T)+prior
  logg=F.log_softmax(g,dim=1);parts=[]
  for c in range(n):parts.append(logg[:,groups[c]]+z[:,c]-torch.logsumexp(z[:,groups==groups[c]],dim=1))
  p=torch.stack(parts,dim=1);assert torch.allclose(p.exp().sum(1),torch.ones(len(p),device=p.device,dtype=p.dtype),atol=1e-6);return p,g
 try:
  for fold in [0,1]:
   fit=[];hold=[]
   for c in range(n):
    ids=torch.where(y==c)[0];fit.extend(ids[fold::2][:64].tolist());hold.extend(ids[1-fold::2][:64].tolist())
   assert not set(fit)&set(hold);means=torch.stack([x[[i for i in fit if int(y[i])==c]].mean(0) for c in range(n)]).cuda();gp=torch.stack([means[groups==g].mean(0) for g in range(ng)])
   for seed in [74002,74003,74004]:
    init_control=None
    for arm in ['native','flat','factorized']:
     proof={};write('STATUS.json',dict(phase='FIT_ONLY_OPTIMIZATION',fold=fold,seed=seed,arm=arm,time=time.time()))
     def update(net):
      original_update(net)
      if arm=='factorized':
       cpu=torch.get_rng_state().clone();cuda=[z.clone() for z in torch.cuda.get_rng_state_all()];net.coarse_prototypes=torch.nn.Parameter(F.normalize(gp,dim=1).clone());assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()));proof['coarse_init_no_rng']=True
      initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if k!='coarse_prototypes'}
      if arm!='native':assert all(torch.equal(v,init_control[k]) for k,v in initial.items());proof['initial_core_exact_native']=True
      else:proof['initial_core']=initial
     INC.hyper_net.update_task=update
     ns['seed_all'](seed+10000*fold);ns['install_feature_model'](INC,text,bank['prompts'],bank['logit_scale']);args=ns['make_args'](seed,'H-unit');args.update(tuned_epoch=2,decision_mode=arm)
     learner=ns['CheckedLearner'](args);learner.seed=seed;learner.arm=arm;learner.start_audit();coarse_grad=[]
     def decision(raw,replay,logits,targets):
      assert replay is None;net=learner._network;z=net.logit_scale.exp()*logits;learner.audit['decision_calls']=learner.audit.get('decision_calls',0)+1
      if arm=='flat':return balanced(F.cross_entropy(z,targets,reduction='none'),targets)
      p,g=posterior(net,raw,z);fine=torch.stack([-z[k,targets[k]]+torch.logsumexp(z[k,groups==groups[targets[k]]],dim=0) for k in range(len(targets))]);coarse=F.cross_entropy(g,groups[targets],reduction='none')
      if learner.audit['decision_calls']==1:
       params=[p for key,p in net.named_parameters() if p.requires_grad and key!='coarse_prototypes'];gr=torch.autograd.grad(balanced(coarse,targets),params,retain_graph=True,allow_unused=True);assert all(t is None for t in gr);assert torch.autograd.grad(balanced(fine,targets),net.coarse_prototypes,retain_graph=True,allow_unused=True)[0] is None;proof['coarse_fine_gradient_separated']=True
      return balanced(-p[torch.arange(len(targets),device='cuda'),targets],targets)
     learner._decision_loss=decision;before=learner._before_step
     def inspect():
      before()
      if arm=='factorized':
       g=learner._network.coarse_prototypes.grad;assert g is not None and torch.isfinite(g).all();coarse_grad.append(float(g.norm()))
     learner._before_step=inspect;learner.incremental_train(ns['DataManager'](x,y,fit,[]));net=learner._network;steps=learner.audit['steps'];assert steps==2*math.ceil(len(fit)/64);steps_total+=steps
     if arm=='native':init_control=proof.pop('initial_core')
     else:assert learner.audit['decision_calls']==steps
     if arm=='factorized':assert len(coarse_grad)==steps and max(coarse_grad)>0
     with torch.no_grad():
      pred=ns['predictions'](learner,x[hold])
      if arm!='native':
       z=pred['hyper'].cuda()*net.logit_scale.exp();pred['final']=(F.log_softmax(z,dim=1) if arm=='flat' else posterior(net,x[hold].cuda(),z)[0]).cpu()
     scores={}
     for key,z in pred.items():
      yy=y[hold];pp=z.argmax(1);nn=torch.bincount(yy,minlength=n).tolist();cc=torch.bincount(yy[pp==yy],minlength=n).tolist();scores[key]=dict(counts=nn,correct=cc,macro_BA=100*sum(v/q for v,q in zip(cc,nn))/n,tail_BA=100*sum(cc[c]/nn[c] for c in tail)/len(tail))
     records.append(dict(fold=fold,seed=seed,arm=arm,optimizer_steps=steps,fit_counts=torch.bincount(y[fit],minlength=n).tolist(),holdout_counts=torch.bincount(y[hold],minlength=n).tolist(),train_counts=counts.tolist(),tail_classes=tail,groups=groups.tolist(),fit_holdout_disjoint=True,fresh_core_fit_only=True,proof=proof,scores=scores,extra_parameters=ng*512 if arm=='factorized' else 0,max_coarse_gradient=max(coarse_grad) if coarse_grad else 0.,original_loss_audit=learner.audit))
     write('PROGRESS.json',dict(completed_cells=len(records),optimizer_steps=steps_total));assert torch.cuda.max_memory_reserved()<8*2**30
     del learner,net,pred;torch.cuda.empty_cache()
 finally:INC.hyper_net.update_task=original_update
 assert len(records)==18
 write('RESULTS.json',dict(dataset=ds,cells=records,optimizer_steps=steps_total,VAL_reads=0,TEST_reads=0,fresh_fit_only_pilot=True,passed=True));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
