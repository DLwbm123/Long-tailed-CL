"""Fresh fit-only TRAIN pilot: class-specific text residuals, matched native control."""
import os,json,csv,time,traceback,math
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ds=cfg['dataset']
ns=dict(__name__='text_residual_pilot',__file__=cfg['base_runner']);exec(compile(Path(cfg['base_runner']).read_text(),'<runner-definitions>','exec'),ns)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 INC=ns['INC'];n=len(ns['STAGES'][0]);order=ns['ORDER'];names=ns['NAMES'];template=ns['LEAF_TEMPLATE']
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 assert all(r['split']=='train' for r in rows);labels=torch.tensor([order.index(int(r['original_label'])) for r in rows]);bank=torch.load(Path(cfg['cache_root'] if ds=='NIH' else cfg['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
 assert bank['ids']==[i for i,y in enumerate(labels.tolist()) if y<n];x=bank['x'];text=bank['text'];x=F.normalize(x,dim=1) if ds=='NIH' else x;text=F.normalize(text,dim=1) if ds=='NIH' else text
 assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);assert torch.allclose(text.norm(dim=1),torch.ones(len(text)),atol=2e-6)
 y=labels[bank['ids']];counts=torch.bincount(y,minlength=n);tail=torch.where(counts<250)[0].tolist();leaf=text[[bank['prompts'].index(template.format(z)) for z in names[:n]]].cuda();assert len(torch.unique(leaf,dim=0))==n
 original_update=INC.hyper_net.update_task;original_text=INC.hyper_net.Text_encode;records=[];steps_total=0
 try:
  for fold in [0,1]:
   fit=[];hold=[]
   for c in range(n):
    ids=torch.where(y==c)[0];fit.extend(ids[fold::2][:64].tolist());hold.extend(ids[1-fold::2][:64].tolist())
   assert not set(fit)&set(hold);assert all(torch.bincount(y[fit],minlength=n)>0);core_initial=None
   for seed in [74002,74003,74004]:
    init_control=None
    for arm in ['native','class_residual']:
     candidate=arm=='class_residual';proof={};write('STATUS.json',dict(phase='FIT_ONLY_OPTIMIZATION',fold=fold,seed=seed,arm=arm,time=time.time()))
     def update(net):
      original_update(net)
      if candidate:
       state=torch.get_rng_state().clone();cuda=[z.clone() for z in torch.cuda.get_rng_state_all()]
       net.class_text_residual=torch.nn.Parameter(torch.zeros(n,512,device='cuda'))
       assert torch.equal(state,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
       proof['zero_init_no_rng']=True
       net.to('cuda')
       assert torch.equal(net.Text_encode(leaf),original_text(net,leaf))
      initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items() if k!='class_text_residual'}
      if candidate:assert all(torch.equal(v,init_control[k]) for k,v in initial.items());proof['initial_core_exact_control']=True
      else:proof['initial_core']=initial
     def encode(net,t):
      z=original_text(net,t);match=(t[:,None,:]==leaf[None,:,:]).all(-1)
      assert (match.sum(1)<=1).all()
      return z+match.to(z.dtype)@net.class_text_residual
     INC.hyper_net.update_task=update;INC.hyper_net.Text_encode=encode if candidate else original_text
     ns['seed_all'](seed+10000*fold);ns['install_feature_model'](INC,text,bank['prompts'],bank['logit_scale']);args=ns['make_args'](seed,'H-unit');args['tuned_epoch']=2
     learner=ns['CheckedLearner'](args);learner.seed=seed;learner.arm=arm;learner.start_audit();before=learner._before_step;gradproof=[]
     def inspect():
      before()
      if candidate:
       g=learner._network.class_text_residual.grad;assert g is not None and torch.isfinite(g).all();gradproof.append(float(g.norm()))
     learner._before_step=inspect
     learner.incremental_train(ns['DataManager'](x,y,fit,[]));net=learner._network;steps=learner.audit['steps'];assert steps==2*math.ceil(len(fit)/64);steps_total+=steps
     if not candidate:init_control=proof.pop('initial_core')
     else:assert len(gradproof)==steps and max(gradproof)>0;assert float(net.class_text_residual.norm())>0
     pred=ns['predictions'](learner,x[hold]);scores={}
     for key,z in pred.items():
      yy=y[hold];pp=z.argmax(1);nn=torch.bincount(yy,minlength=n).tolist();cc=torch.bincount(yy[pp==yy],minlength=n).tolist();scores[key]=dict(counts=nn,correct=cc,macro_BA=100*sum(v/q for v,q in zip(cc,nn))/n,tail_BA=100*sum(cc[c]/nn[c] for c in tail)/len(tail))
     records.append(dict(fold=fold,seed=seed,arm=arm,optimizer_steps=steps,fit_counts=torch.bincount(y[fit],minlength=n).tolist(),holdout_counts=torch.bincount(y[hold],minlength=n).tolist(),train_counts=counts.tolist(),tail_classes=tail,fit_holdout_disjoint=True,fresh_core_fit_only=True,proof=proof,scores=scores,extra_parameters=n*512 if candidate else 0,residual_norm=float(net.class_text_residual.detach().norm()) if candidate else 0.,max_residual_gradient=max(gradproof) if candidate else 0.,original_loss_audit=learner.audit))
     assert torch.cuda.max_memory_reserved()<8*2**30
     del learner,net,pred;torch.cuda.empty_cache()
 finally:INC.hyper_net.update_task=original_update;INC.hyper_net.Text_encode=original_text
 assert len(records)==12
 write('RESULTS.json',dict(dataset=ds,cells=records,optimizer_steps=steps_total,VAL_reads=0,TEST_reads=0,fresh_fit_only_pilot=True,passed=True));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
