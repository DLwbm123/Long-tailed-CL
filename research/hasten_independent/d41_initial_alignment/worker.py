"""Untrained stage-zero adapter alignment probe; TRAIN only, no optimizer."""
import os,json,csv,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);ds=cfg['dataset']
ns=dict(__name__='initial_alignment_probe',__file__=cfg['base_runner']);exec(compile(Path(cfg['base_runner']).read_text(),'<runner-definitions>','exec'),ns)
def write(n,v):(out/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
 G=ns['GEO'];stages=ns['STAGES'];order=ns['ORDER'];names=ns['NAMES'];n=len(stages[0])
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 assert all(r['split']=='train' for r in rows)
 yall=torch.tensor([order.index(int(r['original_label'])) for r in rows])
 bank=torch.load(Path(cfg['cache_root'] if ds=='NIH' else cfg['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False)
 if ds=='NIH':bank['text']=F.normalize(bank['text'],dim=-1)
 assert torch.allclose(bank['text'].norm(dim=1),torch.ones(len(bank['text'])),atol=2e-6)
 expected=[i for i,y in enumerate(yall.tolist()) if y in stages[0]];assert bank['ids']==expected
 y=yall[bank['ids']];chosen=[];train_counts=torch.bincount(y,minlength=n)
 for c in range(n):chosen.extend(torch.where(y==c)[0][:64].tolist())
 x=bank['x'][chosen];x=F.normalize(x,dim=1) if ds=='NIH' else x;x=x.cuda();y=y[chosen].cuda()
 assert torch.allclose(x.norm(dim=1),torch.ones(len(x),device='cuda'),atol=2e-6)
 tail=[c for c in range(n) if train_counts[c]<250]
 records=[]
 for seed in [74002,74003,74004]:
  write('STATUS.json',dict(phase='UNTRAINED_ALIGNMENT',seed=seed,time=time.time()))
  ns['seed_all'](seed);ns['install_feature_model'](ns['INC'],bank['text'],bank['prompts'],bank['logit_scale']);learner=ns['CheckedLearner'](ns['make_args'](seed,'H-unit'));net=learner._network
  ns['seed_all'](seed);net.update_fc(n);net.update_task();net.to('cuda');net.eval()
  named=dict(net.named_parameters());keep={k:p.detach().clone() for k,p in named.items() if not ('Adapter.0.fc.' in k)}
  rng=(torch.get_rng_state().clone(),[t.clone() for t in torch.cuda.get_rng_state_all()])
  with torch.no_grad():
   text=net.model.encode_text(net.tokenizer([ns['LEAF_TEMPLATE'].format(v) for v in names[:n]]).cuda())
   def evaluate():
    ii=net.Image_encode(x);tt=net.Text_encode(text);hi=net.trans_hyper(ii);ht=net.trans_hyper(tt)
    scores={'euclidean':F.normalize(ii,dim=1)@F.normalize(tt,dim=1).T,'hyper':-G.pairwise_dist(hi,ht,net.get_curv())}
    result={}
    for key,z in scores.items():
     assert torch.isfinite(z).all();pred=z.argmax(1);correct=[int(((pred==y)&(y==c)).sum()) for c in range(n)];counts=torch.bincount(y,minlength=n).tolist()
     result[key]=dict(correct_by_class=correct,counts=counts,macro=100*sum(v/q for v,q in zip(correct,counts))/n,tail_macro=None if not tail else 100*sum(correct[c]/counts[c] for c in tail)/len(tail))
    return result
   original=evaluate()
   for adapter in [net.Image_Adapter[0],net.Text_Adapter[0]]:
    fc=adapter.fc[0];assert isinstance(fc,torch.nn.Linear) and fc.weight.shape==(512,512)
    fc.weight.copy_(torch.eye(512,device=fc.weight.device,dtype=fc.weight.dtype));fc.bias.zero_()
   assert torch.allclose(net.Image_encode(x),x*net.Image_Adapter[0].log_alpha.clamp(max=0).exp(),atol=2e-6)
   identity=evaluate()
  assert all(torch.equal(v,named[k]) for k,v in keep.items())
  assert torch.equal(rng[0],torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(rng[1],torch.cuda.get_rng_state_all()))
  records.append(dict(seed=seed,stage=0,rows=len(x),classes=n,tail_classes=tail,training_counts=train_counts.tolist(),original=original,identity=identity,only_first_fc_weight_bias_changed=True,setting_identity_rng_unchanged=True,optimizer_steps=0,not_training_effect=True))
  del learner,net,named,keep;torch.cuda.empty_cache()
 write('RESULTS.json',dict(dataset=ds,cells=records,passed=len(records)==3,optimizer_steps=0,VAL_reads=0,TEST_reads=0));write('STATUS.json',dict(phase='COMPLETE',time=time.time()))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
