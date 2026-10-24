"""Uncapped TRAIN-fold, complete schedule; all checkpoints predeclared, no selection."""
import os,json,csv,time,math,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from worker import ns,cfg
out=Path(cfg['output'])
def main():
 n=len(ns['STAGES'][0]);fold=cfg['audit_fold'];seed=74002
 with open(Path(cfg['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 assert all(r['split']=='train' for r in rows);labels=torch.tensor([ns['ORDER'].index(int(r['original_label'])) for r in rows]);bank=torch.load(Path(cfg['cache_root'] if cfg['dataset']=='NIH' else cfg['shared_cache'])/'stage_0.private.pt',map_location='cpu',weights_only=False);assert bank['ids']==[i for i,y in enumerate(labels.tolist()) if y<n]
 x=bank['x'];text=bank['text'];x=F.normalize(x,dim=1) if cfg['dataset']=='NIH' else x;text=F.normalize(text,dim=1) if cfg['dataset']=='NIH' else text;assert torch.allclose(x.norm(dim=1),torch.ones(len(x)),atol=2e-6);assert torch.allclose(text.norm(dim=1),torch.ones(len(text)),atol=2e-6)
 y=labels[bank['ids']];fit=[];hold=[]
 for c in range(n):
  ids=torch.where(y==c)[0];fit.extend(ids[fold::2].tolist());hold.extend(ids[1-fold::2].tolist())
 assert not set(fit)&set(hold) and len(fit)+len(hold)==len(x)
 ns['seed_all'](seed+10000*fold);ns['install_feature_model'](ns['INC'],text,bank['prompts'],bank['logit_scale']);learner=ns['CheckedLearner'](ns['make_args'](seed,'H-unit'));learner.seed=seed;learner.arm='H-unit';learner.start_audit();learner._curve_holdout=(x[hold],y[hold]);learner._curve_train_counts=torch.bincount(y,minlength=len(ns['TYPE'])).tolist();learner.incremental_train(ns['DataManager'](x,y,fit,[]))
 assert learner.audit['steps']==10*math.ceil(len(fit)/64);assert torch.cuda.max_memory_reserved()<8*2**30
 record=dict(phase='COMPLETE',mode=cfg['decision_mode'],fold=fold,seed=seed,optimizer_steps=learner.audit['steps'],fit_counts=torch.bincount(y[fit],minlength=n).tolist(),holdout_counts=torch.bincount(y[hold],minlength=n).tolist(),train_counts=torch.bincount(y,minlength=n).tolist(),fit_holdout_disjoint=True,fit_holdout_uncapped=True,VAL_reads=0,TEST_reads=0,epochs=10,audit=learner.audit)
 (out/'RESULTS.json').write_text(json.dumps(record,indent=2));(out/'STATUS.json').write_text(json.dumps(dict(phase='COMPLETE',time=time.time())))
try:main()
except BaseException:
 (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
