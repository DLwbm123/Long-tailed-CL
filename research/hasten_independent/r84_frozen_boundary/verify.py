"""Only saved arrays and metadata; no encoder or classifier calls."""
import os,json
from pathlib import Path
import torch
from common import write
from data import train_rows
from folds import assign_folds
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);records=json.loads((r/'RECORDS.private.json').read_text())
seal=json.loads((r/'ALL_HEADS_SEALED.json').read_text());done=json.loads((r/'TRAIN_COMPLETE.json').read_text())
assert seal['cells']==len(records)==12 and seal['fitted_heads']==done['fitted_heads']==24
assert seal['Q_scores']==0 and seal['time']<done['time']
rows,y=train_rows(cfg['domains']['HK']);folds=assign_folds(y.tolist(),[x['identity_component'] for x in rows]);total=0
for rec in records:
 s,b,n=rec['stage'],rec['fold'],rec['seen'];feature=torch.load(r/f'stage{s}.private.pt',map_location='cpu',weights_only=False)
 head=torch.load(r/f's{s}f{b}.private.pt',map_location='cpu',weights_only=False);Q=head['Q'];S=head['S']
 source=torch.load(Path(cfg['source_run'])/f'HK/uniform_controller/74002/head{s}.private.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,source['delta'][k]) for k,v in feature['source_delta'].items())
 assert feature['ids']==[i for stage in cfg['domains']['HK']['stages'][:s+1] for i,z in enumerate(y.tolist()) if z in stage]
 assert all(int(feature['labels'][j])<n for j in range(len(feature['ids'])))
 assert S==[j for j,i in enumerate(feature['ids']) if folds[i]==1-b]
 assert Q==[j for j,i in enumerate(feature['ids']) if folds[i]==b] and not set(S)&set(Q)
 for arm in ['natural','balanced']:
  fit=head['heads'][arm];assert fit['max_abs_gradient']<=1e-5 and abs(fit['data_mass']-n)<1e-10
  assert fit['residual'].shape==(n,513) and torch.isfinite(fit['residual']).all()
 scores=torch.load(r/f'scores_s{s}f{b}.private.pt',map_location='cpu',weights_only=False)
 assert scores['Q']==Q and torch.equal(scores['labels'],feature['labels'][Q])
 for arm in ['NCM','natural','balanced']:
  raw=scores['arms'][arm];assert torch.equal(raw['prediction'],raw['logits'].argmax(1))
  for tag,ids in [('full',None),('Qa',set(head['Qa_ids'])),('Qb',set(head['Qb_ids']))]:
   mask=torch.ones(len(Q),dtype=torch.bool) if ids is None else torch.tensor([i in ids for i in scores['ids']])
   yy=scores['labels'][mask];pred=raw['prediction'][mask];den=torch.bincount(yy,minlength=n);cc=torch.bincount(yy[yy==pred],minlength=n)
   assert rec['query'][arm][tag]==dict(counts=den.tolist(),correct=cc.tolist())
   if tag=='full':total+=len(yy)
assert total==done['new_TRAIN_query_rows_scored']
write(r/'VERIFICATION.json',dict(passed=True,cells=12,fitted_heads=24,exact_source_deltas=True,
    saved_predictions_counts_recomputed=True,S_only_fit_and_component_disjoint_Q=True,
    new_visual_model_scores=0,new_classifier_forward_calls=0,new_optimizer_steps=0,VAL_reads=0,TEST_reads=0))
