"""Saved arrays only; verifies pooling inputs, source bindings and counts."""
import os,json,torch
import torch.nn.functional as F
from pathlib import Path
from common import write
from pooling import channels
from outcome import counts
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);s=Path(cfg['feature_source_run'])
records=json.loads((r/'RECORDS.private.json').read_text());source_records=json.loads((s/'RECORDS.private.json').read_text())
seal=json.loads((r/'ALL_HEADS_SEALED.json').read_text());done=json.loads((r/'TRAIN_COMPLETE.json').read_text())
assert len(records)==seal['cells']==12 and seal['fitted_heads']==done['fitted_heads']==24
assert seal['scientific_Q_rows_scored']==0 and seal['time']<done['time'];total=iterations=evals=0;maximum=0.;reference_error=0.
for rec in records:
 stage,fold,n=rec['stage'],rec['fold'],rec['seen'];head=torch.load(r/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
 feature=torch.load(r/f'stage{stage}.private.pt',map_location='cpu',weights_only=False);old=torch.load(s/f'stage{stage}.private.pt',map_location='cpu',weights_only=False)
 src=torch.load(s/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
 assert head['S']==src['S'] and head['Q']==src['Q'] and not set(head['S'])&set(head['Q'])
 assert feature['ids']==old['ids'] and torch.equal(feature['labels'],old['labels']) and int(feature['labels'].max())<n
 assert all(torch.equal(v,old['source_delta'][k]) for k,v in feature['source_delta'].items())
 error=float((feature['cls']-old['features']).abs().max());assert error<=2e-5;reference_error=max(reference_error,error)
 for z in [feature['cls'],feature['patch']]:assert torch.isfinite(z).all() and torch.allclose(z.norm(dim=1),torch.ones(len(z)),atol=2e-6)
 zz=channels(feature['cls'],feature['patch']);S=head['S'];y=feature['labels'][S];nc=torch.bincount(y,minlength=n)
 for arm,z in zz.items():
  sums=torch.zeros(n,1024,dtype=torch.double);sums.index_add_(0,y,z[S].double());proto=F.normalize((sums/nc[:,None]).float(),dim=1)
  assert torch.equal(proto,head['prototypes'][arm]);h=head['heads'][arm]
  assert h['max_abs_gradient']<=1e-5 and abs(h['data_mass']-n)<1e-10 and h['residual'].shape==(n,1025)
  iterations+=h['iterations'];evals+=h['closure_evaluations'];maximum=max(maximum,h['max_abs_gradient'])
 scores=torch.load(r/f'scores_s{stage}f{fold}.private.pt',map_location='cpu',weights_only=False);Q=head['Q']
 assert scores['Q']==Q and scores['ids']==[feature['ids'][i] for i in Q] and torch.equal(scores['labels'],feature['labels'][Q])
 for arm,h in head['heads'].items():
  raw=scores['arms'][arm];assert torch.equal(raw['prediction'],raw['logits'].argmax(1))
  assert rec['query'][arm]==counts(raw['prediction'],scores['labels'],n,scores['ids'],head['Qa_ids'],head['Qb_ids']);total+=len(Q)
 original=next(v for v in source_records if v['stage']==stage and v['fold']==fold)
 for a,b in [('NCM','NCM'),('CLS','balanced'),('natural','natural')]:assert rec['query'][a]==original['query'][b]
assert total==done['new_classifier_TRAIN_query_rows_scored'] and iterations==seal['CPU_head_iterations'] and evals==seal['CPU_head_closure_evaluations']
write(r/'VERIFICATION.json',dict(passed=True,cells=12,fitted_heads=24,source_feature_ids_labels_deltas_exact=True,
 S_only_prototypes_rebuilt_exact=True,saved_predictions_denominators_exact=True,source_CLS_max_abs=reference_error,
 CPU_head_iterations=iterations,CPU_head_closure_evaluations=evals,max_abs_gradient=maximum,
 new_model_scores=0,new_optimizer_steps=0,new_VAL_TEST_reads=0))
