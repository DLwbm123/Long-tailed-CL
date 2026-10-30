"""Saved tensors/counts only; no new classifier forwards or fitting."""
import os,json,torch
from pathlib import Path
from common import write
from structure import matrices
from train import counts
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);s=Path(cfg['feature_source_run'])
records=json.loads((r/'RECORDS.private.json').read_text());source_records=json.loads((s/'RECORDS.private.json').read_text())
seal=json.loads((r/'ALL_HEADS_SEALED.json').read_text());done=json.loads((r/'TRAIN_COMPLETE.json').read_text())
assert len(records)==seal['cells']==12 and seal['fitted_heads']==done['fitted_heads']==36
assert seal['scientific_Q_rows_scored']==0 and seal['time']<done['time']
total=iterations=evals=0;maximum=0.
for rec in records:
 stage,fold,n=rec['stage'],rec['fold'],rec['seen'];head=torch.load(r/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
 src=torch.load(s/f's{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
 assert head['S']==src['S'] and head['Q']==src['Q'] and not set(head['S'])&set(head['Q'])
 assert torch.equal(head['prototypes'],src['prototypes'])
 ms,nodes=matrices(cfg['domains']['HK']['ordered_names'][:n]);assert head['nodes']==nodes
 for a in ms:assert torch.equal(head['matrices'][a],ms[a])
 scores=torch.load(r/f'scores_s{stage}f{fold}.private.pt',map_location='cpu',weights_only=False)
 assert scores['Q']==head['Q']
 for arm,h in head['heads'].items():
  assert h['max_abs_gradient']<=1e-5 and abs(h['data_mass']-n)<1e-10 and torch.isfinite(h['residual']).all()
  iterations+=h['iterations'];evals+=h['closure_evaluations'];maximum=max(maximum,h['max_abs_gradient'])
  raw=scores['arms'][arm];assert torch.equal(raw['prediction'],raw['logits'].argmax(1))
  assert rec['query'][arm]==counts(raw['prediction'],scores['labels'],n,scores['ids'],head['Qa_ids'],head['Qb_ids'])
  total+=len(scores['labels'])
 original=next(v for v in source_records if v['stage']==stage and v['fold']==fold)
 for a,old in [('NCM','NCM'),('unstructured','balanced'),('natural','natural')]:assert rec['query'][a]==original['query'][old]
assert total==done['new_classifier_TRAIN_query_rows_scored'] and iterations==seal['CPU_head_iterations'] and evals==seal['CPU_head_closure_evaluations']
write(r/'VERIFICATION.json',dict(passed=True,cells=12,fitted_heads=36,source_reference_counts_exact=True,
 declared_graphs_exact=True,S_only_and_fixed_source=True,saved_predictions_and_denominators=True,
 CPU_head_iterations=iterations,CPU_head_closure_evaluations=evals,max_abs_gradient=maximum,
 new_model_scores=0,new_optimizer_steps=0,new_VAL_TEST_reads=0))
