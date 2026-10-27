"""Verify saved class moments/counts and full scope, without models/images."""
from pathlib import Path
import os,json,torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);out=r/'HK';assert json.loads((r/'CONTROL_EXIT.json').read_text())['exit_code']==0
records=json.loads((out/'CENSUS_RECORDS.json').read_text());summaries=json.loads((out/'CLASS_SUMMARIES.json').read_text());seal=json.loads((out/'CENSUS_SEAL.json').read_text());assert len(records)==len(summaries)==18 and seal['TRAIN_endpoint_embedding_rows']==87651 and seal['optimizer_steps']==seal['VAL_TEST_reads']==0
for rec,s in zip(records,summaries):
 assert (rec['stage'],rec['model'])==(s['stage'],s['model']) and rec['optimizer_steps']==rec['backwards']==rec['VAL_TEST_reads']==0
 assert rec['old_folds_preserved'] and rec['original_weights_unchanged'] and rec['LoRA_delta_unchanged'] and rec['gradients_absent'] and rec['opposite_component_fold_support'] and rec['full_prototype_parity_maxabs']<=2e-5
 m=torch.load(out/rec['model']/f"moments{rec['stage']}.private.pt",map_location='cpu',weights_only=False);assert m['counts'].tolist()==rec['counts'] and torch.equal(m['fold_counts'].sum(0),m['counts']);assert torch.allclose(m['fold_sums'].sum(0),m['sums'],atol=1e-9,rtol=1e-12)
 for c,v in enumerate(s['classes']):
  n=int(m['counts'][c]);var=float((m['squared_norm_sums'][c]-m['sums'][c].square().sum()/n)/(n-1));assert v['count']==n and abs(v['unbiased_within_trace']-max(var,0))<1e-9
  for mode in ['raw','natural']:assert 0<=v[mode]['correct']<=n and 0<=v[mode]['correct_probability_mean']<=1
assert sum(v['TRAIN_rows'] for v in records)==87651
c=json.loads((out/'REGENERABLE_CACHE_CLEANUP.json').read_text());assert c['reference_max_abs']<=2e-5 and c['TRAIN_only'] and c['science_preserved']
p=r/'results';p.mkdir(exist_ok=True);x=dict(passed=True,saved_only=True,new_model_scores=0,records=18,TRAIN_endpoint_embedding_rows=87651,TRAIN_scoring_routes=36,optimizer_steps=0,VAL_TEST_reads=0);(p/'VERIFICATION.json').write_text(json.dumps(x,indent=2));print(json.dumps(x))
