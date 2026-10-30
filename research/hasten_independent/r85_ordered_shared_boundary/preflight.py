"""Two true saved TRAIN feature rows, not a new native-pixel test."""
import json,os,time,torch
from pathlib import Path
from common import write,guard
from boundary import logits
from structure import matrices
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);s=Path(cfg['feature_source_run']);t=time.time();guard(cfg);torch.set_num_threads(4)
assert json.loads((s/'VERIFICATION.json').read_text())['passed']
feature=torch.load(s/'stage0.private.pt',map_location='cpu',weights_only=False)
head=torch.load(s/'s0f0.private.pt',map_location='cpu',weights_only=False)
scores=torch.load(s/'scores_s0f0.private.pt',map_location='cpu',weights_only=False)
ii=head['Q'][:2];z=feature['features'][ii];p=head['prototypes'];n=len(p);L,nodes=matrices(cfg['domains']['HK']['ordered_names'][:n])
new=logits(z,p,torch.zeros(n,513,dtype=torch.double));err=float((new-scores['arms']['NCM']['logits'][:2]).abs().max());assert err<=2e-5
assert max(feature['labels'][ii])<n and torch.isfinite(new).all()
write(r/'NATIVE_CHECK.json',dict(passed=True,kind='saved_feature_native_engineering_not_pixel_encoder',
 actual_TRAIN_feature_rows=2,classifier_rows_scored=2,source_NCM_max_abs=err,
 visual_encoding_rows=0,backward_calls=0,optimizer_steps=0,VAL_TEST_reads=0,
 unchanged_R84_encoder_receipt_reused=True,wall_seconds=time.time()-t))
