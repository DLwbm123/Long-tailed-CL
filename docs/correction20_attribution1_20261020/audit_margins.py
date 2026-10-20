"""Align already-frozen native score margins to parent-joint temperature units."""
import time
START=time.time()
import json,os
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);parent=Path(cfg['parent_run']);pred=np.load(parent/'outer_predictions.private.npz');y=pred['labels'];fold=pred['folds'];temperatures=np.array([float(torch.load(parent/f'outer_{f}.private.pt',weights_only=True)['temperature']) for f in range(3)]);rowtemp=temperatures[fold];p=out/'EXISTING_SIGNAL_AUDIT.json';audit=json.loads(p.read_text());original=out/'EXISTING_SIGNAL_AUDIT_native.private.json'
if not original.exists():original.write_text(p.read_text())
for row in audit['outer']:
 c=cfg['order'].index(row['class_id']);mask=y==c;yy=y[mask];n=len(yy);native={a:pred[a][mask] for a in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction']};common={a:s/rowtemp[mask,None] if a in ['cls768_ridge','scale_joint','df_matched_joint'] else s for a,s in native.items()};margins={}
 for a,s in common.items():
  wrong=s.copy();wrong[np.arange(n),yy]=-np.inf;m=s[np.arange(n),yy]-wrong.max(1);margins[a]=m;item=row['arms'][a];item['native_score_margin_quantiles']=item.pop('true_vs_max_wrong_margin_quantiles');item['common_temperature_margin_quantiles']=np.quantile(m,[0,.25,.5,.75,1]).tolist()
 for a in ['plain_correction','utility_correction']:
  harmed=(native['scale_joint'].argmax(1)==yy)&(native[a].argmax(1)!=yy);item=row['harmed_transitions'][a];assert int(harmed.sum())==item['images'];item['common_margin_before_quantiles']=np.quantile(margins['scale_joint'][harmed],[0,.5,1]).tolist() if harmed.any() else None;item['common_margin_after_quantiles']=np.quantile(margins[a][harmed],[0,.5,1]).tolist() if harmed.any() else None
  if harmed.any():assert np.all(margins['scale_joint'][harmed]>=0) and np.all(margins[a][harmed]<=0)
audit.update(margin_units='Parent joint temperature per outer fold. Divide native ridge scores by that positive temperature; corrected scores already in these units. Native margins retained separately.',margin_audit_process_wall_seconds=time.time()-START,margin_audit_new_fits=0);p.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(dict(state='PASS',process_wall_seconds=audit['margin_audit_process_wall_seconds'],new_fits=0,new_encodes=0)))
