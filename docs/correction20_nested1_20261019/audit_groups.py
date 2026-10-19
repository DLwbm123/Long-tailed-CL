"""Direct weighted harm, frequency and gain-retention recount from frozen outer scores."""
import time
START=time.time()
import os,json
from pathlib import Path
import numpy as np

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);d=np.load(out/'outer_predictions.private.npz');y=d['labels'];g=d['groups'];f=d['folds'];arms=['cls768_ridge','scale_joint','df_matched_joint','plain_correction','utility_correction'];correct={a:d[a].argmax(1)==y for a in arms};counts=np.bincount(y,minlength=20);bw=1/(20*counts[y]);pw=np.empty(len(y))
for c in range(20):
 ids=np.flatnonzero(y==c);_,inverse,numbers=np.unique(g[ids],return_inverse=True,return_counts=True);pw[ids]=1/(20*len(numbers)*numbers[inverse])
assert abs(bw.sum()-1)<1e-12 and abs(pw.sum()-1)<1e-12
results=json.loads((out/'results.json').read_text());checks={};groups={};frequency={}
for a in arms:
 ok=correct[a];assert abs(float(bw@ok)-results['metrics'][a]['BA'])<1e-12 and abs(float(pw@ok)-results['metrics'][a]['PP_accuracy'])<1e-12
 for ref in ['cls768_ridge','scale_joint','df_matched_joint','plain_correction']:
  if a==ref:continue
  harm=~ok&correct[ref];rescue=ok&~correct[ref];checks[a+'-'+ref]=dict(rescue=int(rescue.sum()),harm=int(harm.sum()),BA_rescue=float(bw@rescue),BA_harm=float(bw@harm),PP_rescue=float(pw@rescue),PP_harm=float(pw@harm))
for name,mask in [('both_correct',correct['cls768_ridge']&correct['scale_joint']),('CLS_only',correct['cls768_ridge']&~correct['scale_joint']),('joint_only',~correct['cls768_ridge']&correct['scale_joint']),('both_wrong',~correct['cls768_ridge']&~correct['scale_joint'])]:
 groups[name]=dict(images=int(mask.sum()),BA_mass=float(bw@mask),PP_mass=float(pw@mask),arms={a:dict(correct=int((correct[a]&mask).sum()),correct_fraction=float(correct[a][mask].mean()),BA_fraction=float(bw@(correct[a]&mask)/(bw@mask)),PP_fraction=float(pw@(correct[a]&mask)/(pw@mask))) for a in arms})
for name,original_ids in cfg['frequency_bins'].items():
 mask=np.isin(y,[cfg['order'].index(c) for c in original_ids]);frequency[name]=dict(images=int(mask.sum()),classes=original_ids,arms={a:dict(correct=int((correct[a]&mask).sum()),BA=float(bw@(correct[a]&mask)/(bw@mask)),PP=float(pw@(correct[a]&mask)/(pw@mask)),CLS_harm=int((~correct[a]&correct['cls768_ridge']&mask).sum()),CLS_rescue=int((correct[a]&~correct['cls768_ridge']&mask).sum())) for a in arms})
near={a:dict(top_two_gap_below_1e_10=int((np.diff(np.partition(d[a],-2,axis=1)[:,-2:],axis=1).ravel()<1e-10).sum()),minimum_top_two_gap=float(np.diff(np.partition(d[a],-2,axis=1)[:,-2:],axis=1).min())) for a in arms}
r=dict(state='PASS',scope='frozen_outer_predictions_only',direct_metrics_recounted=True,contrasts=checks,groups=groups,frequency=frequency,score_boundary=near,new_encodes=0,new_fits=0,process_wall_seconds=time.time()-START);(out/'WEIGHTED_GROUPS.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(state='PASS',process_wall_seconds=r['process_wall_seconds'])))
