"""Independent frozen-score metric recount using centered-error identity and bincount weights."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from run import validation_rows
start=time.time();torch.set_num_threads(4);root=Path(os.environ['RUN_OUTPUT']);out=Path(os.environ['ANALYSIS_OUTPUT'])
cfg=json.loads((root/'config.private.json').read_text());aa=json.loads((root/'TRAINING_AUDIT.json').read_text());rows=validation_rows(cfg)
saved=json.loads((out/'metrics.json').read_text())['records'];lookup={(x['stage'],x['source'],x['arm'],x['subset']):x for x in saved};maximum=0.;checked=0;coverage=[]
for stage in range(6):
 o=np.load(root/f'OOF_{stage}.private.npz');k=o['CLS'].shape[1];c=o['CLS'];j=o['joint'];delta=j-c;delta-=delta.mean(1,keepdims=True)
 # A common score offset does not affect centered squared error or argmax.
 oof=dict(cls768_ridge=c,scale_joint=j,uniform_shrink=c+delta*aa[stage]['uniform_alpha'],risk_limited_shrink=c+delta*np.array(aa[stage]['risk_limited_alpha']))
 s=torch.load(root/f'stage_{stage}.private.pt',weights_only=True);seen=s['seen'];sel=[r for r in rows if r['label'] in seen];yv=np.array([seen.index(r['label']) for r in sel]);gv=np.array([r['group'] for r in sel]);val={a:s['scores'][a].numpy() for a in [*oof,'df_matched_joint']}
 for source,y,g,scores in [('OOF',o['labels'],o['groups'],oof),('validation',yv,gv,val)]:
  w=np.zeros(len(y))
  for cls in range(k):
   ids=np.flatnonzero(y==cls);_,inverse,counts=np.unique(g[ids],return_inverse=True,return_counts=True);w[ids]=1/(len(counts)*counts[inverse])
  assert abs(w.sum()-k)<1e-10
  base=scores['cls768_ridge'].argmax(1)==y
  for arm,score in scores.items():
   e=score.copy();e[np.arange(len(y)),y]-=1
   loss=np.square(e).mean(1)-np.square(e.mean(1));pred=score.argmax(1);correct=pred==y
   for subset,mask in [('all',np.ones(len(y),dtype=bool)),('CLS_correct',base),('CLS_wrong',~base)]:
    a=lookup[stage,source,arm,subset];n=np.bincount(y[mask],minlength=k);valid=n>0
    expected=dict(PP_risk=float(np.dot(w[mask],loss[mask])/w[mask].sum()),PP_accuracy=float(np.dot(w[mask],correct[mask])/w[mask].sum()),BA=float((np.bincount(y[mask],weights=correct[mask],minlength=k)[valid]/n[valid]).mean()),image_balanced_risk=float((np.bincount(y[mask],weights=loss[mask],minlength=k)[valid]/n[valid]).mean()),micro_accuracy=float(correct[mask].mean()))
    assert a['images']==int(mask.sum()) and a['patients']==len(np.unique(g[mask])) and a['represented_classes']==int(valid.sum()) and a['correct']==int(correct[mask].sum())
    for key,v in expected.items():
     err=abs(v-a[key]);maximum=max(err,maximum);assert err<1e-10
    checked+=1
  if source=='OOF':
   covered=np.zeros(len(y),dtype=bool)
   for group in aa[stage]['constraints']:
    if group['kind']=='CLS_correct':covered|=(o['folds']==group['fold'])&(y==seen.index(group['class_id']))&base
   wrong=scores['risk_limited_shrink'].argmax(1)!=y
   coverage.append(dict(stage=stage,CLS_correct_images=int(base.sum()),protected_CLS_correct_images=int(covered.sum()),protected_images_harmed=int((covered&wrong).sum()),unprotected_CLS_correct_images=int((base&~covered).sum()),unprotected_images_harmed=int((base&~covered&wrong).sum())))
assert checked==162
result=dict(status='PASS',records_recounted=checked,max_metric_error=maximum,risk_recount_method='mean((score-onehot)^2)-mean(score-onehot)^2',weights_recount_method='per-class np.unique patient multiplicities',existing_validation_BA_and_OOF_risk_preserved=True,protected_OOF_correct_subsets=coverage,no_training_no_solver_no_encodes=True,wall_seconds=time.time()-start)
(out/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
