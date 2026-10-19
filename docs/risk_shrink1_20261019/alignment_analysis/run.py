"""Frozen-score diagnostic only: no feature reads, fits, searches, or coefficient selection."""
import csv,json,os,resource,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch


def center(x):return x-x.mean(1,keepdims=True)


def patient_weights(y,g):
 w=np.empty(len(y))
 for c in np.unique(y):
  ids=np.flatnonzero(y==c);counts=Counter(g[ids].tolist());w[ids]=[1/(len(counts)*counts[z]) for z in g[ids]]
 return w


def metrics(score,y,g,w,mask):
 if not mask.any():return dict(images=0,patients=0,represented_classes=0)
 k=score.shape[1];target=np.eye(k)[y];target-=target.mean(1,keepdims=True)
 loss=np.mean((center(score)-target)**2,axis=1);correct=score.argmax(1)==y
 ids=np.flatnonzero(mask);classes=np.unique(y[ids]);byclass=[ids[y[ids]==c] for c in classes]
 return dict(images=len(ids),patients=len(np.unique(g[ids])),represented_classes=len(classes),PP_risk=float(np.average(loss[ids],weights=w[ids])),PP_accuracy=float(np.average(correct[ids],weights=w[ids])),BA=float(np.mean([correct[a].mean() for a in byclass])),image_balanced_risk=float(np.mean([loss[a].mean() for a in byclass])),correct=int(correct[ids].sum()),micro_accuracy=float(correct[ids].mean()))


def validation_rows(cfg):
 with open(Path(cfg['manifest'])/'val.csv') as f:rows=list(csv.DictReader(f))
 with open(Path(cfg['author_labels'])/'val.csv') as f:source={r['id']:r for r in csv.DictReader(f)}
 for r in rows:
  name=Path(r['relative_path']).name;r['label']=int(r['original_label']);r['group']=str(int(source[name]['subject_id']))
  assert r['group']==str(int(name.split('_')[0])) and r['split']=='val'
 assert len(rows)==300
 return rows


def main():
 start=time.time();torch.set_num_threads(4);root=Path(os.environ['RUN_OUTPUT']);out=Path(os.environ['ANALYSIS_OUTPUT']);out.mkdir(exist_ok=True);assert not (out/'metrics.json').exists()
 cfg=json.loads((root/'config.private.json').read_text());audit=json.loads((root/'TRAINING_AUDIT.json').read_text());assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE' and len(audit)==6
 assert json.loads((root/'FIT_SEAL.json').read_text())['query_opened'] is False
 original=json.loads((root/'results.json').read_text());rows=validation_rows(cfg);records=[];n_scores=0
 for stage in range(6):
  k=sum(cfg['task_sizes'][:stage+1]);o=np.load(root/f'OOF_{stage}.private.npz');frozen=audit[stage];c=center(o['CLS']);d=center(o['joint']-o['CLS'])
  oof_scores=dict(cls768_ridge=o['CLS'],scale_joint=o['joint'],uniform_shrink=center(c+d*frozen['uniform_alpha']),risk_limited_shrink=center(c+d*np.array(frozen['risk_limited_alpha'])))
  val=torch.load(root/f'stage_{stage}.private.pt',weights_only=True);seen=val['seen'];selected=[r for r in rows if r['label'] in seen]
  assert seen==cfg['order'][:k] and [r['label'] for r in selected]==val['labels'].tolist()
  yv=np.array([seen.index(r['label']) for r in selected]);gv=np.array([r['group'] for r in selected]);wv=patient_weights(yv,gv)
  val_scores={arm:val['scores'][arm].numpy() for arm in [*oof_scores,'df_matched_joint']}
  for source,y,g,w,scores in [('OOF',o['labels'],o['groups'],o['weights'],oof_scores),('validation',yv,gv,wv,val_scores)]:
   assert np.max(np.abs(w-patient_weights(y,g)))<1e-12
   base=scores['cls768_ridge'].argmax(1)==y
   for arm,score in scores.items():
    assert score.shape==(len(y),k) and np.isfinite(score).all();n_scores+=len(y)
    for subset,mask in [('all',np.ones(len(y),dtype=bool)),('CLS_correct',base),('CLS_wrong',~base)]:
     row=dict(stage=stage,classes=k,source=source,arm=arm,subset=subset,**metrics(score,y,g,w,mask));records.append(row)
    if source=='validation':
     assert abs(records[-3]['BA']-original[stage]['metrics'][arm]['macro'])<1e-12
    elif arm in ['cls768_ridge','uniform_shrink','risk_limited_shrink']:
     key={'cls768_ridge':'OOF_CLS_risk','uniform_shrink':'OOF_uniform_risk','risk_limited_shrink':'OOF_limited_risk'}[arm]
     assert abs(records[-3]['PP_risk']-frozen[key])<1e-10
   assert time.time()-start<300 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*2**30
 result=dict(protocol=dict(arms_OOF=list(oof_scores),arms_validation=list(val_scores),OOF_df_matched_joint='unavailable: fold-specific scores were never generated; no new fits',patient_weights='within each true class equal patients, within patient equal images; full-source weights conditionally renormalized for subsets',BA='image-balanced class mean accuracy; subsets average over represented classes',image_balanced_risk='mean per-image squared centered-score error across outputs, then image mean within class and class mean',OOF_use='coefficient-fitting data; not independent shrinker evaluation',validation_use='previous development validation; no refitting or parameter selection',identity_binding='same frozen manifest row order as original generation; labels and seen order checked; scores lack standalone sample IDs'),records=records)
 (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
 with (out/'metrics.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
 costs=dict(status='COMPLETE',wall_seconds=time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,score_rows_evaluated=n_scores,new_training_encodes=0,new_validation_encodes=0,new_head_fits=0,new_solver_calls=0,coefficient_updates=0,GPU_used=False,history_unchanged=True)
 (out/'costs.json').write_text(json.dumps(costs,indent=2)+'\n');assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<8*2**20
 print(json.dumps(costs))


if __name__=='__main__':main()
