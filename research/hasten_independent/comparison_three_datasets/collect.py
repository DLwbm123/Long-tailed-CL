"""Read existing evidence only. Export no sample identities or private paths."""
import os,json,csv,statistics as st
from pathlib import Path
base=Path(os.environ['EVIDENCE_ROOT']);old=base/'three_dataset1_20261008';new=base/'hasten_independent/r12_three_datasets';out=Path(os.environ['COMPARE_OUT']);out.mkdir(parents=True,exist_ok=True)
newa=json.loads((new/'results/ANALYSIS.json').read_text());oldp=json.loads((old/'public/DATA_PROTOCOL.json').read_text());rows=[];audit={};raw={}
for ds in ['ISIC','HK']:
 cfg=json.loads((new/ds/'config.private.json').read_text());spec=oldp[ds];order=cfg['order'];tail=[c for c,n in enumerate(cfg['train_counts']) if n<250]
 assert spec['shuffled']==order and spec['task_sizes']==cfg['task_sizes'];assert [spec['class_counts'][str(c)] for c in range(len(order))]==cfg['train_counts']
 current={}
 for split in ['train','val']:
  with open(Path(cfg['manifest'])/(split+'.csv')) as f:rr=list(csv.DictReader(f))
  current[split]=rr
  expected=spec['class_counts'] if split=='train' else spec['evaluation_counts']
  assert {str(c):sum(int(r['original_label'])==c for r in rr) for c in range(len(order))}==expected
 assert not({r['identity_component'] for r in current['train']}&{r['identity_component'] for r in current['val']})
 audit[ds]=dict(same_manifest_reference=True,full_train_counts_match=True,all_stage_eval_counts_match=True,order=order,task_sizes=cfg['task_sizes'],train_rows=len(current['train']),eval_rows=len(current['val']),common_tail_original_ids=tail,old_tail_rule='lowest-frequency half (including median for odd class count)',common_tail_rule='full TRAIN count <250',historical_row_level_identity_snapshot_available=False,scope='same recorded manifest reference, class counts and evaluation schedule verified; exact historical per-row membership not independently recoverable from saved aggregates',fit_counts=spec['fit_counts'],meta_counts=spec['meta_counts'])
 raw[ds]={}
 for method in ['cf_linear','cf_prototype','cf_weighted','cf_group']:
  p=old/f'{ds}_shuffled_{method}';ic=json.loads((p/'INPUT.private.json').read_text());m=json.loads((p/'metrics.json').read_text());status=json.loads((p/'STATUS.json').read_text());assert ic['manifest']==cfg['manifest'] and ic['order']==order and ic['task_sizes']==cfg['task_sizes'];assert len(m['stages'])==len(cfg['task_sizes']) and not m['test_accessed']
  diag=json.loads((p/'diagnostics.json').read_text());stages=[]
  for i,s in enumerate(m['stages']):
   seen=order[:sum(cfg['task_sizes'][:i+1])];assert s['seen']==seen
   assert s['per_class_n']=={str(c):spec['evaluation_counts'][str(c)] for c in seen}
   ba=st.mean(s['per_class_recall'].values());assert abs(ba-s['balanced_accuracy'])<1e-10
   stages.append(dict(stage=i,seen=seen,BA=100*ba,tail_BA=100*st.mean(s['per_class_recall'][str(c)] for c in tail if c in seen),counts=s['per_class_n'],recall=s['per_class_recall']))
  rows.append(dict(dataset=ds,method=method,seed=ic['seed'],mean_BA=st.mean(s['BA'] for s in stages),final_BA=stages[-1]['BA'],tail_BA=stages[-1]['tail_BA'],original_report_tail_BA=100*m['final_tail_recall'],stages=stages,epochs=ic['epochs'],optimizer_steps=max(v['steps'] for v in diag),full_train_used_in_final_head=True,feature_training_uses_fit_subset=True))
  raw[ds][method]=m
 for seedrecord in newa[ds]['seeds']:
  rows.append(dict(dataset=ds,method='HASTEN H-unit',seed=seedrecord['seed'],**seedrecord['branches']['final']))
 rows.append(dict(dataset=ds,method='HASTEN H-unit',seed='three-seed mean',**{k:v['mean'] for k,v in newa[ds]['summary']['final'].items()}))
for name,data in [('RESULTS.json',rows),('PROTOCOL_AUDIT.json',audit),('OLD_AGGREGATES.json',raw)]:
 (out/name).write_text(json.dumps(data,indent=2,allow_nan=False))
print(json.dumps([{k:v for k,v in r.items() if k not in ['stages']} for r in rows],indent=2))
