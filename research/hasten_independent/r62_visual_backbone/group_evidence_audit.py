"""TRAIN metadata support feasibility only; no optimization, model, VAL or TEST."""
import os,json,csv
from pathlib import Path
from collections import defaultdict
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out={}
for ds,d in cfg['domains'].items():
 with open(Path(d['manifest'])/'train.csv') as f:rows=list(csv.DictReader(f))
 classes=[]
 for i,label in enumerate(d['order']):
  rr=[r for r in rows if int(r['original_label'])==label];groups=defaultdict(int)
  for r in rr:
   key=r['identity_component'];assert key;groups[key]+=1
  assert len(rr)==d['train_counts_ordered'][i]
  classes.append(dict(class_index=i,TRAIN_rows=len(rr),metadata_identity_components=len(groups),largest_component_rows=max(groups.values()),tail_under250=len(rr)<250,group_disjoint_support_query_possible=len(groups)>=2))
 out[ds]=dict(classes=classes,all_classes_have_two_metadata_groups=all(c['group_disjoint_support_query_possible'] for c in classes),group_semantics='Existing identity_component; patient/lesion/content semantics require provenance review, never assumed universally patient IDs.',no_optimization=True,no_model_scoring=True,no_VAL_TEST=True)
r=Path(cfg['output'])/'results';(r/'TRAIN_GROUP_EVIDENCE.json').write_text(json.dumps(out,indent=2));print(json.dumps({ds:{'all_two_groups':v['all_classes_have_two_metadata_groups'],'tail_rows_groups':[(c['TRAIN_rows'],c['metadata_identity_components']) for c in v['classes'] if c['tail_under250']]} for ds,v in out.items()}))
