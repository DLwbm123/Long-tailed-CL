"""TRAIN metadata and sealed-schema only; zero model/image scoring or optimization."""
import os,json,time,csv,resource,fcntl
from pathlib import Path
import torch
from folds import assign_folds
from common import write
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);dc=cfg['domains']['HK'];src=Path(cfg['source_run'])/'HK/uniform_controller/74002';torch.set_num_threads(4)
rows=list(csv.DictReader(open(Path(dc['manifest'])/'train.csv')));assert all(v['split']=='train' for v in rows);labels=[dc['order'].index(int(v['original_label'])) for v in rows];fold=assign_folds(labels,[v['identity_component'] for v in rows]);blocks=[];schemas=[]
for stage in range(6):
 n=sum(map(len,dc['stages'][:stage+1]));h=torch.load(src/f'head{stage}.private.pt',map_location='cpu',weights_only=False);assert h['seen']==n and h['optimizer_state']['state'];assert all(int(v['step'])==24*(stage+1) for v in h['optimizer_state']['state'].values());assert len(h['delta']) and all(torch.isfinite(v).all() for v in h['delta'].values());assert len(h['optimizer_state']['param_groups'])==1
 schemas.append(dict(stage=stage,seen=n,optimizer_steps=24*(stage+1),saved_lr=h['optimizer_state']['param_groups'][0]['lr'],delta_parameters=sum(v.numel() for v in h['delta'].values()),state_restorable_schema=True))
 for b in [0,1]:
  groups={c:{} for c in range(n)}
  for i,(c,f) in enumerate(zip(labels,fold)):
   if c<n and f==b:groups[c].setdefault(rows[i]['identity_component'],[]).append(i)
  unsupported=[c for c,g in groups.items() if len(g)<2];qa=[];qb=[];cover=[]
  for c,g in groups.items():
   if c in unsupported:continue
   sizes=[0,0]
   for component,ii in sorted(g.items(),key=lambda z:(-len(z[1]),z[0])):
    side=0 if sizes[0]<=sizes[1] else 1;(qa if side==0 else qb).extend(ii);sizes[side]+=len(ii)
   assert min(sizes)>0;cover.append(dict(class_index=c,Q_images=sum(map(len,g.values())),Q_components=len(g),Qa_images=sizes[0],Qb_images=sizes[1],tail=dc['train_counts_ordered'][c]<250))
  assert not set(qa)&set(qb);assert not {rows[i]['identity_component'] for i in qa}&{rows[i]['identity_component'] for i in qb};assert cover and any(v['tail'] for v in cover)
  blocks.append(dict(stage=stage,fold=b,coverage=cover,unsupported_classes=unsupported,unsupported_tail=[c for c in unsupported if dc['train_counts_ordered'][c]<250],Qa_ids=qa,Qb_ids=qb))
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4_000_000_000
private=dict(schemas=schemas,blocks=blocks,new_model_scores=0,new_visual_optimizer_steps=0,VAL_reads=0);write(r/'FEASIBILITY.private.json',private)
safe=dict(passed_partial_coverage=True,schemas=schemas,blocks=[{k:v for k,v in x.items() if k not in ['Qa_ids','Qb_ids']} for x in blocks],all_tail_dual_query_identifiable=all(not x['unsupported_tail'] for x in blocks),new_model_scores=0,new_visual_optimizer_steps=0,VAL_reads=0)
write(r/'FEASIBILITY.json',safe);print(json.dumps(safe))
