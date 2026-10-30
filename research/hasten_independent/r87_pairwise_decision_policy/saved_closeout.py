"""Describe saved expected/modal decisions and all declared slices; no forward/update."""
import os,json,time,resource,torch
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);start=time.time();cpu=time.process_time();torch.set_num_threads(4);rows=json.loads((r/'RECORDS.private.json').read_text());arms=['G','CE','E','H','U','NCM','balanced'];readouts=[];groups=[];classes=[];behavior=[]
def mean(x):return sum(x)/len(x) if x else None
for v in rows:
 meta={k:v[k] for k in ['stage','fold','direction','stream','query','heldout_tag']};readouts.append(dict(**meta,arms={a:{k:b for k,b in v['arms'][a].items() if k not in ['counts','expected_correct','modal_correct','modal_actions']} for a in arms}))
 old=sum(map(len,cfg['stages'][:v['stage']]))
 for group in ['old_head','old_tail','new_head','new_tail']:
  cc=[c for c,n in enumerate(v['arms']['NCM']['counts']) if n>0 and (c<old)==group.startswith('old') and (cfg['train_counts'][c]<250)==group.endswith('tail')];out={}
  for a in arms:
   aa=v['arms'][a];out[a]={typ:None if not cc else 100*mean([aa[key][c]/aa['counts'][c] for c in cc]) for typ,key in [('expected_macro','expected_correct'),('modal_macro','modal_correct')]}
  groups.append(dict(**meta,group=group,covered_classes=len(cc),arms=out))
 tag=v['heldout_tag'] if v['query']=='heldout' else 'full';ctx=torch.load(r/f"eval_s{v['stage']}f{v['fold']}d{v['direction']}_{tag}.private.pt",map_location='cpu',weights_only=False);pp=torch.load(r/f"distributions_s{v['stage']}f{v['fold']}d{v['direction']}r{v['stream']}_{tag}.private.pt",map_location='cpu',weights_only=False)
 for a in ['G','CE','E','H','U']:
  if a=='U':m=len(ctx['actions']);p1=torch.full((m,),1/m,dtype=torch.double);p2=ctx['mask'].double()/ctx['mask'].sum(1,keepdim=True)
  else:p1,p2=pp[a]['p1'],pp[a]['p2']
  j=p1[:,None]*p2;second_nonzero=float(j[:,1:].sum());mode=v['arms'][a]['modal_actions'];behavior.append(dict(**meta,arm=a,first_stop_probability=float(p1[0]),expected_number_of_swaps=1-float(p1[0])+second_nonzero,modal_is_stop=mode[0]==[-1,-1],expected_utility=v['arms'][a]['expected_reward'],base_utility=v['arms']['NCM']['expected_macro']/100))
for query in ['heldout','full']:
 for stage in range(6):
  rr=[v for v in rows if v['query']==query and v['stage']==stage]
  for c in range(sum(map(len,cfg['stages'][:stage+1]))):
   cc=[v for v in rr if v['arms']['NCM']['counts'][c]>0];result={}
   for a in arms:result[a]={typ:100*mean([v['arms'][a][key][c]/v['arms'][a]['counts'][c] for v in cc]) for typ,key in [('expected_recall','expected_correct'),('modal_recall','modal_correct')]}
   classes.append(dict(query=query,stage=stage,class_index=c,covered_cells=len(cc),class1_fold1_unsupported=(query=='heldout' and c==1),arms=result))
summary=[]
for query in ['heldout','full']:
 for a in ['G','CE','E','H','U']:
  rr=[v for v in behavior if v['query']==query and v['arm']==a];summary.append(dict(query=query,arm=a,cells=len(rr),modal_stop_cells=sum(v['modal_is_stop'] for v in rr),mean_first_stop_probability=mean([v['first_stop_probability'] for v in rr]),mean_expected_swaps=mean([v['expected_number_of_swaps'] for v in rr]),mean_expected_utility_gain_vs_NCM=mean([v['expected_utility']-v['base_utility'] for v in rr]),utility_gain_positive_cells=sum(v['expected_utility']>v['base_utility']+1e-12 for v in rr)))
for name,data in [('CELL_METRICS',readouts),('OLD_NEW_HEAD_TAIL',groups),('CLASS_STAGE_METRICS',classes),('POLICY_BEHAVIOR_SUMMARY',summary)]: (r/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
(r/'POLICY_BEHAVIOR.private.json').write_text(json.dumps(behavior,indent=2))
receipt=dict(saved_only=True,records=len(rows),new_policy_encoder_classifier_forward_rows=0,new_optimizer_steps=0,new_VAL_TEST_reads=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(r/'SAVED_CLOSEOUT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(receipt=receipt,policy_behavior=summary)))
