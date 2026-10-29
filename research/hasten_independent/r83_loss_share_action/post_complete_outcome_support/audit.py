"""Cardinalities of saved aggregate outcomes, not a new selector or model score."""
import os,json,time,resource
from fractions import Fraction as F
from pathlib import Path
r=Path(__file__).resolve().parent;t=time.time();cpu=time.process_time();assert t<1793373083.959357-60;resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20));rows=json.loads((r.parent/'RECORDS.private.json').read_text());out=[];class_rows=[]
for v in rows:
 bs=[z for z in v['branches'] if not z['duplicate'] and not z['fixed_mean']];base=bs[0]
 for tag in ['full','Qa','Qb']:
  cc=[c for c,n in enumerate(base['query'][tag]['counts']) if n];tt=[c for c in cc if v['tail'][c]];ba=[];tail=[];vary=[]
  for b in bs:
   m=b['query'][tag];ba.append(sum(F(m['correct'][c],m['counts'][c]) for c in cc)/len(cc));tail.append(sum(F(m['correct'][c],m['counts'][c]) for c in tt)/len(tt))
  for c in cc:
   outcomes={b['query'][tag]['correct'][c] for b in bs};own=next(b for b in bs if b['action']==c+1);d=own['query'][tag]['correct'][c]-base['query'][tag]['correct'][c];class_rows.append(dict(stage=v['stage'],fold=v['fold'],repeat=v['repeat'],query=tag,class_index=c,distinct_correct_counts=len(outcomes),own_correct_change_sign=0 if d==0 else (1 if d>0 else -1),actions_with_correct_count_change=sum(b['query'][tag]['correct'][c]!=base['query'][tag]['correct'][c] for b in bs)))
   if len(outcomes)>1:vary.append(c)
  out.append(dict(stage=v['stage'],fold=v['fold'],repeat=v['repeat'],query=tag,unique_actions=len(bs),distinct_macro_BA=len(set(ba)),distinct_tail_BA=len(set(tail)),classes_with_varying_correct_counts=vary,covered_classes=len(cc),covered_tail_classes=len(tt)))
summary={'queries':{tag:dict(blocks=36,macro_BA_constant_blocks=sum(x['distinct_macro_BA']==1 for x in out if x['query']==tag),tail_BA_constant_blocks=sum(x['distinct_tail_BA']==1 for x in out if x['query']==tag),mean_distinct_macro_BA=sum(x['distinct_macro_BA'] for x in out if x['query']==tag)/36,mean_distinct_tail_BA=sum(x['distinct_tail_BA'] for x in out if x['query']==tag)/36) for tag in ['full','Qa','Qb']},'critical_classes':{},'all_classes':[]}
for c in range(23):
 item={'class_index':c}
 for tag in ['full','Qa','Qb']:
  vv=[x for x in class_rows if x['class_index']==c and x['query']==tag];item[tag]=dict(covered_blocks=len(vv),constant_correct_count_blocks=sum(x['distinct_correct_counts']==1 for x in vv),own_priority_improvement_blocks=sum(x['own_correct_change_sign']>0 for x in vv),own_priority_harm_blocks=sum(x['own_correct_change_sign']<0 for x in vv),own_priority_same_blocks=sum(x['own_correct_change_sign']==0 for x in vv))
 summary['all_classes'].append(item)
 if c in [1,2,12]:summary['critical_classes'][str(c)]=item
summary.update(new_model_scores=0,new_optimizer_steps=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-t,peak_RSS_macOS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert summary['peak_RSS_macOS_bytes']<1_000_000_000
(r/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(r/'BLOCK_CARDINALITIES.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='all_classes'}))
