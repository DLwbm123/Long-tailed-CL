"""One exploratory macro-BA selector on saved endpoints, no model/optimizer."""
import json,time,resource,os,math
from pathlib import Path
r=Path(__file__).resolve().parent;t=time.time();cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(600,600));resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20));assert t<1793359827.550853-60
rows=json.loads((r.parent/'RECORDS.private.json').read_text());out=[]
for v in rows:
 bs=[z for z in v['branches'] if not z['duplicate']];item={'stage':v['stage'],'fold':v['fold'],'repeat':v['repeat']}
 for source,target,tag in [('Qa','Qb','forward'),('Qb','Qa','reverse')]:
  scores=[z['stats'][source]['BA'] for z in bs];best=[i for i,x in enumerate(scores) if max(scores)-x<=1e-10];prot=[z['stats'][source]['utility'] for z in bs];pb=[i for i,x in enumerate(prot) if max(prot)-x<=1e-7];result={'macro_BA_best_actions':[bs[i]['action'] for i in best]}
  for m in ['BA','tail_BA']:
   c=sum(bs[i]['stats'][target][m] for i in best)/len(best);p=sum(bs[i]['stats'][target][m] for i in pb)/len(pb);u=sum(z['stats'][target][m] for z in bs)/len(bs);result[m]={'candidate_minus_uniform':c-u,'candidate_minus_action0':c-bs[0]['stats'][target][m],'candidate_minus_protected_Brier':c-p}
  item[tag]=result
 out.append(item)
stages=[]
for s in range(6):
 vv=[z for z in out if z['stage']==s];stages.append({'stage':s,**{tag:{m:{k:sum(z[tag][m][k] for z in vv)/6 for k in ['candidate_minus_uniform','candidate_minus_action0','candidate_minus_protected_Brier']} for m in ['BA','tail_BA']} for tag in ['forward','reverse']}})
summary={tag:{m:{k:sum(z[tag][m][k] for z in out)/36 for k in ['candidate_minus_uniform','candidate_minus_action0','candidate_minus_protected_Brier']} for m in ['BA','tail_BA']} for tag in ['forward','reverse']};summary.update(stage_results=stages,exploratory_not_primary=True,new_model_scores=0,new_optimizer_steps=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-t,peak_RSS_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert summary['peak_RSS_platform_units']<1_000_000_000
(r/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');(r/'BLOCK_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary))
