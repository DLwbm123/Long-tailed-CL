"""Ceiling/headroom disambiguation for the three prespecified critical classes only."""
import json,time,resource
from pathlib import Path
r=Path(__file__).resolve().parent;t=time.time();cpu=time.process_time();assert t<1793373083.959357-60;resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20));rows=json.loads((r.parent/'RECORDS.private.json').read_text());out={}
for c in [1,2,12]:
 out[str(c)]={}
 for tag in ['full','Qa','Qb']:
  bs=[z['branches'][0]['query'][tag] for z in rows if z['branches'][0]['query'][tag]['counts'][c]>0];out[str(c)][tag]=dict(covered_blocks=len(bs),action0_ceiling_blocks=sum(z['correct'][c]==z['counts'][c] for z in bs),action0_headroom_blocks=sum(z['correct'][c]<z['counts'][c] for z in bs))
payload=dict(critical_classes=out,scope='post-descriptive ceiling check, no selection/gate changes',new_model_scores=0,new_optimizer_steps=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-t,peak_RSS_macOS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);(r/'HEADROOM_SUMMARY.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
