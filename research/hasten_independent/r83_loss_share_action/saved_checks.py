"""Supplemental saved-only checks for slice identities, utility and prototype accounting."""
import os,json,math,time,resource,fcntl
from pathlib import Path
import torch
from common import write,guard
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-4:]));cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(cfg['output']);lock=Path(cfg['shared_memory_lock']).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);t=time.time();start_cpu=time.process_time();records=json.loads((r/'HK/RECORDS.private.json').read_text());feas=json.loads((r/'FEASIBILITY.private.json').read_text());maxproto=0.;maxslice=0.;maxutility=0.
for v in records:
 n=v['seen'];assert v['tail']==[c<250 for c in cfg['domains']['HK']['train_counts_ordered'][:n]];assert v['protected']==[c<v['old_seen'] or v['tail'][c] for c in range(n)];f=next(z for z in feas['blocks'] if z['stage']==v['stage'] and z['fold']==v['fold']);cover={z['class_index']:z for z in f['coverage']}
 for z in v['branches']:
  guard(cfg)
  for tag,m in z['query'].items():
   assert all(math.isfinite(x) and x>=0 for x in m['brier']);cc=[c for c,num in enumerate(m['counts']) if num];tt=[c for c in cc if v['tail'][c]];pp=[c for c in cc if v['protected'][c]];damage=sum(max(0,m['brier'][c]-v['anchors'][tag][c]) for c in pp)/len(pp);u=-sum(m['brier'][c] for c in cc)/len(cc)-damage;err=max(abs(u-z['stats'][tag]['utility']),abs(damage-z['stats'][tag]['damage']));maxutility=max(maxutility,err);assert err<1e-12;assert abs(100*sum(m['correct'][c]/m['counts'][c] for c in tt)/len(tt)-z['stats'][tag]['tail_BA'])<1e-10
   if tag!='full':assert cc==sorted(cover)
  for c,x in cover.items():
   a=z['query']['Qa'];b=z['query']['Qb'];full=z['query']['full'];na,nb=a['counts'][c],b['counts'][c];assert (na,nb)==(x['Qa_images'],x['Qb_images']);assert na+nb==full['counts'][c] and a['correct'][c]+b['correct'][c]==full['correct'][c];err=abs((a['brier'][c]*na+b['brier'][c]*nb)/(na+nb)-full['brier'][c]);maxslice=max(maxslice,err);assert err<1e-12
  h=torch.load(r/'HK/endpoints'/z['endpoint'],map_location='cpu',weights_only=False);features=h['features'];labels=h['labels'];counts=torch.bincount(labels,minlength=n);assert (counts>0).all() and torch.allclose(features.norm(dim=1),torch.ones(len(features)),atol=2e-6,rtol=0);sums=torch.zeros(n,512,dtype=torch.float64);sums.index_add_(0,labels,features.double());proto=torch.nn.functional.normalize((sums/counts[:,None]).float(),dim=1);err=float((proto-h['prototypes']).abs().max());maxproto=max(maxproto,err);assert err<=2e-5
summary=dict(passed=True,all_saved_slice_class_coverage=True,all_full_Q_preserved=True,same_endpoint_slice_risk_count_identity=True,protected_utility_recomputed=True,prototype_max_abs=maxproto,slice_identity_max_abs=maxslice,utility_max_abs=maxutility,new_model_scores=0,new_optimizer_steps=0,CPU_seconds=time.process_time()-start_cpu,wall_seconds=time.time()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(r/'results/SUPPLEMENTAL_VERIFICATION.json',summary);print(json.dumps(summary))
