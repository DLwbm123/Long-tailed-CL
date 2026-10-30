import os,time,json,resource,itertools
from pathlib import Path
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import torch,numpy as np
from scipy.stats import kendalltau
r=Path(os.environ['R87_SAVED_ROOT']);d=r/'post_complete_effect_transfer';d.mkdir(exist_ok=True)
assert not (d/'RESULTS.json').exists(), 'already computed'
start=time.time();cpu=time.process_time();deadline=1793448588.90611
assert start<deadline-120
resource.setrlimit(resource.RLIMIT_CPU,(120,120));torch.set_num_threads(4)
rows=[]
def guard():
 assert time.time()-start<120 and time.time()<deadline
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<1000000000
for s in range(6):
 for f in range(2):
  guard();v=[torch.load(r/f'context_s{s}f{f}d{q}.private.pt',map_location='cpu',weights_only=False) for q in (0,1)]
  maps=[{tuple(a):i for i,a in enumerate(c['actions'])} for c in v];common=sorted(set(maps[0])&set(maps[1]));stop=(-1,-1);aa=[a for a in common if a!=stop]
  endpoints=[(stop,stop)]+[(a,stop) for a in aa]+[(a,b) for a,b in itertools.combinations(aa,2) if set(a)!=set(b)]
  effect={}
  for metric in ['utility','macro']:
   z=[]
   for q,c in enumerate(v):
    o=c['outcome'];base=float(o['reward'][0,0]) if metric=='utility' else float((o['base_correct'][o['eligible']]/o['count'][o['eligible']]).mean())
    vals=[]
    for a,b in endpoints:
     i,j=maps[q][a],maps[q][b];assert c['mask'][i,j]
     val=float(o['reward'][i,j]) if metric=='utility' else float(((o['base_correct']+o['delta_correct'][i]+o['delta_correct'][j])[o['eligible']]/o['count'][o['eligible']]).mean())
     vals.append(val-base)
    z.append(np.array(vals))
   sign=[np.where(x>1e-10,1,np.where(x < -1e-10,-1,0)) for x in z];tau=kendalltau(np.round(z[0],10),np.round(z[1],10),variant='b').statistic
   effect[metric]={'tau_b':None if not np.isfinite(tau) else float(tau),'Qa_sign_counts':{str(k):int((sign[0]==k).sum()) for k in [-1,0,1]},'Qb_sign_counts':{str(k):int((sign[1]==k).sum()) for k in [-1,0,1]},'Qa_positive_to_Qb':{str(k):int(((sign[0]==1)&(sign[1]==k)).sum()) for k in [-1,0,1]},'Qb_positive_to_Qa':{str(k):int(((sign[1]==1)&(sign[0]==k)).sum()) for k in [-1,0,1]}}
  rows.append(dict(stage=s,fold=f,common_nonstop_actions=len(aa),unique_endpoints=len(endpoints),effects=effect));guard()
receipt=dict(CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_forward_optimizer_VAL_TEST=0,original_deadline=deadline)
result=dict(strata=rows,cost=receipt,interpretation='Descriptive same-endpoint effect transfer; not efficacy or an action selector. All endpoints reused and correlated.')
p=d/'RESULTS.json';p.write_text(json.dumps(result,indent=2,allow_nan=False));assert p.stat().st_size<5000000
print(json.dumps(result))
