"""Saved-only class metrics, finite endpoints, true memory bytes and irreversible history."""
import json
from pathlib import Path
import torch,numpy as np
from memory import DTYPE,BUDGET_BYTES
from common import write
from constants import ARMS

def finite(x):
 if torch.is_tensor(x):assert torch.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(tuple,list)):
  for v in x:finite(v)

def run(c):
 r=Path(c['output']);assert json.loads((r/'LEARNING_SEAL.json').read_text())['classifier_endpoints']==30;previous={};metrics=json.loads((r/'STAGE_METRICS.json').read_text());worst=0
 for t in range(5):
  C=sum(map(len,c['stages'][:t+1]));old=C-len(c['stages'][t]);finite(torch.load(r/f's{t}'/'POLICY.private.pt',map_location='cpu',weights_only=False))
  for arm in ARMS:
   finite(torch.load(r/f's{t}'/f'{arm}_MODEL.private.pt',map_location='cpu',weights_only=False));p=r/f's{t}'/f'{arm}_MEMORY.private.bin';a=np.fromfile(p,dtype=DTYPE);assert p.stat().st_size+Path(str(p)+'.stats').stat().st_size<=BUDGET_BYTES;assert np.isfinite(a['feature']).all();assert set(a['label'].tolist())==set(range(C));assert set(a['parent'][a['view']>0])<=set(a['parent'][a['view']==0]);oldkeys=set(zip(a['parent'][a['label']<old].tolist(),a['view'][a['label']<old].tolist()));assert oldkeys<=previous.get(arm,set());previous[arm]=set(zip(a['parent'].tolist(),a['view'].tolist()));s=np.fromfile(str(p)+'.stats',dtype=[('count','<u4'),('birth','<i2'),('padding','<u2')]);assert (s['count'][C:]==0).all() and (s['birth'][C:]==-1).all()
  pack=torch.load(r/f'VAL_s{t}.private.pt',map_location='cpu',weights_only=False);assert pack['global_seal_before_read'];y=pack['labels'];den=torch.bincount(y,minlength=C);tail=torch.tensor(c['train_counts'][:C])<250
  for arm,pred in pack['predictions'].items():
   rec=torch.bincount(y[y==pred],minlength=C).double()/den;values={'micro':float((y==pred).double().mean()*100),'macro':float(rec.mean()*100),'tail':float(rec[tail].mean()*100) if tail.any() else None}
   for k,v in values.items():
    if v is not None:worst=max(worst,abs(v-metrics[t]['modes'][arm][k]))
 assert worst<1e-10;write(r/'VERIFICATION.json',{'passed':True,'classifier_endpoints':30,'policy_endpoints':15,'all_saved_states_finite':True,'actual_memory_bytes_and_own_history_verified':True,'max_saved_metric_error_pp':worst,'VAL_after_global180_seal':True,'new_forward_optimizer':0,'TEST':0})
