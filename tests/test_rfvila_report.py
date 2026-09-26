"""Full logical-matrix finalization on synthetic labels, never medical evidence."""
import tempfile,json
from pathlib import Path
import numpy as np
from tools.finalize_rfvila import finalize
from tools.run_nb2_vlm_r1 import _write,SIZES
from route_a.run_ct13_real import _stage_metrics,_write_csv

def main():
 with tempfile.TemporaryDirectory() as td:
  out=Path(td);(out/'private').mkdir();(out/'backup_acks').mkdir();p=json.loads((Path(__file__).resolve().parents[1]/'docs/rfvila_plan/03_PROTOCOL_PROPOSAL.json').read_text());p['admitted_tiers']=p['tier_priority'];_write(out/'PROTOCOL_LOCK.json',p)
  sr=[];cr=[];events=[]
  for ds,sizes in SIZES.items():
   labels=np.arange(sum(sizes));pred=[];index=[]
   for seed in [1993,1994,1995]:
    pos=0
    for t,n in enumerate(sizes,1):
     current=list(range(pos,pos+n));pos+=n;seen=list(range(pos));sp=out/'stages'/ds/str(seed)/f'task_{t:02d}';sp.mkdir(parents=True)
     _write(sp/'STATE_LOCK.json',{'dataset':ds,'seed':seed,'task':t,'residuals':{'x':1e-14},'lambdas':{'F.LIN':.001},'cv':{'F.LIN':{'scores':[],'status':'TASK1_GROUP_CV_UNAVAILABLE'}}})
     events.append({'kind':'stage_sealed','peak_gpu_bytes':1,'peak_rss_kib':1})
     for m in p['methods']:
      score=np.eye(sum(sizes))[:,:pos];metric,rows,pr=_stage_metrics(score,labels,seen,current,set(labels[-2:]));metric={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metric.items()}
      common={'dataset':ds,'seed':seed,'task':t,'method':m['method']};sr.append({**common,**metric});cr.extend({**common,**r} for r in rows);pred.append(pr);index.append(common)
   np.savez(out/'private'/f'{ds}_PREDICTIONS.npz',labels=labels,components=np.array([str(x) for x in labels]),predictions=np.stack(pred),index=np.array([json.dumps(v) for v in index]))
  _write_csv(out/'stage_metrics.csv',sr);_write_csv(out/'class_metrics.csv',cr);(out/'ACCESS_EVENTS.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
  _write(out/'CLOCK_LOCK.json',{'t0_epoch':0,'t0_utc':'synthetic'});_write(out/'QUALIFICATION.json',{'gpu_name':'synthetic'})
  r=finalize(out,{})
  assert r['stage_rows']==1440 and r['class_rows']==14688
  assert all(g['status']=='FAIL' for g in r['gates'])
  assert (out/'FINAL_REPORT_ZH.md').is_file()
  print('SYNTHETIC_FULL_MATRIX_REPORT_PASS')
if __name__=='__main__':main()
