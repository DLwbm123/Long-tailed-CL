import time,resource,os,json
from pathlib import Path
t=time.time();cpu=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20))
import torch
torch.set_num_threads(4)
r=Path(os.environ['AUDIT_ROOT']);out=r/'post_complete_error_transfer';clock=json.loads((r/'FIRST_ACTUAL.private.json').read_text());assert time.time()<clock['deadline_unix']-60
records=json.loads((r/'RECORDS.private.json').read_text());summary={a:[dict(class_index=c,opportunities=0,stable_correct=0,corrected=0,new_error=0,persistent_error=0,wrong_label_changed=0,any_prediction_changed=0) for c in range(23)] for a in ['balanced','natural']}
private={a:torch.zeros(23,23,dtype=torch.long) for a in summary}
for rec in records:
 s,b=rec['stage'],rec['fold'];z=torch.load(r/f'scores_s{s}f{b}.private.pt',map_location='cpu',weights_only=False);y=z['labels'];base=z['arms']['NCM']['prediction'];oldok=base==y
 for a in summary:
  pred=z['arms'][a]['prediction'];ok=pred==y
  for c in range(rec['seen']):
   mask=y==c;v=summary[a][c]
   items=dict(stable_correct=mask&oldok&ok,corrected=mask&~oldok&ok,new_error=mask&oldok&~ok,persistent_error=mask&~oldok&~ok,wrong_label_changed=mask&~oldok&~ok&(pred!=base),any_prediction_changed=mask&(pred!=base))
   v['opportunities']+=int(mask.sum())
   for k,x in items.items():v[k]+=int(x.sum())
   assert int(items['corrected'].sum()-items['new_error'].sum())==rec['query'][a]['full']['correct'][c]-rec['query']['NCM']['full']['correct'][c]
   wrong=mask&~ok;private[a][c]+=torch.bincount(pred[wrong],minlength=23)
for a,rows in summary.items():
 for v in rows:assert v['opportunities']==sum(v[k] for k in ['stable_correct','corrected','new_error','persistent_error'])
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<1_000_000_000
cost=dict(CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,new_model_scores=0,new_optimizer_steps=0,new_VAL_TEST_reads=0,original_clock_retained=True)
public=dict(classes=summary,cost=cost,counts_are_repeated_stage_fold_opportunities_not_unique_images=True,main_R84_necessary_gate=False,main_R84_gate_changed=False)
(out/'SUMMARY.json').write_text(json.dumps(public,indent=2)+'\n')
torch.save(private,out/'CONFUSIONS.private.pt')
print(json.dumps(dict(cost=cost,tiny_balanced=[summary['balanced'][c] for c in [1,2,12]])))
