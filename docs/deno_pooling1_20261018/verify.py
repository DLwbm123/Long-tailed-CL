"""Recount metrics from saved predictions without importing the experiment code."""
import json
import os
from pathlib import Path
import time
import torch


root=Path(os.environ['RUN_OUTPUT'])
start=time.time()
assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
cfg=json.loads((root/'config.private.json').read_text())
records=json.loads((root/'results.json').read_text())
assert len(records)==6
checked=0
clipping_changes=[]
for stage,record in enumerate(records):
    saved=torch.load(root/f'stage_{stage}.private.pt',weights_only=True)
    seen=saved['seen'];labels=saved['labels'].tolist()
    assert seen==cfg['order'][:sum(cfg['task_sizes'][:stage+1])]
    predictions={}
    for arm,score in saved['scores'].items():
        assert torch.isfinite(score).all()
        prediction=[seen[i] for i in score.argmax(1).tolist()]
        predictions[arm]=prediction
        counts=[labels.count(c) for c in seen]
        correct=[sum(y==c and p==c for y,p in zip(labels,prediction)) for c in seen]
        confusion=[[sum(y==c and p==d for y,p in zip(labels,prediction)) for d in seen] for c in seen]
        metric=record['metrics'][arm]
        assert counts==metric['counts'] and correct==metric['correct'] and confusion==metric['confusion']
        macro=sum(c/n for c,n in zip(correct,counts))/len(seen)
        assert abs(macro-metric['macro'])<1e-12
        assert abs(sum(correct)/len(labels)-metric['micro'])<1e-12
        checked+=len(prediction)
    for comparator,v in record['paired'].items():
        a,b=predictions['deno_mixture'],predictions[comparator]
        assert v['rescue']==sum(p==y and q!=y for p,q,y in zip(a,b,labels))
        assert v['harm']==sum(p!=y and q==y for p,q,y in zip(a,b,labels))
        assert v['changed']==sum(p!=q for p,q in zip(a,b))
    assert predictions['uniform_raw']==predictions['reference_mean_raw']
    assert predictions['reference_mean_clipped']==predictions['identity_mixture']
    clipping_changes.append(sum(a!=b for a,b in zip(predictions['reference_mean_raw'],predictions['reference_mean_clipped'])))
numerical=json.loads((root/'NUMERICAL_CHECKS.json').read_text())
assert len(numerical['uniform_replication'])==20
assert max(v['uniform_replication_error'] for v in numerical['uniform_replication'])<1e-7
assert numerical['prefix_recompute_error']<1e-10
assert numerical['clip_total']==sum(numerical['per_class_clip'])
if 'clipping_prediction_changes_per_stage' in numerical:
    assert clipping_changes==numerical['clipping_prediction_changes_per_stage']
costs=json.loads((root/'costs.json').read_text())
assert costs['new_training_images']<=80 and costs['optimizer_updates']==0
assert costs['gpu_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
result=dict(status='PASS',stage_count=6,predictions_recounted=checked,
            confusion_counts_and_BA_verified=True,rescue_harm_verified=True,
            mathematical_controls_verified=True,wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
