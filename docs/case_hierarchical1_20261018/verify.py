"""Recount all completed case-model metrics without importing its implementation."""
import json
import os
from pathlib import Path
import time
import torch

start=time.time()
root=Path(os.environ['RUN_OUTPUT'])
cfg=json.loads((root/'config.private.json').read_text())
assert json.loads((root/'status.private.json').read_text())['state']=='COMPLETE'
results=json.loads((root/'results.json').read_text())
assert len(results)==6
count=0
for stage,record in enumerate(results):
    saved=torch.load(root/f'stage_{stage}.private.pt',weights_only=True)
    seen=saved['seen'];labels=saved['labels'].tolist()
    assert seen==cfg['order'][:sum(cfg['task_sizes'][:stage+1])]
    predictions={}
    for arm,score in saved['scores'].items():
        assert torch.isfinite(score).all()
        prediction=[seen[i] for i in score.argmax(1).tolist()];predictions[arm]=prediction
        counts=[labels.count(c) for c in seen]
        correct=[sum(y==c and p==c for y,p in zip(labels,prediction)) for c in seen]
        confusion=[[sum(y==c and p==d for y,p in zip(labels,prediction)) for d in seen] for c in seen]
        target=record['metrics'][arm]
        assert counts==target['counts'] and correct==target['correct'] and confusion==target['confusion']
        assert abs(sum(a/b for a,b in zip(correct,counts))/len(seen)-target['macro'])<1e-12
        assert abs(sum(correct)/len(labels)-target['micro'])<1e-12
        count+=len(prediction)
    for comparator,expected in record['paired'].items():
        a,b=predictions['hierarchical'],predictions[comparator]
        assert expected['rescue']==sum(p==y and q!=y for p,q,y in zip(a,b,labels))
        assert expected['harm']==sum(p!=y and q==y for p,q,y in zip(a,b,labels))
        assert expected['changed']==sum(p!=q for p,q in zip(a,b))
    historical=torch.load(Path(cfg['prior_run'])/f'stage_{stage}.private.pt',weights_only=True)
    assert saved['scores']['patient_uniform'].argmax(1).tolist()==historical['scores']['uniform'].argmax(1).tolist()
costs=json.loads((root/'costs.json').read_text())
assert costs['training_images_extracted']==cfg['train_count']
assert costs['gpu_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
result=dict(status='PASS',stages=6,predictions_recounted=count,confusion_and_BA_verified=True,
            rescue_harm_verified=True,historical_patient_uniform_predictions_reproduced=True,
            wall_seconds=time.time()-start)
(root/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
