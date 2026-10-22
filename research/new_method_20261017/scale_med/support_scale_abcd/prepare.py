"""Freeze only support metadata before any fitting or encoding."""
import json, os
from pathlib import Path
from design import make_plan, EXPECTED
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output'])
rows=[r for r in json.loads(Path(cfg['support_rows']).read_text()) if r['label']<13]
plan=make_plan(rows)
private=out/'PLAN.private.json';assert not private.exists(),'DO_NOT_RESEAL_EXISTING_PLAN'
private.write_text(json.dumps(plan))
def counts(ids):return [sum(rows[i]['label']==c for i in ids) for c in range(13)]
public=dict(support_rows=len(rows),support_groups=EXPECTED,split_seed=86420,singleton_training_only_class=1,
            folds=[dict(outer_fold=k,train_counts=counts(f['train']),test_counts=counts(f['test']),
                        inner=[dict(train_counts=counts(g['train']),validation_counts=counts(g['validation'])) for g in f['inner']]) for k,f in enumerate(plan['folds'])],
            group_disjoint=True,all_eligible_groups_outer_once=True,query_access=False,patient_independence='UNKNOWN')
(out/'SPLIT_SEAL.json').write_text(json.dumps(public,indent=2)+'\n')
print(json.dumps(public))
