"""Read existing predictions only; publish class and anonymous patient coverage."""
import json,os,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(os.environ['RUN_CONFIG']).resolve().parents[1]/'case_hierarchical1_20261018'))
from data import read_rows
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
val=read_rows(cfg,'val');out=Path(cfg['output']).parent
records=[]
for stage in range(6):
 s=torch.load(Path(cfg['matched_run'])/f'stage_{stage}.private.pt',weights_only=True)
 seen=s['seen'];rows=[r for r in val if r['label'] in seen];y=s['labels']
 assert [r['label'] for r in rows]==y.tolist()
 cls=torch.tensor(seen)[s['scores']['cls768_ridge'].argmax(1)]
 patch=torch.tensor(seen)[s['scores']['patchmean768_ridge'].argmax(1)]
 a=cls==y;b=patch==y
 rescue=(~a)&b;harm=a&(~b)
 perclass=[]
 for c in seen:
  mask=y==c
  perclass.append(dict(class_id=c,count=int(mask.sum()),rescue=int((rescue&mask).sum()),
   harm=int((harm&mask).sum()),both_wrong=int(((~a)&(~b)&mask).sum())))
 groups={r['group'] for r in rows}
 rp={rows[i]['group'] for i in rescue.nonzero().flatten().tolist()}
 hp={rows[i]['group'] for i in harm.nonzero().flatten().tolist()}
 records.append(dict(stage=stage,images=len(y),patients=len(groups),CLS_correct=int(a.sum()),
  patch_correct=int(b.sum()),both_correct=int((a&b).sum()),both_wrong=int(((~a)&(~b)).sum()),
  rescue=int(rescue.sum()),harm=int(harm.sum()),changed=int((cls!=patch).sum()),
  rescue_classes=sum(d['rescue']>0 for d in perclass),harm_classes=sum(d['harm']>0 for d in perclass),
  rescue_patients=len(rp),harm_patients=len(hp),patients_with_both=len(rp&hp),per_class=perclass,
  oracle_union_BA=float((a|b).double().mean()),oracle_is_label_dependent_not_a_method=True))
(out/'COMPLEMENT.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records[-1]))
