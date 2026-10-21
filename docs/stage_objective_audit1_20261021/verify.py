"""Independent block-moment audit; no full joint Gram assembly."""
import os,json,time
from pathlib import Path
import torch,numpy as np
from core import write,inputs,moments,block_record
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);torch.set_num_threads(4);start=time.time();heads,fac,alpha=inputs(cfg);res=json.loads((root/'RESULTS.json').read_text());error=0.
for j,c in enumerate(cfg['order']):
    assert time.time()<cfg['hard_deadline_epoch']-60
    a,b,cross=moments(cfg,c);assert res['records'][j]['class']==c
    for name,w in heads.items():
        rec=block_record(a,b,cross,fac[:,None]*w,j)
        error=max(error,max(abs(value-res['records'][j]['metrics'][name][key]) for key,value in rec.items()))
assert error<1e-8
for name in heads:
    for group,cs in res['bins'].items():
        for metric,value in res['groups'][name][group].items():assert abs(value-np.mean([row['metrics'][name][metric] for row in res['records'] if row['class'] in cs]))<1e-12
write(root/'VERIFICATION.json',dict(state='PASS',independent_block_max_error=error,classes=20,heads_read=2,new_image_encodes=0,training_sample_features_read=0,query_rows_scored=0,wall_seconds=time.time()-start))
write(root/'status.private.json',dict(state='COMPLETE_VERIFIED',updated=time.time()))
