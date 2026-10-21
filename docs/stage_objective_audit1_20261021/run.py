"""Final-only moment diagnostic: no fitting or per-sample access."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from core import write,assembled,scalar_record,self_check,inputs,moments
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);root.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(4);self_check()
heads,fac,alpha=inputs(cfg);gsum=np.zeros((1537,1537));mu=[];records=[]
for j,c in enumerate(cfg['order']):
    assert time.time()<cfg['hard_deadline_epoch']-60
    a,b,cross=moments(cfg,c);g=assembled(a,b,cross);gsum+=g*fac[:,None]*fac[None,:];mu.append(g[:,-1]*fac)
    row={'class':c,'metrics':{}}
    for name,w in heads.items():
        v=fac[:,None]*w;row['metrics'][name]=scalar_record(v.T@g@v,g[:,-1]@v,j,1.)
    records.append(row);write(root/'status.private.json',dict(state='RUNNING',classes=j+1,updated=time.time()))
t=np.stack(mu,1);t-=t.mean(1,keepdims=True);grad=gsum@heads['ridge']-t+alpha*heads['ridge'];assert np.max(np.abs(grad))<1e-6
objectives={name:sum(row['metrics'][name]['half_squared_risk'] for row in records)+alpha/2*float((w*w).sum()) for name,w in heads.items()}
d=heads['offline_CE']-heads['ridge'];excess=.5*float((d*(gsum@d)).sum())+alpha/2*float((d*d).sum());err=abs(objectives['offline_CE']-objectives['ridge']-excess);assert err<1e-8 and excess>=0
support=json.loads((Path(cfg['prior'])/'support.json').read_text());order=sorted(cfg['order'],key=lambda c:(support[str(c)]['images'],c));bins=dict(tail7=order[:7],middle6=order[7:13],head7=order[13:],initial10=cfg['order'][:10],later10=cfg['order'][10:]);groups={}
for name in heads:
    groups[name]={group:{metric:float(np.mean([row['metrics'][name][metric] for row in records if row['class'] in cs])) for metric in records[0]['metrics'][name]} for group,cs in bins.items()}
write(root/'RESULTS.json',dict(scope='final20_training_moments_only',records=records,groups=groups,bins=bins,regularized_objectives=objectives,objective_excess=excess,excess_identity_error=err,ridge_gradient_max=float(np.max(np.abs(grad))),alpha=alpha))
write(root/'COSTS.json',dict(worker_seconds=time.time()-start,classes=20,sealed_heads_read=2,new_heads=0,optimizer_updates=0,new_image_encodes=0,training_sample_features_read=0,query_rows_scored=0,GPU_usage=False,CPU_threads=4))
write(root/'status.private.json',dict(state='COMPLETE_PENDING_VERIFICATION',updated=time.time()))
