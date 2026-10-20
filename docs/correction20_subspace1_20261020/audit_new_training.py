"""All training samples of the already-sealed new heads; no fitting or selection."""
import time
START=time.time()
import os,json
from pathlib import Path
import numpy as np
import torch
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);assert (out/'FIT_SEAL.json').exists();torch.set_num_threads(4);idx=np.load(Path(cfg['cache_run'])/'train_index.private.npz');y=idx['labels'].astype(int);fold=idx['folds'];cache=np.load(Path(cfg['cache_run'])/'train768.private.npy',mmap_mode='r');results=[]
for f in range(3):
 h=torch.load(out/f'outer_{f}.private.pt',weights_only=True);h={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in h.items()};ids=np.flatnonzero(fold!=f)
 for arm,q in [('score_subspace',h['score_basis']),('pca_subspace',h['PCA_basis'])]:
  W=h['joint_anchor']+q@h[arm];scores=np.empty((len(ids),20))
  for off in range(0,len(ids),2048):
   rows=ids[off:off+2048];x=np.column_stack([np.array(cache[rows],dtype=np.float64),np.ones(len(rows))]);scores[off:off+len(rows)]=(x*h['factors'])@W
  yy=y[ids];other=scores.copy();other[np.arange(len(ids)),yy]=-np.inf;margin=scores[np.arange(len(ids)),yy]-other.max(1);ok=scores.argmax(1)==yy;rows=[]
  for c in range(20):
   m=yy==c;rows.append(dict(class_id=cfg['order'][c],images=int(m.sum()),correct=int(ok[m].sum()),accuracy=float(ok[m].mean()),margin_quantiles=np.quantile(margin[m],[0,.25,.5,.75,1]).tolist()))
  results.append(dict(outer=f,arm=arm,classes=rows))
p=out/'SUBSPACE_TRAINING_AUDIT.json';p.write_text(json.dumps(dict(state='COMPLETE_READONLY',new_fits=0,new_encodes=0,margin_units='parent_joint_temperature',records=results,process_wall_seconds=time.time()-START),indent=2)+'\n');print(json.dumps(dict(state='COMPLETE_READONLY',process_wall_seconds=time.time()-START)))
