import os,sys,json,time
from pathlib import Path
START=time.time();cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code'])
from runtime import Data,write
import numpy as np
root=Path(cfg['output']).parent;probe=root/'mount_probe.private.tmp';probe.write_bytes(b'probe-ok\n');assert probe.read_bytes()==b'probe-ok\n';probe.unlink();d=Data(cfg);coverage=[]
for h in range(3):
 ids=np.flatnonzero(d.fold==h);assert len(np.unique(d.y[ids]))==20;sub=d.subfold(ids);assert len(np.unique(sub))==2
 for half in range(2):
  train=ids[sub!=half];held=ids[sub==half];assert not np.intersect1d(d.g[train],d.g[held]).size;coverage.append(dict(fold=h,held_half=half,missing_classes=np.flatnonzero(np.bincount(d.y[train],minlength=20)==0).tolist()))
result=dict(state='PASS',images=len(d.y),classes=20,subteacher_coverage=coverage,probe_read_write=True,engineering_seconds=time.time()-START);write(root/'preflight.private.json',result);print(json.dumps(result))
