import time
START=time.time()
import os,sys,json
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());sys.path.insert(0,cfg['parent_code']);sys.path.insert(0,cfg['selector_code'])
from runtime import Data,write
import numpy as np
root=Path(cfg['output']).parent;probe=root/'probe.private.tmp'
with probe.open('xb') as f:f.write(b'probe-ok\n')
assert probe.read_bytes()==b'probe-ok\n';probe.unlink();data=Data(cfg);old=Path(cfg['selector_run']);assert json.loads((old/'VERIFICATION.json').read_text())['state']=='PASS'
for outer in range(3):
 s=np.load(old/f'signals_{outer}.private.npz');assert np.array_equal(s['rows'],np.flatnonzero(data.fold!=outer));assert s['features'].shape==(len(s['rows']),5);assert (old/f'singleton_{outer}.private.pt').exists()
result=dict(state='PASS',engineering_seconds=time.time()-START,new_encodes=0,old_cache_and_signals_ready=True,mount_probe_read_write=True);write(root/'preflight.private.json',result);print(json.dumps(result))
