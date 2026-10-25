"""One low-cost arrival check, no hashes or repeat full scans."""
import json,sys
from pathlib import Path
import numpy as np
from common import write
cfg=json.loads(Path('/tmp/q1095/config.json').read_text());j=json.load(sys.stdin)['id'];assert isinstance(j,int);r=Path(cfg['output'])/'cache'/str(j);m=json.loads((r/'META.json').read_text());assert m['dtype']=='float32' and m['token_shape']==[197,768] and json.loads((r/'SOURCE_READY.json').read_text())['rows']==len(m['ids'])
for c in m['chunks']:
 p=r/c['file'];assert p.stat().st_size==c['rows']*197*768*4;v=np.memmap(p,dtype=np.float32,mode='r',shape=(c['rows'],197,768));assert np.isfinite(v[0,:2,:8]).all();v._mmap.close();b=np.load(r/c['base'],mmap_mode='r');assert b.shape==(c['rows'],512);del b
write(r/'READY.json',dict(id=j,rows=len(m['ids']),size_shape_probe_passed=True,source_input_parity=True))
