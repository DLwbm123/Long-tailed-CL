import io,json,time,torch,os
from pathlib import Path
p=Path(os.environ['P22_SERIALIZATION_CHECKPOINT']);t=time.monotonic();v=torch.load(p,map_location='cpu',weights_only=False)
b=io.BytesIO();torch.save(v,b,pickle_protocol=4);n=b.tell();b.seek(0);w=torch.load(b,map_location='cpu',weights_only=False)
import numpy as np
def equal(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(type(a),a,b)
equal(v,w)
x=dict(status='PASS',old_bytes=p.stat().st_size,protocol4_bytes=n,all_fields_exact=True,CPU_seconds=time.monotonic()-t)
print(json.dumps(x))
