"""Reproduce hidden backing storage and validate the bounded save contract; zero image calls."""
import json,os,time,resource
from pathlib import Path
import torch
from run import save
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());p=Path(cfg['output']).parent;start=time.time()
x=torch.zeros(160,1536);view=x[:8];target=p/'storage_check.private.pt'
assert view.untyped_storage().nbytes()>view.numel()*view.element_size()
try:save(target,dict(features=view))
except AssertionError:pass
else:raise AssertionError('Unbounded view was not rejected')
assert not target.exists();save(target,dict(features=view.clone()));loaded=torch.load(target,weights_only=True)['features']
assert loaded.shape==(8,1536) and loaded.untyped_storage().nbytes()==loaded.numel()*loaded.element_size()
r=dict(state='PASS',reproduced_full_backing_storage_bytes=view.untyped_storage().nbytes(),retained_tensor_bytes=loaded.numel()*loaded.element_size(),actual_file_bytes=target.stat().st_size,images=0,model_forward=0,optimization=0,seconds=time.time()-start,RSS_process_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(p/'STORAGE_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
