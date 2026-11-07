import json,time,os
from pathlib import Path

def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)

def guard(cfg):
    assert time.time()<cfg['deadline']-120,'ORIGINAL_CLOCK'
    import torch
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_reserved()<12*2**30,'GPU_RESERVED'

class Counts(dict):
    phase='native'
    def key(self,key):
        if key.startswith('engineering_virtual_'):return self.phase+'_virtual_'+key[len('engineering_virtual_'):]
        if key.startswith('paired_') or key=='query_rows':return self.phase+'_'+key
        return key
    def get(self,key,default=None):return super().get(self.key(key),default)
    def __getitem__(self,key):return super().__getitem__(self.key(key))
    def __setitem__(self,key,value):return super().__setitem__(self.key(key),value)

