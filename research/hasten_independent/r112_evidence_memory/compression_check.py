"""Exact old/new ZERO state checks and a bounded synthetic GPU cost check."""
import json,os,time,resource
from pathlib import Path
import numpy as np
import torch
from . import evidence as E
from . import native


def old_compress(e,seed):return e.with_support(E.anchors(e,seed)[0])


def same(a,b):
    assert torch.equal(a.bank.pid*8+a.bank.view,b.bank.pid*8+b.bank.view)
    assert torch.equal(a.bank.z,b.bank.z) and torch.equal(a.bank.y,b.bank.y) and torch.equal(a.flags,b.flags)
    assert np.array_equal(a.stats,b.stats)


def run(output,device='cpu'):
    start=time.time();torch.set_num_threads(1 if device=='cpu' else 2)
    if device=='cuda':
        os.sched_setaffinity(0,{124,125});resource.setrlimit(resource.RLIMIT_CPU,(120,120));resource.setrlimit(resource.RLIMIT_FSIZE,(2**30,2**30))
        torch.cuda.set_per_process_memory_fraction((12*2**30-64*2**20)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(112);z=torch.nn.functional.normalize(torch.randn(30,6,512,device=device),dim=-1)
    if device=='cpu':
        rows=[{'original_label':str(i//24),'identity_component':f'unit_{i}'} for i in range(552)]
        full=E.mem.from_views(torch.nn.functional.normalize(torch.randn(552,6,512),dim=-1),torch.arange(552),torch.arange(552)//24)
        stages=[range(5),range(5,10),range(10,15),range(15,19),range(19,23)];checks=0
        for seed in [74003,74004]:
            a=b=None
            for stage,cs in enumerate(stages):
                f=full.take(torch.where(torch.isin(full.y,torch.tensor(list(cs))))[0]);current=E.born(f,rows,stage)
                a=old_compress(E.join(a,current),seed+70000+stage);b=E.compressed(E.join(b,current),seed+70000+stage);same(a,b);checks+=1
                prefix=E.classes(a,list(range(min(a.C,3))))
                same(old_compress(prefix,seed+stage),E.compressed(prefix,seed+stage));checks+=1
        result={'passed':True,'synthetic_only':True,'device':device,'all_flags_features_stats_and_sets_equal':True,'sequential_checks':checks,'science_updates':0,'encoder_rows':0,'wall_seconds':time.time()-start}
    else:
        initial=native.fixture(z,3331,0,torch.cat([torch.arange(20),torch.full((3311,),20)]));kept=E.compressed(initial,112)
        pool=E.join(kept,native.fixture(z,64,10000,torch.full((64,),21),1));torch.cuda.synchronize()
        t=time.time();a=old_compress(pool,31001);torch.cuda.synchronize();old=time.time()-t
        t=time.time();b=E.compressed(pool,31001);torch.cuda.synchronize();new=time.time()-t;same(a,b)
        result={'passed':True,'synthetic_only':True,'device':device,'all_flags_features_stats_and_sets_equal':True,'old_compression_seconds':old,'new_compression_seconds':new,'science_updates':0,'encoder_rows':0,'wall_seconds':time.time()-start}
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':run(os.environ['Q_OUTPUT'],os.environ.get('Q_DEVICE','cpu'))
