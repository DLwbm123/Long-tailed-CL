"""Small checks of sampling, evaluation permission, and unchanged base/local prior."""
import json
import random
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from prepare_pathmnist_lt import counts,select
from run_module_additions import official_cifar,synthetic_path
import module_additions as modules
import prototype_coherent as native
from summarize_module_additions import summarize


def check():
    begun=time.monotonic();rank=list(range(9));random.Random(74003).shuffle(rank)
    y=np.repeat(np.arange(9),6000);a=select(y,rank);b=select(y,rank)
    assert a==b and len(a)==len(set(a))==11358
    assert {c:int((y[a]==c).sum()) for c in rank}==counts(rank)
    try:select(np.repeat(np.arange(9),49),rank)
    except ValueError:pass
    else:raise AssertionError('Insufficient class supply accepted')
    valid=dict(name='PathMNISTLT',synthetic_path_authorized=True,evaluation_split='official_validation',image_size=224,imbalance_factor=100)
    assert synthetic_path(valid) and not official_cifar(valid)
    for bad in (dict(valid,evaluation_split='official_test'),dict(valid,synthetic_path_authorized=False),dict(valid,name='ISIC'),dict(valid,image_size=28)):
        try:synthetic_path(bad)
        except ValueError:pass
        else:raise AssertionError('Dataset permission escaped')
    torch.manual_seed(139);torch.set_num_threads(2)
    x=torch.randn(45,7,dtype=torch.float64);labels=torch.arange(9).repeat_interleave(5)
    bank=native.append(native.empty(7),x,labels,labels,torch.full((45,),.2,dtype=x.dtype))
    w,r,_=modules.head(bank,'base','PathMNISTLT',list(range(9)));wn,rn=native.head(bank)
    assert torch.equal(w,wn) and torch.equal(r,rn)
    memory=modules.local_memory(None,x.new_zeros(4,7),torch.randn(45,4,7,dtype=x.dtype),labels)
    _,r,a=modules.head(bank,'local_transport','PathMNISTLT',list(range(9)),memory)
    assert a['extra_columns']==36 and a['prior_delta']>0 and torch.linalg.eigvalsh(r).min()>0
    fields=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall')
    metrics=dict(stages=[dict(per_class_n={'0':100})],test_accessed=False,evaluation_split='official_validation',**{f:.5 for f in fields})
    snapshot=dict(datasets=['PathMNISTLT'],synthetic_path_authorized=True,modules=['base','local_transport'],program=dict(costs=[]),runs={f'PathMNISTLT_{m}':dict(status=dict(status='COMPLETE'),metrics=metrics.copy()) for m in ('base','local_transport')})
    with tempfile.TemporaryDirectory() as tmp:
        summarize(snapshot,Path(tmp)/'valid')
        snapshot['runs']['PathMNISTLT_local_transport']['metrics']['test_accessed']=True
        try:summarize(snapshot,Path(tmp)/'invalid')
        except ValueError:pass
        else:raise AssertionError('PathMNIST test aggregate accepted')
    return dict(status='PASS',sampling_deterministic_without_replacement=True,insufficient_supply_rejected=True,official_test_rejected=True,dataset_permission_scoped=True,original_base_head_exact=True,local_prior_9_classes_positive_definite=True,aggregate_test_rejected=True,cpu_wall_seconds=time.monotonic()-begun)


if __name__=='__main__':print(json.dumps(check()))
