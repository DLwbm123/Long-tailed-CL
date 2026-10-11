"""CPU sampling, scoped permissions, grayscale loader and unchanged-head checks."""
import json
import math
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from prepare_tissuemnist_lt import RANK,ORDER,SOURCE_TRAIN,counts,select
from run_module_additions import synthetic_tissue,synthetic_path,official_cifar,verify_historical_baseline
from run_prototype_single import Images
from run_prototype_coherent import fit_split
from summarize_module_additions import summarize
import module_additions as modules
import prototype_coherent as native


def check():
    start=time.monotonic();r=list(range(8));o=r.copy()
    random.Random(74003).shuffle(r);random.Random(74004).shuffle(o);assert r==RANK and o==ORDER
    cap=min(5000,min(SOURCE_TRAIN[c]*100**(i/7) for i,c in enumerate(RANK)))
    assert 100*math.floor(cap/100)==5000
    y=np.concatenate([np.full(n,c) for c,n in enumerate(SOURCE_TRAIN)])
    ids=select(y);assert ids==select(y) and len(ids)==len(set(ids))==10315
    assert {c:int((y[ids]==c).sum()) for c in RANK}==counts()
    try:select(np.repeat(np.arange(8),30))
    except ValueError:pass
    else:raise AssertionError('Insufficient supply accepted')
    fit={}
    for c,n in counts().items():
        rows=[dict(label=c,identity_component=f'tissuemnist_train_{i:06d}') for i in range(n)]
        a,b=fit_split(rows,74002);assert not set(a)&set(b);fit[c]=len(a)
    sizes=[2,2,2,2];at=0;steps=[]
    for k in sizes:
        steps.append(2*math.ceil(sum(fit[c] for c in ORDER[at:at+k])/64));at+=k
    assert steps==[40,128,74,20] and sum(steps)==262
    valid=dict(name='TissueMNISTLT',synthetic_tissue_authorized=True,evaluation_split='official_validation',image_size=224,imbalance_factor=100,n_max=5000,classes=8)
    assert synthetic_tissue(valid) and not synthetic_path(valid) and not official_cifar(valid)
    for bad in [dict(valid,name='ISIC'),dict(valid,name='PathMNISTLT'),dict(valid,evaluation_split='official_test'),dict(valid,n_max=3100),dict(valid,synthetic_tissue_authorized=False)]:
        try:synthetic_tissue(bad)
        except ValueError:pass
        else:raise AssertionError('Tissue permission escaped')
    torch.manual_seed(142);torch.set_num_threads(2)
    x=torch.randn(40,7,dtype=torch.float64);labels=torch.arange(8).repeat_interleave(5)
    bank=native.append(native.empty(7),x,labels,labels,torch.full((40,),.2,dtype=x.dtype))
    w,r,_=modules.head(bank,'base','TissueMNISTLT',list(range(8)));wn,rn=native.head(bank)
    assert torch.equal(w,wn) and torch.equal(r,rn)
    memory=modules.local_memory(None,x.new_zeros(4,7),torch.randn(40,4,7,dtype=x.dtype),labels)
    _,r,a=modules.head(bank,'local_transport','TissueMNISTLT',list(range(8)),memory)
    assert a['extra_columns']==32 and a['prior_delta']>0 and torch.linalg.eigvalsh(r).min()>0
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        reports=[dict(per_class_recall={'0':.5})]
        for config in [dict(valid,module='base'),dict(name='PathMNISTLT',module='base',synthetic_path_authorized=True,
                evaluation_split='official_validation',image_size=224,imbalance_factor=100),
                dict(name='CIFAR100LT',module='base',auxiliary_cifar_authorized=True,evaluation_split='official_test',imbalance_factor=100)]:
            m={};verify_historical_baseline(config,reports,m);assert m=={}
        medical=dict(name='ISIC',module='base')
        try:verify_historical_baseline(medical,reports,{})
        except KeyError:pass
        else:raise AssertionError('Medical history requirement bypassed')
        history=p/'history.json';history.write_text(json.dumps(dict(stages=reports)))
        medical['historical_metrics']=str(history);m={};verify_historical_baseline(medical,reports,m)
        assert m['historical_baseline_recalls_equal']
        try:verify_historical_baseline(medical,[dict(per_class_recall={'0':.6})],{})
        except ValueError:pass
        else:raise AssertionError('Medical historical mismatch accepted')
        gray=np.arange(16,dtype=np.uint8).reshape(4,4);Image.fromarray(gray).save(p/'a.png')
        pixel,label=Images([dict(relative_path='a.png',label=3)],p,lambda image:np.asarray(image),74002)[0]
        assert pixel.shape==(4,4,3) and np.array_equal(pixel[:,:,0],gray) and np.array_equal(pixel[:,:,2],gray) and label==3
        fields=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall','forgetting','final_old_recall','final_new_recall')
        metrics=dict(stages=[dict(per_class_n={'0':100})],test_accessed=False,evaluation_split='official_validation',**{f:.5 for f in fields})
        snapshot=dict(datasets=['TissueMNISTLT'],synthetic_tissue_authorized=True,modules=['base','local_transport'],program=dict(costs=[]),runs={f'TissueMNISTLT_{m}':dict(status=dict(status='COMPLETE'),metrics=metrics.copy()) for m in ('base','local_transport')})
        summarize(snapshot,p/'valid')
        snapshot['runs']['TissueMNISTLT_local_transport']['metrics']['test_accessed']=True
        try:summarize(snapshot,p/'invalid')
        except ValueError:pass
        else:raise AssertionError('Tissue test aggregate accepted')
    return dict(status='PASS',deterministic_unique_sampling=True,source_supply_rule=True,stage_updates=steps,total_per_arm=262,original_base_head_exact=True,local_prior_32_columns_positive=True,grayscale_rgb_loader=True,test_and_cross_dataset_permissions_rejected=True,new_dataset_baselines_need_no_medical_history=True,medical_history_required_matching_and_mismatch_rejected=True,cpu_wall_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(check()))
