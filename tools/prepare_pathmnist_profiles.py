"""Frozen IF10/50 and IF100-size balanced control; reuse the official archive."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from prepare_pathmnist_lt import counts,select
from run_module_cycle import save


PROFILES={'IF10':(10,5000),'IF50':(50,5000),'BALANCED11358':(1,1262)}


def prepare(config):
    root=Path(config['root']);begun=time.monotonic()
    def budget():
        if time.time()>=config['deadline']:raise TimeoutError('Frozen campaign deadline')
    def status(**values):
        save(root/'DATA_STATUS.private.json',dict(elapsed_seconds=time.monotonic()-begun,**values))
    try:
        source=Path(config['reuse_data']);audit=json.loads((source.parent/'DATA_AUDIT.json').read_text())
        assert audit['test_arrays_opened'] is False
        images=root/'data'/'images';images.mkdir(parents=True)
        (images/'val').symlink_to(source/'images'/'val',target_is_directory=True)
        train_images=images/'train';train_images.mkdir()
        status(status='RUNNING',phase='EXTRACT_TRAIN_FROM_CACHED_ARCHIVE')
        with np.load(source/'pathmnist_224.npz',allow_pickle=False) as arrays:
            y=arrays['train_labels'].reshape(-1);v=arrays['val_labels'].reshape(-1)
            for split,labels in [('train',y),('val',v)]:
                reference=np.load(Path(config['reference_labels'])/(split+'_labels.npy'),allow_pickle=False).reshape(-1)
                assert np.array_equal(labels,reference),'Official label version mismatch'
            selected={k:select(y,config['frequency_rank'],*args) for k,args in PROFILES.items()}
            union=sorted(set().union(*selected.values()));data=arrays['train_images']
            assert data.shape==(89996,224,224,3) and data.dtype==np.uint8
            reused=0
            for done,i in enumerate(union,1):
                budget();name=f'{i:06d}.png';old=source/'images'/'train'/name
                if old.exists():
                    (train_images/name).symlink_to(old);reused+=1
                else:Image.fromarray(data[i]).save(train_images/name)
                if done%1000==0:status(status='RUNNING',phase='PREPARE_SELECTED_TRAIN',images_completed=done,images_total=len(union))
            del data
        from run_prototype_single import manifests
        from run_prototype_coherent import fit_split
        profiles={}
        for key,args in PROFILES.items():
            folder=root/'data'/key;folder.mkdir()
            for split,indices,labels in [('train',selected[key],y),('val',range(len(v)),v)]:
                with (folder/(split+'.csv')).open('x',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['split','relative_path','identity_component','original_label']);writer.writeheader()
                    for i in indices:writer.writerow(dict(split=split,relative_path=f'{split}/{i:06d}.png',identity_component=f'pathmnist_{split}_{i:06d}',original_label=int(labels[i])))
            jobs=[j for j in config['jobs'] if j['path_profile']==key];assert len(jobs)==2
            train,val=manifests(jobs[0]);assert len(train)==sum(counts(config['frequency_rank'],*args).values()) and len(val)==10004
            fit_counts={};meta_counts={}
            for c in range(9):
                rows=[row for row in train if row['label']==c];fit,meta=fit_split(rows,74002)
                assert set(fit).isdisjoint(meta)
                fit_counts[str(c)]=len(fit);meta_counts[str(c)]=len(meta)
            offset=0;updates=[]
            for k in (3,2,2,2):
                cs=jobs[0]['order'][offset:offset+k];offset+=k
                updates.append(2*__import__('math').ceil(sum(fit_counts[str(c)] for c in cs)/64))
            assert sum(updates)==jobs[0]['expected_steps']==jobs[1]['expected_steps']
            profiles[key]=dict(class_counts={str(c):n for c,n in counts(config['frequency_rank'],*args).items()},fit_counts=fit_counts,meta_counts=meta_counts,stage_updates=updates,train_n=len(train))
        save(root/'DATA_AUDIT.json',dict(test_arrays_opened=False,source_counts=audit['source_counts'],profiles=profiles,union_images_n=len(union),reused_images_n=reused,new_images_n=len(union)-reused,official_validation_n=10004,patient_identifiers_available=False))
        status(status='READY',phase='DATA_PREPARED',profiles_n=3,test_accessed=False)
    except BaseException as exc:
        status(status='INCOMPLETE',phase='STOPPED_DATA_PREPARATION',error=str(exc));raise


def check():
    from run_module_additions import synthetic_path
    rank=[4,2,7,6,0,5,3,8,1];y=np.repeat(np.arange(9),6000)
    old=select(y,rank);assert len(old)==11358
    chosen={k:select(y,rank,*args) for k,args in PROFILES.items()}
    assert set(old)<=set(chosen['IF50'])<=set(chosen['IF10'])
    assert all(n==1262 for n in counts(rank,1,1262).values()) and len(chosen['BALANCED11358'])==11358
    for key,args in PROFILES.items():
        a=chosen[key];assert a==select(y,rank,*args) and len(a)==len(set(a))
        assert {c:int((y[a]==c).sum()) for c in rank}==counts(rank,*args)
        cfg=dict(name='PathMNISTLT',synthetic_path_authorized=True,evaluation_split='official_validation',image_size=224,path_profile=key,imbalance_factor=args[0],n_max=args[1])
        assert synthetic_path(cfg)
        for bad in [dict(cfg,evaluation_split='official_test'),dict(cfg,n_max=5001),dict(cfg,path_profile='IF20')]:
            try:synthetic_path(bad)
            except ValueError:pass
            else:raise AssertionError('Unfrozen profile accepted')
    return dict(status='PASS',nested_samples=True,balanced_total_matches_if100=True,deterministic_unique=True,test_and_unfrozen_profiles_rejected=True,profile_counts={k:counts(rank,*args) for k,args in PROFILES.items()})


if __name__=='__main__':prepare(json.loads(Path(os.environ['Q128_CONFIG']).read_text()))
