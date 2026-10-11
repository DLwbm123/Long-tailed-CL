"""Frozen kidney-cell IF100 pair; official test arrays remain unopened."""
import csv
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from prepare_pathmnist_lt import download
from run_module_cycle import save

RANK=[4,2,7,5,6,3,0,1]
ORDER=[7,3,4,0,6,2,1,5]
SOURCE_TRAIN=[53075,7814,5866,15406,11789,7705,39203,24608]
SOURCE_VAL=[7582,1117,838,2201,1684,1101,5601,3516]


def counts():
    return {c:math.floor(5000*100**(-i/7)) for i,c in enumerate(RANK)}


def select(labels):
    y=np.asarray(labels).reshape(-1)
    if set(y.tolist())!=set(range(8)):raise ValueError('Source class coverage differs')
    result=[]
    for c,n in counts().items():
        available=np.flatnonzero(y==c).tolist()
        if len(available)<n:raise ValueError('Insufficient source; no clipping or replacement')
        random.Random(74002+c*2003).shuffle(available);result.extend(available[:n])
    assert len(result)==len(set(result))==10315
    return sorted(result)


def prepare(config):
    root=Path(config['root']);begun=time.monotonic();data=root/'data';data.mkdir()
    def status(**values):save(root/'DATA_STATUS.private.json',dict(elapsed_seconds=time.monotonic()-begun,**values))
    def budget():
        if time.time()>=config['deadline']:raise TimeoutError('Original campaign deadline')
    try:
        archive=data/'tissuemnist_224.npz';partial=data/'tissuemnist_224.part';length=3433703243
        status(status='RUNNING',phase='DOWNLOAD',downloaded_bytes=0,expected_bytes=length)
        download(config['data_url'],partial,length,budget,status);partial.replace(archive)
        images=data/'images';manifest=data/'manifests';images.mkdir();manifest.mkdir()
        audit=dict(test_arrays_opened=False,patient_identifiers_available=False,image_size=224,
                   sampling_without_replacement=True,source_counts={},selected_counts={},fit_counts={},meta_counts={},stage_updates=[])
        with np.load(archive,allow_pickle=False) as arrays:
            for split,expected in [('train',SOURCE_TRAIN),('val',SOURCE_VAL)]:
                budget();status(status='RUNNING',phase='EXTRACT_'+split.upper(),downloaded_bytes=length)
                labels=arrays[split+'_labels'].reshape(-1)
                reference=np.load(Path(config['reference_labels'])/(split+'_labels.npy'),allow_pickle=False).reshape(-1)
                observed={str(c):int((labels==c).sum()) for c in range(8)}
                if not np.array_equal(labels,reference) or list(observed.values())!=expected:
                    raise ValueError('Official train/val labels disagree with frozen supply')
                audit['source_counts'][split]=observed
                indices=select(labels) if split=='train' else list(range(len(labels)))
                pixels=arrays[split+'_images']
                if pixels.dtype!=np.uint8 or pixels.shape not in ((len(labels),224,224),(len(labels),224,224,1)):
                    raise ValueError('Official grayscale 224 shape differs')
                folder=images/split;folder.mkdir()
                with (manifest/(split+'.csv')).open('x',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['split','relative_path','identity_component','original_label']);writer.writeheader()
                    for i in indices:
                        budget();name=f'{i:06d}.png';Image.fromarray(pixels[i].reshape(224,224)).save(folder/name)
                        writer.writerow(dict(split=split,relative_path=split+'/'+name,
                            identity_component=f'tissuemnist_{split}_{i:06d}',original_label=int(labels[i])))
                audit['selected_counts'][split]={str(c):int((labels[indices]==c).sum()) for c in range(8)}
                del pixels
        from run_prototype_single import manifests,Images
        from run_prototype_coherent import fit_split
        train,val=manifests(config['jobs'][0]);assert len(train)==10315 and len(val)==23640
        for c,n in counts().items():
            rows=[r for r in train if r['label']==c];fit,meta=fit_split(rows,74002)
            assert len(rows)==n and set(fit).isdisjoint(meta)
            audit['fit_counts'][str(c)]=len(fit);audit['meta_counts'][str(c)]=len(meta)
        offset=0
        for k in (2,2,2,2):
            cs=ORDER[offset:offset+k];offset+=k
            audit['stage_updates'].append(2*math.ceil(sum(audit['fit_counts'][str(c)] for c in cs)/64))
        assert audit['stage_updates']==[40,128,74,20]
        # Exercise the real loader before dispatch; no external image symlinks.
        pixel,label=Images(train,images,lambda image:np.asarray(image),74002)[0]
        assert pixel.shape==(224,224,3) and np.array_equal(pixel[:,:,0],pixel[:,:,2]) and label==train[0]['label']
        save(root/'DATA_AUDIT.json',audit)
        status(status='READY',phase='DATA_PREPARED',train_n=10315,validation_n=23640,test_accessed=False)
    except BaseException as exc:
        status(status='INCOMPLETE',phase='STOPPED_DATA_PREPARATION',error=str(exc));raise


if __name__=='__main__':prepare(json.loads(Path(os.environ['Q128_CONFIG']).read_text()))
