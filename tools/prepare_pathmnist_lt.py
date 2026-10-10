"""One frozen PathMNIST-224 sampling protocol; official test is never opened."""
import csv
import json
import math
import os
from pathlib import Path
import random
import time
import urllib.request

import numpy as np
from PIL import Image

from run_module_cycle import save


def counts(rank):
    if sorted(rank)!=list(range(9)):
        raise ValueError('Nine distinct classes required')
    return {c:math.floor(5000*100**(-i/8)) for i,c in enumerate(rank)}


def select(labels,rank):
    labels=np.asarray(labels).reshape(-1)
    if set(labels.tolist())!=set(range(9)):
        raise ValueError('Source class coverage differs')
    result=[]
    for c,n in counts(rank).items():
        available=np.flatnonzero(labels==c).tolist()
        if len(available)<n:
            raise ValueError('Source cannot supply frozen count; no clipping or replacement')
        random.Random(74002+c*2003).shuffle(available)
        result.extend(available[:n])
    assert len(result)==11358 and len(set(result))==len(result)
    return sorted(result)


def prepare(config):
    root=Path(config['root']);begun=time.monotonic();source=root/'data';source.mkdir()
    def status(**values):
        save(root/'DATA_STATUS.private.json',dict(elapsed_seconds=time.monotonic()-begun,**values))
    def budget():
        if time.time()>=config['deadline']:raise TimeoutError('Original campaign deadline')
    try:
        archive=source/'pathmnist_224.npz';partial=archive.with_suffix('.part')
        status(status='RUNNING',phase='DOWNLOAD',downloaded_bytes=0,expected_bytes=12629854322)
        with urllib.request.urlopen(config['data_url'],timeout=90) as response,partial.open('xb') as target:
            length=int(response.headers['Content-Length']);assert length==12629854322
            size=0;last=0.
            while True:
                budget();chunk=response.read(8*1024**2)
                if not chunk:break
                target.write(chunk);size+=len(chunk)
                if time.monotonic()-last>15:
                    status(status='RUNNING',phase='DOWNLOAD',downloaded_bytes=size,expected_bytes=length);last=time.monotonic()
        if size!=length:raise ValueError('Download length mismatch')
        partial.replace(archive)
        images=source/'images';manifest=source/'manifests';images.mkdir();manifest.mkdir()
        audit={'test_arrays_opened':False,'patient_identifiers_available':False,'image_size':224,'sampling_without_replacement':True,'source_counts':{},'selected_counts':{},'fit_counts':{},'meta_counts':{},'stage_updates':[]}
        with np.load(archive,allow_pickle=False) as arrays:
            for split in ('train','val'):
                budget();status(status='RUNNING',phase='EXTRACT_'+split.upper(),downloaded_bytes=size)
                labels=arrays[split+'_labels'].reshape(-1)
                reference=np.load(Path(config['reference_labels'])/(split+'_labels.npy'),allow_pickle=False).reshape(-1)
                if not np.array_equal(labels,reference):raise ValueError('Official labels disagree across source versions')
                audit['source_counts'][split]={str(c):int((labels==c).sum()) for c in range(9)}
                indices=select(labels,config['frequency_rank']) if split=='train' else list(range(len(labels)))
                data=arrays[split+'_images']
                if data.shape!=(len(labels),224,224,3) or data.dtype!=np.uint8:raise ValueError('Official 224 RGB shape differs')
                folder=images/split;folder.mkdir()
                with (manifest/(split+'.csv')).open('x',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['split','relative_path','identity_component','original_label']);writer.writeheader()
                    for i in indices:
                        budget();name=f'{i:06d}.png';Image.fromarray(data[i]).save(folder/name)
                        writer.writerow(dict(split=split,relative_path=split+'/'+name,identity_component=f'pathmnist_{split}_{i:06d}',original_label=int(labels[i])))
                audit['selected_counts'][split]={str(c):int((labels[indices]==c).sum()) for c in range(9)}
                del data
        from run_prototype_single import manifests
        from run_prototype_coherent import fit_split
        train,val=manifests(config['jobs'][0]);assert len(train)==11358 and len(val)==10004
        for c,n in counts(config['frequency_rank']).items():
            rows=[v for v in train if v['label']==c];fit,meta=fit_split(rows,74002)
            assert len(rows)==n and set(fit).isdisjoint(meta)
            audit['fit_counts'][str(c)]=len(fit);audit['meta_counts'][str(c)]=len(meta)
        offset=0
        for k in (3,2,2,2):
            cs=config['jobs'][0]['order'][offset:offset+k];offset+=k
            audit['stage_updates'].append(2*math.ceil(sum(audit['fit_counts'][str(c)] for c in cs)/64))
        assert sum(audit['stage_updates'])==config['jobs'][0]['expected_steps']
        save(root/'DATA_AUDIT.json',audit);status(status='READY',phase='DATA_PREPARED',train_n=11358,validation_n=10004,test_accessed=False)
    except BaseException as exc:
        status(status='INCOMPLETE',phase='STOPPED_DATA_PREPARATION',error=str(exc));raise


if __name__=='__main__':prepare(json.loads(Path(os.environ['Q128_CONFIG']).read_text()))
