"""Audit author train/val only and reuse the existing NIH image cache read-only."""
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

LABELS = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
          'Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule',
          'Pleural_Thickening','Pneumonia','Pneumothorax','Pneumoperitoneum',
          'Pneumomediastinum','Subcutaneous Emphysema','Tortuous Aorta',
          'Calcification of the Aorta','No Finding']
ORDER = [13,19,6,14,17,1,9,4,15,8,18,2,0,12,16,3,7,11,5,10]


def read_labels(path):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != ['id',*LABELS,'subject_id']:
            raise ValueError('Author schema differs; subject_id is metadata, not a label')
        rows=list(reader)
    ids=set();counts=[0]*20
    for row in rows:
        name=row['id'];p=Path(name)
        if p.name!=name or p.suffix!='.png' or name in ids:
            raise ValueError('Unsafe or duplicate image identifier')
        if str(int(name.split('_')[0]))!=str(int(row['subject_id'])):
            raise ValueError('Author subject_id does not match NIH image identity')
        values=[int(row[c]) for c in LABELS]
        if set(values)-{0,1} or sum(values)!=1:
            raise ValueError('Require exactly one author label; no argmax or label filtering')
        row['label']=values.index(1);counts[row['label']]+=1;ids.add(name)
    return rows,counts


def prepare(config):
    begun=time.monotonic();root=Path(config['root']);cache=Path(config['image_cache'])
    mount=subprocess.check_output(['findmnt','-T',str(root.parent),'-n','-o','FSTYPE'],text=True).strip()
    assert mount.startswith('nfs') and shutil.disk_usage(root.parent).free>10*1024**3
    root.mkdir(exist_ok=False)
    probe=root/'.probe';probe.write_text('probe');assert probe.read_text()=='probe';probe.unlink()
    raw=root/'author_labels';raw.mkdir();manifest=root/'manifests';manifest.mkdir()
    source={};counts={}
    for split in ('train','val'):
        p=raw/(split+'.csv');p.write_bytes(Path(config[split+'_labels']).read_bytes())
        source[split],counts[split]=read_labels(p)
    assert len(source['train'])==68058 and len(source['val'])==300 and counts['val']==[15]*20
    images={s:{r['id'] for r in rows} for s,rows in source.items()}
    patients={s:{r['subject_id'] for r in rows} for s,rows in source.items()}
    assert not images['train']&images['val'] and not patients['train']&patients['val']
    ready=json.loads((cache/'prepared/READY.json').read_text());assert ready['status']=='READY'
    # One filename listing checks actual coverage without decoding or hashing all images.
    available={e.name for e in os.scandir(cache/'images')}
    missing=(images['train']|images['val'])-available
    if missing:raise ValueError(f'{len(missing)} required author images absent; do not silently subset')
    for split,rows in source.items():
        with (manifest/(split+'.csv')).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['split','relative_path','identity_component','original_label'])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(split=split,relative_path='images/'+row['id'],
                    identity_component=row['id'],original_label=row['label']))
    import sys
    sys.path.insert(0,str(Path(config['source_tools'])))
    from run_prototype_single import Images,manifests
    from run_prototype_coherent import fit_split
    train,val=manifests(dict(manifest=str(manifest),order=ORDER))
    fit={}
    for c in range(20):
        a,b=fit_split([r for r in train if r['label']==c],74002)
        fit[c]=len(a);assert len(a)+len(b)==counts['train'][c] and set(a).isdisjoint(b)
    for rows in (train,val):
        shape,label=Images(rows,cache,lambda im:im.size,74002)[0]
        assert min(shape)>0 and label==rows[0]['label']
    tasks=[ORDER[:10]]+[ORDER[i:i+2] for i in range(10,20,2)]
    steps=[2*math.ceil(sum(fit[c] for c in block)/64) for block in tasks]
    assert steps==[1504,26,84,12,22,58]
    audit=dict(status='READY',labels=LABELS,train_n=68058,validation_n=300,
        train_counts=counts['train'],validation_counts=counts['val'],
        train_patients=len(patients['train']),validation_patients=len(patients['val']),
        patient_overlap=0,image_overlap=0,missing_images=0,source_images_modified=False,
        test_labels_opened=False,test_images_opened=False,order=ORDER,task_sizes=[10,2,2,2,2,2],
        fit_counts=fit,stage_updates=steps,updates_per_arm=sum(steps),
        head_classes=[i for i,n in enumerate(counts['train']) if n>1000],
        middle_classes=[i for i,n in enumerate(counts['train']) if 100<=n<=1000],
        tail_classes=[i for i,n in enumerate(counts['train']) if n<100],
        grouping_difference='Released CSV gives 7/9/4 under paper thresholds, paper says 7/10/3',
        cpu_process_seconds=time.monotonic()-begun,
        identity='Patient-disjoint author train/val; fit/meta and training tasks are image based, not patient-disjoint')
    (root/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    (root/'READY.json').write_text(json.dumps(dict(status='READY',training_started=False,test_accessed=False))+'\n')
    return audit


if __name__=='__main__':
    print(json.dumps(prepare(json.load(__import__('sys').stdin))))
