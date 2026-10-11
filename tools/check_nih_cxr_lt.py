"""Small CPU checks for author-label parsing and NIH-only train/validation scope."""
import csv
import json
from pathlib import Path
import tempfile
import time

import torch

from prepare_nih_cxr_lt import LABELS, ORDER, read_labels
from run_module_additions import author_nih, verify_historical_baseline
from summarize_module_additions import summarize
import module_additions as modules
import prototype_coherent as native


def check():
    start=time.monotonic()
    valid=dict(name='NIHCXRLT',author_nih_authorized=True,evaluation_split='author_balanced_validation',
               classes=20,image_size=224,seed=74002,task_sizes=[10,2,2,2,2,2],module='base')
    assert author_nih(valid)
    for changed in ({'name':'ISIC'},{'evaluation_split':'official_test'},
                    {'author_nih_authorized':False},{'classes':14},{'module':'local_readout'}):
        try:author_nih(dict(valid,**changed))
        except ValueError:pass
        else:raise AssertionError('NIH permission escaped')
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);raw=p/'labels.csv'
        row=dict(id='00000001_000.png',subject_id='1',**{k:int(i==0) for i,k in enumerate(LABELS)})
        def write(value):
            with raw.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['id',*LABELS,'subject_id']);w.writeheader();w.writerow(value)
        write(row);rows,counts=read_labels(raw)
        assert rows[0]['label']==0 and counts==[1]+[0]*19
        for change in ({LABELS[1]:1},{LABELS[0]:0},{LABELS[0]:2},
                       {'subject_id':'2'},{'id':'../00000001_000.png'}):
            write(dict(row,**change))
            try:read_labels(raw)
            except ValueError:pass
            else:raise AssertionError('Invalid author label accepted')
        verify_historical_baseline(valid,[dict(per_class_recall={'0':.5})],{})
        try:verify_historical_baseline(dict(name='ISIC',module='base'),[],{})
        except KeyError:pass
        else:raise AssertionError('Medical historical guard bypassed')
        fields=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall',
                'forgetting','final_old_recall','final_new_recall')
        metrics=dict(stages=[dict(per_class_n={'0':15})],test_accessed=False,
                     evaluation_split='author_balanced_validation',**{f:.5 for f in fields})
        s=dict(datasets=['NIHCXRLT'],author_nih_authorized=True,modules=['base','local_transport'],
               program=dict(costs=[]),runs={f'NIHCXRLT_{m}':dict(status=dict(status='COMPLETE'),
               metrics=metrics.copy()) for m in ('base','local_transport')})
        summarize(s,p/'valid')
        for i,changed in enumerate(({'test_accessed':True},{'evaluation_split':'official_test'})):
            s['runs']['NIHCXRLT_local_transport']['metrics']=dict(metrics,**changed)
            try:summarize(s,p/f'invalid{i}')
            except ValueError:pass
            else:raise AssertionError('NIH test aggregate accepted')
    torch.set_num_threads(2);torch.manual_seed(145)
    x=torch.randn(60,7,dtype=torch.float64);labels=torch.arange(20).repeat_interleave(3)
    bank=native.append(native.empty(7),x,labels,labels,torch.full((60,),1/3,dtype=x.dtype))
    w,r,_=modules.head(bank,'base','NIHCXRLT',ORDER);wn,rn=native.head(bank)
    assert torch.equal(w,wn) and torch.equal(r,rn)
    memory=modules.local_memory(None,x.new_zeros(4,7),torch.randn(60,4,7,dtype=x.dtype),labels)
    _,r,a=modules.head(bank,'local_transport','NIHCXRLT',ORDER,memory)
    assert a['extra_columns']==80 and a['prior_delta']>0 and torch.linalg.eigvalsh(r).min()>0
    return dict(status='PASS',author_one_hot_and_subject_metadata_checked=True,
                invalid_labels_and_paths_rejected=True,test_and_cross_dataset_permissions_rejected=True,
                medical_history_guard_retained=True,original_base_head_exact=True,
                local_prior_80_columns_positive=True,cpu_wall_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(check()))
