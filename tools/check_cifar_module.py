"""CPU checks for the explicit auxiliary-test boundary and unchanged local prior."""
import copy
import json
from pathlib import Path
import tempfile
import time

import torch
import module_additions as modules
import prototype_coherent as native
from run_module_additions import official_cifar
from summarize_module_additions import summarize


def check():
    started=time.monotonic()
    config=dict(name='CIFAR100LT',auxiliary_cifar_authorized=True,
                evaluation_split='official_test',imbalance_factor=100)
    assert official_cifar(config)
    assert not official_cifar(dict(name='ISIC',evaluation_split='development_validation'))
    for invalid in (dict(config,auxiliary_cifar_authorized=False),dict(config,name='ISIC'),
                    dict(config,evaluation_split='development_validation'),dict(config,imbalance_factor=50)):
        try:official_cifar(invalid)
        except ValueError:pass
        else:raise AssertionError('Evaluation permission escaped its scope')
    torch.manual_seed(137);torch.set_num_threads(2)
    x=torch.randn(500,7,dtype=torch.float64);y=torch.arange(100).repeat_interleave(5)
    bank=native.append(native.empty(7),x,y,y,torch.full((500,),.2,dtype=x.dtype))
    w,r,_=modules.head(bank,'base','CIFAR100LT',list(range(100)));wn,rn=native.head(bank)
    assert torch.equal(w,wn) and torch.equal(r,rn)
    parts=torch.randn(500,4,7,dtype=x.dtype)
    memory=modules.local_memory(None,x.new_zeros(4,7),parts,y)
    _,r,a=modules.head(bank,'local_transport','CIFAR100LT',list(range(100)),memory)
    assert a['extra_columns']==400 and a['prior_delta']>0 and torch.linalg.eigvalsh(r).min()>0
    shift=torch.randn(4,7,dtype=x.dtype)/100
    moved=modules.local_memory(memory,shift,parts,y)
    assert torch.allclose(moved[:100],memory+shift) and torch.equal(moved[100:],memory)
    fields=('final_balanced_accuracy','average_incremental_balanced_accuracy','final_tail_recall',
            'forgetting','final_old_recall','final_new_recall')
    metrics=dict(stages=[dict(per_class_n={'0':100})],test_accessed=True,evaluation_split='official_test',
                 **{f:.5 for f in fields})
    snapshot=dict(datasets=['CIFAR100LT'],auxiliary_cifar_authorized=True,modules=['base','local_transport'],
                  program=dict(costs=[]),runs={f'CIFAR100LT_{m}':dict(status=dict(status='COMPLETE'),metrics=copy.deepcopy(metrics)) for m in ('base','local_transport')})
    with tempfile.TemporaryDirectory() as folder:
        destination=Path(folder)/'valid';summarize(snapshot,destination)
        assert json.loads((destination/'RESULTS.json').read_text())['test_accessed'] is True
        snapshot['auxiliary_cifar_authorized']=False
        try:summarize(snapshot,Path(folder)/'invalid')
        except ValueError:pass
        else:raise AssertionError('Unapproved test aggregate accepted')
        snapshot.update(datasets=['ISIC','HK'],auxiliary_cifar_authorized=True)
        snapshot['runs'].update({f'{ds}_{m}':dict(status=dict(status='COMPLETE'),metrics=copy.deepcopy(metrics)) for ds in ('ISIC','HK') for m in ('base','local_transport')})
        try:summarize(snapshot,Path(folder)/'medical')
        except ValueError:pass
        else:raise AssertionError('Medical test aggregate accepted')
    return dict(status='PASS',explicit_cifar_permission=True,medical_test_still_rejected=True,
                original_base_head_exact=True,local_prior_100_classes_positive_definite=True,
                regional_transport_algebra=True,aggregate_test_metadata_correct=True,
                cpu_wall_seconds=time.monotonic()-started)


if __name__=='__main__':print(json.dumps(check()))
