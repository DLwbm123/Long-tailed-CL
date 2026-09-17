"""CT1 early resource gate, before formal release; never accesses held-out images.

This is a measured resource admission only, not the complete P0 engineering gate.
Paths come from a private JSON file via P20_CONFIG; no trained parent is loaded.
"""
import csv
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from run_medical_v2 import (ROOT, Learner, Images, seed_all, extend_embedding,
                            effective_optimizer, write_json, sha, WEIGHT_SHA)
from prepare_hyperkvasir_hk1_m1 import ORDERS as HK_ORDERS
from check_isic_a1_linear_native import install_batched_linear

ISIC_ORDERS = {1993: [4,0,3,7,5,6,2,1], 1994: [3,7,4,5,1,0,6,2],
               1995: [1,3,0,6,4,5,2,7]}


def task_lock(cfg):
    result = {}
    for name, orders, sizes in [('HK', HK_ORDERS, [2]*10+[3]),
                                 ('ISIC', ISIC_ORDERS, [2]*4)]:
        spec = cfg['datasets'][name]
        rows = {}
        for split in ('train', 'val'):
            path = Path(spec['manifests']) / (split+'.csv')
            assert sha(path) == spec['manifest_sha256'][split], 'BLOCKED_MANIFEST_DRIFT'
            with path.open() as f: rows[split] = list(csv.DictReader(f))
            assert len(rows[split]) == spec['n_'+split]
        counts = {c: sum(int(r['original_label']) == c for r in rows['train'])
                  for c in range(sum(sizes))}
        assert min(counts.values()) > 0
        if name == 'HK':
            assert all(int(r['original_label']) == int(r['official_name_sorted_id'])
                       for rs in rows.values() for r in rs), 'BLOCKED_LABEL_NAMESPACE'
        runs = {}
        for seed, order in orders.items():
            assert sorted(order) == list(range(sum(sizes)))
            tasks = []; start = 0
            for index, n in enumerate(sizes):
                stop = start+n; classes = order[start:stop]
                nt = sum(counts[c] for c in classes)
                nv = sum(int(r['original_label']) in order[:stop] for r in rows['val'])
                tasks.append(dict(task=index+1, classes=classes, seen=stop, n_train=nt,
                                  n_val_seen=nv, steps=10*math.ceil(nt/48),
                                  probe_batches=11*math.ceil(nt/48),
                                  val_batches=math.ceil(nv/48)))
                start = stop
            runs[str(seed)] = tasks
        result[name] = dict(orders=orders, task_sizes=sizes, train_counts=counts,
                            runs=runs, manifest_sha256=spec['manifest_sha256'])
    assert sum(len(ts) for x in result.values() for ts in x['runs'].values())*2 == 90
    return result


def main():
    cfg = json.loads(Path(os.environ['P20_CONFIG']).read_text())
    out = Path(cfg['root'])/'output/public'; out.mkdir(parents=True, exist_ok=True)
    lock = task_lock(cfg); write_json(out/'DATA_AND_TASK_LOCK.json', lock)
    assert sha(cfg['weight']) == WEIGHT_SHA
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    allowed = set()
    for spec in cfg['datasets'].values():
        with (Path(spec['manifests'])/'train.csv').open() as f:
            rows = list(csv.DictReader(f))
        first = lock[spec['name']]['orders'][1993][:2]
        allowed.update(str((Path(spec['images'])/r['relative_path']).resolve())
                       for r in rows if int(r['original_label']) in first)
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0]))
        if p.suffix.lower() in ('.jpg','.jpeg','.png'):
            assert str(p.resolve()) in allowed, 'BLOCKED_OLD_FUTURE_OR_HELDOUT_IMAGE'
    import sys
    sys.addaudithook(guard)
    began = time.monotonic(); results = {}; training_steps=0; image_rows=0
    with threadpool_limits(limits=4):
        for name in ('HK','ISIC'):
            spec=cfg['datasets'][name]; order=lock[name]['orders'][1993]
            seed_all(1993)
            a=json.loads((ROOT/'third_party/APART/exps/apart_cifar_shuffle.json').read_text())
            a.update(nb_classes=len(order),nb_tasks=len(lock[name]['task_sizes']),
                     init_cls=2,increment=2,seed=1993,device=[torch.device('cuda:0')],
                     medical_v2=True,locked_weight_path=cfg['weight'],tuned_epoch=1,
                     concm_stage1=False,concm_stage1_eval_calibration=False,calibration_rule='none',
                     lt_list=[lock[name]['train_counts'][c] for c in order],
                     weight_decay=.01,real_ce_scope='all_seen',save_task_checkpoints=False)
            learner=Learner(a);extend_embedding(learner._network.backbone.assigner,1993)
            learner._cur_task=0;learner._known_classes=0;learner._total_classes=2
            net=learner._network.to('cuda');opt=effective_optimizer(net.backbone)
            nonshared={'backbone.'+k for k in net.backbone.weight_load_audit['allowed_missing']}
            assert {k for k,p in net.named_parameters() if p.requires_grad} == nonshared
            tensor_bytes=sum(t.numel()*t.element_size() for k,t in net.state_dict().items() if k in nonshared)
            train_ds=Images(Path(spec['manifests'])/'train.csv',spec['images'],order,range(2),True)
            loader=DataLoader(train_ds,batch_size=48,shuffle=True,num_workers=4,drop_last=False,
                              generator=torch.Generator().manual_seed(1993))
            batches=[]
            for batch in loader:
                if len(batch[1]) == 48: batches.append(batch)
                if len(batches)==6:break
            assert len(batches)>=3
            timings=[]
            for batch in batches:
                torch.cuda.synchronize();t=time.monotonic()
                learner._init_train([batch],None,opt,None)
                torch.cuda.synchronize();timings.append(time.monotonic()-t)
                training_steps+=1;image_rows+=len(batch[1])
            del opt,batches,loader,train_ds
            # This instance is discarded; the inference patch never enters training.
            net.requires_grad_(False).eval();install_batched_linear(net)
            net.backbone.pool.batchwise_prompt=False;net.backbone.pool_few.batchwise_prompt=False
            ds=Images(Path(spec['manifests'])/'train.csv',spec['images'],order,range(2),False)
            loader=DataLoader(ds,batch_size=48,shuffle=False,num_workers=4,drop_last=False,
                              generator=torch.Generator().manual_seed(1993))
            probes=[];last=time.monotonic()
            with torch.inference_mode():
                for j,(_,x,y) in enumerate(loader):
                    o=net(x.cuda(),train=False,weight=None)
                    raw=torch.stack([o['pre_logits'],o['pre_logits_few']],1).cpu().numpy()
                    assert np.isfinite(raw).all()
                    torch.cuda.synchronize();now=time.monotonic();probes.append(now-last);last=now
                    image_rows+=len(x)
                    if j>=10:break
            results[name]=dict(train_batch48_seconds=timings,pointwise_batch48_seconds=probes,
                               nonshared_tensor_bytes=tensor_bytes,probe_rows=len(ds),
                               peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
            write_json(out/'RESOURCE_TIMINGS.json',results)
            del learner,net,ds,loader,x,y,o,raw;gc.collect();torch.cuda.empty_cache()
    projected={}
    for name,x in lock.items():
        tasks=[t for ts in x['runs'].values() for t in ts]
        train_rate=float(np.median(results[name]['train_batch48_seconds'][1:]))
        probe_rate=float(np.median(results[name]['pointwise_batch48_seconds'][1:]))
        steps=2*sum(t['steps'] for t in tasks)
        probes=2*sum(t['probe_batches'] for t in tasks)
        vals=2*sum(t['val_batches'] for t in tasks)
        projected[name]=dict(optimizer_steps=steps,probe_batches=probes,val_batches=vals,
                             train_batch_seconds=train_rate,probe_batch_seconds=probe_rate,
                             train_seconds=steps*train_rate,probe_seconds=probes*probe_rate,
                             val_seconds=vals*probe_rate)
    base=sum(sum(x[k] for k in ('train_seconds','probe_seconds','val_seconds')) for x in projected.values())
    residence=time.monotonic()-began
    # Explicit modest allowance for transport, checkpoints and boundary overhead.
    total=residence+1.1*base+300
    report=dict(status='RESOURCE_GATE_PASS' if total<=43200 else 'BLOCKED_GPU_BUDGET',
                projected=projected,measured_GPU_process_residence_seconds=residence,
                projected_GPU_seconds_without_reserve=base,projected_GPU_seconds_with_reserve=total,
                reserve_fraction=.1,boundary_reserve_seconds=300,GPU_budget_seconds=43200,
                new_engineering_optimizer_steps=training_steps,engineering_image_rows=image_rows,
                formal_training_started=False,full_P0_passed=False,new_test_image_reads=0,
                new_test_feature_reads=0,new_test_model_forwards=0,new_test_predictions=0,
                discarded_smoke_models=True,free_bytes=shutil.disk_usage(cfg['root']).free)
    write_json(out/'EARLY_RESOURCE_ADMISSION.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
