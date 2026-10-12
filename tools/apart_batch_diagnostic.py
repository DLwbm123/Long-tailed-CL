"""Frozen-checkpoint validation only: batch composition, per-query routing and w."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from run_external_medical import (MedicalImages, Observation, make_apart,
    checkpoint_metadata, stage_report, aggregate, VERSION)
from run_module_cycle import save

MODES = ('mixed_batch', 'class_concat', 'individual_routing')
DIAGNOSTIC_VERSION = 'apart_val_batch_w_v1'


def validate(config):
    lock = json.loads(Path(config['diagnostic_lock']).read_text())
    old = config['original_job']
    data = lock['data'][old['dataset']]
    if (config['version'] != DIAGNOSTIC_VERSION or config['modes'] != list(MODES)
            or config['batch_size'] != 48 or config['train_allowed']
            or config['test_accessed'] or old['method'] != 'APART_full'
            or old['source_commit'] != lock['checkpoint_source_commit']
            or old['version'] != VERSION or old['seed'] != 74002
            or old['order'] != data['order'] or old['task_sizes'] != data['task_sizes']
            or old['expected_steps'] != data['APART_total_updates']
            or old['test_accessed'] or old['evaluation_split'] != 'development_validation'):
        raise ValueError('Frozen diagnostic scope differs')
    parent = Path(config['parent_root']).resolve()
    if Path(old['output']).resolve() != parent/'runs'/(old['dataset']+'_APART_full'):
        raise ValueError('Only original APART checkpoints allowed')
    old_config = json.loads((parent/'CONFIG.private.json').read_text())
    canonical = next(j for j in old_config['jobs'] if j['dataset'] == old['dataset'] and j['method'] == 'APART_full')
    if old != canonical:
        raise ValueError('Original inputs must match launched source config exactly')
    status = json.loads((parent/'PROGRAM_STATE.json').read_text())
    if status['status'] != 'COMPLETE' or status['active']:
        raise ValueError('Original campaign must be complete')
    out = Path(config['output']).resolve()
    if out != Path(config['root']).resolve()/'runs'/old['dataset']:
        raise ValueError('Independent diagnostic output required')
    return lock, data


def ordered_rows(rows, mode):
    if mode not in MODES:
        raise ValueError('Unknown evaluation mode')
    return sorted(rows, key=lambda r:r['target']) if mode == 'class_concat' else list(rows)


def describe_w(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all() or np.any((x < 0)|(x > 1)):
        raise ValueError('Invalid w support')
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()),
        minimum=float(x.min()), maximum=float(x.max()),
        quantiles={str(q):float(np.quantile(x,q)) for q in (.05,.25,.5,.75,.95)},
        n_le_001=int((x <= .01).sum()), n_le_005=int((x <= .05).sum()),
        n_ge_095=int((x >= .95).sum()),
        near_zero_screen=bool((x <= .05).mean() >= .9 and x.mean() <= .05))


def paired(a, b, targets, routes_a, routes_b, seen, order):
    values = {}
    for name, mask in [('all',np.ones(len(targets),dtype=bool))]+[
            (str(c), targets == order.index(c)) for c in seen]:
        ca = a[mask] == targets[mask]; cb = b[mask] == targets[mask]
        values[name] = dict(n=int(mask.sum()), predictions_changed=int((a[mask] != b[mask]).sum()),
            rescued=int((~ca & cb).sum()), harmed=int((ca & ~cb).sum()),
            both_correct=int((ca & cb).sum()), both_wrong=int((~ca & ~cb).sum()),
            either_pool_route_changed=int((routes_a[mask] != routes_b[mask]).any(axis=1).sum()))
    return values


class IndexedImages(MedicalImages):
    def __getitem__(self, i):
        _, x, y = super().__getitem__(i)
        return self.rows[i]['canonical_index'], x, y


def run(config):
    started = time.monotonic()
    lock, data = validate(config)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1') or not torch.cuda.is_available():
        raise ValueError('GPU01 only')
    torch.set_num_threads(4)
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=False)
    result = dict(dataset=config['original_job']['dataset'],version=DIAGNOSTIC_VERSION,
        modes={m:[] for m in MODES},paired_stages=[],w_stages=[],
        optimizer_updates=0,train_accessed=False,test_accessed=False)
    def budget():
        if time.time() >= config['deadline']:
            raise TimeoutError('Original diagnostic deadline')
    def state(status, **kw):
        save(out/'STATUS.json',dict(status=status,phase='DIAGNOSTIC_VALIDATION',
            optimizer_updates=0,train_accessed=False,test_accessed=False,**kw))
    state('RUNNING')
    try:
        old=config['original_job']; rows=[]
        # Open only the already-authorized validation manifest, never train/test.
        with (Path(old['manifest'])/'val.csv').open() as f:
            for i,r in enumerate(csv.DictReader(f)):
                label=int(r['original_label']);p=Path(r['relative_path'])
                if (r['split']!='val' or label not in old['order'] or p.is_absolute()
                        or '..' in p.parts or not r['identity_component']):
                    raise ValueError('Validation scope/label/path differs')
                r.update(target=old['order'].index(label),canonical_index=i);rows.append(r)
        counts={str(c):sum(int(r['original_label'])==c for r in rows) for c in old['order']}
        if counts!=data['val_counts'] or len({r['relative_path'] for r in rows})!=len(rows):
            raise ValueError('Frozen validation support differs')
        observation=Observation(budget,lambda *a:None,True)
        learner=make_apart(dict(old,data=data),torch.device('cuda:0'),observation)
        model=learner._network.requires_grad_(False).eval()
        captured={}
        handle=model.original_backbone.register_forward_hook(lambda module,args,value:captured.update(cls=value))
        comparisons=[('mixed_batch','class_concat'),('mixed_batch','individual_routing'),('class_concat','individual_routing')]
        for task in range(1,len(data['task_sizes'])+1):
            budget()
            checkpoint=torch.load(str(Path(old['output'])/f'stage_{task}.pt'),map_location='cpu',weights_only=False)
            expected_steps=sum(t['APART_updates_10epoch'] for t in data['tasks'][:task])
            if checkpoint['metadata']!=checkpoint_metadata(old,task,expected_steps,0):
                raise ValueError('Checkpoint provenance differs')
            model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
            versions={n:p._version for n,p in model.named_parameters()}
            total=sum(data['task_sizes'][:task]);seen=old['order'][:total]
            selected=[r for r in rows if r['target']<total]
            global_to_stage={r['canonical_index']:i for i,r in enumerate(selected)}
            targets=np.array([r['target'] for r in selected])
            predictions={};routes={};ws=np.full(len(selected),np.nan)
            for mode in MODES:
                state('RUNNING',task=task,mode=mode)
                individual=mode=='individual_routing'
                for pool in (model.backbone.pool,model.backbone.pool_few):
                    pool.batchwise_prompt=not individual
                stream=DataLoader(IndexedImages(ordered_rows(selected,mode),old['images'],False,74002),
                    batch_size=48,shuffle=False,num_workers=0,drop_last=False)
                pred=np.full(len(selected),-1,dtype=np.int64)
                route=np.full((len(selected),2),-1,dtype=np.int64)
                with torch.no_grad():
                    for indices,x,y in stream:
                        budget();v=model(x.to('cuda:0'),task_id=task-1)
                        logits=(v['logits']+v['logits_few'])[:,:total]
                        if not torch.isfinite(logits).all() or logits.shape[1]!=total:
                            raise ValueError('Invalid seen logits')
                        ix=np.array([global_to_stage[int(i)] for i in indices])
                        if not np.array_equal(y.numpy(),targets[ix]):
                            raise ValueError('Canonical paired targets differ')
                        pred[ix]=logits.argmax(1).cpu().numpy()
                        route[ix]=torch.cat([v['prompt_idx'],v['prompt_idx_few']],dim=1).cpu().numpy()
                        if mode=='mixed_batch':
                            # True-label frequency is used only for w diagnosis after prediction.
                            freq=torch.tensor([data['train_counts'][str(old['order'][int(t)])] for t in y],device='cuda:0')
                            ws[ix]=model.backbone.forward_pool(captured.pop('cls'),freq).flatten().cpu().numpy()
                        captured.clear()
                if np.any(pred<0) or np.any(route<0):
                    raise ValueError('Incomplete paired predictions')
                if any(p._version!=versions[n] for n,p in model.named_parameters()) or observation.steps:
                    raise ValueError('Diagnostic modified parameters')
                report=stage_report(task,seen,targets,pred,None,old['order'],data['train_counts'])
                if report['per_class_n']!={str(c):data['val_counts'][str(c)] for c in seen}:
                    raise ValueError('Paired validation denominators differ')
                result['modes'][mode].append(report);predictions[mode]=pred;routes[mode]=route
            result['paired_stages'].append(dict(task=task,comparisons={a+'__'+b:
                paired(predictions[a],predictions[b],targets,routes[a],routes[b],seen,old['order']) for a,b in comparisons}))
            known=sum(data['task_sizes'][:task-1])
            groups={'all':np.ones(len(targets),dtype=bool),'new':targets>=known,
                'native_rare_le100':np.array([data['train_counts'][str(old['order'][t])]<=100 for t in targets]),
                'native_common_gt100':np.array([data['train_counts'][str(old['order'][t])]>100 for t in targets])}
            if known:groups['old']=targets<known
            result['w_stages'].append(dict(task=task,
                groups={k:describe_w(ws[mask]) if mask.any() else None for k,mask in groups.items()},
                per_class={str(c):describe_w(ws[targets==old['order'].index(c)]) for c in seen}))
            save(out/'PARTIAL_AGGREGATES.json',result)
        handle.remove()
        tasks=[];at=0
        for n in data['task_sizes']:tasks.append(old['order'][at:at+n]);at+=n
        result['metrics']={m:aggregate(stages,tasks,data['train_counts']) for m,stages in result['modes'].items()}
        result['checkpoint_metadata_all_pass']=True
        result['parameters_unchanged']=True
        result['model_source_commit']=old['source_commit']
        result['diagnostic_source_commit']=config['source_commit']
        result['validation_image_exposures']=3*sum(s['validation_n'] for s in result['modes']['mixed_batch'])
        result['peak_gpu_bytes']=torch.cuda.max_memory_allocated()
        result['worker_wall_seconds']=time.monotonic()-started
        save(out/'RESULTS.json',result);state('COMPLETE',stages=3*len(tasks))
    except BaseException as exc:
        state('INCOMPLETE',error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    campaign=json.loads(Path(os.environ['Q128_CONFIG']).read_text())
    job=campaign['jobs'][int(os.environ['Q157_SLOT'])]
    run(dict(job,root=campaign['root'],source_commit=campaign['source_commit'],deadline=campaign['deadline']))
