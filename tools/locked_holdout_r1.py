"""One locked R1 evaluation. No training entry point; private paths come from runtime locks."""
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import numpy as np
import torch
from torchvision import transforms as T
from run_medical_v2 import ROOT, MedicalLearner, Images, args_for, seed_all, sha, write_json, network_hash, rng_equal
from diagnose_medical_v3 import loader, scored, SEEDS
from stationary_medical_v4 import fork, HEADS
from frozen_medical_v3 import Arrivals, solve_stats, DINO_SHA
from utils.medical_v2 import FrozenOriginal, WEIGHT_SHA
from evaluate_medical_v2 import write_csv

METHODS=('C','H','K','RA','RD')
MANIFESTS={'train':'6c9113aaac70a3d0c976177d28d54fbaa3f0138f0fa74715c225014de6ee8abd','val':'f485459bb5636ef85bf5c023d3c8c7b988df2e51cdada25967e1888ab7581956','test':'11143bd5fece6cdd09168ec2caa00f8bc432663e30e660363af9e37e8e3f3c0c'}
COUNTS=[10529,3263,2835,1306,458,256,44,27]

def read(p):return json.loads(Path(p).read_text())
def csvrows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def tensor_digest(value):
    h=hashlib.sha256()
    def add(x):
        if isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());add(x[k])
        elif torch.is_tensor(x):h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,(list,tuple)):
            for v in x:add(v)
        else:h.update(str(x).encode())
    add(value);return h.hexdigest()
def no_step(*a,**k):raise RuntimeError('R1_FORBIDS_OPTIMIZER_STEP')

def dataset(c,order,stage,split):
    assert split in ('val','test')
    if split=='test':
        release=read(Path(c['out'])/'TEST_RELEASE_R1.json')
        assert release['eval_lock_sha256']==sha(Path(c['out'])/'EVAL_LOCK_R1.json')
    d=Images(Path(c['protocol'])/(split+'.csv'),c['images'],order,range((4,6,8)[stage]),False)
    d.rows.sort(key=lambda r:r['sample_id'])
    assert len({r['sample_id'] for r in d.rows})==len(d)
    return d

def layout(rows):return digest([[r['sample_id'] for r in rows[i:i+48]] for i in range(0,len(rows),48)])

def record_access(c,method,seed,stage,batch,images,kind):
    p=Path(c['out'])/'private/TEST_ACCESS.jsonl'
    with p.open('a') as f:
        f.write(json.dumps(dict(timestamp=time.time(),method=method,seed=seed,stage=stage,batch=batch,images=images,kind=kind))+'\n');f.flush();os.fsync(f.fileno())

def save_npz(path,**arrays):
    path=Path(path);assert not path.exists(),'NO_OVERWRITE '+str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.part')
    with tmp.open('wb') as f:np.savez(f,**arrays)
    tmp.replace(path)

def save_prediction(c,method,seed,stage,split,rows,raw,provenance,main=None,few=None):
    path=Path(c['out'])/'private'/split/f'{method}_{seed}_s{stage}.npz'
    assert np.isfinite(raw).all() and raw.shape==(len(rows),(4,6,8)[stage])
    ids=np.array([r['sample_id'] for r in rows]);y=np.array([r['target'] for r in rows]);order=c['class_orders'][str(seed)]
    values=dict(raw=raw,y=y,ids=ids,original=np.array([int(r['original_label']) for r in rows]),
                component=np.array([r['identity_component'] for r in rows]),lesion=np.array([r['lesion_id'] for r in rows]),order=np.array(order))
    if main is not None:values.update(main=main,few=few)
    save_npz(path,**values)
    p=dict(method=method,seed=seed,stage=stage,split=split,images=len(rows),layout_sha256=layout(rows),manifest_sha256=MANIFESTS[split],
           prediction_sha256=sha(path),path=str(path),**provenance)
    write_json(path.with_suffix('.json'),p)
    return p

def asset_inventory(c):
    out=Path(c['out']);assets=[];entries=[]
    def check(path,expected,kind):
        p=Path(path);assert p.is_file(),'BLOCKED_MISSING_ASSET '+str(p)
        actual=sha(p);assert actual==expected,'BLOCKED_ASSET_SHA '+str(p)
        assets.append(dict(path=str(p),resolved_path=str(p.resolve()),kind=kind,sha256=actual,bytes=p.stat().st_size))
    v2=read(c['v2_runtime']);c['protocol']=v2['protocol'];c['images']=v2['images'];c['weight']=v2['weight'];c['class_orders']=v2['class_orders']
    check(c['weight'],WEIGHT_SHA,'AugReg');check(Path(c['v3_root'])/'weights/model.safetensors',DINO_SHA,'DINOv2')
    for split,n in [('train',18718),('val',295),('test',764)]:
        p=Path(c['protocol'])/(split+'.csv');check(p,MANIFESTS[split],'manifest')
        rs=csvrows(p);assert len(rs)==n and all(r['split']==split for r in rs)
        assert len({r['sample_id'] for r in rs})==n
        if split=='train':assert [sum(int(r['original_label'])==k for r in rs) for k in range(8)]==COUNTS
        groups={}
        for r in rs:
            assert r['identity_component'] and r['lesion_id'],'BLOCKED_GROUP_ID'
            groups.setdefault(r['identity_component'],set()).add(int(r['original_label']))
        assert all(len(g)==1 for g in groups.values()),'BLOCKED_GROUP_LABEL_CONFLICT'
    for e in read(Path(v2['output'])/'ALL_CHECKPOINTS_LOCK.json')['checkpoints']:
        if e['branch']!='C':continue
        check(e['path'],e['sha256'],'C_checkpoint');entries.append(dict(e,method='C'))
    for method,key in [('H','h_runtime'),('K','k_runtime')]:
        cfg=read(c[key]);c[method+'_config']=cfg
        for e in read(Path(cfg['output'])/'CHECKPOINT_LOCK.json')['sessions']:
            check(e['path'],e['sha256'],method+'_delta');seed=int(Path(e['path']).parent.name.split('_')[0])
            entries.append(dict(e,seed=seed,branch=method,method=method))
    assert len(entries)==21
    for encoder in ('A','B'):
        d=Path(c['v3_complete_output'])/'p1'/encoder;lock=read(d/'CACHE_LOCK.json')
        for split in ('train','val'):
            check(d/(split+'.npz'),lock['splits'][split]['cache_sha256'],'V3_feature_cache')
            check(d/(split+'_associations.json'),lock['splits'][split]['associations_sha256'],'V3_cache_association')
            assert lock['splits'][split]['manifest_sha256']==MANIFESTS[split]
        if encoder=='B':
            for name,h in lock['metadata_sha256'].items():check(Path(c['v3_root'])/'weights'/name,h,'DINO_metadata')
    c['entries']=entries
    write_json(out/'ASSET_INVENTORY.json',{'status':'FILE_HASH_PASS','assets':assets,'new_neural_epochs':0,'new_optimizer_steps':0})
    write_json(out/'runtime_resolved.json',c)
    return c

def restore(c,e):
    seed=e['seed'];method=e['method'];stage=e['session'];out=Path(c['out'])/'private/restore'/f'{method}_{seed}'
    if method=='C':
        cfg=read(c['v2_runtime']);a,order=args_for(cfg,seed,'C',False);seed_all(seed)
        l=MedicalLearner(a,cfg,order,out);state=l.restore(e['path'])
    else:
        l=fork(c[method+'_config'],seed,out=out);state=l.restore(e['path']);l.assert_stationary()
    assert (state['task'],state['known'],state['total'],state['epoch'],state['phase'])==(stage,(0,4,6)[stage],(4,6,8)[stage],10,'session_complete')
    if method=='C':expected=read(Path(c['v3_output'])/'p0'/f'{seed}_C_s{stage}.json')['network_hash']
    else:expected=read(Path(e['path']).with_name(f'session{stage}_training.json'))['network_sha256']
    actual=network_hash(l._network);assert actual==expected,'BLOCKED_RECONSTRUCTED_NETWORK'
    l._network.eval();l._network.requires_grad_(False)
    lineage=dict(method=method,seed=seed,stage=stage,checkpoint_sha256=e['sha256'],network_sha256=actual,
                 parent=state.get('immutable_parent'),source_commit=state['config']['code_commit'],head_mapping=l.order[:l._total_classes],
                 strict_restore='PASS',reconstructor_sha256=sha(ROOT/'tools/stationary_medical_v4.py'),
                 inference_bytes=sum(v.numel()*v.element_size() for v in l._network.state_dict().values()),
                 incremental_memory_bytes=sum(v.numel()*v.element_size() for m in l.concm_stage1_memory.values() for v in m.values() if torch.is_tensor(v)))
    del state
    return l,lineage

def original_val(c,method,seed,stage):
    if method=='C' or stage==0:return read(Path(c['v3_output'])/'p0'/f'{seed}_C_s{stage}.json')['diagnostics']['val']
    return read(Path(c[method+'_config']['output'])/f'{seed}_{method}'/f'session{stage}_val.json')

def check_reproduction(raw,rows,order,stage,ref):
    y=np.array([r['target'] for r in rows]);m,pc=scored(raw,y,rows,order,(0,4,6)[stage])
    target={int(r['original_label']):r for r in ref['per_class'] if r.get('head','sum')=='sum'}
    assert len(target)==len(pc)
    for r in pc:
        assert r['n_images']==int(target[r['original_label']]['n_images'])
        assert round(r['recall'],3)==round(float(target[r['original_label']]['recall']),3),'BLOCKED_REPRODUCIBILITY_PER_CLASS'
    rm=next(r for r in ref['metrics'] if r.get('head','sum')=='sum')
    for k in ('balanced_accuracy','old_macro_recall','current_macro_recall','tail_rank2'):
        if m[k] is None:assert rm[k] in (None,'')
        else:assert round(m[k],3)==round(float(rm[k]),3),'BLOCKED_REPRODUCIBILITY_'+k
    return {'balanced_accuracy':m['balanced_accuracy'],'old_macro_recall':m['old_macro_recall'],'current_macro_recall':m['current_macro_recall'],'tail_rank2':m['tail_rank2'],
            'worst_class_recall':min(r['recall'] for r in pc),'status':'PASS','comparison':'all class recalls and main aggregate metrics round-to-3; historical neural logits unavailable'}

@torch.inference_mode()
def neural(c,e,split):
    l,line=restore(c,e);seed=e['seed'];stage=e['session'];method=e['method'];d=dataset(c,l.order,stage,split)
    h=network_hash(l._network);mh=tensor_digest(l.concm_stage1_memory);rng=l._capture_rng_state();sy=l.synth_rng.clone();lr=l.loader_generator.get_state().clone()
    a=[];b=[];timing=None;start=time.monotonic()
    for bi,(_,x,_) in enumerate(loader(d)):
        x=x.cuda();o=l._network(x,train=False)
        if split=='test':record_access(c,method,seed,stage,bi,len(x),'neural_forward')
        aa=o['logits'][:,:l._total_classes];bb=o['logits_few'][:,:l._total_classes]
        assert torch.isfinite(aa).all() and torch.isfinite(bb).all();a.append(aa.cpu().numpy());b.append(bb.cpu().numpy())
        if split=='val' and seed==1993 and stage==2 and bi==0:
            # Same images, swapped frequency/label proxies: prediction cannot use ground truth.
            p=l._network(x,train=False,weight=torch.ones(len(x),device='cuda'))
            q=l._network(x,train=False,weight=torch.full((len(x),),10529,device='cuda'))
            for k in ('logits','logits_few'):assert torch.equal(o[k],p[k]) and torch.equal(o[k],q[k]),'BLOCKED_LABEL_DEPENDENCE'
            timing=benchmark(lambda:l._network(x,train=False))
    a=np.concatenate(a);b=np.concatenate(b);raw=a+b
    assert network_hash(l._network)==h and tensor_digest(l.concm_stage1_memory)==mh,'BLOCKED_EVAL_STATE_MUTATION'
    assert rng_equal(rng,l._capture_rng_state()) and torch.equal(sy,l.synth_rng) and torch.equal(lr,l.loader_generator.get_state()),'BLOCKED_EVAL_RNG_MUTATION'
    line.update(eval_state_unchanged=True,rng_unchanged=True,seconds=time.monotonic()-start,latency=timing)
    if split=='val':line['reproduction']=check_reproduction(raw,d.rows,l.order,stage,original_val(c,method,seed,stage))
    prov=save_prediction(c,method,seed,stage,split,d.rows,raw,dict(network_sha256=h,checkpoint_sha256=e['sha256'],precision='float32; no AMP',kind='new_neural_forward',**({'eval_lock_sha256':sha(Path(c['out'])/'EVAL_LOCK_R1.json')} if split=='test' else {})),main=a,few=b)
    del l,o,x;gc.collect();torch.cuda.empty_cache()
    return line,prov

@torch.inference_mode()
def benchmark(call):
    for _ in range(5):call()
    torch.cuda.synchronize();ts=[]
    for _ in range(20):
        t=time.perf_counter();call();torch.cuda.synchronize();ts.append(time.perf_counter()-t)
    return dict(batch=48,warmup=5,measured=20,median_seconds=float(np.median(ts)),mean_seconds=float(np.mean(ts)),cache='fixed decoded tensor on GPU; forward only')

def ridge_reconstruct(c):
    out=Path(c['out']);audits=[];lineage=[]
    for method,encoder in [('RA','A'),('RD','B')]:
        folder=Path(c['v3_complete_output'])/'p1'/encoder
        with np.load(folder/'train.npz') as f:z=f['z'];y=f['y']
        with np.load(folder/'val.npz') as f:v=f['z'].astype(np.float64);vy=f['y']
        rows=read(folder/'val_associations.json');refs=csvrows(folder/'frozen_val_metrics.csv');pcs=csvrows(folder/'frozen_per_class_metrics.csv')
        finals=[]
        for seed in SEEDS:
            order=c['class_orders'][str(seed)];arrival=Arrivals(z,y,order);stats={}
            for stage,seen in enumerate((4,6,8)):
                for label,current in arrival.next():stats[label]={'n':len(current),'s':current.sum(0,dtype=np.float64),'Q':current.T@current}
                assert set(stats)==set(order[:seen])
                w,_,diag=solve_stats(stats,order[:seen]);mask=np.isin(vy,order[:seen]);rs=[dict(rows[i],target=order.index(int(vy[i]))) for i in np.where(mask)[0]]
                assert [r['sample_id'] for r in rs]==sorted(r['sample_id'] for r in rs)
                raw=v[mask]@w
                ref={'metrics':[r for r in refs if r['classifier']=='CBRidge' and int(r['order_seed'])==seed and int(r['session'])==stage],
                     'per_class':[r for r in pcs if r['classifier']=='CBRidge' and int(r['order_seed'])==seed and int(r['session'])==stage]}
                rep=check_reproduction(raw,rs,order,stage,ref)
                path=out/'private/classifiers'/f'{method}_{seed}_s{stage}.npz';save_npz(path,W=w,order=np.array(order[:seen]))
                prov=save_prediction(c,method,seed,stage,'val',rs,raw,dict(classifier_sha256=sha(path),kind='existing_val_features_exact_original_recipe'))
                audits.append(dict(method=method,seed=seed,stage=stage,reproduction=rep,current_access=arrival.access[-1],stats_classes=list(stats),**diag))
                lineage.append(dict(method=method,seed=seed,stage=stage,classifier_path=str(path),classifier_sha256=sha(path),status='ANALYTIC_RECONSTRUCTION',
                     train_cache_sha256=read(folder/'CACHE_LOCK.json')['splits']['train']['cache_sha256'],lambda_value=1e-3,bias=False,statistics_dtype='float64',statistic_bytes=sum(s['s'].nbytes+s['Q'].nbytes+8 for s in stats.values())))
            finals.append(w[:,[order.index(k) for k in range(8)]])
        for w in finals[1:]:
            assert np.allclose(w,finals[0],atol=1e-10,rtol=1e-9)
            assert np.array_equal((v@w).argmax(1),(v@finals[0]).argmax(1))
        # Reuse the pinned same-cache V3 offline streaming/batch equivalence, never fit val/test.
        engineering=read(folder/'engineering.json');assert engineering['equivalence']['batch_stream_max_abs']<1e-10
        write_json(out/f'{method}_ANALYTIC_AUDIT.json',dict(status='PASS',equivalence_source_sha256=sha(folder/'engineering.json'),
            existing_same_cache_batch_stream_equivalence=engineering['equivalence'],final_order_invariance=True,fit_split='train',fit_complete_at=time.time()))
    return audits,lineage

@torch.inference_mode()
def encoder(c,method,split):
    assert method in ('RA','RD');weight=Path(c['weight']) if method=='RA' else Path(c['v3_root'])/'weights/model.safetensors'
    if method=='RA':model=FrozenOriginal(weight).cuda().eval();mean=std=[.5]*3
    else:
        from transformers import Dinov2Model
        model=Dinov2Model.from_pretrained(str(weight.parent),local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').cuda().eval()
        mean=[.485,.456,.406];std=[.229,.224,.225]
    model.requires_grad_(False);h=network_hash(model);rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    d=dataset(c,list(range(8)),2,split);d.transform=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.ToTensor(),T.Normalize(mean,std)])
    def forward(x):
        f=model(x)['pre_logits'] if method=='RA' else model(pixel_values=x).last_hidden_state[:,0]
        return torch.nn.functional.normalize(f,dim=1)
    parts=[];timing=None
    for bi,(_,x,_) in enumerate(loader(d)):
        x=x.cuda();parts.append(forward(x).cpu().numpy())
        if split=='test':record_access(c,method,None,None,bi,len(x),'encoder_forward')
        if split=='val' and bi==0:timing=benchmark(lambda:forward(x))
    z=np.concatenate(parts);assert np.isfinite(z).all() and z.shape==(len(d),768)
    assert network_hash(model)==h and torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
    if split=='val':
        folder=Path(c['v3_complete_output'])/'p1'/('A' if method=='RA' else 'B')
        with np.load(folder/'val.npz') as f:old=f['z']
        assert np.allclose(z,old,atol=1e-6,rtol=1e-5),'BLOCKED_ENCODER_REPRODUCIBILITY'
        result=dict(method=method,status='PASS',max_abs_feature_difference=float(np.abs(z-old).max()),atol=1e-6,rtol=1e-5,latency=timing,
                    inference_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),network_sha256=h)
        write_json(Path(c['out'])/(method+'_ENCODER_AUDIT.json'),result)
    else:
        save_npz(Path(c['out'])/'private/test'/f'{method}_features.npz',z=z,ids=np.array([r['sample_id'] for r in d.rows]))
        for seed in SEEDS:
            order=c['class_orders'][str(seed)]
            for stage,seen in enumerate((4,6,8)):
                p=Path(c['out'])/'private/classifiers'/f'{method}_{seed}_s{stage}.npz'
                with np.load(p) as f:w=f['W'];assert f['order'].tolist()==order[:seen]
                idx=[i for i,r in enumerate(d.rows) if int(r['original_label']) in order[:seen]]
                rs=[dict(d.rows[i],target=order.index(int(d.rows[i]['original_label']))) for i in idx]
                if stage==2 and seed!=1993:
                    with np.load(Path(c['out'])/'private/test'/f'{method}_1993_s2.npz') as f:
                        original_order=f['order'].tolist();raw=f['raw'][:,[original_order.index(k) for k in order]]
                else:raw=z[idx].astype(np.float64)@w
                save_prediction(c,method,seed,stage,split,rs,raw,dict(network_sha256=h,classifier_sha256=sha(p),kind='fixed_native_test_features; stage-specific locked ridge',
                            eval_lock_sha256=sha(Path(c['out'])/'EVAL_LOCK_R1.json'),shared_final_reference=stage==2))
    del model,x;gc.collect();torch.cuda.empty_cache()


def resources(start):
    return dict(process_seconds=time.monotonic()-start,cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)

def preflight(c):
    out=Path(c['out']);assert not (out/'TEST_RELEASE_R1.json').exists();start=time.monotonic();torch.set_num_threads(4)
    torch.optim.AdamW.step=no_step
    c=asset_inventory(c);lineage=[];audit=[]
    for e in c['entries']:
        print('VAL',e['method'],e['seed'],e['session'],flush=True)
        line,prov=neural(c,e,'val');lineage.append(line);audit.append(dict(method=e['method'],seed=e['seed'],stage=e['session'],**line['reproduction']))
        assert time.monotonic()-start<5400,'BLOCKED_E1_BUDGET'
        write_json(out/'E1_PROGRESS.json',dict(completed=len(audit),last=audit[-1],test_predictions=0))
    # Shared S0: exact parent network, mapping and layout. Predictions are references, not extra forwards.
    for method in ('H','K'):
        for seed in SEEDS:
            original=next(x for x in lineage if (x['method'],x['seed'],x['stage'])==('C',seed,0))
            lineage.append(dict(original,method=method,shared_with='C',latency=None))
            v=next(x for x in audit if (x['method'],x['seed'],x['stage'])==('C',seed,0));audit.append(dict(v,method=method,shared_with='C'))
    ra,rl=ridge_reconstruct(c);lineage+=rl;audit += [dict(method=r['method'],seed=r['seed'],stage=r['stage'],**r['reproduction']) for r in ra]
    for method in ('RA','RD'):encoder(c,method,'val')
    assert len(audit)==45
    resource_record=resources(start);remaining_estimate=resource_record['process_seconds']*3+120
    assert resource_record['process_seconds']+remaining_estimate<=7200,'BLOCKED_R1_BUDGET'
    assert shutil.disk_usage(out).free>512*1024**2+100*1024**2,'BLOCKED_DISK'
    env={p:importlib.metadata.version(p) for p in ('torch','torchvision','timm','transformers','numpy','safetensors','pillow')}
    env.update(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
               matmul_precision=torch.get_float32_matmul_precision(),amp=False,dtype='float32',device=torch.cuda.get_device_name(),cuda=torch.version.cuda)
    write_json(out/'ENVIRONMENT_R1.json',env);write_json(out/'MODEL_LINEAGE.json',dict(models=lineage,status='PASS'))
    write_json(out/'REPRODUCTION_AUDIT.json',dict(status='PASS',rows=audit,logical_rows=45,new_test_predictions=0,
         neural_logits_historical_availability=False,neural_equality='per-class and aggregate round-to-3',ridge_source='same pinned feature cache and exact V3 analytic objective',
         label_blind='network receives images and shared stage only; frequency proxy swap exact',new_neural_epochs=0,new_optimizer_steps=0))
    write_json(out/'RESOURCE_E1.json',dict(**resource_record,conservative_remaining_seconds=remaining_estimate,budget_seconds=7200))
    write_json(out/'E1_COMPLETE.json',dict(status='PASS',timestamp=time.time(),test_released=False))
    print('E1_COMPLETE',resource_record,flush=True)


def lock_and_release(c):
    out=Path(c['out']);assert read(out/'E1_COMPLETE.json')['status']=='PASS'
    assert shutil.disk_usage(out).free>512*1024**2+100*1024**2,'BLOCKED_RELEASE_DISK'
    assert sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file())<1024**3,'BLOCKED_NEW_FILE_BUDGET'
    assert read(out/'ENGINEERING_R1.json')['status']=='PASS'
    source={str(p.relative_to(ROOT)):sha(p) for folder in ('tools','third_party/APART') for p in (ROOT/folder).rglob('*.py')}
    assert (ROOT/'tools/report_locked_holdout_r1.py').exists(),'ANALYSIS_MUST_BE_FROZEN'
    private_layout={}
    for split in ('val','test'):
        rows=csvrows(Path(c['protocol'])/(split+'.csv'))
        private_layout[split]={}
        for seed in SEEDS:
            order=c['class_orders'][str(seed)]
            for stage,seen in enumerate((4,6,8)):
                rs=sorted((r for r in rows if int(r['original_label']) in order[:seen]),key=lambda r:r['sample_id'])
                private_layout[split][f'{seed}_{stage}']={'layout_sha256':layout(rs),'ids':[r['sample_id'] for r in rs],'images':len(rs)}
    write_json(out/'private/LAYOUT.json',private_layout)
    files=['ASSET_INVENTORY.json','MODEL_LINEAGE.json','REPRODUCTION_AUDIT.json','ENVIRONMENT_R1.json','RESOURCE_E1.json','RA_ANALYTIC_AUDIT.json','RD_ANALYTIC_AUDIT.json','RA_ENCODER_AUDIT.json','RD_ENCODER_AUDIT.json','ENGINEERING_R1.json']
    lock=dict(protocol='R1_LOCKED_HISTORICALLY_EXPOSED_HOLDOUT',timestamp=time.time(),source_commit=c['source_commit'],code_sha256=source,
        evidence_sha256={f:sha(out/f) for f in files},manifest_sha256=MANIFESTS,methods=METHODS,orders=c['class_orders'],stages=[0,1,2],seen=[4,6,8],
        primary_predictor={'C':'raw_sum','H':'raw_sum','K':'raw_sum','RA':'original_V3_CBRidge','RD':'original_V3_CBRidge'},
        primary_endpoint='Final BA',primary_contrast='K-C',secondary_contrasts=['K-H','K-RA','K-RD'],
        main_rows=45,per_class_rows=270,layout={s:{k:{a:b for a,b in v.items() if a!='ids'} for k,v in d.items()} for s,d in private_layout.items()},
        batch_size=48,drop_last=False,sort='sample_id ascending after seen-class filter',tie_rule='first head index argmax in locked order',
        preprocessing={'C_H_K_RA':'RGB 224 square bicubic antialias mean/std .5; RA sample L2 CLS','RD':'same geometry; ImageNet mean/std; sample L2 CLS'},
        metrics='report_locked_holdout_r1.py frozen hash; image-within-class recall is primary',frequency_groups={'head':[0,1],'mid':[2,3,4,5],'tail':[6,7]},
        bootstrap={'unit':'class-stratified identity_component','resamples':2000,'seed':91001,'interval':'percentile 2.5/97.5','training_seeds':'fixed, never resampled','paired_weights':'shared methods/seeds/stages'},
        gates={'mean_final_ba_delta_gt':0,'mean_final_current_delta_ge':10,'mean_final_old_delta_ge':-5,'paired_ba_current_improvements_ge':2,'new_current_zero_recall':0},
        new_neural_epochs=0,new_optimizer_steps=0,existing_holdout_exposed_by_V2=True,independent_confirmation=False,
        gpu_process_budget_seconds=7200,workers=1,new_file_budget_bytes=1024**3,min_free_bytes=512*1024**2,
        numerical_tolerance={'neural_metrics':'round to 3 decimals','encoder_features_atol':1e-6,'encoder_features_rtol':1e-5,'ridge_atol':1e-10,'ridge_rtol':1e-9},
        c_historical_predictions='not reused: historical unsorted manifest and task_id call lack this layout provenance; approved historical-control reevaluation',
        final_ridge='deterministic reference; independent of class arrival order within verified tolerance; no artificial training SD',
        release_scope='one complete fixed matrix, technical recovery incomplete units only',further_run_authorized=False)
    assert not (out/'EVAL_LOCK_R1.json').exists();write_json(out/'EVAL_LOCK_R1.json',lock)
    write_json(out/'TEST_RELEASE_R1.json',dict(timestamp=time.time(),eval_lock_sha256=sha(out/'EVAL_LOCK_R1.json'),scope=lock['release_scope'],authorized=True))
    assert read(out/'TEST_RELEASE_R1.json')['timestamp']>lock['timestamp']


def test(c):
    out=Path(c['out']);start=time.monotonic();torch.set_num_threads(4);torch.optim.AdamW.step=no_step
    lock=read(out/'EVAL_LOCK_R1.json');assert read(out/'TEST_RELEASE_R1.json')['eval_lock_sha256']==sha(out/'EVAL_LOCK_R1.json')
    for name,h in lock['code_sha256'].items():assert sha(ROOT/name)==h,'BLOCKED_RELEASE_CODE_DRIFT'
    records=[]
    for e in c['entries']:
        key=f"{e['method']}_{e['seed']}_s{e['session']}";p=out/'private/test'/f'{key}.json'
        if p.exists():
            prov=read(p);assert prov['eval_lock_sha256']==sha(out/'EVAL_LOCK_R1.json') and sha(prov['path'])==prov['prediction_sha256']
        else:
            print('TEST',key,flush=True);_,prov=neural(c,e,'test')
        records.append(prov)
        assert resources(start)['process_seconds']+read(out/'RESOURCE_E1.json')['process_seconds']<7200,'BLOCKED_GPU_BUDGET'
        write_json(out/'E2_PROGRESS.json',dict(complete_neural_units=len(records),new_neural_forward_images=sum(x['images'] for x in records)))
    for method in ('RA','RD'):
        marker=out/(method+'_TEST_COMPLETE.json')
        if not marker.exists():
            print('TEST',method,flush=True);encoder(c,method,'test');write_json(marker,dict(status='PASS',timestamp=time.time()))
    # Explicit logical references avoid duplicated S0 blobs.
    references={f'{m}_{s}_s0':f'C_{s}_s0' for m in ('H','K') for s in SEEDS}
    write_json(out/'private/SHARED_REFERENCES.json',references)
    # Protocol explicitly requires asset files to remain unchanged across evaluation.
    for asset in read(out/'ASSET_INVENTORY.json')['assets']:
        assert sha(asset['path'])==asset['sha256'],'BLOCKED_POST_EVAL_ASSET_DRIFT'
    logical_images=sum(lock['layout']['test'][f'{s}_{t}']['images'] for m in METHODS for s in SEEDS for t in range(3))
    neural_images=sum(x['images'] for x in records);reference_images=sum(lock['layout']['test'][f'{s}_0']['images'] for m in ('H','K') for s in SEEDS)
    ridge_scores=sum(lock['layout']['test'][f'{s}_{t}']['images'] for m in ('RA','RD') for s in SEEDS for t in range(3))
    accesses=[json.loads(x) for x in (out/'private/TEST_ACCESS.jsonl').read_text().splitlines()]
    assert sum(x['images'] for x in accesses if x['kind']=='neural_forward')==neural_images
    assert sum(x['images'] for x in accesses if x['kind']=='encoder_forward')==2*764
    write_json(out/'PREDICTION_USAGE_R1.json',dict(access_journal_sha256=sha(out/'private/TEST_ACCESS.jsonl'),actual_forward_batches=len(accesses),historical_prediction_reuse_records=0,new_raw_neural_forward_records=neural_images,
        new_encoder_image_accesses=2*764,new_image_forward_accesses_total=neural_images+2*764,within_R1_shared_S0_logical_records=reference_images,
        derived_ridge_score_records=ridge_scores,logical_prediction_records=logical_images,unique_test_images=764,logical_main_rows=45,
        new_test_access_occurred=True,evaluation_activities=1,new_neural_epochs=0,new_optimizer_steps=0))
    write_json(out/'RESOURCE_E2.json',resources(start));write_json(out/'E2_COMPLETE.json',dict(status='PASS',timestamp=time.time()))
    print('E2_COMPLETE',flush=True)

if __name__=='__main__':
    c=read(os.environ['R1_CONFIG']);phase=os.environ['R1_PHASE'];out=Path(c['out']);out.mkdir(parents=True,exist_ok=True)
    try:
        if phase=='preflight':preflight(c)
        elif phase=='release':lock_and_release(read(out/'runtime_resolved.json'))
        elif phase=='test':test(read(out/'runtime_resolved.json'))
        else:raise ValueError('Unknown phase')
    except Exception as error:
        write_json(out/f'BLOCKED_{phase}_{int(time.time())}.json',dict(status='BLOCKED',phase=phase,error=repr(error),test_released=(out/'TEST_RELEASE_R1.json').exists()))
        raise
