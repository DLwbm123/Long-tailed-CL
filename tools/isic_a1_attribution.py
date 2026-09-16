"""A1 fixed S0 readout attribution. Runtime paths stay outside the repository.

Inference restores the complete network with strict metadata/tensor checks but
does not construct an optimizer or the training learner. The only routing
changes are the two explicitly authorized batchwise_prompt flags.
"""
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from isic_g1_controlled_mixup import Increment, moments, solve as original_solve, close, sha, digest
from report_locked_holdout_r1 import read, write, csvwrite, analyze_unit, predict_columns

ROOT=Path(__file__).resolve().parents[1]
CORE=('A-U','A-CB','S-J-U','S-J-CB')
METHODS=CORE+('S-M-U','S-M-CB','S-F-U','S-F-CB')
FIXED=dict(neural_training_epochs=0,optimizer_steps=0,new_S0_training=0,
           new_test_predictions=0,new_test_feature_reads=0,new_test_image_reads=0,
           routing_mode='S0_POINTWISE_PROBE',routing_configuration_changed=True,
           checkpoint_tensor_parameters_changed=False,validation_adaptively_reused=True,
           independent_confirmation=False,full_GSR_reproduction=False,further_experiments_started=False)
SOLVE_CALLS=0


def solve(G,R):
    global SOLVE_CALLS
    SOLVE_CALLS+=1
    return original_solve(G,R)


def view(raw,name):
    raw=np.asarray(raw,dtype=np.float64)
    z=raw[:,0] if name=='M' else raw[:,1] if name=='F' else raw.reshape(len(raw),1536)
    norm=np.linalg.norm(z,axis=1)
    assert np.isfinite(z).all() and np.all(norm>0),'BLOCKED_FEATURE_NORM'
    return z/norm[:,None]


def duplicate_embedding(G,R,W,z):
    g=np.block([[G,G],[G,G]])/2
    r=np.concatenate([R,R])/np.sqrt(2);w=np.concatenate([W,W])/np.sqrt(2)
    residual=float(np.linalg.norm((g+.001*np.eye(len(g)))@w-r)/np.linalg.norm(r))
    assert residual<1e-10
    scores=z@W;dupscores=(np.concatenate([z,z],1)/np.sqrt(2))@w
    return dict(score_max_abs=close(scores,dupscores),weight_norm_difference=close(np.array([np.linalg.norm(w)]),np.array([np.linalg.norm(W)])),
                W_structure_verified=True,solve_relative_residual=residual,score_ties=int(np.sum(np.sum(scores==scores.max(1,keepdims=True),axis=1)>1)))


def engineering(private):
    rng=np.random.default_rng(43);xs=[rng.normal(size=(n,7)) for n in (9,5,4)];checks=[]
    for kind in ('U','CB'):
        inc=Increment([2,0,7],7,'R0' if kind=='U' else 'B0');allx=[];y=[];weights=[]
        for j,x in enumerate(xs):
            inc.arrive([2,0,7][j],moments(x));allx.extend(x);y.extend([j]*len(x));weights.extend([1 if kind=='U' else 1/len(x)]*len(x))
            if j==0:
                inc.save(private/'toy_state.npz');inc=Increment.restore(private/'toy_state.npz')
        G,R=inc.system();W,res=solve(G,R);X=np.array(allx);w=np.array(weights,dtype=float);w/=w.sum()
        close(G,X.T@(w[:,None]*X));close(R,X.T@(w[:,None]*np.eye(3)[y]))
        # Repeated samples and an independent direct solve in twice the dimension.
        repeated=Increment([2,0,7],7,inc.method)
        for c,x in zip([2,0,7],xs):repeated.arrive(c,moments(np.repeat(x,2,axis=0)))
        close(solve(*repeated.system())[0],W)
        dup=np.concatenate([X,X],1)/np.sqrt(2);dw,_=solve(dup.T@(w[:,None]*dup),dup.T@(w[:,None]*np.eye(3)[y]))
        close(dw,np.concatenate([W,W])/np.sqrt(2));close(dup@dw,X@W)
        checks.append(dict(kind=kind,**duplicate_embedding(G,R,W,X)))
    (private/'toy_state.npz').unlink()
    inc=Increment([2,0,7],7,'B0')
    try:inc.arrive(7,moments(xs[0]))
    except AssertionError:pass
    else:raise AssertionError('FUTURE_ACCESS_NOT_REJECTED')
    try:Run.images_at(None,'test',[])
    except AssertionError:pass
    else:raise AssertionError('TEST_SPLIT_NOT_REJECTED')
    raw=np.array([[1.,1.,0.],[0.,2.,2.]])
    assert np.array_equal(predict_columns(raw,np.array([7,2,0])),[1,2])
    sample=rng.normal(size=(4,2,768));j=view(sample,'J');assert np.max(np.abs(np.linalg.norm(j,axis=1)-1))<1e-12
    assert np.linalg.norm(j[:,:768].T@j[:,768:])>0
    return dict(status='PASS',toy_duplicate_controls=checks,future_class_rejected=True,test_split_rejected=True,tie_smallest_original_label=True,full_joint_cross_block=True,restore_then_next_arrival=True)


class Run:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root']);self.out=self.root/'output';self.pub=self.out/'public';self.private=self.out/'private'
        for p in (self.pub,self.private):p.mkdir(parents=True,exist_ok=True)
        self.protocol=read(ROOT/'exps/isic_a1_protocol.json');self.orders={int(k):v for k,v in self.protocol['class_orders'].items()}
        self.prior=cfg.get('prior_resources',{})
        self.start=time.monotonic();self.cpu_seconds=self.prior.get('analytic_CPU_seconds',0.);self.gpu_start=None;self.gpu_seconds=0.;self.solves=0;self.reused_fits=0
        if cfg.get('linear_native_per_image'):
            self.protocol['numerical_implementation']='Native FP32 Linear per-image GEMM at 3D B>1; network batch unchanged; native B1 untouched'
        self.data={};self.metrics=[];self.pc=[];self.errors=[];self.numeric=[];self.duplicates=[];self.features=[];self.parent_locks=[];self.parity=[]
        self.access=dict(wrapper_batch_calls=0,wrapper_image_rows=0,new_train_val_feature_rows=0,image_reads={'probe':0,'formal':0},
            module_batch_calls={m:0 for m in ('original','main','few')},module_image_rows={m:0 for m in ('original','main','few')},
            phase_calls={},formal_arrivals=[],analytic_train_rows=0,fit_val_rows=0,train_forensics_rows=0)
        previous_access=cfg.get('prior_access',{})
        for key in self.access:
            if key in previous_access:self.access[key]=json.loads(json.dumps(previous_access[key]))
        assert self.access['new_train_val_feature_rows']==0 and not self.access['formal_arrivals'],'BLOCKED_DUPLICATE_EXTRACTION'
        self.phase='probe';self.peak_files=0;self.min_free=shutil.disk_usage(self.root).free;self.peak_allocated=0
        self.verified_parents=set();self.blocked=[]

    def resource_check(self,enforce=True):
        storage_root=Path(self.cfg.get('storage_root',self.root))
        size=sum(p.stat().st_size for p in storage_root.rglob('*') if p.is_file() and not p.is_symlink())
        free=shutil.disk_usage(self.root).free;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        self.peak_files=max(self.peak_files,size);self.min_free=min(self.min_free,free)
        if enforce:
            assert size<640*1024**2 and free>=1024**3,'BLOCKED_RESOURCE_DISK'
            assert rss<8*1024**3 and self.cpu_seconds+self.prior.get('CPU_test_budget_reserve_seconds',0)<7200,'BLOCKED_RESOURCE_CPU'
        if self.gpu_start is not None:
            import torch
            self.peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated())
            if enforce:assert self.peak_allocated<8*1024**3 and self.prior.get('GPU_budget_seconds',0)+time.monotonic()-self.gpu_start<7200,'BLOCKED_RESOURCE_GPU'
        worker_gpu=self.gpu_seconds if self.gpu_start is None else time.monotonic()-self.gpu_start
        return dict(wall_seconds=time.monotonic()-self.start,analytic_CPU_seconds=self.cpu_seconds,
            CPU_budget_seconds=self.cpu_seconds+self.prior.get('CPU_test_budget_reserve_seconds',0),
            GPU_process_residence_seconds=self.prior.get('GPU_observed_seconds',0)+worker_gpu,
            worker_GPU_process_residence_seconds=worker_gpu,GPU_budget_seconds=self.prior.get('GPU_budget_seconds',0)+worker_gpu,
            peak_RSS_bytes=max(rss,self.prior.get('peak_RSS_bytes',0)),peak_GPU_allocated_bytes=max(self.peak_allocated,self.prior.get('peak_GPU_allocated_bytes',0)),new_file_bytes_observed=self.peak_files,min_free_bytes=self.min_free,
            threads=4,workers=1,loader_workers=0,batch_size=48,analytic_fits=self.solves+self.reused_fits,new_analytic_fits=self.solves,reused_A_fits=self.reused_fits,
            total_solve_calls_including_engineering=SOLVE_CALLS+self.prior.get('solve_calls',0))

    def audit(self):
        r1=read(self.cfg['r1_runtime']);self.v2=read(r1['v2_runtime']);self.images=Path(r1['images']);self.cache=Path(r1['v3_complete_output'])/'p1/A'
        self.entries=[e for e in r1['entries'] if e['method']=='C' and e['session']==0];assert len(self.entries)==3
        assert self.v2['code_commit']=='6d8bd3e700924c864e229c47efe5f108ce728c78'
        old=read(ROOT/'docs/isic_locked_holdout_r1/MODEL_LINEAGE.json')['models']
        self.expected={r['seed']:r for r in old if r['method']=='C' and r['stage']==0}
        lock=read(self.cache/'CACHE_LOCK.json');g1=read(ROOT/'docs/isic_g1_controlled_spherical_mixup/ASSET_AND_CACHE_AUDIT.json');audit={}
        for split in ('train','val'):
            sp=lock['splits'][split];files=[self.cache/(split+'.npz'),self.cache/(split+'_associations.json'),Path(r1['protocol'])/(split+'.csv')]
            hashes=[sp['cache_sha256'],sp['associations_sha256'],sp['manifest_sha256']]
            assert hashes==[g1['splits'][split][k] for k in ('cache_sha256','associations_sha256','manifest_sha256')]
            assert sp['manifest_sha256']==self.protocol['manifest_sha256'][split]
            for p,h in zip(files,hashes):assert sha(p)==h,'BLOCKED_FEATURE_CACHE'
            rows=read(files[1]);ids=[r['sample_id'] for r in rows]
            manifest={r['sample_id']:r for r in csv.DictReader(files[2].open())};assert set(manifest)==set(ids)
            with np.load(files[0],allow_pickle=False) as f:z=f['z'].copy();y=f['y'].copy()
            assert ids==sorted(ids) and len(ids)==len(set(ids))==sp['n'] and z.shape==(sp['n'],768) and z.dtype==np.float32
            assert np.array_equal(y,[int(r['original_label']) for r in rows]) and np.array_equal(y,[int(r['target']) for r in rows])
            for row in rows:
                assert row['split']==split
                for k in ('original_label','identity_component','lesion_id'):assert row[k] and row[k]==manifest[row['sample_id']][k]
            assert np.isfinite(z).all() and np.max(abs(np.linalg.norm(z.astype(float),axis=1)-1))<=2e-6
            if split=='train':assert np.bincount(y,minlength=8).tolist()==self.protocol['train_counts']
            self.data[split]=(z,y,rows);audit[split]=dict(n=len(z),cache_sha256=hashes[0],association_sha256=hashes[1],manifest_sha256=hashes[2])
        for key in ('sample_id','identity_component','lesion_id'):
            assert not({r[key] for r in self.data['train'][2]}&{r[key] for r in self.data['val'][2]})
        # Validate all parent files before creating any GPU model.
        for e in self.entries:
            assert e['sha256']==self.expected[e['seed']]['checkpoint_sha256']
            assert sha(e['path'])==e['sha256'],'BLOCKED_PARENT_STATE_FILE'
            self.verified_parents.add(e['seed'])
        for file in ('backbone/vision_transformer_adapter_pool_a.py','utils/inc_net.py','utils/medical_v2.py','backbone/linears.py','backbone/prompt.py'):
            key='third_party/APART/'+file
            assert sha(ROOT/key)==self.v2['code_sha256'][key],'BLOCKED_NATIVE_SOURCE'
        assert sha(self.v2['weight'])==lock['model_sha256'],'BLOCKED_WEIGHT_SHA'
        self.audit_record=dict(status='PASS',splits=audit,parent_file_hashes_verified=3,native_source_matches_V2=True,
                               parent_order_not_crossed=True,protected_history=True,cache_reuse='A only; no matching existing S0 pointwise raw cache identified')
        write(self.pub/'ASSET_AUDIT.json',self.audit_record)
        # All image reads are restricted to the verified train/val association universe.
        allowed_images={str((self.images/r['relative_path']).resolve()) for s in ('train','val') for r in self.data[s][2]}
        allowed_features={str((self.cache/(s+'.npz')).resolve()) for s in ('train','val')}
        if self.cfg.get('reuse_A_output'):
            old_private=Path(self.cfg['reuse_A_output'])/'private'
            allowed_features.update(str((old_private/f'{m}_{seed}_s{stage}.npz').resolve()) for m in ('A-U','A-CB') for seed in self.orders for stage in range(3))
        allowed_weights={str(Path(e['path']).resolve()) for e in self.entries}|{str(Path(self.v2['weight']).resolve())}
        private=str(self.private.resolve())+os.sep
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            path=Path(os.fsdecode(args[0]));suffix=path.suffix.lower()
            if suffix not in ('.jpg','.jpeg','.png','.npy','.npz','.pt','.pth','.safetensors','.csv','.json'):return
            full=str(path.resolve())
            if suffix in ('.jpg','.jpeg','.png'):assert full in allowed_images,'FORBIDDEN_IMAGE_ACCESS'
            if suffix in ('.npy','.npz'):assert full in allowed_features or full.startswith(private),'FORBIDDEN_FEATURE_ACCESS'
            if suffix in ('.pt','.pth','.safetensors'):assert full in allowed_weights,'FORBIDDEN_WEIGHT_ACCESS'
            assert path.name not in ('test.csv','test.npz','test_associations.json') and not ('test' in path.parts and suffix in ('.npz','.csv')), 'FORBIDDEN_TEST_ACCESS'
        sys.addaudithook(guard)

    def setup_torch(self):
        global torch
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        self.gpu_start=time.monotonic();torch.cuda.reset_peak_memory_stats()

    def restore(self,e):
        from run_medical_v2 import args_for, network_hash
        from utils.inc_net import AdapterVitNet
        from utils.medical_v2 import extend_embedding,WEIGHT_SHA
        assert e['seed'] in self.verified_parents
        # mmap leaves unused optimizer storages on disk; no optimizer is instantiated,
        # moved to a device, or restored. Training metadata is validated before use.
        state=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True)
        for key in ('optimizer','scheduler','memory','rng','loader_rng','synth_rng'):state.pop(key,None)
        seed=e['seed'];a,order=args_for(self.v2,seed,'C',False)
        summary=read(Path(self.v2['protocol'])/'V2_DATASET_SUMMARY.json')
        assert order==self.orders[seed]==state['order'] and state['training_args']==dict(a,device=[str(d) for d in a['device']])
        assert state['manifest_sha256']==summary['manifest_sha256'] and state['protocol_sha256']==self.v2['protocol_sha256']
        assert state['code_sha256']==self.v2['code_sha256'] and state['weight_sha256']==WEIGHT_SHA
        assert (state['task'],state['known'],state['total'],state['epoch'],state['phase'],state['train_branch'],state['train_seed'])==(0,0,4,10,'session_complete','C',seed)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed);net=AdapterVitNet(a,True);extend_embedding(net.backbone.assigner,seed)
        net.load_state_dict(state['network'],strict=True);del state
        net.requires_grad_(False);net.eval().cuda()
        expected=self.expected[seed]['network_sha256'];assert network_hash(net)==expected,'BLOCKED_PARENT_STATE_TENSORS'
        assert net.backbone.pool.batchwise_prompt and net.backbone.pool_few.batchwise_prompt
        assert net.backbone.pool.top_k==net.backbone.pool_few.top_k==1
        line=dict(parent_id=seed,checkpoint_sha256=e['sha256'],network_sha256=expected,train_source_commit=self.v2['code_commit'],
                  order=order,base_labels=order[:4],epoch=10,session=0,strict_restore=True,optimizer_constructed=False,
                  original_frequency_embedding_shape=list(net.backbone.assigner.cls_emb.weight.shape),
                  routing_diff={'backbone.pool.batchwise_prompt':[True,False],'backbone.pool_few.batchwise_prompt':[True,False]})
        # Counters describe actual module invocations, including engineering calls.
        def original_hook(module,args):self.count_module('original',len(args[0]))
        net.original_backbone.register_forward_pre_hook(original_hook)
        native=net.backbone.forward_features
        def features(x,*args,**kwargs):
            self.count_module('few' if kwargs.get('few',0)==1 else 'main',len(x));return native(x,*args,**kwargs)
        net.backbone.forward_features=features
        if self.cfg.get('linear_native_per_image'):
            from check_isic_a1_linear_native import install_batched_linear
            install_batched_linear(net)
            line['numerical_implementation']=self.protocol['numerical_implementation']
        return net,line

    def count_module(self,name,n):
        self.access['module_batch_calls'][name]+=1;self.access['module_image_rows'][name]+=n
        d=self.access['phase_calls'].setdefault(self.phase,{})
        d[name+'_calls']=d.get(name+'_calls',0)+1;d[name+'_image_rows']=d.get(name+'_image_rows',0)+n

    def forward(self,net,x):
        self.access['wrapper_batch_calls']+=1;self.access['wrapper_image_rows']+=len(x)
        with torch.inference_mode():out=net(x,train=False,weight=None)
        for k in ('pre_logits','pre_logits_few'):
            assert out[k].shape==(len(x),768) and out[k].dtype==torch.float32 and torch.isfinite(out[k]).all()
        return {k:out[k].detach().cpu().numpy() for k in ('pre_logits','pre_logits_few','prompt_idx','prompt_idx_few')}

    def images_at(self,split,indices):
        assert split in ('train','val'),'FORBIDDEN_SPLIT'
        from PIL import Image
        from run_medical_v2 import transform
        prep=transform(False);rows=self.data[split][2];images=[]
        for i in indices:
            with Image.open(self.images/rows[i]['relative_path']) as im:images.append(prep(im.convert('RGB')))
            self.access['image_reads'][self.phase]+=1
        return torch.stack(images).cuda()

    def pointwise(self,net,enabled):
        net.backbone.pool.batchwise_prompt=not enabled;net.backbone.pool_few.batchwise_prompt=not enabled

    def probe(self,e):
        from run_medical_v2 import network_hash
        net,line=self.restore(e);seed=e['seed'];base=self.orders[seed][:4];xs=[];indices={};start=time.monotonic()
        for split in ('train','val'):
            ix=np.flatnonzero(np.isin(self.data[split][1],base))[:16];assert len(ix)==16
            indices[split]=ix;xs.append(self.images_at(split,ix))
        x=torch.cat(xs);before_torch=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state();before_np=np.random.get_state();before_py=random.getstate()
        singles=[self.forward(net,x[i:i+1]) for i in range(len(x))]
        ref={k:np.concatenate([r[k] for r in singles]) for k in singles[0]}
        self.pointwise(net,True);point=self.forward(net,x)
        records=[]
        # Changed companions, reversed rows, and a full B=48 with duplicated peers.
        layouts=[np.arange(32),np.arange(31,-1,-1),np.r_[np.arange(16),np.arange(16)],np.r_[np.arange(16,32),np.arange(16,32)],np.r_[np.arange(32),np.arange(16)]]
        for layout in layouts:
            result=self.forward(net,x[layout]);record=dict(batch=len(layout),companion_layout_sha256=digest(layout),features={})
            for key in ref:
                target=ref[key][layout];actual=result[key]
                if key.startswith('prompt_idx'):assert np.array_equal(actual,target),'BLOCKED_ROUTING_PARITY_IDS'
                else:
                    error=float(np.max(np.abs(actual-target)));relative=float(np.linalg.norm(actual-target)/np.linalg.norm(target))
                    record['features'][key]=dict(max_abs=error,relative_l2=relative)
                    if not np.allclose(actual,target,atol=1e-5,rtol=1e-5):
                        write(self.pub/'ROUTING_PARITY_FAILURE.json',dict(parent=seed,layout=record,key=key,atol=1e-5,rtol=1e-5))
                        raise AssertionError('BLOCKED_ROUTING_PARITY_FEATURES')
            records.append(record)
        # Evaluator labels are changed outside the model: input and permitted kwargs
        # are identical, and no labels/frequencies/task/adapter IDs enter forward.
        ignored_labels=np.arange(32)[::-1];assert len(ignored_labels)==len(x)
        again=self.forward(net,x)
        for k in point:assert np.array_equal(point[k],again[k]),'BLOCKED_LABEL_DEPENDENCE'
        self.pointwise(net,False);native48=self.forward(net,x[layouts[-1]])
        diff={k:int(np.sum(native48[k]!=ref[k][layouts[-1]])) for k in ('prompt_idx','prompt_idx_few')}
        self.pointwise(net,True)
        with torch.inference_mode():original=net.original_backbone(xs[0])['pre_logits'];original=torch.nn.functional.normalize(original,dim=1).cpu().numpy()
        a=self.data['train'][0][indices['train']];assert np.allclose(original,a,atol=1e-5,rtol=1e-5),'BLOCKED_ORIGINAL_BACKBONE_CACHE_PARITY'
        assert network_hash(net)==line['network_sha256'] and all(not m.training for m in net.modules()),'BLOCKED_STATE_MUTATION'
        assert torch.equal(before_torch,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
        assert all(np.array_equal(a,b) for a,b in zip(before_np,np.random.get_state())) and before_py==random.getstate()
        elapsed=time.monotonic()-start
        self.parity.append(dict(parent_id=seed,status='PASS',native_single_rows=32,layouts=records,native_batch48_route_differences=diff,
            original_backbone_cache_max_abs=float(np.max(np.abs(original-a))),original_backbone_cache_relative_l2=float(np.linalg.norm(original-a)/np.linalg.norm(a)),
            label_blind=True,parameters_and_buffers_unchanged=True,rng_unchanged=True,seconds=elapsed))
        self.parent_locks.append(line)
        # End-to-end fixed probe: includes image decoding, transfer and raw inference.
        torch.cuda.synchronize();t=time.monotonic();xx=self.images_at('train',indices['train']);self.forward(net,xx);torch.cuda.synchronize()
        seconds=time.monotonic()-t
        self.resource_check();del net,x,xs,xx,singles;gc.collect();torch.cuda.empty_cache()
        return dict(parent_id=seed,rows=16,end_to_end_seconds=seconds,projected_all_57039_rows_seconds=seconds/16*57039*1.5)

    def prediction(self,method,seed,stage,W,z,split='val'):
        assert split=='val'
        order=self.orders[seed];seen=(4,6,8)[stage];_,labels,rows=self.data[split]
        ix=np.flatnonzero(np.isin(labels,order[:seen]));inverse={c:i for i,c in enumerate(order[:seen])}
        assert len(z)==len(ix)
        raw=z@W;assert raw.shape==(len(ix),seen) and np.isfinite(raw).all()
        p=dict(raw=raw,y=np.array([inverse[int(c)] for c in labels[ix]]),original=labels[ix],order=np.array(order),
               ids=np.array([rows[i]['sample_id'] for i in ix]),component=np.array([rows[i]['identity_component'] for i in ix]),lesion=np.array([rows[i]['lesion_id'] for i in ix]))
        m,pc,errors=analyze_unit(p,method,seed,stage,tie_original_label=True)
        pred=predict_columns(raw,p['order']);other=raw.copy();other[np.arange(len(raw)),p['y']]=-np.inf;margin=raw[np.arange(len(raw)),p['y']]-other.max(1)
        for row in [m]+pc+errors:row.update(split='val',predictor='A1_fixed_analytic',parent_id=seed if method.startswith('S') else None,core=method in CORE)
        for row in pc:
            mask=p['y']==row['head_index'];row.update(in_parent_base=row['original_label'] in order[:4],n_correct=int(np.count_nonzero(pred[mask]==p['y'][mask])),
                raw_margin_quantiles=np.quantile(margin[mask],[0,.05,.25,.5,.75,.95,1]).tolist(),margin_positive_fraction=float(np.mean(margin[mask]>0)))
        for name,subset in [('base',order[:4]),('post_base',order[4:]),('tail_base',[c for c in order[:4] if c in (6,7)]),('tail_post_base',[c for c in order[4:] if c in (6,7)])]:
            vals=[r['recall'] for r in pc if r['original_label'] in subset]
            m[name+'_macro_recall']=float(np.mean(vals)) if vals else None;m[name+'_n_classes']=len(vals)
        for c in (6,7):
            record=next((r for r in pc if r['original_label']==c),None)
            for k in ('recall','n_correct','n_images'):m[f'label_{c}_{k}']=record[k] if record else None
        m['raw_margin_quantiles']=np.quantile(margin,[0,.05,.25,.5,.75,.95,1]).tolist();m['exact_argmax_ties']=int(np.sum(np.sum(raw==raw.max(1,keepdims=True),axis=1)>1))
        assert abs(m['balanced_accuracy']-np.mean([r['recall'] for r in pc]))<1e-10
        np.savez_compressed(self.private/f'{method}_{seed}_s{stage}.npz',W=W,**p)
        self.metrics.append(m);self.pc+=pc;self.errors+=errors
        return raw

    def fit_stage(self,inc,method,seed,stage,val):
        t=time.monotonic();G,R=inc.system();W,res=solve(G,R);self.solves+=1
        e=np.linalg.eigvalsh((G+G.T)/2);condition=float((e[-1]+.001)/(e[0]+.001))
        self.numeric.append(dict(method=method,order_seed=seed,session=stage,dimension=len(G),trace_G=float(np.trace(G)),regularized_condition=condition,
            solve_relative_residual=res,W_norm=float(np.linalg.norm(W)),G_sha256=digest(G),R_sha256=digest(R),W_sha256=digest(W),finite=True,
            joint_cross_block_norm=float(np.linalg.norm(G[:768,768:])) if len(G)==1536 else None,
            sufficient_statistic_bytes=inc.H.nbytes+sum(x.nbytes for x in inc.cols),future_classes_in_statistics=False))
        raw=self.prediction(method,seed,stage,W,val)
        if method.startswith('A-'):
            control=duplicate_embedding(G,R,W,val);assert np.array_equal(predict_columns(raw,np.array(inc.order)),predict_columns(raw,None)),'G1_TIE_COMPATIBILITY'
            self.duplicates.append(dict(method=method,order_seed=seed,session=stage,**control))
        inc.save(self.private/'active_statistics.npz');restored=Increment.restore(self.private/'active_statistics.npz')
        close(restored.H,inc.H);close(np.stack(restored.cols),np.stack(inc.cols));close(solve(*restored.system())[0],W)
        self.cpu_seconds+=time.monotonic()-t
        return restored

    def baseline(self):
        if self.cfg.get('reuse_A_output'):
            from report_isic_g1 import records
            old=Path(self.cfg['reuse_A_output']);audit=read(old/'public/ASSET_AUDIT.json')
            assert audit['splits']==self.audit_record['splits'],'BLOCKED_REUSE_DATA'
            assert read(old/'public/COMPLETION_AUDIT.json')['completed_methods']==['A-U','A-CB']
            for attr,name in [('metrics','val_metrics'),('pc','val_per_class_metrics'),('errors','error_decomposition'),('numeric','numeric_diagnostics')]:
                table=records(old/'public'/f'{name}.csv')
                for row in table:
                    assert row['method'] in ('A-U','A-CB')
                    for k,v in row.items():
                        if isinstance(v,str) and v in ('True','False'):row[k]=v=='True'
                setattr(self,attr,table)
            assert len(self.metrics)==18 and len(self.pc)==108
            for method in ('A-U','A-CB'):
                for seed in self.orders:
                    for stage in range(3):
                        name=f'{method}_{seed}_s{stage}.npz';source=old/'private'/name
                        assert source.is_file()
                        (self.private/name).symlink_to(source)
            controls=read(old/'public/DUPLICATE_EMBEDDING_CONTROL.json');assert controls['status']=='PASS'
            self.duplicates=controls['actual_A_controls'];self.reused_fits=18;self.flush()
            write(self.pub/'DUPLICATE_EMBEDDING_CONTROL.json',controls)
            write(self.pub/'A_REUSE.json',dict(status='PASS',source_commit='fba6a39dacc95bf7cb6c33449849d58d8a240b46',fits=18,new_A_fits=0,original_private_scores_referenced=True,source_manifest_and_cache_hashes_match=True))
            print('A_BASELINES_REUSED',flush=True);return
        t=time.monotonic();cpu_before=self.cpu_seconds;train,labels,_=self.data['train'];val,vy,_=self.data['val'];final={}
        for seed,order in self.orders.items():
            learners={kind:Increment(order,768,'R0' if kind=='U' else 'B0') for kind in ('U','CB')};known=0
            for stage,seen in enumerate((4,6,8)):
                for c in order[known:seen]:
                    stat=moments(train[labels==c].astype(np.float64))
                    for inc in learners.values():inc.arrive(c,stat)
                for kind,inc in learners.items():
                    learners[kind]=self.fit_stage(inc,'A-'+kind,seed,stage,val[np.isin(vy,order[:seen])].astype(np.float64))
                    if stage==2:
                        G,R=inc.system();W,_=solve(G,R);canonical=(G,R[:,np.argsort(order)],W[:,np.argsort(order)])
                        if kind in final:
                            for a,b in zip(final[kind],canonical):close(a,b)
                        else:final[kind]=tuple(a.copy() for a in canonical)
                known=seen
        ref=list(csv.DictReader((ROOT/'docs/isic_g1_controlled_spherical_mixup/val_metrics.csv').open()));pc=list(csv.DictReader((ROOT/'docs/isic_g1_controlled_spherical_mixup/val_per_class_metrics.csv').open()))
        for row in self.metrics:
            method='R0' if row['method']=='A-U' else 'B0'
            old=next(r for r in ref if (r['method'],int(r['order_seed']),int(r['session']))==(method,row['order_seed'],row['session']))
            for field in ('balanced_accuracy','tail_rank2','accuracy','macro_f1'):
                assert abs(row[field]-float(old[field]))<1e-10,'BLOCKED_REPRODUCIBILITY'
        for row in self.pc:
            method='R0' if row['method']=='A-U' else 'B0'
            old=next(r for r in pc if (r['method'],int(r['order_seed']),int(r['session']),int(r['original_label']))==(method,row['order_seed'],row['session'],row['original_label']))
            assert row['n_correct']==int(old['n_correct']) and abs(row['recall']-float(old['recall']))<1e-10
        self.flush();write(self.pub/'DUPLICATE_EMBEDDING_CONTROL.json',dict(status='PASS',actual_A_controls=self.duplicates))
        self.cpu_seconds=cpu_before+time.monotonic()-t
        print('A_BASELINES_REPRODUCED',flush=True)

    def diagnostic_features(self,seed,split,c,raw,indices):
        m=raw[:,0].astype(float);f=raw[:,1].astype(float);mn=np.linalg.norm(m,axis=1);fn=np.linalg.norm(f,axis=1)
        a=self.data[split][0][indices].astype(float);mm=m/mn[:,None];ff=f/fn[:,None]
        vals={'main_norm':mn,'few_norm':fn,'main_to_few_norm_ratio':mn/fn,'main_few_cosine':np.sum(mm*ff,axis=1),
              'A_main_cosine':np.sum(a*mm,axis=1)/np.linalg.norm(a,axis=1),'A_few_cosine':np.sum(a*ff,axis=1)/np.linalg.norm(a,axis=1)}
        for k,v in vals.items():
            assert np.isfinite(v).all()
            self.features.append(dict(parent_id=seed,split=split,original_label=c,in_parent_base=c in self.orders[seed][:4],n=len(raw),metric=k,mean=float(v.mean()),quantiles=np.quantile(v,[0,.05,.25,.5,.75,.95,1]).tolist()))

    def extract_parent(self,e):
        from run_medical_v2 import network_hash
        self.phase='formal';seed=e['seed'];order=self.orders[seed];net,line=self.restore(e);self.pointwise(net,True)
        before_torch=torch.get_rng_state();before_cuda=torch.cuda.get_rng_state();before_np=np.random.get_state();before_py=random.getstate()
        maps={};paths={};written={s:set() for s in ('train','val')}
        for s in ('train','val'):
            p=self.private/f'{seed}_{s}_raw.partial.npy';assert not p.exists() and not p.with_name(p.name.replace('.partial','')).exists(),'BLOCKED_DUPLICATE_EXTRACTION'
            maps[s]=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(len(self.data[s][0]),2,768));paths[s]=p
        learners={(v,k):Increment(order,1536 if v=='J' else 768,'R0' if k=='U' else 'B0') for v in ('J','M','F') for k in ('U','CB')}
        known=0
        for stage,seen in enumerate((4,6,8)):
            for c in order[known:seen]:
                assert c not in written['train'] and c==order[len(written['train'])]
                for s in ('train','val'):
                    ix=np.flatnonzero(self.data[s][1]==c)
                    for start in range(0,len(ix),48):
                        batch=ix[start:start+48];x=self.images_at(s,batch);out=self.forward(net,x)
                        maps[s][batch]=np.stack([out['pre_logits'],out['pre_logits_few']],axis=1)
                        self.access['new_train_val_feature_rows']+=len(batch)
                    written[s].add(c);maps[s].flush()
                    self.diagnostic_features(seed,s,c,maps[s][ix],ix)
                t=time.monotonic();ix=np.flatnonzero(self.data['train'][1]==c)
                for v in ('J','M','F'):
                    z=view(maps['train'][ix],v);stat=moments(z)
                    for k in ('U','CB'):learners[v,k].arrive(c,stat)
                self.access['analytic_train_rows']+=len(ix);self.access['formal_arrivals'].append(dict(parent_id=seed,session=stage,label=c,n_train=len(ix),visited=order[:len(written['train'])]))
                self.cpu_seconds+=time.monotonic()-t;self.resource_check()
            known=seen;ix=np.flatnonzero(np.isin(self.data['val'][1],order[:seen]))
            for (v,k),inc in learners.items():
                assert inc.visited==order[:seen] and set(inc.visited)==written['train']
                learners[v,k]=self.fit_stage(inc,f'S-{v}-{k}',seed,stage,view(maps['val'][ix],v))
            self.flush();print(json.dumps(dict(event='parent_stage_complete',parent=seed,stage=stage,rows=len(self.metrics))),flush=True)
        assert network_hash(net)==line['network_sha256'] and all(not m.training for m in net.modules()),'BLOCKED_STATE_MUTATION'
        assert torch.equal(before_torch,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
        assert all(np.array_equal(a,b) for a,b in zip(before_np,np.random.get_state())) and before_py==random.getstate()
        del net,x,out;gc.collect();torch.cuda.empty_cache()
        for s in ('train','val'):
            assert written[s]==set(range(8));maps[s].flush();del maps[s]
            target=paths[s].with_name(paths[s].name.replace('.partial',''));paths[s].replace(target)
        self.resource_check()
        # Isolated, after-all-stages reaggregation. Reverse only within each class;
        # never use another parent's task order and never feed this back to fits.
        t=time.monotonic();raw=np.load(self.private/f'{seed}_train_raw.npy',mmap_mode='r');labels=self.data['train'][1]
        equivalence=[]
        for v in ('J','M','F'):
            check={k:Increment(order,1536 if v=='J' else 768,'R0' if k=='U' else 'B0') for k in ('U','CB')}
            for c in order:
                ix=np.flatnonzero(labels==c)[::-1];n=len(ix);s=np.zeros(check['U'].H.shape[0]);q=np.zeros_like(check['U'].H)
                for start in range(0,n,257):
                    z=view(raw[ix[start:start+257]],v);s+=z.sum(0);q+=z.T@z
                for inc in check.values():inc.arrive(c,dict(n=n,s=s,T=q))
            for k in ('U','CB'):
                G,R=check[k].system();oldG,oldR=learners[v,k].system();dg=close(G,oldG);dr=close(R,oldR);W,res=solve(G,R)
                with np.load(self.private/f'S-{v}-{k}_{seed}_s2.npz') as saved:
                    dw=close(W,saved['W']);vv=np.load(self.private/f'{seed}_val_raw.npy',mmap_mode='r');scores=view(vv,v)@W
                    close(scores,saved['raw']);assert np.array_equal(predict_columns(scores,np.array(order)),predict_columns(saved['raw'],np.array(order)))
                equivalence.append(dict(view=v,weighting=k,G_max_abs=dg,R_max_abs=dr,W_max_abs=dw,relative_solve_residual=res))
        self.cpu_seconds+=time.monotonic()-t
        write(self.pub/f'PARENT_{seed}_EXTRACTION.json',dict(status='PASS',parent_id=seed,new_feature_rows=19013,network_tensor_hash_unchanged=True,rng_unchanged=True,
            cache_sha256={s:sha(self.private/f'{seed}_{s}_raw.npy') for s in ('train','val')},within_parent_reblocking_and_row_reverse=equivalence,
            diagnostic_train_feature_reaccess_rows=18718,forensic_scores_generated=False))
        del learners,raw,check;gc.collect()

    def flush(self):
        for name,rows in [('val_metrics',self.metrics),('val_per_class_metrics',self.pc),('val_core_metrics',[r for r in self.metrics if r['core']]),
            ('val_core_per_class',[r for r in self.pc if r['core']]),('error_decomposition',self.errors),('numeric_diagnostics',self.numeric),('feature_diagnostics',self.features)]:
            if rows:csvwrite(self.pub/(name+'.csv'),rows)
        write(self.pub/'ACCESS_AUDIT.json',dict(**FIXED,**self.access,encoder_forward_calls=sum(self.access['module_batch_calls'].values()),
            encoder_image_forward_rows=sum(self.access['module_image_rows'].values()),no_labels_frequency_task_adapter_passed=True))

    def all(self):
        self.audit();t=time.monotonic();self.engineering=engineering(self.private);self.cpu_seconds+=time.monotonic()-t
        self.baseline();self.setup_torch();bench=[]
        for e in self.entries:
            bench.append(self.probe(e));write(self.pub/'ROUTING_PARITY.json',dict(status='PASS',parents=self.parity));self.flush()
        projected=max(b['projected_all_57039_rows_seconds'] for b in bench)+300
        assert projected+self.prior.get('GPU_budget_seconds',0)+(time.monotonic()-self.gpu_start)<7200,'BLOCKED_RESOURCE_PROJECTED_GPU'
        projected_cpu=self.cpu_seconds*20+300+self.prior.get('CPU_test_budget_reserve_seconds',0)
        assert projected_cpu<7200,'BLOCKED_RESOURCE_PROJECTED_CPU'
        self.engineering.update(A_baselines_reproduced=True,routing_parity=True,budget_probe=bench,projected_GPU_seconds=projected,
            projected_CPU_seconds=projected_cpu,projected_new_file_bytes=450*1024**2,preexisting_free_bytes=self.min_free,parameter_buffers_immutable=True)
        write(self.pub/'ENGINEERING_A1.json',self.engineering);write(self.pub/'PARENT_LOCKS.json',dict(status='PASS',parents=self.parent_locks))
        write(self.pub/'PROTOCOL_A1.json',self.protocol)
        write(self.pub/'FEATURE_VIEW_LOCK.json',dict(routing_mode='S0_POINTWISE_PROBE',raw_dtype='float32',derive_dtype='float64',batch_size=48,
             views={'J':'concat raw main,few / sqrt(sum main**2 + sum few**2)','M':'main / norm(main)','F':'few / norm(few)','A':'unaltered original float32 cached values cast to float64'},
             output_fields=['pre_logits','pre_logits_few'],parameter_changes=False,pointwise_flags=['backbone.pool.batchwise_prompt=false','backbone.pool_few.batchwise_prompt=false'],
             original_A_cache_sha256={s:self.audit_record['splits'][s]['cache_sha256'] for s in ('train','val')},atol=1e-5,rtol=1e-5,
             numerical_implementation=self.protocol.get('numerical_implementation','unchanged native operators')))
        files=['tools/isic_a1_attribution.py','tools/report_isic_a1.py','tools/isic_g1_controlled_mixup.py','tools/report_locked_holdout_r1.py','tools/run_medical_v2.py','tests/test_isic_a1.py','exps/isic_a1_protocol.json',
               'third_party/APART/backbone/vision_transformer_adapter_pool_a.py','third_party/APART/utils/inc_net.py','third_party/APART/utils/medical_v2.py']
        if self.cfg.get('linear_native_per_image'):files.append('tools/check_isic_a1_linear_native.py')
        write(self.pub/'CODE_LOCK_A1.json',dict(source_commit=self.cfg['source_commit'],sha256={f:sha(ROOT/f) for f in files},locked_before_S_candidate_scoring=True,
            versions={k:importlib.metadata.version(k) for k in ('torch','torchvision','timm','numpy','scipy','Pillow')},tf32=False,AMP=False,
            numerical_implementation=self.protocol.get('numerical_implementation','unchanged native operators'),prior_resources=self.prior,reused_A_fits=self.reused_fits,**FIXED))
        (self.pub/'P0_AUDIT.md').write_text('# A1 P0\n\nAsset hashes, strict full-network restoration, native routing and feature parity, A/G1 baseline reproduction, duplicate controls and budget gates PASS. Exactly two routing flags differ in disposable probes. No S candidate scored before this lock. See structured evidence alongside this audit.\n')
        print(json.dumps(dict(event='P0_P1_ENGINEERING_PASS',projected_GPU_seconds=projected)),flush=True)
        for e in self.entries:self.extract_parent(e)
        self.gpu_seconds=time.monotonic()-self.gpu_start
        assert len(self.metrics)==72 and len(self.pc)==432 and sum(r['core'] for r in self.metrics)==36 and sum(r['core'] for r in self.pc)==216
        assert self.access['new_train_val_feature_rows']==57039
        self.flush();write(self.pub/'P2_COMPLETE.json',dict(status='PASS',all_val_metric_rows=72,all_val_per_class_rows=432,core_val_metric_rows=36,core_val_per_class_rows=216))
        from report_isic_a1 import complete,render
        t=time.monotonic();complete(self.out);self.cpu_seconds+=time.monotonic()-t
        resources=self.resource_check();resources.update(continued_increment_all_views_two_targets_bytes=2*(1536**2+2*768**2)*8+6*1536*8*8,
            raw_cache_payload_bytes=57039*2*768*4,physical_formal_stage_solves=72,engineering_or_equivalence_solves_excluded_from_formal_count=True,
            source_commit=self.cfg['source_commit'],GPU_residence_upper_bound_including_CPU_analysis=self.prior.get('GPU_budget_seconds',0)+resources['wall_seconds'])
        write(self.pub/'RESOURCE_REPORT.json',resources)
        completion=dict(status='COMPLETE_A1_ATTRIBUTION',**FIXED,analytic_fits=self.solves+self.reused_fits,new_analytic_fits=self.solves,reused_A_fits=self.reused_fits,encoder_forward_calls=sum(self.access['module_batch_calls'].values()),
            new_train_val_feature_rows=57039,core_val_metric_rows=36,core_val_per_class_rows=216,all_val_metric_rows=72,all_val_per_class_rows=432)
        write(self.pub/'COMPLETION_AUDIT.json',completion);render(self.pub)
        print(json.dumps(completion),flush=True)


def main():
    run=Run(read(os.environ['P17_CONFIG']))
    try:
        with threadpool_limits(limits=4):run.all()
    except Exception as e:
        run.flush();write(run.pub/'BLOCKED.json',dict(status=str(e) if str(e).startswith('BLOCKED') else 'BLOCKED_ENGINEERING',reason=str(e),exception=type(e).__name__,
           completed_metric_rows=len(run.metrics),completed_per_class_rows=len(run.pc),**FIXED))
        write(run.pub/'RESOURCE_REPORT.json',run.resource_check(enforce=False))
        raise


if __name__=='__main__':main()
