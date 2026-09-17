"""CT1 fixed two-stream continuous adapter training; private P20_CONFIG paths.

The inherited APART loop supplies all real losses. Probes use a separate model,
and neither analytic readout ever feeds back into neural training.
"""
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits

from run_medical_v2 import (ROOT,Learner,Images,seed_all,extend_embedding,
                            effective_optimizer,write_json,sha,WEIGHT_SHA,network_hash,rng_equal)
from run_hyperkvasir_hk1_m1 import tensor_digest
from check_isic_a1_linear_native import install_batched_linear
from preflight_ct1 import task_lock
from ct1_statistics import empty,append,fit_map,transport,ridge,risk,joint,selfcheck


def read(p):return json.loads(Path(p).read_text())
def jsonl(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def npz(p,**x):
    tmp=Path(str(p)+'.part')
    with tmp.open('wb') as f:np.savez(f,**x)
    tmp.replace(p)


class Run:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root']);self.out=self.root/'output'
        self.pub=self.out/'public';self.private=self.out/'private'
        for p in (self.pub,self.private,self.private/'sealed',self.private/'archive_ack'):p.mkdir(parents=True,exist_ok=True)
        self.mode=cfg['mode'];self.lock=task_lock(cfg);self.access={}
        self.started=time.monotonic();self.phase='engineering';self.minfree=shutil.disk_usage(self.root).free
        self.maxbytes=0;self.lineage=[];self.sealed=[];self.gpu_start=None
        self.prior=read(self.pub/'RESOURCE_LEDGER.json') if (self.pub/'RESOURCE_LEDGER.json').exists() else {}
        self.preflight=read(self.pub/'EARLY_RESOURCE_ADMISSION.json')
        assert self.preflight['status']=='RESOURCE_GATE_PASS'
        self.code={k:sha(ROOT/k) for k in cfg['code_files']}
        assert self.code==cfg['code_sha256'],'BLOCKED_CODE_DRIFT'
        assert sha(cfg['weight'])==WEIGHT_SHA
        self.rows={}
        for name,s in cfg['datasets'].items():
            self.rows[name]={}
            for split in ('train','val'):
                with (Path(s['manifests'])/(split+'.csv')).open() as f:rs=list(csv.DictReader(f))
                self.rows[name][split]=sorted(rs,key=lambda r:r['sample_id'])
        self.allowed=set();self.forbidden_attempts=0
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0]))
            if p.suffix.lower() in ('.jpg','.jpeg','.png'):
                if str(p.resolve()) not in self.allowed:
                    self.forbidden_attempts+=1;raise AssertionError('BLOCKED_OLD_FUTURE_OR_HELDOUT_IMAGE_ACCESS')
            if p.name in ('test.csv','reserved.csv','test.npz','reserved.npz'):
                raise AssertionError('BLOCKED_TEST_ASSET_ACCESS')
        sys.addaudithook(guard)

    def setup(self):
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');self.gpu_start=time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def count(self,k,n=1):self.access[k]=self.access.get(k,0)+n

    def resources(self,enforce=True):
        size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
        self.maxbytes=max(self.maxbytes,size);self.minfree=min(self.minfree,shutil.disk_usage(self.root).free)
        elapsed=0 if self.gpu_start is None else time.monotonic()-self.gpu_start
        prior=self.prior.get('GPU_process_residence_seconds',self.preflight['measured_GPU_process_residence_seconds'])
        record=dict(GPU_process_residence_seconds=prior+elapsed,current_worker_seconds=elapsed,
                    active_bytes=size,active_peak_bytes=self.maxbytes,min_free_bytes=self.minfree,
                    peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if self.gpu_start else 0,
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    GPU_budget_seconds=43200,CPU_budget_seconds=21600,single_worker=True,
                    forbidden_image_attempts=self.forbidden_attempts,mode=self.mode)
        write_json(self.pub/'RESOURCE_LEDGER.json',record)
        write_json(self.pub/f'ACCESS_{self.mode}.json',dict(self.access,new_test_image_reads=0,
                   new_test_feature_reads=0,new_test_model_forwards=0,new_test_predictions=0,
                   old_train_image_reads=0,future_train_image_reads=0))
        if enforce:
            assert prior+elapsed<43200,'BLOCKED_GPU_BUDGET'
            assert size<=3*1024**3 and self.minfree>=1024**3,'BLOCKED_STORAGE'
        return record

    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert split in ('train','val')
        s=self.cfg['datasets'][name];order=self.lock[name]['orders'][str(seed)] if str(seed) in self.lock[name]['orders'] else self.lock[name]['orders'][seed]
        ds=Images(Path(s['manifests'])/(split+'.csv'),s['images'],order,classes,train,smoke)
        ds.rows.sort(key=lambda r:r['sample_id'])
        self.allowed={str((Path(s['images'])/r['relative_path']).resolve()) for r in ds.rows}
        return ds

    def extract(self,l,classes,split='train',smoke=False):
        rng=l._capture_rng_state();mode=[m.training for m in l._network.modules()]
        l.probe.load_state_dict(l._network.state_dict(),strict=True);l.probe.eval()
        ds=self.dataset(l.name,l.seed,classes,False,split,smoke or getattr(l,'bounded_probe',False))
        loader=DataLoader(ds,batch_size=48,shuffle=False,num_workers=4,drop_last=False,
                          generator=torch.Generator().manual_seed(l.seed+1000003))
        raw=[];pt=[];scores=[];ys=[];t=time.monotonic()
        captured={}
        hook=l.probe.original_backbone.register_forward_hook(lambda m,a,o:captured.update(A=o['pre_logits']))
        with torch.inference_mode():
            for _,x,y in loader:
                o=l.probe(x.cuda(),train=False,weight=None)
                raw.append(torch.stack([o['pre_logits'],o['pre_logits_few']],1).cpu().numpy())
                pt.append(torch.nn.functional.normalize(captured['A'],dim=1).cpu().numpy())
                scores.append((o['logits']+o['logits_few'])[:,:l._total_classes].cpu().numpy());ys.append(y.numpy())
                self.count(self.phase+'_'+split+'_probe_rows',len(x));self.count(self.phase+'_wrapper_calls')
                self.count(self.phase+'_internal_encoder_calls',3)
        hook.remove();assert mode==[m.training for m in l._network.modules()]
        l._restore_rng_state(rng)
        z=np.concatenate(raw);assert np.isfinite(z).all()
        self.count(self.phase+'_probe_milliseconds',int(1000*(time.monotonic()-t)))
        return z,np.concatenate(pt),np.concatenate(scores),np.concatenate(ys),ds.rows

    def write_scores(self,l,method,W,features,ys,rows,order):
        scores=features@W if W is not None else features
        assert scores.shape==(len(rows),l._total_classes) and np.isfinite(scores).all()
        stem=f'{l.name}_{l.seed}_{method}_t{l.task+1:02d}'
        p=self.private/'sealed'/(stem+'.npz')
        components=[r.get('identity_component',r.get('verified_group',r.get('component_id',''))) for r in rows]
        assert all(components),'BLOCKED_COMPONENT_DEPENDENCY'
        npz(p,raw=scores,y=ys,original=np.array(order)[ys],order=np.array(order),
            ids=np.array([r['sample_id'] for r in rows]),component=np.array(components))
        self.sealed.append(dict(dataset=l.name,seed=l.seed,task=l.task+1,method=method,file=p.name,
                                sha256=sha(p),seen=l._total_classes,known=l._known_classes,n=len(rows)))
        write_json(self.private/'SEALED_INDEX.json',self.sealed)


class CTLearner(Learner):
    def __init__(self,r,name,seed,stream):
        self.r=r;self.name=name;self.seed=seed;self.stream=stream;self.task=0
        self.order=r.lock[name]['orders'][seed];self.tasks=r.lock[name]['runs'][str(seed)]
        seed_all(seed)
        args=read(ROOT/'third_party/APART/exps/apart_cifar_shuffle.json')
        args.update(nb_classes=len(self.order),nb_tasks=len(self.tasks),init_cls=2,increment=2,
                    seed=seed,device=[torch.device('cuda:0')],medical_v2=True,
                    locked_weight_path=r.cfg['weight'],tuned_epoch=10,concm_stage1=stream=='R',
                    concm_stage1_loss_weight=0.,concm_stage1_eval_calibration=False,calibration_rule='none',
                    lt_list=[r.lock[name]['train_counts'][c] for c in self.order],
                    weight_decay=.01,real_ce_scope='all_seen',save_task_checkpoints=False,dataset='ct1_'+name,
                    shared_prompt_pool=False,shared_prompt_key=False)
        super().__init__(args);extend_embedding(self._network.backbone.assigner,seed)
        self.nonshared={'backbone.'+k for k in self._network.backbone.weight_load_audit['allowed_missing']}
        assert {k for k,p in self._network.named_parameters() if p.requires_grad}==self.nonshared
        self.initial_hash=network_hash(self._network)
        self.shared_hash=tensor_digest((k,v) for k,v in self._network.state_dict().items() if k not in self.nonshared)
        self._network.cuda();self.probe=copy.deepcopy(self._network).requires_grad_(False).eval()
        install_batched_linear(self.probe)
        self.probe.backbone.pool.batchwise_prompt=False;self.probe.backbone.pool_few.batchwise_prompt=False
        self.loader_generator=torch.Generator().manual_seed(seed)
        self.synth=torch.Generator(device='cuda').manual_seed(seed+2000003)
        self.stats=empty(1536);self.stale=empty(1536);self.pt=empty(768)
        self.raw_memory={};self.epoch_memory={};self.steps=0;self.records=[];self.smoke=False
        self.optimizer=None;self.scheduler=None;self.current_epoch=0;self.last_anchor=None
        self._cur_task=0;self._known_classes=0;self._total_classes=2

    def task_setup(self,task):
        self.task=task;self._cur_task=task;self._known_classes=0 if task==0 else self.tasks[task-1]['seen']
        self._total_classes=self.tasks[task]['seen'];self.current=range(self._known_classes,self._total_classes)
        self.optimizer=effective_optimizer(self._network.backbone)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=10,eta_min=1e-5)
        self._v2_start_epoch=0;self.epoch_memory=copy.deepcopy(self.raw_memory)
        self.anchor_stats=copy.deepcopy(self.stats);self.anchor_raw=copy.deepcopy(self.raw_memory)
        self.adapter_before={k:p.detach().cpu().clone() for k,p in self._network.named_parameters()
                             if '.pool.' in k and ('up_proj' in k or 'down_proj' in k)}
        self.epoch_parts=[];self.epoch_start=time.monotonic();self.lr=[g['lr'] for g in self.optimizer.param_groups]

    def _init_prompt(self):pass

    def _concm_stage1_effective_weight(self,epoch):
        return self._known_classes/len(self.current) if self.stream=='R' else 0.

    def _concm_stage1_sample_memory(self,*args,**kwargs):
        if self.stream!='R' or self._known_classes==0:return None
        m=[];f=[];labels=[]
        for c in range(self._known_classes):
            s=self.epoch_memory[c]
            for path,target in [(0,m),(1,f)]:
                mu=torch.as_tensor(s['mu'][path],device='cuda',dtype=torch.float32)
                var=torch.as_tensor(s['var'][path],device='cuda',dtype=torch.float32)
                target.append(mu+torch.randn((4,768),device='cuda',generator=self.synth)*torch.sqrt(var.clamp_min(0)+1e-6))
            labels.extend([c]*4)
        m=torch.cat(m);f=torch.cat(f);y=torch.tensor(labels,device='cuda')
        if len(y)>48:
            ix=torch.randperm(len(y),device='cuda',generator=self.synth)[:48];m=m[ix];f=f[ix];y=y[ix]
        return m,f,y

    def _v2_input_hook(self,epoch,batch,x,y):
        assert y.min()>=self._known_classes and y.max()<self._total_classes
        assert not torch.is_autocast_enabled() and x.dtype==torch.float32
        self.r.count(self.r.phase+'_train_rows',len(x));self.r.count(self.r.phase+'_wrapper_calls')
        self.r.count(self.r.phase+'_internal_encoder_calls',3)

    def _v2_gradient_hook(self,epoch,batch,loss):
        assert torch.isfinite(loss),'BLOCKED_NONFINITE_LOSS'
        nonzero=0
        for k,p in self._network.named_parameters():
            if p.grad is None:continue
            assert k in self.nonshared and torch.isfinite(p.grad).all(),'BLOCKED_GRADIENT'
            if k.startswith(('backbone.head.','backbone.head_few.')):
                assert torch.count_nonzero(p.grad[self._total_classes:])==0,'BLOCKED_FUTURE_GRADIENT'
            if '.pool.' in k and torch.count_nonzero(p.grad):nonzero+=1
        self.r.count(self.r.phase+'_optimizer_steps');self.steps+=1
        self.epoch_gradient_nonzero=max(getattr(self,'epoch_gradient_nonzero',0),nonzero)
        if self.smoke:self.last_gradient=tensor_digest((k,p.grad) for k,p in self._network.named_parameters() if p.grad is not None)

    def _v3_batch_hook(self,epoch,batch,o,y,lsum,lfew,lassign,lreplay,weight):
        with torch.no_grad():
            main=o['logits'][:,:self._total_classes];summed=main+o['logits_few'][:,:self._total_classes]
            self.epoch_parts.append(dict(main_CE=float(torch.nn.functional.cross_entropy(main,y)),
                sum_CE=float(lsum),pool_weighted_few=float(lfew),assignment=float(lassign),
                pull=-.1*float(o['reduce_sim']+o['reduce_sim_few']),
                replay_CE=0. if lreplay is None else float(lreplay),replay_weight=weight,
                all_seen_correct=int((summed.argmax(1)==y).sum()),
                restricted_correct=int((summed[:,self._known_classes:].argmax(1)==y-self._known_classes).sum()),n=len(y)))

    def _v2_epoch_hook(self,epoch,optimizer,scheduler,stats):
        if self.smoke:return
        torch.cuda.synchronize();training_seconds=time.monotonic()-self.epoch_start
        raw,pt,_,ys,_=self.r.extract(self,self.current)
        if self._known_classes:
            z0=joint(self.last_anchor);z1=joint(raw)
            a,b,error,diag=fit_map(z0,z1,ys)
            # Always map the immutable pre-task bank, never yesterday's transported bank.
            self.epoch_stats=transport(self.anchor_stats,a,b,error,self.task+1)
            self.epoch_memory={}
            rawdiag=[]
            for path in range(2):
                ar,br,rr,dr=fit_map(self.last_anchor[:,path],raw[:,path],ys);rawdiag.append(dr)
                for c,s in self.anchor_raw.items():
                    dest=self.epoch_memory.setdefault(c,dict(mu=np.empty((2,768)),var=np.empty((2,768)),n=s['n'],arrival=s['arrival']))
                    dest['mu'][path]=s['mu'][path]*ar+br;dest['var'][path]=s['var'][path]*ar**2
            jsonl(self.r.pub/('transport_audit.jsonl' if self.r.phase=='formal' else 'engineering_transport.jsonl'),dict(dataset=self.name,seed=self.seed,stream=self.stream,
                  task=self.task+1,epoch=epoch+1,anchor_version=self.task,target_version=f'{self.task+1}.{epoch+1}',
                  old_classes=self._known_classes,committed=epoch==9,joint=diag,raw=rawdiag,
                  old_mean_norms=np.linalg.norm(self.epoch_stats['mu'],axis=0).tolist(),
                  old_trace=float(np.trace(self.epoch_stats['S'])),e_mean=float(self.epoch_stats['e'].mean())))
        else:self.epoch_stats=copy.deepcopy(self.anchor_stats)
        self.last_features=(raw,pt,ys)
        parts=self.epoch_parts;row=dict(dataset=self.name,seed=self.seed,stream=self.stream,task=self.task+1,
              epoch=epoch+1,optimizer_steps=len(parts),cumulative_steps=self.steps,train_seconds=training_seconds,
              lr_used=self.lr,lr_next=[g['lr'] for g in optimizer.param_groups],loss=stats['loss'],
              all_seen_train_accuracy=100*sum(x['all_seen_correct'] for x in parts)/sum(x['n'] for x in parts),
              current_restricted_accuracy=100*sum(x['restricted_correct'] for x in parts)/sum(x['n'] for x in parts),
              adapter_gradient_tensors=self.epoch_gradient_nonzero,future_data_gradient_max=0.,
              **{k:float(np.mean([x[k] for x in parts])) for k in ('main_CE','sum_CE','pool_weighted_few','assignment','pull','replay_CE','replay_weight')})
        assert self.epoch_gradient_nonzero>0,'BLOCKED_NO_ADAPTER_GRADIENT'
        self.records.append(row);jsonl(self.r.pub/('train_epoch_metrics.jsonl' if self.r.phase=='formal' else 'engineering_epoch_metrics.jsonl'),row)
        self.current_epoch=epoch+1;self.save(self.r.private/'active_resume.pt',resume=True)
        self.r.resources();self.epoch_parts=[];self.epoch_gradient_nonzero=0
        self.lr=row['lr_next'];self.epoch_start=time.monotonic()
        print('EPOCH',self.name,self.seed,self.stream,self.task+1,epoch+1,flush=True)

    def payload(self,resume=False):
        p=dict(delta={k:t.detach().cpu().clone() for k,t in self._network.state_dict().items() if k in self.nonshared},
               nonshared_keys=sorted(self.nonshared),network_sha256=network_hash(self._network),shared_sha256=self.shared_hash,
               weight_sha256=WEIGHT_SHA,code_sha256=self.r.code,dataset=self.name,seed=self.seed,stream=self.stream,
               task=self.task,epoch=self.current_epoch,known=self._known_classes,seen=self._total_classes,
               order=self.order,steps=self.steps,records=self.records,stats=self.stats,
               stale=self.stale if self.stream=='U' else None,pt=self.pt if self.stream=='U' else None,
               raw_memory=self.raw_memory,rng=self._capture_rng_state(),loader_rng=self.loader_generator.get_state(),
               synthesis_rng=self.synth.get_state(),manifest_sha256=self.r.lock[self.name]['manifest_sha256'],
               args=dict(self.args,device=[str(x) for x in self.args['device']]))
        if resume:p.update(optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),
                           anchor=self.last_anchor,anchor_stats=self.anchor_stats,anchor_raw=self.anchor_raw,
                           epoch_stats=getattr(self,'epoch_stats',self.stats),epoch_memory=self.epoch_memory,
                           last_features=getattr(self,'last_features',None))
        return p

    def save(self,path,resume=False):
        self.r.resources();p=self.payload(resume);tmp=Path(str(path)+'.part')
        torch.save(p,tmp);tmp.replace(path);return p

    def restore(self,path,resume=False):
        p=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in [('dataset',self.name),('seed',self.seed),('stream',self.stream),('order',self.order),
                    ('weight_sha256',WEIGHT_SHA),('code_sha256',self.r.code),('shared_sha256',self.shared_hash),
                    ('manifest_sha256',self.r.lock[self.name]['manifest_sha256']),('nonshared_keys',sorted(self.nonshared))]:
            assert p[k]==v,'BLOCKED_CHECKPOINT_'+k
        assert p['args']==dict(self.args,device=[str(x) for x in self.args['device']])
        state=self._network.state_dict();state.update(p['delta']);self._network.load_state_dict(state,strict=True)
        assert network_hash(self._network)==p['network_sha256'],'BLOCKED_STRICT_RESTORE'
        self.task=p['task'];self._cur_task=p['task'];self.current_epoch=p['epoch']
        self._known_classes=p['known'];self._total_classes=p['seen'];self.current=range(p['known'],p['seen'])
        self.steps=p['steps'];self.records=p['records'];self.stats=p['stats'];self.raw_memory=p['raw_memory']
        if self.stream=='U':self.stale=p['stale'];self.pt=p['pt']
        self.loader_generator.set_state(p['loader_rng']);self.synth.set_state(p['synthesis_rng']);self._restore_rng_state(p['rng'])
        if resume:
            self.optimizer.load_state_dict(p['optimizer']);self.scheduler.load_state_dict(p['scheduler'])
            self.last_anchor=p['anchor'];self.anchor_stats=p['anchor_stats'];self.anchor_raw=p['anchor_raw']
            self.epoch_stats=p['epoch_stats'];self.epoch_memory=p['epoch_memory'];self.last_features=p['last_features']
        return p


def commit_statistics(l):
    task=l.task;raw,pt,y=l.last_features
    z=joint(raw);l.stats=append(l.epoch_stats,z,y,l.current,task+1)
    l.raw_memory=copy.deepcopy(l.epoch_memory)
    for c in l.current:
        x=raw[y==c].astype(np.float64);mu=x.mean(0)
        l.raw_memory[c]=dict(mu=mu,var=((x-mu)**2).mean(0),n=len(x),arrival=task+1)
    if l.stream=='U':
        l.stale=append(l.stale,z,y,l.current,task+1)
        l.pt=append(l.pt,pt.astype(np.float64),y,l.current,task+1)


def archive(r,l,path,p):
    """Publish only a strictly restored, immutable task state to the archive queue."""
    # Restore the complete nonshared state against its pinned core before transfer.
    before=p['network_sha256'];l.restore(path,False)
    assert network_hash(l._network)==before
    digest=sha(path);request=dict(name=path.name,sha256=digest,bytes=path.stat().st_size,
                                network_sha256=before,strict_source_restore=True,
                                nonshared_tensor_sha256=tensor_digest(p['delta'].items()))
    write_json(r.private/'ARCHIVE_REQUEST.json',request)
    ack=r.private/'archive_ack'/(path.name+'.json');deadline=time.monotonic()+180
    while not ack.exists():
        assert time.monotonic()<deadline,'BLOCKED_ARCHIVE_TIMEOUT'
        if (r.private/'ARCHIVE_FAILURE.json').exists():raise AssertionError('BLOCKED_ARCHIVE_FAILURE')
        time.sleep(2)
    received=read(ack)
    assert received['status']=='PASS' and received['sha256']==digest
    assert received['nonshared_tensor_sha256']==request['nonshared_tensor_sha256']
    assert received['readable_tensor_keys']==len(l.nonshared)
    r.lineage.append(dict(dataset=l.name,seed=l.seed,stream=l.stream,task=l.task+1,
                         name=path.name,sha256=digest,network_sha256=before,
                         archive=received['archive'],strict_restore=True,transfer_SHA_verified=True,
                         initializer_network_sha256=l.initial_hash,previous_task_network_sha256=getattr(l,'previous_network',None)))
    l.previous_network=before;write_json(r.pub/'MODEL_LINEAGE.json',r.lineage)
    path.unlink()  # Authorized only this run's verified temporary task copy.
    jsonl(r.pub/'STORAGE_LEDGER.jsonl',dict(**request,action='archived_new_task_and_removed_local_temporary'))


def parity(r,l):
    ds=r.dataset(l.name,l.seed,l.current,False,smoke=True)
    x=torch.stack([ds[i][1] for i in range(min(8,len(ds)))]).cuda()
    rng=l._capture_rng_state();before=network_hash(l._network)
    l._network.eval();refs=[]
    with torch.inference_mode():
        for i in range(len(x)):
            refs.append(l._network(x[i:i+1],train=False,weight=None))
        ref={k:torch.cat([o[k] for o in refs]) for k in ('pre_logits','pre_logits_few','prompt_idx','prompt_idx_few')}
        l.probe.load_state_dict(l._network.state_dict());l.probe.eval()
        errors={}
        for idx in (torch.arange(len(x)),torch.arange(len(x)-1,-1,-1)):
            o=l.probe(x[idx],train=False,weight=None)
            for k in ref:
                if k.startswith('prompt'):assert torch.equal(o[k],ref[k][idx])
                else:
                    torch.testing.assert_close(o[k],ref[k][idx],atol=1e-5,rtol=1e-5)
                    errors[k]=max(errors.get(k,0.),float((o[k]-ref[k][idx]).abs().max()))
        a=l.probe(x,train=False,weight=torch.ones(len(x),device='cuda'))
        b=l.probe(x,train=False,weight=torch.full((len(x),),999,device='cuda'))
        assert torch.equal(a['logits'],b['logits']) and torch.equal(a['pre_logits'],b['pre_logits'])
    assert before==network_hash(l._network) and rng_equal(rng,l._capture_rng_state())
    r.count('engineering_wrapper_calls',len(x)+4);r.count('engineering_internal_encoder_calls',3*(len(x)+4))
    return dict(status='PASS',native_B1_pointwise_max_abs=errors,label_frequency_independent=True)


def engineering(r):
    t=time.monotonic();math_checks=selfcheck();math_seconds=time.monotonic()-t
    records=[];first_hash={};first_inputs={}
    for name in ('HK','ISIC'):
        for stream in ('U','R'):
            l=CTLearner(r,name,1993,stream);l.smoke=True
            for task in range(2):
                l.task_setup(task);ds=r.dataset(name,1993,l.current,True,smoke=True)
                loader=DataLoader(ds,batch_size=48,shuffle=True,num_workers=0,generator=l.loader_generator)
                batches=list(loader);assert len(batches)>=1
                l.args['tuned_epoch']=1
                if task==0:
                    first_inputs[name,stream]=hashlib.sha256(batches[0][1].numpy().tobytes()).hexdigest()
                if task and stream=='R':
                    # Toy old-class memory contains no old image features and is discarded.
                    l.epoch_memory={c:dict(mu=np.zeros((2,768)),var=np.full((2,768),.02),n=4,arrival=1)
                                    for c in range(l._known_classes)}
                l._init_train([batches[0]],None,l.optimizer,None)
                if task==0:first_hash[name,stream]=network_hash(l._network)
                changed=[k for k,p in l._network.named_parameters() if k in l.adapter_before and not torch.equal(p.detach().cpu(),l.adapter_before[k])]
                assert changed and l.epoch_gradient_nonzero>0,'BLOCKED_FIRST_TASK_FREEZE'
                if task==1:
                    l.args['tuned_epoch']=10;path=r.private/'engineering_resume.pt'
                    l.current_epoch=0;l.save(path,True);base=l._capture_rng_state()
                    batch=batches[-1];results=[]
                    for _ in range(2):
                        l.restore(path,True);assert rng_equal(base,l._capture_rng_state())
                        l.args['tuned_epoch']=1;l._init_train([batch],None,l.optimizer,None);l.args['tuned_epoch']=10
                        results.append((network_hash(l._network),l.last_gradient))
                    assert results[0]==results[1],'BLOCKED_RESUME_NEXT_UPDATE'
                    path.unlink()
                    probe=parity(r,l)
                    records.append(dict(dataset=name,stream=stream,task2_adapter_updates=len(changed),
                                        strict_compact_next_step=True,pointwise=probe))
            del l,batches,loader,ds;gc.collect();torch.cuda.empty_cache();r.resources()
        assert first_hash[name,'U']==first_hash[name,'R'],'BLOCKED_TASK1_PAIRING'
        assert first_inputs[name,'U']==first_inputs[name,'R'],'BLOCKED_INPUT_PAIRING'
    # Exercise epoch-boundary transport and the actual checkpoint/statistics path.
    flow=CTLearner(r,'HK',1993,'R');flow.bounded_probe=True;flow.args['tuned_epoch']=2
    for task in (0,1):
        flow.task_setup(task)
        flow.last_anchor,_,_,anchor_y,_=r.extract(flow,flow.current)
        ds=r.dataset('HK',1993,flow.current,True,smoke=True)
        loader=DataLoader(ds,batch_size=48,shuffle=True,num_workers=0,generator=flow.loader_generator)
        flow._init_train(loader,None,flow.optimizer,flow.scheduler)
        assert np.array_equal(flow.last_features[2],anchor_y)
        commit_statistics(flow);W,_=ridge(flow.stats)
        if task==1:
            assert flow.stats['mu'].shape==(1536,4) and len(flow.raw_memory)==4
            assert np.any(flow.stats['e'][:,:2]>0) and np.all(flow.stats['e'][:,2:]==0)
            # Persisted final state must also reconstruct all transported statistics.
            p=r.private/'engineering_task_final.pt';flow.save(p);saved=flow.restore(p)
            np.testing.assert_array_equal(saved['stats']['S'],flow.stats['S']);p.unlink()
    (r.private/'active_resume.pt').unlink()
    del flow,loader,ds;gc.collect();torch.cuda.empty_cache()
    # The reference parity uses existing frozen-A train/val caches only. No images are opened.
    refs=[]
    for name,spec in r.cfg['datasets'].items():
        cache=spec['pt_cache'];order=r.lock[name]['orders'][1993]
        arrays={}
        for split in ('train','val'):
            p=Path(cache[split]);assert sha(p)==cache[split+'_sha256'],'BLOCKED_PT_CACHE_DRIFT'
            if p.suffix=='.npy':z=np.load(p)
            else:
                with np.load(p) as f:
                    z=f['z'].copy()
                    assert np.array_equal(f['y'],[int(x['original_label']) for x in r.rows[name][split]])
                assoc=read(cache[split+'_associations'])
                assert [x['sample_id'] for x in assoc]==[x['sample_id'] for x in r.rows[name][split]]
            arrays[split]=z.astype(np.float64)
        ys={s:np.array([int(x['original_label']) for x in r.rows[name][s]]) for s in ('train','val')}
        # Validate row identities using source cache association/manifest locks, not sizes alone.
        assert cache['row_order']=='sample_id ascending'
        st=empty(768)
        for c in order:st=append(st,arrays['train'][ys['train']==c],np.full(np.sum(ys['train']==c),len(st['n'])),[len(st['n'])],1)
        w,_=ridge(st);scores=arrays['val']@w
        from report_locked_holdout_r1 import predict_columns
        pred=np.array(order)[predict_columns(scores,np.array(order))]
        ba=100*np.mean([(pred[ys['val']==c]==c).mean() for c in order])
        assert abs(ba-cache['expected_final_BA'])<1e-5,'BLOCKED_PT_REFERENCE_REPRODUCTION'
        refs.append(dict(dataset=name,status='PASS',final_BA=ba,reference_only=True))
    resource=r.resources();early=r.preflight['projected_GPU_seconds_with_reserve']
    admission=dict(status='READY',projected_GPU_seconds=early+resource['current_worker_seconds'],
                   risk_max23_toy_seconds=math_checks['max23_toy']['seconds'],
                   risk_all45_conservative_seconds=45*math_checks['max23_toy']['seconds']*1.5,
                   archive_limit_bytes=10*1024**3,active_limit_bytes=3*1024**3)
    if admission['projected_GPU_seconds']+admission['risk_all45_conservative_seconds']>43200:
        admission['status']='BLOCKED_GPU_BUDGET'
    write_json(r.pub/'P0_ENGINEERING_TESTS.json',dict(status='PASS',neural=records,
               Task1_U_R_identical=True,epoch_boundary_transport_flow=True,transported_resume_flow=True,
               math=math_checks,math_wall_seconds=math_seconds,PT_reproduction=refs))
    write_json(r.pub/'RESOURCE_ADMISSION.json',admission)
    assert admission['status']=='READY',admission['status']
    write_json(r.pub/'READY_CT1.json',dict(status='READY',code_sha256=r.code,
               task_lock_sha256=sha(r.pub/'DATA_AND_TASK_LOCK.json'),weight_sha256=WEIGHT_SHA,
               source_commit=r.cfg['source_commit']))


def formal(r):
    ready=read(r.pub/'READY_CT1.json');assert ready['status']=='READY' and ready['code_sha256']==r.code
    assert not (r.pub/'FORMAL_STARTED.json').exists(),'BLOCKED_ALREADY_RELEASED'
    write_json(r.pub/'FORMAL_STARTED.json',dict(status='RUNNING',unix=time.time(),source_commit=r.cfg['source_commit']))
    r.phase='formal';init={};task1={};risk_fail=[]
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for stream in ('U','R'):
                l=CTLearner(r,name,seed,stream)
                init[name,seed,stream]=l.initial_hash
                if stream=='R':assert init[name,seed,'U']==l.initial_hash
                write_json(r.pub/f'ACTUAL_CONFIG_{name}_{seed}_{stream}.json',dict(l.args,device=['cuda:0'],locked_weight_path='PINNED_SHARED_AUGREG'))
                for task in range(len(l.tasks)):
                    l.task_setup(task)
                    l.last_anchor,_,_,anchor_y,_=r.extract(l,l.current)
                    ds=r.dataset(name,seed,l.current,True)
                    loader=DataLoader(ds,batch_size=48,shuffle=True,num_workers=4,drop_last=False,generator=l.loader_generator)
                    step_before=l.steps;l._init_train(loader,None,l.optimizer,l.scheduler)
                    assert l.steps-step_before==l.tasks[task]['steps']
                    changed=[k for k,p in l._network.named_parameters() if k in l.adapter_before and not torch.equal(p.detach().cpu(),l.adapter_before[k])]
                    assert changed,'BLOCKED_NO_ADAPTER_PARAMETER_UPDATE'
                    assert np.array_equal(l.last_features[2],anchor_y)
                    commit_statistics(l)
                    if task==0:
                        task1[name,seed,stream]=network_hash(l._network)
                        if stream=='R':assert task1[name,seed,'U']==task1[name,seed,'R'],'BLOCKED_TASK1_FORMAL_PAIRING'
                    W,diag=ridge(l.stats);weights={'CT-J-CB' if stream=='U' else 'CT-ConCM-CB':W}
                    if stream=='U':
                        weights['CT-J-Stale']=ridge(l.stale)[0];weights['PT-CB']=ridge(l.pt)[0]
                        try:solution,risk_record=risk(l.stats,W)
                        except (ImportError,ModuleNotFoundError) as e:
                            solution=None;risk_record=dict(status='PARTIAL_SOLVER',reason=str(e))
                        if solution is not None:
                            weights['CT-Risk']=solution['W']
                            npz(r.private/'sealed'/f'{name}_{seed}_risk_t{task+1:02d}.npz',**solution)
                        else:risk_fail.append(dict(dataset=name,seed=seed,task=task+1,**risk_record))
                        jsonl(r.pub/'risk_solver_audit.jsonl',dict(dataset=name,seed=seed,task=task+1,**risk_record))
                    npz(r.private/'sealed'/f'{name}_{seed}_{stream}_W_t{task+1:02d}.npz',**weights)
                    rawv,ptv,headv,yv,rows=r.extract(l,range(l._total_classes),'val')
                    zv=joint(rawv)
                    for method,w in weights.items():r.write_scores(l,method,w,ptv.astype(np.float64) if method=='PT-CB' else zv,yv,rows,l.order)
                    if stream=='R':
                        r.write_scores(l,'CT-ConCM',None,headv,yv,rows,l.order)
                    # Only diagnostics about train/engineering are public before full release.
                    jsonl(r.pub/'TASK_TECHNICAL_AUDIT.jsonl',dict(dataset=name,seed=seed,stream=stream,task=task+1,
                          steps=l.steps-step_before,adapter_parameter_updates=len(changed),ridge=diag,
                          raw_variance_max=float(max(s['var'].max() for s in l.raw_memory.values())),
                          ages=[task+1-a for a in l.stats['arrival']],input_classes=l.tasks[task]['classes']))
                    path=r.private/f'{name}_{seed}_{stream}_t{task+1:02d}.pt'
                    l.current_epoch=10;p=l.save(path);archive(r,l,path,p)
                    resume=r.private/'active_resume.pt'
                    if resume.exists():resume.unlink()
                    del p,rawv,zv,ptv,headv,loader,ds;l.last_anchor=None;l.last_features=None
                    r.resources()
                del l;gc.collect();torch.cuda.empty_cache()
    assert len(r.lineage)==90
    expected=270-len(risk_fail);assert len(r.sealed)==expected
    write_json(r.pub/'TRAJECTORIES_LOCK.json',dict(status='PARTIAL_SOLVER' if risk_fail else 'PASS',
               checkpoints=r.lineage,sealed_units=r.sealed,risk_failures=risk_fail,
               formal_task_epochs=900,optimizer_steps=r.access['formal_optimizer_steps'],
               new_test_predictions=0,source_commit=r.cfg['source_commit']))
    write_json(r.pub/'GPU_PHASE_COMPLETE.json',dict(status='PARTIAL_SOLVER' if risk_fail else 'PASS',
               tasks=90,formal_epochs=900,neural_training_stopped=True))
    write_json(r.private/'ARCHIVE_STOP.json',dict(status='STOP',expected_tasks=90))


def main():
    cfg=read(os.environ['P20_CONFIG']);r=Run(cfg)
    try:
        with threadpool_limits(limits=4):
            r.setup()
            if r.mode=='engineering':engineering(r)
            elif r.mode=='formal':formal(r)
            else:raise ValueError(r.mode)
    except BaseException as e:
        write_json(r.pub/f'FAILURE_{r.mode}.json',dict(status='BLOCKED',exception=type(e).__name__,reason=str(e),new_test_predictions=0))
        if r.mode=='formal':write_json(r.private/'ARCHIVE_STOP.json',dict(status='BLOCKED',reason=str(e)))
        raise
    finally:r.resources(enforce=False)


if __name__=='__main__':main()
