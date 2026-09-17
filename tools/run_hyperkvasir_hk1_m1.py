"""Fixed HK1-M1 S0 training and extraction; private runtime via P19_CONFIG.

Reuses the original S0 loss and optimizer. Never constructs a test loader.
The CPU readout/report entry is a separate process, so GPU residence ends here.
"""
import ast
import csv
import gc
import hashlib
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

from run_medical_v2 import (ROOT, Learner, Images, seed_all, network_hash, rng_equal,
                            write_json, sha, extend_embedding, effective_optimizer,
                            effective_scheduler, WEIGHT_SHA)
from prepare_hyperkvasir_hk1_m1 import ORDERS, SEEN, json_hash
from check_isic_a1_linear_native import install_batched_linear


def read(p): return json.loads(Path(p).read_text())
def tensor_digest(items):
    h=hashlib.sha256()
    for k,v in sorted(items):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Run:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root']);self.out=self.root/'output'
        self.pub=self.out/'public';self.private=self.out/'private'
        self.pub.mkdir(parents=True,exist_ok=True);self.private.mkdir(parents=True,exist_ok=True)
        self.protocol=read(self.pub/'PROTOCOL_HK1.json');self.mode=cfg['mode']
        self.previous=read(self.pub/'GPU_RESOURCE_LEDGER.json') if (self.pub/'GPU_RESOURCE_LEDGER.json').exists() else {}
        self.access=read(self.pub/'ACCESS_AUDIT.json');self.start=time.monotonic();self.gpu_start=None
        self.peak_bytes=self.previous.get('artifact_peak_bytes',0);self.min_free=shutil.disk_usage(self.root).free
        self.phase='engineering';self.parents=[];self.features=[];self.restores=[];self.parity=[]
        self.code={k:sha(ROOT/k) for k in cfg['code_files']}
        assert self.code==cfg['code_sha256'],'BLOCKED_CODE_DRIFT'
        assert sha(cfg['reference_s0_source'])==cfg['reference_s0_source_sha256'],'BLOCKED_REFERENCE_SOURCE_DRIFT'
        for split in ('train','val'):
            assert sha(self.root/'inputs'/(split+'.csv'))==self.protocol['manifest_sha256'][split]
        assert self.protocol['class_orders']=={str(k):v for k,v in ORDERS.items()}
        assert self.protocol['seen_classes']==SEEN
        assert read(self.pub/'INPUT_STAGING_AUDIT_HK1_M1.json')['status']=='PASS'
        assert sha(cfg['weight'])==WEIGHT_SHA
        self.rows={s:list(csv.DictReader((self.root/'inputs'/(s+'.csv')).open())) for s in ('train','val')}
        assert all([r['sample_id'] for r in rs]==sorted(r['sample_id'] for r in rs) for rs in self.rows.values())
        # An explicit two-manifest image universe; reserved stays on the source host.
        allowed={str((self.root/'images'/r['relative_path']).resolve()) for rs in self.rows.values() for r in rs}
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0]))
            if p.suffix.lower() in ('.jpg','.jpeg','.png'):assert str(p.resolve()) in allowed,'FORBIDDEN_IMAGE_ACCESS'
            assert p.name not in ('test.csv','reserved.csv','test.npz','test_associations.json'),'FORBIDDEN_TEST_ACCESS'
        sys.addaudithook(guard)

    def args(self,seed):
        a=read(ROOT/'third_party/APART/exps/apart_cifar_shuffle_concm_stage1_capped_headnorm_calibrated_full_seed1993_gpu0.json')
        a.update(nb_classes=23,nb_tasks=6,init_cls=13,increment=2,seed=seed,device=[torch.device('cuda:0')],
            medical_v2=True,locked_weight_path=self.cfg['weight'],tuned_epoch=10,
            concm_stage1=False,concm_stage1_eval_calibration=False,calibration_rule='none',
            lt_list=[self.protocol['fit_counts'][str(c)] for c in ORDERS[seed]],
            dataset=self.protocol['protocol_id'],weight_decay=.01,optimizer_profile='legacy_effective_v1',
            scheduler='S0_cosine',save_task_checkpoints=False,class_order=ORDERS[seed],
            embedding_seed=seed+1000003,data_loader_seed=seed,real_ce_scope='current')
        for k in ('longtail','order','task_checkpoint_dir'):a.pop(k,None)
        return a

    def serial_args(self,a):
        return dict(a,device=[str(d) for d in a['device']],locked_weight_path='LOCKED_PRETRAINED_REFERENCE')

    def count(self,key,n=1): self.access[key]=self.access.get(key,0)+n

    def audit_save(self,enforce=True):
        # This run's small output tree only, never scan historical assets or images.
        size=self.cfg['external_artifact_bytes']+sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        self.peak_bytes=max(self.peak_bytes,size);self.min_free=min(self.min_free,shutil.disk_usage(self.root).free)
        elapsed=0. if self.gpu_start is None else time.monotonic()-self.gpu_start
        prior=self.previous.get('GPU_process_residence_seconds',0.)
        record=dict(GPU_process_residence_seconds=prior+elapsed,worker_GPU_seconds=elapsed,
            artifact_peak_bytes=self.peak_bytes,min_free_bytes=self.min_free,
            input_staging_bytes=self.protocol['prospective_input_staging_bytes'],artifact_limit_bytes=768*1024**2,
            GPU_limit_seconds=14400,CPU_analytic_limit_seconds=7200,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if self.gpu_start else 0,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            batch_size=48,AMP=False,BLAS_threads=4,torch_threads=4,loader_workers=8,mode=self.mode)
        self.access['encoder_forward_calls']=sum(v for k,v in self.access.items() if k.endswith(('_original_calls','_main_calls','_few_calls')))
        write_json(self.pub/'GPU_RESOURCE_LEDGER.json',record);write_json(self.pub/'ACCESS_AUDIT.json',self.access)
        if enforce:
            assert size<768*1024**2 and self.min_free>=1024**3,'BLOCKED_STORAGE'
            assert prior+elapsed<14400,'BLOCKED_GPU_BUDGET'
        return record

    def setup(self):
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');self.gpu_start=time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def dataset(self,seed,split,smoke=False,all_classes=False):
        assert split in ('train','val')
        return Images(self.root/'inputs'/(split+'.csv'),self.root/'images',ORDERS[seed],
                      range(23 if all_classes else 13),split=='train' and not all_classes,smoke)

    def input(self,seed,indices,split='train'):
        ds=self.dataset(seed,split,all_classes=True)
        self.count(self.phase+'_image_reads',len(indices))
        return torch.stack([ds[i][1] for i in indices]).cuda()

    def forward(self,net,x):
        capture={}
        h=net.original_backbone.register_forward_hook(lambda m,a,v:capture.update(A=v['pre_logits']))
        with torch.inference_mode():o=net(x,train=False,weight=None)
        h.remove()
        result={k:o[k].detach().cpu().numpy() for k in ('pre_logits','pre_logits_few','prompt_idx','prompt_idx_few')}
        result['original_A']=torch.nn.functional.normalize(capture['A'],dim=1).cpu().numpy()
        return result

    def probe(self,l):
        net=l._network;net.requires_grad_(False).eval();original_hash=network_hash(net)
        # Only this discarded/restored inference instance receives the accepted patch.
        install_batched_linear(net)
        rows=self.rows['train'];ix=[i for i,r in enumerate(rows) if int(r['original_label']) in ORDERS[l.args['seed']][:13]][:48]
        t=time.monotonic();x=self.input(l.args['seed'],ix);torch.cuda.synchronize();input_seconds=time.monotonic()-t
        rng=l._capture_rng_state()
        refparts=[self.forward(net,x[i:i+1]) for i in range(48)]
        ref={k:np.concatenate([r[k] for r in refparts]) for k in refparts[0]}
        net.backbone.pool.batchwise_prompt=False;net.backbone.pool_few.batchwise_prompt=False
        records=[]
        for ids in (np.arange(48),np.arange(47,-1,-1),np.r_[np.arange(24),np.arange(24)]):
            torch.cuda.synchronize();t=time.monotonic();actual=self.forward(net,x[ids]);torch.cuda.synchronize()
            checks={}
            for k in ref:
                if k.startswith('prompt_idx'):assert np.array_equal(actual[k],ref[k][ids]),'BLOCKED_ROUTING_IDS'
                else:
                    checks[k]=float(np.max(np.abs(actual[k]-ref[k][ids])))
                    np.testing.assert_allclose(actual[k],ref[k][ids],atol=1e-5,rtol=1e-5)
            records.append(dict(batch=48,layout_sha256=json_hash(ids.tolist()),max_abs=checks,seconds=time.monotonic()-t))
        before=self.forward(net,x);after=self.forward(net,x)
        assert all(np.array_equal(before[k],after[k]) for k in before)
        assert original_hash==network_hash(net) and rng_equal(rng,l._capture_rng_state())
        assert all(not m.training for m in net.modules())
        row=dict(seed=l.args['seed'],status='PASS',layouts=records,unmodified_native_B1_reference=True,
                 labels_and_frequency_not_passed=True,parameters_buffers_and_RNG_unchanged=True,atol=1e-5,rtol=1e-5)
        self.parity.append(row)
        return max(r['seconds'] for r in records)+input_seconds

    def remove_temporary(self,p):
        assert p.parent==self.private and p.name in ('engineering_resume.pt','active_resume.pt')
        n=p.stat().st_size;p.unlink()
        with (self.pub/'STORAGE_LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(path=p.name,bytes=n,action='remove_this_run_registered_temporary_after_restore_PASS'))+'\n')


class HKLearner(Learner):
    def __init__(self,r,seed):
        self.r=r;seed_all(seed);super().__init__(r.args(seed))
        extend_embedding(self._network.backbone.assigner,seed)
        self._cur_task=0;self._known_classes=0;self._total_classes=13
        self.loader_generator=torch.Generator().manual_seed(seed)
        self.nonshared={'backbone.'+k for k in self._network.backbone.weight_load_audit['allowed_missing']}
        # Includes every non-shared tensor, even unused pool slots and future head rows.
        state=self._network.state_dict();assert self.nonshared<=set(state)
        self.shared_hash=tensor_digest((k,v) for k,v in state.items() if k not in self.nonshared)
        self.optimizer=effective_optimizer(self._network.backbone);self.scheduler=effective_scheduler(self.optimizer,0)
        self.records=[];self.steps=0;self.gradient_hashes=[];self.losses=[];self.smoke=False;self.observed_frequency_rows=set()
        def stepped(optimizer,args,kwargs):
            self.steps+=1
            r.count('engineering_optimizer_steps' if self.smoke else 'formal_training_optimizer_steps')
        self.optimizer.register_step_post_hook(stepped)
        allowed={'backbone.'+k for k in self._network.backbone.weight_load_audit['allowed_missing']}
        assert {k for k,p in self._network.named_parameters() if p.requires_grad}==allowed
        assert {id(p) for g in self.optimizer.param_groups for p in g['params']}=={id(p) for p in self._network.parameters() if p.requires_grad}
        self._network.to(self._device)
        def hook(module,args):
            r.count(r.phase+'_wrapper_calls');r.count(r.phase+'_wrapper_image_rows',len(args[0]))
        self._network.register_forward_pre_hook(hook)
        for module,name in ((self._network.original_backbone,'original'),):
            module.register_forward_pre_hook(lambda m,a,n=name:(r.count(r.phase+'_'+n+'_calls'),r.count(r.phase+'_'+n+'_image_rows',len(a[0]))) and None)
        native=self._network.backbone.forward_features
        def features(x,*args,**kwargs):
            name='few' if kwargs.get('few',0)==1 else 'main'
            r.count(r.phase+'_'+name+'_calls');r.count(r.phase+'_'+name+'_image_rows',len(x))
            return native(x,*args,**kwargs)
        self._network.backbone.forward_features=features

    def payload(self,epoch,resume):
        state=self._network.state_dict()
        assert tensor_digest((k,v) for k,v in state.items() if k not in self.nonshared)==self.shared_hash,'BLOCKED_SHARED_CORE_CHANGED'
        p=dict(delta={k:state[k].detach().cpu().clone() for k in sorted(self.nonshared)},
            shared_core_sha256=self.shared_hash,network_sha256=network_hash(self._network),seed=self.args['seed'],
            order=ORDERS[self.args['seed']],training_args=self.r.serial_args(self.args),epoch=epoch,steps=self.steps,
            weight_sha256=WEIGHT_SHA,protocol_sha256=self.r.cfg['protocol_sha256'],code_sha256=self.r.code,
            manifest_sha256={s:self.r.protocol['manifest_sha256'][s] for s in ('train','val')},
            nonshared_keys=sorted(self.nonshared),records=self.records,phase='resume' if resume else 'S0_final',
            rng=self._capture_rng_state(),loader_rng=self.loader_generator.get_state())
        if resume:p.update(optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict())
        return p

    def save(self,path,epoch,resume=True):
        p=self.payload(epoch,resume);temp=path.with_suffix('.pt.part')
        estimate=sum(t.numel()*t.element_size() for t in p['delta'].values())*(3 if resume else 1)+1024**2
        current=self.r.cfg['external_artifact_bytes']+sum(q.stat().st_size for q in self.r.out.rglob('*') if q.is_file())
        assert current+estimate<=768*1024**2 and shutil.disk_usage(self.r.root).free-estimate>=1024**3,'BLOCKED_STORAGE'
        torch.save(p,temp);self.r.audit_save();temp.replace(path)
        return p

    def restore(self,path,resume):
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['seed']==self.args['seed'] and p['order']==ORDERS[self.args['seed']]
        assert p['weight_sha256']==WEIGHT_SHA and p['code_sha256']==self.r.code and p['protocol_sha256']==self.r.cfg['protocol_sha256']
        assert p['training_args']==self.r.serial_args(self.args) and p['nonshared_keys']==sorted(self.nonshared)
        assert p['shared_core_sha256']==self.shared_hash
        state=self._network.state_dict();state.update(p['delta']);self._network.load_state_dict(state,strict=True)
        assert network_hash(self._network)==p['network_sha256'],'BLOCKED_COMPACT_TENSOR_RESTORE'
        self.steps=p['steps'];self.records=p['records']
        if resume:
            self.optimizer.load_state_dict(p['optimizer']);self.scheduler.load_state_dict(p['scheduler'])
            self.loader_generator.set_state(p['loader_rng']);self._restore_rng_state(p['rng'])
        return p

    def _v2_input_hook(self,epoch,batch,inputs,targets):
        assert targets.min()>=0 and targets.max()<13
        assert inputs.shape[0]<=48 and inputs.dtype==torch.float32 and not torch.is_autocast_enabled()
        self.r.count(self.r.phase+'_image_reads',len(inputs));self.r.audit_save()

    def _v2_gradient_hook(self,epoch,batch,loss):
        assert torch.isfinite(loss),'BLOCKED_NONFINITE_LOSS'
        for k,p in self._network.named_parameters():
            if p.grad is not None:
                assert k in self.nonshared and torch.isfinite(p.grad).all(),'BLOCKED_GRADIENT'
                if k in ('backbone.head.weight','backbone.head.bias','backbone.head_few.weight','backbone.head_few.bias'):
                    assert torch.count_nonzero(p.grad[13:])==0,'BLOCKED_FUTURE_CE_GRADIENT'
        if self.smoke:
            self.gradient_hashes.append(tensor_digest((k,p.grad) for k,p in self._network.named_parameters() if p.grad is not None))
            self.losses.append(float(loss))
        grad=self._network.backbone.assigner.cls_emb.weight.grad
        if grad is not None:
            observed=set(torch.where(grad.abs().sum(1)>0)[0].tolist())
            assert observed<=set(self.args['lt_list'][:13]),'BLOCKED_FREQUENCY_INDEX'
            self.observed_frequency_rows.update(observed)

    def _v2_epoch_hook(self,epoch,optimizer,scheduler,stats):
        if self.smoke:return
        torch.cuda.synchronize()
        row=dict(seed=self.args['seed'],epoch=epoch+1,seconds=time.monotonic()-self.epoch_start,
                 steps=self.steps,lr_used=self.lr,lr_next=[g['lr'] for g in optimizer.param_groups],
                 gradients_finite_all_steps=True,future_head_CE_gradient_max_abs=0.,
                 frequency_gradient_rows=sorted(self.observed_frequency_rows),**stats)
        self.records.append(row);self.lr=row['lr_next'];self.r.count('new_formal_neural_training_epochs')
        self.save(self.r.private/'active_resume.pt',epoch+1)
        with (self.r.pub/'S0_TRAINING_EPOCHS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print('EPOCH',self.args['seed'],epoch+1,stats['loss'],flush=True)
        self.r.audit_save();self.epoch_start=time.monotonic()


def reference_train(path):
    tree=ast.parse(Path(path).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Learner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_init_train')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    import models.apart
    namespace=dict(vars(models.apart));exec(compile(module,str(path),'exec'),namespace)
    return namespace['_init_train']


def engineering(r):
    from report_hyperkvasir_hk1_m1 import engineering as analytic_engineering
    t=time.monotonic();analytic=analytic_engineering(r.private)
    analytic['wall_seconds']=time.monotonic()-t
    timings=[];probe_seconds=[];records=[];sizes=[]
    oldtrain=reference_train(r.cfg['reference_s0_source'])
    for seed in ORDERS:
        r.phase='engineering';l=HKLearner(r,seed);l.smoke=True
        initial=network_hash(l._network)
        ds=r.dataset(seed,'train',smoke=True)
        batches=list(DataLoader(ds,batch_size=48,shuffle=True,num_workers=0,generator=l.loader_generator))
        assert len(batches)==2 and all(len(b[1])==48 for b in batches)
        start_rng=l._capture_rng_state();l.args['tuned_epoch']=1
        torch.cuda.synchronize();t=time.monotonic();l._init_train([batches[0]],None,l.optimizer,None);torch.cuda.synchronize()
        timings.append(time.monotonic()-t)
        after_one=network_hash(l._network);first_gradient=l.gradient_hashes[-1];first_loss=l.losses[-1]
        l.args['tuned_epoch']=10
        if seed==1993:
            path=r.private/'engineering_resume.pt';l.save(path,0);saved_rng=l._capture_rng_state()
        l.args['tuned_epoch']=1;torch.cuda.synchronize();t=time.monotonic()
        l._init_train([batches[1]],None,l.optimizer,None);torch.cuda.synchronize();timings.append(time.monotonic()-t)
        after_two=network_hash(l._network);second_gradient=l.gradient_hashes[-1];second_loss=l.losses[-1];l.args['tuned_epoch']=10
        if seed==1993:
            del l;gc.collect();torch.cuda.empty_cache()
            l=HKLearner(r,seed);l.smoke=True;assert network_hash(l._network)==initial
            l.concm_stage1_enabled=True # Empty S0 replay must remain algebraically inactive.
            l._restore_rng_state(start_rng);l.args['tuned_epoch']=1
            oldtrain(l,[batches[0]],None,l.optimizer,None)
            assert network_hash(l._network)==after_one and l.gradient_hashes[-1]==first_gradient and l.losses[-1]==first_loss,'BLOCKED_V2_S0_SEMANTICS'
            l.args['tuned_epoch']=10;l.restore(path,True);assert rng_equal(saved_rng,l._capture_rng_state())
            l.args['tuned_epoch']=1;l._init_train([batches[1]],None,l.optimizer,None);l.args['tuned_epoch']=10
            assert network_hash(l._network)==after_two and l.gradient_hashes[-1]==second_gradient and l.losses[-1]==second_loss,'BLOCKED_SAME_NEXT_STEP_RESTORE'
            r.remove_temporary(path)
        byte_count=sum(p.numel()*p.element_size() for k,p in l._network.state_dict().items() if k in l.nonshared);sizes.append(byte_count)
        gradlist=[dict(name=k,shape=list(p.shape),numel=p.numel()) for k,p in l._network.named_parameters() if p.requires_grad]
        write_json(r.pub/f'TRAINABLE_PARAMETERS_{seed}.json',gradlist)
        probe_seconds.append(r.probe(l));records.append(dict(seed=seed,initial_tensor_sha256=initial,nonshared_tensor_bytes=byte_count,
            future_head_CE_gradient_zero=True,allowlist_exact=True,frequency_embedding_shape=[12726,16],
            gradient_sha256=l.gradient_hashes,smoke_losses=l.losses,smoke_steps=4 if seed==1993 else 2))
        del l,batches;gc.collect();torch.cuda.empty_cache();r.audit_save()
    n=[sum(r.protocol['fit_counts'][str(c)] for c in o[:13]) for o in ORDERS.values()]
    steps=[10*math.ceil(x/48) for x in n];nall=r.protocol['n_train']+r.protocol['n_val']
    projected_train=max(timings[1::2])*sum(steps)*1.5
    projected_extract=max(probe_seconds)*math.ceil(nall/48)*3*1.5
    # Retain all deltas, one A + three raw S caches, final sufficient stats, W/scores.
    training_peak=8*max(sizes)+16*1024**2+r.cfg['external_artifact_bytes']
    analytic_peak=3*max(sizes)+nall*(768+3*1536)*4+3*(1536**2+2*768**2)*8+32*1024**2+r.cfg['external_artifact_bytes']
    used=r.audit_save();storage=max(training_peak,analytic_peak)
    admission=dict(status='READY',true_S0_counts=n,formal_steps_by_seed=steps,step_seconds=timings,
        extraction_batch48_seconds=probe_seconds,projected_train_seconds=projected_train,projected_extract_seconds=projected_extract,
        projected_total_GPU_seconds=used['GPU_process_residence_seconds']+projected_train+projected_extract+600,
        artifact_peak_projected_bytes=storage,post_staging_free_bytes=shutil.disk_usage(r.root).free)
    if admission['projected_total_GPU_seconds']>14400:admission['status']='BLOCKED_GPU_BUDGET'
    if storage>768*1024**2 or shutil.disk_usage(r.root).free-storage<1024**3:admission['status']='BLOCKED_STORAGE'
    write_json(r.pub/'RESOURCE_ADMISSION_HK1.json',admission)
    write_json(r.pub/'ROUTING_PARITY_HK1.json',r.parity)
    write_json(r.pub/'ENGINEERING_HK1.json',dict(status='PASS',parents=records,analytic=analytic,
        locked_V2_loss_gradient_update_bitwise_equal=True,empty_replay_callback_disabled_after_equivalence=True,
        compact_same_next_step_bitwise_equal=True,native_training_Linear=True,engineering_updates_discarded=True))
    assert admission['status']=='READY',admission['status']
    mapping=read(r.pub/'MAPPING_AUDIT_HK1_M1.json');mapping['full_neural_engineering']='PASS'
    write_json(r.pub/'MAPPING_AUDIT_HK1_M1.json',mapping)
    write_json(r.pub/'READY_HK1.json',dict(status='READY',code_sha256=r.code,protocol_sha256=r.cfg['protocol_sha256'],
        data_manifest_sha256={s:r.protocol['manifest_sha256'][s] for s in ('train','val')},weight_sha256=WEIGHT_SHA))


def formal(r):
    ready=read(r.pub/'READY_HK1.json');assert ready['code_sha256']==r.code and ready['status']=='READY'
    assert not (r.pub/'ALL_S0_LOCK_HK1.json').exists(),'FORMAL_ALREADY_STARTED'
    for seed in ORDERS:
        r.phase='formal_training';l=HKLearner(r,seed)
        config=r.serial_args(l.args);write_json(r.pub/f'ACTUAL_TRAIN_CONFIG_{seed}.json',config)
        write_json(r.pub/f'INITIALIZATION_LOCK_{seed}.json',dict(seed=seed,network_sha256=network_hash(l._network),
            shared_core_sha256=l.shared_hash,weight_sha256=WEIGHT_SHA,order=ORDERS[seed],no_supervised_parent=True))
        ds=r.dataset(seed,'train');loader=DataLoader(ds,batch_size=48,shuffle=True,num_workers=8,generator=l.loader_generator,drop_last=False)
        l.lr=[g['lr'] for g in l.optimizer.param_groups];l.epoch_start=time.monotonic()
        l._init_train(loader,None,l.optimizer,l.scheduler)
        assert l.steps==10*math.ceil(len(ds)/48) and len(l.records)==10
        path=r.private/f'parent_{seed}.pt';p=l.save(path,10,False);r.count('new_formal_S0_runs')
        r.phase='final_restore_probe';x=r.input(seed,list(range(48)));l._network.eval()
        ref=r.forward(l._network,x);del l,loader,ds;gc.collect();torch.cuda.empty_cache()
        restored=HKLearner(r,seed);restored.restore(path,False);restored._network.eval()
        actual=r.forward(restored._network,x)
        assert all(np.array_equal(actual[k],ref[k]) for k in ref),'BLOCKED_COMPACT_FEATURE_RESTORE'
        line=dict(seed=seed,epoch=10,steps=p['steps'],checkpoint=path.name,checkpoint_sha256=sha(path),
                  network_sha256=p['network_sha256'],strict_restore=True,feature_bitwise_equal=True,
                  config_sha256=json_hash(config),order=ORDERS[seed],only_own_S0_fit=True)
        r.parents.append(line);write_json(r.pub/'CHECKPOINT_RESTORE_AUDIT.json',r.parents)
        r.remove_temporary(r.private/'active_resume.pt')
        del restored,p,x;gc.collect();torch.cuda.empty_cache();r.audit_save()
    assert r.access['new_formal_neural_training_epochs']==30
    write_json(r.pub/'ALL_S0_LOCK_HK1.json',dict(status='PASS',parents=r.parents,neural_training_closed=True,
        code_sha256=r.code,source_commit=r.cfg['source_commit'],manifest_sha256=ready['data_manifest_sha256']))
    r.phase='formal_extraction'
    for seed in ORDERS:
        l=HKLearner(r,seed);l.restore(r.private/f'parent_{seed}.pt',False)
        # Drop unused optimizer objects. No subsequent optimizer.step is permitted.
        del l.optimizer,l.scheduler;l._network.requires_grad_(False).eval();r.phase='final_parity';r.probe(l)
        write_json(r.pub/'FINAL_ROUTING_PARITY_HK1.json',r.parity);r.phase='formal_extraction'
        h=network_hash(l._network);rng=l._capture_rng_state()
        for split in ('train','val'):
            ds=r.dataset(seed,split,all_classes=True);path=r.private/f'raw_{seed}_{split}.npy'
            raw=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(ds),2,768))
            apath=r.private/f'A_{split}.npy'
            a=np.lib.format.open_memmap(apath,mode='w+',dtype='float32',shape=(len(ds),768)) if seed==1993 else None
            loader=DataLoader(ds,batch_size=48,shuffle=False,num_workers=8,drop_last=False,generator=torch.Generator().manual_seed(seed))
            offset=0;t=time.monotonic()
            for _,x,_ in loader:
                r.count('formal_extraction_image_reads',len(x));o=r.forward(l._network,x.cuda())
                z=np.stack([o['pre_logits'],o['pre_logits_few']],1);assert np.isfinite(z).all()
                raw[offset:offset+len(x)]=z
                if a is not None:a[offset:offset+len(x)]=o['original_A']
                offset+=len(x);r.count('new_train_val_raw_feature_rows',len(x));r.audit_save()
            assert offset==len(ds);raw.flush();del raw
            if a is not None:a.flush();del a
            lock=dict(seed=seed,split=split,rows=offset,raw_sha256=sha(path),manifest_sha256=r.protocol['manifest_sha256'][split],
                      parent_network_sha256=h,dtype='float32',shape=[offset,2,768],seconds=time.monotonic()-t,
                      routing='S0_POINTWISE_PROBE',linear_3D='per-image native, B1 unchanged',batch=48)
            if seed==1993:lock['A_sha256']=sha(apath)
            r.features.append(lock);write_json(r.pub/'FEATURE_CACHE_LOCK_HK1.json',r.features)
        assert network_hash(l._network)==h and rng_equal(rng,l._capture_rng_state())
        del l,loader,ds;gc.collect();torch.cuda.empty_cache();r.audit_save()
    write_json(r.pub/'GPU_PHASE_COMPLETE.json',dict(status='PASS',parents=3,raw_feature_caches=6,A_caches=2,
        incremental_optimizer_steps_S1_to_S5=0,new_test_model_forwards=0,neural_phase_stopped=True))


def main():
    cfg=read(os.environ['P19_CONFIG']);r=Run(cfg)
    try:
        with threadpool_limits(limits=4):
            r.setup()
            if r.mode=='engineering':engineering(r)
            elif r.mode=='formal':formal(r)
            else:raise ValueError(r.mode)
    except BaseException as e:
        write_json(r.pub/f'FAILURE_{r.mode}.json',dict(status='BLOCKED',reason=str(e),exception=type(e).__name__,formal_released=r.mode=='formal'))
        raise
    finally:r.audit_save(enforce=False)


if __name__=='__main__':main()
