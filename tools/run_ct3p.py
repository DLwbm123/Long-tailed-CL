"""CT3-P bounded prefix. CT1 Run/formal are never constructed or called."""
import copy,csv,gc,json,math,multiprocessing as mp,os,shutil,sys,time,types
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from run_ct1 import CTLearner,Run,read,jsonl,npz
from run_medical_v2 import ROOT,Images,sha,network_hash,write_json,WEIGHT_SHA,rng_equal
from run_hyperkvasir_hk1_m1 import tensor_digest
from preflight_ct1 import task_lock
from ct1_statistics import empty,append,fit_map,transport,ridge,joint
from ct3p_core import fd,translation,math_check

class CountedImages(Images):
    def __getitem__(self,i):
        value=super().__getitem__(i)
        with self.counter.get_lock():
            self.counter[0 if self.split=='train' else 1]+=1
            if self.offline and self.split=='train' and value[2]<4:self.counter[2]+=1
        return value

class Prefix:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['root']);self.pub=self.root/'output/public';self.private=self.root/'output/private'
        for p in [self.pub,self.private,self.private/'sealed',self.private/'banks',self.private/'tmp',self.private/'parents',self.private/'control_features']:p.mkdir(parents=True,exist_ok=True)
        self.old=Path(cfg['ct1_root']);self.lock=task_lock(cfg);self.oldcode=read(self.old/'output/public/SOURCE_COMMIT_BINDING.json')['qualified_code_sha256']
        self.code={f:sha(ROOT/f) for f in cfg['qualified_files']};assert self.code==cfg['qualified_sha256']
        self.mode=cfg['mode'];self.phase='engineering' if self.mode=='gate' else self.mode
        self.prior=read(self.pub/'RESOURCE_AND_ACCESS_LEDGER.json') if (self.pub/'RESOURCE_AND_ACCESS_LEDGER.json').exists() else {}
        self.access=dict(self.prior.get('calls',cfg.get('early_calls',{})));self.counter=mp.Array('q',[0,0,0]);self.gpu_start=None;self.started=time.monotonic()
        self.allowed=set();self.scope=None;self.maxbytes=self.prior.get('active_peak_bytes',0);self.archived=self.prior.get('archive_bytes',0)
        self.entries=read(self.private/'CANDIDATES.json') if (self.private/'CANDIDATES.json').exists() else []
        self.lineage=read(self.pub/'PARENT_AND_FORK_LINEAGE.json') if (self.pub/'PARENT_AND_FORK_LINEAGE.json').exists() else []
        self.sealed=read(self.private/'SEALED_INDEX.json') if (self.private/'SEALED_INDEX.json').exists() else []
        self.minfree=shutil.disk_usage(self.root).free
        oldroot=self.old.resolve()
        def guard(event,args):
            if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
            p=Path(os.fsdecode(args[0]));rp=p.resolve();s=str(rp)
            if p.suffix.lower() in ('.jpg','.jpeg','.png'):assert s in self.allowed,'BLOCKED_OLD_FUTURE_IMAGE'
            if any(x.lower().startswith(('test','reserved')) for x in p.parts):raise AssertionError('BLOCKED_TEST')
            assert not ('/ct2d/' in s and '/private/' in s),'BLOCKED_ORACLE_ASSET'
            flags=args[2] if len(args)>2 else 0
            if flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert not rp.is_relative_to(oldroot),'BLOCKED_CT1_WRITE'
            if rp.is_relative_to(oldroot/'output/private/sealed'):
                assert any(p.name==f'{n}_{seed}_{method}_t{t:02d}.npz' for n in ('HK','ISIC') for seed in (1993,1994,1995) for t in (1,2,3) for method in ('U_W','CT-J-CB','PT-CB')),'BLOCKED_NONPREFIX_ASSET'
        sys.addaudithook(guard)
        self.parent_view=types.SimpleNamespace(cfg=cfg,lock=self.lock,code=self.oldcode,phase=self.phase,count=self.count)
    def count(self,k,n=1):self.access[k]=self.access.get(k,0)+n
    def setup(self):
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');self.gpu_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def resources(self,enforce=True):
        size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink());self.maxbytes=max(self.maxbytes,size)
        self.minfree=min(self.minfree,shutil.disk_usage(self.root).free)
        early=self.cfg['early_GPU_seconds'];gpu=self.prior.get('GPU_process_residence_seconds',early)+(time.monotonic()-self.gpu_start if self.gpu_start else 0)
        x=dict(status='RUNNING',phase=self.mode,calls=self.access,GPU_process_residence_seconds=gpu,active_bytes=size,active_peak_bytes=self.maxbytes,
            archive_bytes=self.archived,persistent_archive_upper_bytes=size+self.archived,min_free_bytes=self.minfree,
            peak_GPU_allocated_bytes=max(self.prior.get('peak_GPU_allocated_bytes',0),torch.cuda.max_memory_allocated() if self.gpu_start else 0),
            online_old_fit_reads=0,online_future_fit_reads=0,offline_diagnostic_old_fit_reads=self.prior.get('offline_diagnostic_old_fit_reads',0)+self.counter[2],
            fit_image_reads=self.prior.get('fit_image_reads',self.cfg.get('early_image_reads',0))+self.counter[0],val_image_reads=self.prior.get('val_image_reads',0)+self.counter[1],
            new_test_image_reads=0,new_test_feature_reads=0,new_test_prediction_reads=0,new_test_model_forwards=0,new_test_predictions=0,
            encoder_forward_calls=sum(v for k,v in self.access.items() if k.endswith('_internal_encoder_calls')),formal_optimizer_steps=self.access.get('formal_optimizer_steps',0),engineering_optimizer_steps=self.cfg['early_steps']+self.access.get('engineering_optimizer_steps',0),
            formal_task_epochs=self.access.get('formal_task_epochs',0),new_checkpoints=len(self.lineage),
            CPU_archive_and_engineering_reserve_seconds=600,CPU_analytic_seconds=self.prior.get('CPU_analytic_seconds',0)+self.access.get(self.mode+'_analytic_ms',0)/1000)
        write_json(self.pub/'RESOURCE_AND_ACCESS_LEDGER.json',x)
        if enforce:assert gpu<14400 and x['CPU_analytic_seconds']+600<7200 and size<2*1024**3 and size+self.archived<3*1024**3 and self.minfree>=1024**3,'BLOCKED_RESOURCE'
        return x
    def permit(self,name,seed,task):self.scope=(name,seed,task)
    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert self.scope is not None and (name,seed)==self.scope[:2];task=self.scope[2];classes=list(classes)
        limit=list(range(2*(task-1),2*task)) if split=='train' and self.mode!='oracle' else list(range(2*task))
        assert set(classes)<=set(limit),'BLOCKED_OLD_OR_FUTURE_CLASSES'
        if self.mode=='oracle':assert task==3 and seed==1993 and (self.pub/'ONLINE_PREDICTIONS_LOCK.json').exists()
        if split=='val' and self.mode not in ('gate','controls','evaluate','oracle'):raise AssertionError('BLOCKED_EARLY_VAL')
        spec=self.cfg['datasets'][name];ds=CountedImages(Path(spec['manifests'])/(split+'.csv'),spec['images'],self.lock[name]['orders'][seed],classes,train,smoke)
        ds.rows.sort(key=lambda r:r['sample_id']);ds.counter=self.counter;ds.split=split;ds.offline=self.mode=='oracle'
        self.allowed={str((Path(spec['images'])/x['relative_path']).resolve()) for x in ds.rows};return ds
    def extract(self,l,classes,split='train',smoke=False):
        routes=[]
        h=l.probe.register_forward_hook(lambda m,a,o: routes.append(torch.stack([o['prompt_idx'].flatten(),o['prompt_idx_few'].flatten()],1).detach().cpu().numpy()))
        try:result=Run.extract(self,l,classes,split,smoke)
        finally:h.remove()
        self.last_routes=np.concatenate(routes);return result
    def exchange(self,op,name,digest,**extra):
        started=time.monotonic();key=str(time.time_ns());write_json(self.private/'REQUEST.json',dict(id=key,op=op,name=name,sha256=digest,**extra));end=time.monotonic()+240
        while time.monotonic()<end:
            p=self.private/'ACK.json'
            if p.exists():
                ack=read(p)
                if ack['id']==key:
                    assert ack['status']=='PASS',ack;self.count(self.mode+'_transfer_ms',int(1000*(time.monotonic()-started)));return ack
            time.sleep(1)
        raise TimeoutError('BLOCKED_TRANSFER')
    def acquire(self,name,seed,task,stream='U'):
        if stream=='U':e=next(x for x in read(self.old/'output/public/MODEL_LINEAGE.json') if (x['dataset'],x['seed'],x['stream'],x['task'])==(name,seed,'U',task))
        else:e=next(x for x in self.lineage if (x['dataset'],x['seed'],x['task'])==(name,seed,task))
        self.exchange('GET',e['name'],e['sha256']);p=self.private/'parent.part';assert sha(p)==e['sha256'];p.replace(self.private/'parent.pt')
        return self.private/'parent.pt',e
    def original(self,name,seed,task):
        path,e=self.acquire(name,seed,task);l=CTLearner(self.parent_view,name,seed,'U');p=l.restore(path);l.r=self
        assert p['task']==task-1 and p['epoch']==10 and p['seen']==2*task
        return l,p,e
    def save_candidate(self,l,method,state,source):
        W,diag=ridge(state);self.count('analytic_fits');stem=f'{l.name}_{l.seed}_{method}_t{l.task+1:02d}'
        npz(self.private/'banks'/(stem+'.npz'),W=W)
        e=dict(dataset=l.name,seed=l.seed,task=l.task+1,method=method,W_file=stem+'.npz',W_sha256=sha(self.private/'banks'/(stem+'.npz')),seen=l._total_classes,known=l._known_classes,source=source,network_sha256=network_hash(l._network))
        self.entries.append(e);write_json(self.private/'CANDIDATES.json',self.entries)
        jsonl(self.pub/'READOUT_AUDIT.jsonl',dict(dataset=l.name,seed=l.seed,task=l.task+1,method=method,**diag))
    def archive(self,l):
        name=f'{l.name}_{l.seed}_F_t{l.task+1:02d}.pt';path=self.private/name;p=l.save_new(path,False);l.restore_new(path,False)
        digest=sha(path);ack=self.exchange('PUT',name,digest,bytes=path.stat().st_size,network_sha256=p['network_sha256'],delta_sha256=tensor_digest(p['delta'].items()))
        assert ack['readable'];self.archived=ack['archive_bytes']
        e=dict(dataset=l.name,seed=l.seed,task=l.task+1,name=name,sha256=digest,network_sha256=p['network_sha256'],teacher_source={k:l.teacher_source[k] for k in ('name','sha256','network_sha256','task')},source_commit=self.cfg['source_commit'])
        self.lineage.append(e);write_json(self.pub/'PARENT_AND_FORK_LINEAGE.json',self.lineage);path.unlink();self.resources();return e

class FDStudent(CTLearner):
    def __init__(self,r,name,seed):
        super().__init__(r.parent_view,name,seed,'U');self.r=r;self.beta=10.;self.teacher=None;self.T=empty(1536);self.fdparts=[]
    def fork_from_ct1_task1(self,path,entry):
        self.r,self.r_save=self.r.parent_view,self.r
        try:p=super().restore(path)
        finally:self.r=self.r_save;del self.r_save
        assert p['task']==0 and p['epoch']==10 and p['seen']==2 and sha(path)==entry['sha256']
        self.T=copy.deepcopy(self.stats);self.steps=0;self.records=[];self.teacher_source=entry;self.parent_steps=p['steps']
        jsonl(self.r.pub/'FORK_AUDIT.jsonl',dict(phase=self.r.phase,dataset=self.name,seed=self.seed,parent_SHA=entry['sha256'],parent_network=p['network_sha256'],
            parent_source='8e63785896d17126114a436027d1cb0890b4658f',ordinary_CT1_restore_passed=True,args_changes={},new_protocol=dict(FD_beta=10.,transport_banks=['A','T'],only_tasks=[2,3]),all_RNG_restored=True))
    def task_setup(self,task):
        super().task_setup(task);self.current_epoch=0;self.anchor_T=copy.deepcopy(self.T);self.teacher=copy.deepcopy(self._network).requires_grad_(False).eval()
        self.teacher_hash=network_hash(self.teacher);self.fdparts=[]
        assert not any(a.data_ptr()==b.data_ptr() for a,b in zip(self._network.parameters(),self.teacher.parameters()))
    def _ct3p_add_loss(self,real,x,epoch,batch):
        if self.beta==0:return real
        loss,dist=fd(self._network,self.teacher,x,self)
        self.r.count(self.r.phase+'_wrapper_calls',2);self.r.count(self.r.phase+'_internal_encoder_calls',6)
        self.fdparts.append(dict(FD=float(loss.detach()),FD_p95=float(torch.quantile(dist,.95)),batch=len(x)))
        if batch==0:
            names=[(k,v) for k,v in self._network.named_parameters() if v.requires_grad and ('.pool.pool.' in k or '.pool_few.pool.' in k)]
            g0=torch.autograd.grad(real,[v for _,v in names],retain_graph=True,allow_unused=True)
            g1=torch.autograd.grad(loss,[v for _,v in names],retain_graph=True,allow_unused=True)
            rows={}
            for group,part in [('main','.pool.pool.'),('few','.pool_few.pool.')]:
                ix=[i for i,(k,_) in enumerate(names) if part in k];n0=sum(float(g0[i].detach().square().sum()) for i in ix if g0[i] is not None)**.5;n1=sum(float(g1[i].detach().square().sum()) for i in ix if g1[i] is not None)**.5
                dot=sum(float((g0[i].detach()*g1[i].detach()).sum()) for i in ix if g0[i] is not None and g1[i] is not None)
                rows[group]=dict(real_norm=n0,FD_norm=n1,weighted_ratio=None if n0==0 else 10*n1/n0,cosine=None if n0*n1==0 else dot/(n0*n1))
            jsonl(self.r.pub/('FEATURE_DISTILLATION_AUDIT.jsonl' if self.r.phase=='formal' else 'engineering_FD.jsonl'),dict(dataset=self.name,seed=self.seed,task=self.task+1,epoch=epoch+1,groups=rows))
        return real+10*loss
    def _v2_epoch_hook(self,epoch,optimizer,scheduler,stats):
        if self.smoke:return
        raw,pt,_,ys,_=self.r.extract(self,self.current);z0=joint(self.last_anchor);z1=joint(raw)
        began=time.monotonic();a,b,e,diag=fit_map(z0,z1,ys);at,bt,et,dt=translation(z0,z1,ys)
        self.epoch_stats=transport(self.anchor_stats,a,b,e,self.task+1);self.epoch_T=transport(self.anchor_T,at,bt,et,self.task+1)
        self.last_features=(raw,pt,ys)
        for i,label in enumerate(('main','few')):
            jsonl(self.r.pub/'ROUTING_AUDIT.jsonl',dict(dataset=self.name,seed=self.seed,task=self.task+1,epoch=epoch+1,stream='F',scope='current_fit_only',pool=label,
                pre_usage=np.bincount(self.anchor_routes[:,i],minlength=5).tolist(),post_usage=np.bincount(self.r.last_routes[:,i],minlength=5).tolist(),switch_rate=float(np.mean(self.anchor_routes[:,i]!=self.r.last_routes[:,i]))))
        self.r.count(self.r.mode+'_analytic_ms',int(1000*(time.monotonic()-began)))
        for label,state,d in [('A',self.epoch_stats,diag),('T',self.epoch_T,dt)]:
            mu=state['mu'];d.update(mean_norms=np.linalg.norm(mu,axis=0).tolist(),centered_between_scatter=float(np.mean(np.sum((mu-mu.mean(1,keepdims=True))**2,axis=0))),trace=float(np.trace(state['S'])))
            jsonl(self.r.pub/'TRANSPORT_A_VS_T_AUDIT.jsonl',dict(dataset=self.name,seed=self.seed,stream='F',task=self.task+1,epoch=epoch+1,bank=label,immutable_anchor_task=self.task,committed=epoch==9,**d))
        assert network_hash(self.teacher)==self.teacher_hash and all(v.grad is None for v in self.teacher.parameters())
        parts=self.epoch_parts;row=dict(dataset=self.name,seed=self.seed,task=self.task+1,epoch=epoch+1,optimizer_steps=len(parts),loss=stats['loss'],
            FD_mean=float(np.mean([v['FD'] for v in self.fdparts])),FD_p95_mean=float(np.mean([v['FD_p95'] for v in self.fdparts])),teacher_unchanged=True,
            adapter_gradient_tensors=self.epoch_gradient_nonzero,**{k:float(np.mean([v[k] for v in parts])) for k in ('main_CE','sum_CE','pool_weighted_few','assignment','pull')})
        assert self.epoch_gradient_nonzero>0
        self.records.append(row);jsonl(self.r.pub/'TRAIN_EPOCH_METRICS.jsonl',row);self.r.count('formal_task_epochs')
        self.current_epoch=epoch+1;self.save_new(self.r.private/'active_resume.pt',True);self.r.resources()
        self.fdparts=[];self.epoch_parts=[];self.epoch_gradient_nonzero=0
        print('EPOCH',self.name,self.seed,self.task+1,epoch+1,flush=True)
    def save_new(self,path,resume):
        p=dict(delta={k:v.detach().cpu().clone() for k,v in self._network.state_dict().items() if k in self.nonshared},nonshared_keys=sorted(self.nonshared),
            network_sha256=network_hash(self._network),shared_sha256=self.shared_hash,weight_sha256=WEIGHT_SHA,code_sha256=self.r.code,
            dataset=self.name,seed=self.seed,stream='F',task=self.task,epoch=self.current_epoch,known=self._known_classes,seen=self._total_classes,order=self.order,
            stats=self.stats,T=self.T,optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),rng=self._capture_rng_state(),loader_rng=self.loader_generator.get_state(),
            synthesis_rng=self.synth.get_state(),steps=self.steps,records=self.records,teacher_source=self.teacher_source,teacher_hash=self.teacher_hash,
            manifest_sha256=self.r.lock[self.name]['manifest_sha256'],args=dict(self.args,device=['cuda:0']),beta=10.)
        if resume:p.update(anchor=self.last_anchor,anchor_stats=self.anchor_stats,anchor_T=self.anchor_T,epoch_stats=self.epoch_stats,epoch_T=self.epoch_T,last_features=self.last_features,
            anchor_routes=self.anchor_routes,teacher_delta={k:v.detach().cpu().clone() for k,v in self.teacher.state_dict().items() if k in self.nonshared})
        temp=Path(str(path)+'.part');torch.save(p,temp);self.r.resources();temp.replace(path);return p
    def restore_new(self,path,resume):
        p=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in [('dataset',self.name),('seed',self.seed),('stream','F'),('order',self.order),('weight_sha256',WEIGHT_SHA),('code_sha256',self.r.code),('shared_sha256',self.shared_hash),('manifest_sha256',self.r.lock[self.name]['manifest_sha256']),('nonshared_keys',sorted(self.nonshared)),('beta',10.)]:assert p[k]==v,('BLOCKED_RESTORE',k)
        assert p['args']==dict(self.args,device=['cuda:0'])
        state=self._network.state_dict();state.update(p['delta']);self._network.load_state_dict(state,strict=True);assert network_hash(self._network)==p['network_sha256']
        self.task=p['task'];self._cur_task=self.task;self.current_epoch=p['epoch'];self._known_classes=p['known'];self._total_classes=p['seen'];self.current=range(p['known'],p['seen'])
        self.stats=p['stats'];self.T=p['T'];self.steps=p['steps'];self.records=p['records'];self.teacher_source=p['teacher_source'];self.teacher_hash=p['teacher_hash']
        if self.optimizer is not None:self.optimizer.load_state_dict(p['optimizer']);self.scheduler.load_state_dict(p['scheduler'])
        self.loader_generator.set_state(p['loader_rng']);self.synth.set_state(p['synthesis_rng']);self._restore_rng_state(p['rng'])
        if resume:
            self.anchor_routes=p['anchor_routes'];self.last_anchor=p['anchor'];self.anchor_stats=p['anchor_stats'];self.anchor_T=p['anchor_T'];self.epoch_stats=p['epoch_stats'];self.epoch_T=p['epoch_T'];self.last_features=p['last_features']
            st=self.teacher.state_dict();st.update(p['teacher_delta']);self.teacher.load_state_dict(st,strict=True);assert network_hash(self.teacher)==self.teacher_hash
            self._v2_start_epoch=p['epoch']
        return p

def restore_original(r,l,path):
    l.r=r.parent_view
    try:return l.restore(path)
    finally:l.r=r

def controls(r):
    assert read(r.pub/'P0_ENGINEERING_TESTS.json')['status']=='PASS'
    assert not r.entries,'BLOCKED_CONTROLS_ALREADY_STARTED'
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            l,p,_=r.original(name,seed,1);T=copy.deepcopy(p['stats']);previous=p
            os.link(r.private/'parent.pt',r.private/'parents'/f'{name}_{seed}.pt')
            for task in (1,2,3):
                r.permit(name,seed,task)
                if task>1:
                    current=range(2*(task-1),2*task)
                    before,_,_,yb,rb=r.extract(l,current);routes_before=r.last_routes.copy()
                    path,_=r.acquire(name,seed,task);p=restore_original(r,l,path)
                    after,_,_,ya,ra=r.extract(l,current)
                    for i,label in enumerate(('main','few')):
                        jsonl(r.pub/'ROUTING_AUDIT.jsonl',dict(dataset=name,seed=seed,task=task,stream='U',scope='current_fit_only',pool=label,
                            pre_usage=np.bincount(routes_before[:,i],minlength=5).tolist(),post_usage=np.bincount(r.last_routes[:,i],minlength=5).tolist(),switch_rate=float(np.mean(routes_before[:,i]!=r.last_routes[:,i]))))
                    assert [v['sample_id'] for v in rb]==[v['sample_id'] for v in ra] and np.array_equal(yb,ya)
                    x=joint(before);z=joint(after);a,b,e,da=fit_map(x,z,ya);at,bt,et,dt=translation(x,z,ya)
                    check=append(transport(previous['stats'],a,b,e,task),z,ya,current,task)
                    for k in ('mu','v','S'):np.testing.assert_allclose(check[k],p['stats'][k],atol=1e-5,rtol=1e-5)
                    T=append(transport(T,at,bt,et,task),z,ya,current,task)
                    for label,d in [('A',da),('T',dt)]:jsonl(r.pub/'TRANSPORT_A_VS_T_AUDIT.jsonl',dict(dataset=name,seed=seed,stream='U',task=task,bank=label,committed=True,**d))
                    del before,after,x,z,check
                with np.load(r.old/'output/private/sealed'/f'{name}_{seed}_U_W_t{task:02d}.npz') as f:
                    for method,state,key in [('C0',p['stats'],'CT-J-CB'),('P',p['pt'],'PT-CB')]:
                        np.testing.assert_allclose(ridge(state)[0],f[key],atol=1e-9,rtol=1e-9)
                        r.save_candidate(l,method,state,'U')
                # Baseline-only reproduction precedes new training; no new candidate score is released.
                vr,vpt,_,vy,vrows=r.extract(l,range(2*task),'val')
                from ct2d_math import classify
                for method,oldmethod,zval in [('C0','CT-J-CB',joint(vr)),('P','PT-CB',vpt.astype(np.float64))]:
                    entry=next(x for x in r.entries if (x['dataset'],x['seed'],x['task'],x['method'])==(name,seed,task,method))
                    with np.load(r.private/'banks'/entry['W_file']) as f:score=zval@f['W']
                    with np.load(r.old/'output/private/sealed'/f'{name}_{seed}_{oldmethod}_t{task:02d}.npz') as f:
                        assert list(f['ids'])==[x['sample_id'] for x in vrows] and np.array_equal(vy,f['y'])
                        np.testing.assert_allclose(score,f['raw'],atol=1e-5,rtol=1e-5)
                        assert np.array_equal(classify(score,np.array(l.order[:2*task])),classify(f['raw'],np.array(l.order[:2*task])))
                jsonl(r.pub/'BASELINE_REPRODUCTION.jsonl',dict(dataset=name,seed=seed,task=task,C0_P_identity_logits_predictions=True))
                cache=r.private/'control_features'/f'{name}_{seed}_{task}.npz'
                npz(cache,raw=vr,pt=vpt,y=vy,order=np.array(l.order),ids=np.array([x['sample_id'] for x in vrows]),component=np.array([x.get('identity_component',x.get('verified_group','')) for x in vrows]))
                jsonl(r.private/'CONTROL_FEATURE_CACHE.jsonl',dict(dataset=name,seed=seed,task=task,file=cache.name,sha256=sha(cache),network_sha256=network_hash(l._network),purpose='baseline reproduction features; defer all new candidate scoring until state/W lock'))
                del vr,vpt,score
                r.save_candidate(l,'C1',T,'U')
                if task==1:
                    for method in ('C2','C3'):r.save_candidate(l,method,p['stats'],'U')
                if task==3 and seed==1993:torch.save(T,r.private/'banks'/f'{name}_1993_U_T.pt')
                previous=p
            del l,p,previous,T;gc.collect();torch.cuda.empty_cache();(r.private/'parent.pt').unlink(missing_ok=True);r.resources()
    assert len(r.entries)==66
    write_json(r.pub/'P1_COMPLETE.json',dict(status='PASS',original_prefix_states=18,original_C0_P_weights_reproduced=True,no_neural_updates=True))


def train(r):
    assert read(r.pub/'P1_COMPLETE.json')['status']=='PASS'
    assert not (r.pub/'TRAINING_STARTED.json').exists(),'BLOCKED_ALREADY_STARTED_USE_EXPLICIT_SAME_LOCK_RESUME'
    write_json(r.pub/'TRAINING_STARTED.json',dict(source_commit=r.cfg['source_commit'],unix=time.time()))
    r.phase='formal'
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            path=r.private/'parents'/f'{name}_{seed}.pt'
            e=next(x for x in read(r.old/'output/public/MODEL_LINEAGE.json') if (x['dataset'],x['seed'],x['stream'],x['task'])==(name,seed,'U',1))
            assert sha(path)==e['sha256'];l=FDStudent(r,name,seed);l.fork_from_ct1_task1(path,e);path.unlink()
            write_json(r.pub/f'ACTUAL_CONFIG_{name}_{seed}.json',dict(l.args,device=['cuda:0'],locked_weight_path='PINNED_SHARED_AUGREG',FD_beta=10.,prefix_tasks=[1,2,3]))
            for task in (1,2):
                r.permit(name,seed,task+1);l.task_setup(task);l.last_anchor,_,_,yb,rb=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
                ds=r.dataset(name,seed,l.current,True);loader=DataLoader(ds,batch_size=48,shuffle=True,num_workers=4,drop_last=False,generator=l.loader_generator)
                step=l.steps;l._init_train(loader,None,l.optimizer,l.scheduler)
                assert l.steps-step==l.tasks[task]['steps'] and l.current_epoch==10
                raw,_,y=l.last_features;assert np.array_equal(y,yb);z=joint(raw)
                l.stats=append(l.epoch_stats,z,y,l.current,task+1);l.T=append(l.epoch_T,z,y,l.current,task+1)
                changed=[k for k,v in l._network.named_parameters() if k in l.adapter_before and not torch.equal(v.detach().cpu(),l.adapter_before[k])]
                assert changed and network_hash(l.teacher)==l.teacher_hash
                jsonl(r.pub/'TASK_UPDATE_AUDIT.jsonl',dict(dataset=name,seed=seed,task=task+1,adapter_changed_names=changed,teacher_hash=l.teacher_hash,teacher_source=l.teacher_source['network_sha256']))
                for method,state in [('C2',l.stats),('C3',l.T)]:r.save_candidate(l,method,state,'F')
                e=r.archive(l);l.teacher_source=e
                (r.private/'active_resume.pt').unlink();del l.teacher;l.teacher=None
                del raw,z,loader,ds;gc.collect();torch.cuda.empty_cache();r.resources()
            del l;gc.collect();torch.cuda.empty_cache();(r.private/'parent.pt').unlink(missing_ok=True)
    assert len(r.entries)==90 and len(r.lineage)==12 and r.access['formal_optimizer_steps']==6620 and r.access['formal_task_epochs']==120
    write_json(r.pub/'ONLINE_STATE_W_LOCK.json',dict(status='LOCKED',source_commit=r.cfg['source_commit'],candidates=r.entries,control_features=[json.loads(x) for x in (r.private/'CONTROL_FEATURE_CACHE.jsonl').read_text().splitlines()],checkpoints=r.lineage,epochs=120,optimizer_steps=6620))


def evaluate(r):
    lock=read(r.pub/'ONLINE_STATE_W_LOCK.json');assert lock['status']=='LOCKED' and len(lock['candidates'])==90
    assert not r.sealed,'BLOCKED_ALREADY_EVALUATED'
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            for task in (1,2,3):
                for source in (['U'] if task==1 else ['U','F']):
                    r.permit(name,seed,task)
                    if source=='U':
                        entry=next(x for x in lock['control_features'] if (x['dataset'],x['seed'],x['task'])==(name,seed,task))
                        cache=r.private/'control_features'/entry['file'];assert sha(cache)==entry['sha256']
                        with np.load(cache) as f:p={k:f[k].copy() for k in f.files}
                        raw=p['raw'];pt=p['pt'];y=p['y'];rows=[dict(sample_id=sid,identity_component=component) for sid,component in zip(p['ids'],p['component'])]
                        l=types.SimpleNamespace(order=p['order'].tolist())
                        assert all(x['network_sha256']==entry['network_sha256'] for x in r.entries if (x['dataset'],x['seed'],x['task'],x['source'])==(name,seed,task,'U'))
                    else:
                        path,_=r.acquire(name,seed,task,'F');l=FDStudent(r,name,seed);p=l.restore_new(path,False)
                    if source=='F':raw,pt,_,y,rows=r.extract(l,range(2*task),'val')
                    order=np.array(l.order[:2*task])
                    for e in [x for x in r.entries if (x['dataset'],x['seed'],x['task'],x['source'])==(name,seed,task,source)]:
                        path=r.private/'banks'/e['W_file'];assert sha(path)==e['W_sha256']
                        with np.load(path) as f:W=f['W'].copy()
                        score=(pt.astype(np.float64) if e['method']=='P' else joint(raw))@W
                        if e['method'] in ('C0','P'):
                            original='CT-J-CB' if e['method']=='C0' else 'PT-CB'
                            with np.load(r.old/'output/private/sealed'/f'{name}_{seed}_{original}_t{task:02d}.npz') as f:
                                assert list(f['ids'])==[v['sample_id'] for v in rows]
                                np.testing.assert_allclose(score,f['raw'],atol=1e-5,rtol=1e-5)
                                from ct2d_math import classify
                                assert np.array_equal(classify(score,order),classify(f['raw'],order)),'BLOCKED_ORIGINAL_PREDICTION_DRIFT'
                        stem=f'{name}_{seed}_{e["method"]}_t{task:02d}.npz'
                        comp=np.array([v.get('identity_component',v.get('verified_group','')) for v in rows]);assert np.all(comp!='')
                        npz(r.private/'sealed'/stem,raw=score,y=y,original=order[y],order=order,ids=np.array([v['sample_id'] for v in rows]),component=comp)
                        r.sealed.append(dict(e,file=stem,sha256=sha(r.private/'sealed'/stem)));write_json(r.private/'SEALED_INDEX.json',r.sealed)
                    del l,p,raw,pt;gc.collect();torch.cuda.empty_cache();(r.private/'parent.pt').unlink(missing_ok=True);r.resources()
    assert len(r.sealed)==90
    write_json(r.pub/'ONLINE_PREDICTIONS_LOCK.json',dict(status='LOCKED',units=r.sealed,source_commit=r.cfg['source_commit'],original_C0_P_predictions_reproduced=True))


def oracle(r):
    assert read(r.pub/'ONLINE_PREDICTIONS_LOCK.json')['status']=='LOCKED'
    assert not (r.pub/'ORACLE_COMPLETE.json').exists()
    for name in ('HK','ISIC'):
        for source in ('U','F'):
            r.permit(name,1993,3)
            if source=='U':
                l,p,_=r.original(name,1993,3);A=p['stats'];T=torch.load(r.private/'banks'/f'{name}_1993_U_T.pt',weights_only=False)
            else:
                path,_=r.acquire(name,1993,3,'F');l=FDStudent(r,name,1993);p=l.restore_new(path,False);A=l.stats;T=l.T
            raw,_,_,y,_=r.extract(l,range(6));state=append(empty(1536),joint(raw),y,range(6),3);W,diag=ridge(state);r.count('offline_analytic_fits')
            vals,_,_,vy,rows=r.extract(l,range(6),'val');order=np.array(l.order[:6]);stem=f'{name}_1993_{source}_Q11.npz'
            npz(r.private/'sealed'/stem,raw=joint(vals)@W,y=vy,original=order[vy],order=order,ids=np.array([v['sample_id'] for v in rows]),component=np.array([v.get('identity_component',v.get('verified_group','')) for v in rows]))
            for label,bank in [('A',A),('T',T)]:
                mm=bank['mu'][:,:4];truth=state['mu'][:,:4]
                scatter=lambda m:float(np.mean(np.sum((m-m.mean(1,keepdims=True))**2,axis=0)))
                jsonl(r.pub/'ORACLE_GEOMETRY.jsonl',dict(dataset=name,seed=1993,source=source,bank=label,
                    old_mean_errors=np.linalg.norm(mm-truth,axis=0).tolist(),stored_between=scatter(mm),real_between=scatter(truth),
                    old_diagonal_covariance_trace_error=(bank['v'][:,:4].sum(0)-state['v'][:,:4].sum(0)).tolist()))
            jsonl(r.pub/'ORACLE_INDEX.jsonl',dict(dataset=name,seed=1993,source=source,file=stem,sha256=sha(r.private/'sealed'/stem),online_CIL=False,**diag))
            del l,p,raw,vals,A,T;gc.collect();torch.cuda.empty_cache();(r.private/'parent.pt').unlink(missing_ok=True);r.resources()
    write_json(r.pub/'ORACLE_COMPLETE.json',dict(status='PASS',models=4,new_fit_classes_each=6,online_artifacts_modified=False))
    write_json(r.private/'STOP_TRANSFER.json',dict(status='STOP'))


def gate(r):
    assert not (r.pub/'P0_ENGINEERING_TESTS.json').exists()
    results=[]
    for name in ('HK','ISIC'):
        for seed in (1993,1994,1995):
            r.permit(name,seed,1);l,p,e=r.original(name,seed,1)
            assert p['known']==0 and p['seen']==2 and p['epoch']==10
            with np.load(r.old/'output/private/sealed'/f'{name}_{seed}_U_W_t01.npz') as f:
                np.testing.assert_allclose(ridge(p['stats'])[0],f['CT-J-CB'],atol=1e-9,rtol=1e-9)
                np.testing.assert_allclose(ridge(p['pt'])[0],f['PT-CB'],atol=1e-9,rtol=1e-9)
            results.append(dict(dataset=name,seed=seed,parent_SHA=e['sha256'],network_SHA=p['network_sha256'],strict_restore=True,shared_Task1_W=True))
            del l,p;gc.collect();torch.cuda.empty_cache()
    r.permit('HK',1993,2);path,e=r.acquire('HK',1993,1);l=FDStudent(r,'HK',1993);l.fork_from_ct1_task1(path,e);l.task_setup(1);l.smoke=True
    ds=r.dataset('HK',1993,l.current,True,smoke=True);batch=next(iter(DataLoader(ds,batch_size=48,shuffle=True,num_workers=0,generator=l.loader_generator)))
    l.last_anchor,pt,_,y,_=r.extract(l,l.current,smoke=True);l.last_features=(l.last_anchor,pt,y);l.anchor_routes=r.last_routes.copy();l.epoch_stats=l.anchor_stats;l.epoch_T=l.anchor_T
    snap=r.private/'engineering_resume.pt';l.save_new(snap,True)
    # beta=0 follows the untouched CT1 path, including optimizer and RNG.
    rng=l._capture_rng_state();l.beta=0.;l.args['tuned_epoch']=1;l._init_train([batch],None,l.optimizer,None);zero=(network_hash(l._network),l.last_gradient)
    del l;gc.collect();torch.cuda.empty_cache()
    control=CTLearner(r.parent_view,'HK',1993,'U');control.restore(path);control.r=r;control.task_setup(1);control.smoke=True;control.args['tuned_epoch']=1;control._restore_rng_state(rng)
    control._init_train([batch],None,control.optimizer,None);assert zero==(network_hash(control._network),control.last_gradient),'BLOCKED_BETA_ZERO'
    del control;gc.collect();torch.cuda.empty_cache()
    l=FDStudent(r,'HK',1993);l.fork_from_ct1_task1(path,e);l.task_setup(1);l.smoke=True;out=[]
    for _ in range(2):
        l.args['tuned_epoch']=10;l.restore_new(snap,True);l.args['tuned_epoch']=1;l._init_train([batch],None,l.optimizer,None);out.append((network_hash(l._network),l.last_gradient))
    assert out[0]==out[1],'BLOCKED_SAME_NEXT_FD_UPDATE'
    l.args['tuned_epoch']=10;l.current_epoch=10;p=l.save_new(r.private/'engineering_final.pt',False);l.restore_new(r.private/'engineering_final.pt',False)
    size=(r.private/'engineering_final.pt').stat().st_size
    refusals=[]
    for cls in ([0],[4]):
        try:r.dataset('HK',1993,cls,False)
        except AssertionError:refusals.append(cls)
        else:raise AssertionError('BLOCKED_GUARD_DID_NOT_REJECT')
    for path in (r.old/'output/private/sealed/HK_1993_CT-J-CB_t11.npz',Path('/tmp/p21root/output/private/scores/HK_1993_U_Q11.npz'),r.old/'test.csv'):
        try:path.open('rb')
        except AssertionError:pass
        else:raise AssertionError('BLOCKED_ASSET_GUARD_DID_NOT_REJECT')
    projected=read(r.pub/'FD_THROUGHPUT_GATE.json')['projected_GPU_seconds']+r.resources()['GPU_process_residence_seconds']+300
    assert projected<14400 and size*12+700*1024**2<3*1024**3,'BLOCKED_ADMISSION'
    for p in (snap,r.private/'engineering_final.pt',r.private/'parent.pt'):p.unlink()
    write_json(r.pub/'P0_ENGINEERING_TESTS.json',dict(status='PASS',parents=results,math=math_check(),beta0_exact_next_step=True,
        FD_same_lock_resume_next_step=True,task_checkpoint_bytes=size,projected_GPU_seconds=projected,engineering_steps_total=r.resources()['engineering_optimizer_steps'],
        teacher_is_independent=True,old_future_oracle_test_access_guard=True))
    assert r.resources()['engineering_optimizer_steps']<=16


def main():
    cfg=read(os.environ['P22_CONFIG']);r=Prefix(cfg);r.setup()
    global ridge
    original_ridge=ridge
    def timed_ridge(*args,**kwargs):
        start=time.monotonic()
        try:return original_ridge(*args,**kwargs)
        finally:r.count(r.mode+'_analytic_ms',int(1000*(time.monotonic()-start)))
    ridge=timed_ridge
    try:
        with threadpool_limits(limits=4):{'gate':gate,'controls':controls,'train':train,'evaluate':evaluate,'oracle':oracle}[cfg['mode']](r)
    except BaseException as e:
        write_json(r.pub/('FAILURE_'+cfg['mode']+'.json'),dict(status='BLOCKED',error=type(e).__name__,reason=str(e)));raise
    finally:r.resources(False)
if __name__=='__main__':main()
