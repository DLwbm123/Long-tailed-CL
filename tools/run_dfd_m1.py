"""M1 storage/training adapter over immutable CT3-P scientific dependencies."""
import copy,csv,gc,gzip,hashlib,json,multiprocessing as mp,os,shutil,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ['N76_ROOT']); REC=ROOT.parent/'recovery_v1'; LEGACY=REC/'source/legacy'
sys.path.insert(0,str(LEGACY/'tools'));sys.path.insert(0,str(ROOT/'source'))
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
from run_ct3p import FDStudent,Prefix,CountedImages
from run_ct1 import CTLearner,jsonl
from run_medical_v2 import sha,network_hash,rng_equal,WEIGHT_SHA,write_json
from preflight_ct1 import task_lock
from ct1_statistics import ridge,joint,append
from ct3p_core import features,fd
from dfd_t4_core import penalty,directional_fd,old_directions,random_directions
CFG=json.loads((REC/'source/INPUT.json').read_text()); PUB=ROOT/'public'; PRIVATE=ROOT/'private'
def read(p):return json.loads(Path(p).read_text())
def save(n,v):write_json(PUB/n,v)
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def load(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rb') as f:return torch.load(f,map_location='cpu',weights_only=False)
    return torch.load(p,map_location='cpu',weights_only=False)
def dump(p,path):
    path=Path(path);tmp=Path(str(path)+'.part')
    if str(path).endswith('.gz'):
        with gzip.open(tmp,'wb',compresslevel=1) as f:torch.save(p,f,pickle_protocol=4)
    else:torch.save(p,tmp,pickle_protocol=4)
    tmp.replace(path)
def context_state(l):
    return ([m.training for m in l._network.modules()],
            [l._network.backbone.pool.batchwise_prompt,l._network.backbone.pool_few.batchwise_prompt],
            [(hasattr(m,'adapt_list'),getattr(m,'adapt_list',None)) for m in l._network.modules() if hasattr(m,'shared')],l._capture_rng_state())
def same_context(a,b):
    assert a[:2]==b[:2] and rng_equal(a[3],b[3])
    for x,y in zip(a[2],b[2]):
        assert x[0]==y[0]
        assert x[1] is y[1]  # The original context restores the exact saved routing object.
class EngineeringEnd(Exception):pass
class Run:
    extract=Prefix.extract
    def __init__(self):
        self.cfg=CFG['legacy_config'];self.lock=task_lock(self.cfg);self.code=CFG['qualified_sha256']
        assert {k:sha(LEGACY/k) for k in self.code}==self.code,'BLOCKED_SOURCE'
        self.phase=self.mode='engineering';self.pub=PUB;self.private=PRIVATE;self.scope=None;self.allowed=set()
        self.started=time.monotonic();self.counter=mp.Array('q',[0,0,0]);self.access={};self.entries=[];self.lineage=[]
        self.prior=read(PUB/'RESOURCE_LEDGER.json') if (PUB/'RESOURCE_LEDGER.json').exists() else {}
        self.access=self.prior.get('calls',{}).copy();self.gpu_prior=self.prior.get('GPU_process_residence_seconds',0)
        self.parent_view=SimpleNamespace(cfg=self.cfg,lock=self.lock,code=CFG['ct1_code'],phase=self.phase,count=self.count)
        self.source_sha=sha(ROOT/'source/run_dfd_m1.py');self.protocol_sha=sha(PUB/'PROTOCOL_LOCK.json')
        self.backup_bytes=self.prior.get('independent_backup_bytes',0);self.last_actual=0
        self.gpu_limit=read(PUB/'PROTOCOL_LOCK.json')['GPU_seconds_limit']
    def count(self,k,n=1):self.access[k]=self.access.get(k,0)+n
    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert self.scope==(name,seed,4) and not smoke
        assert list(classes)==([6,7] if split=='train' else list(range(8)))
        if split=='val':assert self.phase=='evaluate' and (PUB/'STATE_W_LOCK.json').exists() and not train
        else:assert split=='train'
        s=self.cfg['datasets'][name];d=CountedImages(Path(s['manifests'])/(split+'.csv'),s['images'],self.lock[name]['orders'][seed],classes,train,False)
        d.rows.sort(key=lambda x:x['sample_id'])
        if self.phase=='engineering' and not train and split=='train':d.rows=d.rows[:96]
        d.counter=self.counter;d.split=split;d.offline=False
        self.allowed={str((Path(s['images'])/x['relative_path']).resolve()) for x in d.rows};return d
    def resources(self,enforce=True):
        files=[p for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink()]
        total=sum(p.stat().st_size for p in files);active=sum(p.stat().st_size for p in files if p.parent==PRIVATE and ('.pt' in p.name or '.part' in p.name))
        x=dict(phase=self.phase,calls=self.access,GPU_process_residence_seconds=self.gpu_prior+time.monotonic()-self.started,
          primary_new_bytes=total,independent_backup_bytes=self.backup_bytes,persistent_plus_backup_bytes=total+self.backup_bytes,
          active_temporary_bytes=active,free_bytes=shutil.disk_usage(ROOT).free,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
          fit_image_reads=self.prior.get('fit_image_reads',0)+int(self.counter[0]),val_image_reads=self.prior.get('val_image_reads',0)+int(self.counter[1]),
          old_fit_reads=0,future_fit_reads=0,test_reserved_reads=0,oracle_reads=0,
          P0_P0R_usage='Separate historical ledgers referenced by AMENDMENT_LOCK; existing verified assets are reused')
        save('RESOURCE_LEDGER.json',x)
        if enforce:
            assert x['GPU_process_residence_seconds']<self.gpu_limit,'BLOCKED_GPU_BUDGET'
            assert total+self.backup_bytes<3*1024**3 and active<2*1024**3 and x['free_bytes']>=1024**3,'BLOCKED_STORAGE_BUDGET'
        return x
    def backup(self,paths,label):
        # One request/ack channel, bounded to 15 minutes; no retry/retraining loop.
        key=str(time.time_ns());items=[dict(file=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in paths]
        write_json(PRIVATE/'TRANSFER_REQUEST.json',dict(id=key,items=items,label=label))
        start=time.monotonic()
        while time.monotonic()-start<900:
            ack=PRIVATE/'TRANSFER_ACK.json'
            if ack.exists():
                a=read(ack)
                if a['id']==key:
                    assert a['status']=='PASS',a
                    assert a['items']==items
                    self.backup_bytes=a['backup_total_bytes'];jsonl(PUB/'BACKUP_RECEIPTS.jsonl',a);return a
            time.sleep(2)
        raise TimeoutError('BLOCKED_INDEPENDENT_BACKUP')
    def fork(self,e,method):
        self.scope=e['dataset'],e['seed'],4
        l=Student(self,e['dataset'],e['seed']);p=FDStudent.restore_new(l,e['path'],False)
        assert (p['task'],p['epoch'],p['seen'],p['beta'])==(2,10,6,10.)
        assert network_hash(l._network)==e['network_sha256'] and rng_equal(l._capture_rng_state(),p['rng'])
        assert torch.equal(l.loader_generator.get_state(),p['loader_rng']) and torch.equal(l.synth.get_state(),p['synthesis_rng'])
        l.parent=e;l.parent_steps=p['steps'];l.method=method;l.teacher_source=e;l.bind_q();l.task_setup(3)
        assert l.teacher_hash==e['network_sha256'] and rng_equal(l._capture_rng_state(),p['rng'])
        jsonl(PUB/'FORK_AUDIT.jsonl',dict(dataset=l.name,seed=l.seed,method=method,phase=self.phase,parent_sha256=e['sha256'],
            parent_network_sha256=e['network_sha256'],inherited_steps=l.parent_steps,ordinary_restore=True,RNG_loader_synthesis=True))
        return l,p
class Student(FDStudent):
    def bind_q(self):
        p=ROOT.parent/'private'/f'{self.name}_{self.seed}_Q.npz';self.q_sha=sha(p)
        with np.load(p) as f:self.q=torch.tensor(f['Q_'+self.method],device='cuda',dtype=torch.float32)
    def binding(self):
        return dict(schema='DFD_M1_v1',method=self.method,Q_sha256=self.q_sha,coefficients=[10.,1.],
            worker_sha256=self.r.source_sha,protocol_sha256=self.r.protocol_sha,parent_sha256=self.parent['sha256'])
    def validate(self,p):
        for k,v in self.binding().items():assert p['binding'].get(k)==v,('BLOCKED_M1_RESTORE',k)
    def task_setup(self,task):
        super().task_setup(task);self.optimizer.register_step_post_hook(self.after_step);self.pending=None
    def _v2_input_hook(self,epoch,batch,x,y):
        self.batch_started=time.monotonic()
        return super()._v2_input_hook(epoch,batch,x,y)
    def _ct3p_add_loss(self,real,x,epoch,batch):
        before=context_state(self) if self.r.phase=='engineering' else None
        loss,delta=directional_fd(self._network,self.teacher,x,self,self.q)
        if before:
            same_context(before,context_state(self))
            if self.steps==self.parent_steps:assert float(loss.detach())==0.,'BLOCKED_INITIAL_FD'
        self.r.count(self.r.phase+'_wrapper_calls',2);self.r.count(self.r.phase+'_internal_encoder_calls',6)
        self.last_x=x.detach();part=self.epoch_parts[-1]
        reconstructed=(part['main_CE']+part['sum_CE']+part['pool_weighted_few'])/3+part['assignment']+part['pull']
        assert np.isclose(float(real.detach()),reconstructed,atol=1e-6,rtol=1e-5),'BLOCKED_REAL_LOSS_REGRESSION'
        self.fdparts.append(dict(FD=float(loss.detach()),FD_p95=float(torch.quantile(delta.square().sum(1),.95)),batch=len(x)))
        if batch==0:
            names=[(k,v) for k,v in self._network.named_parameters() if v.requires_grad and ('.pool.pool.' in k or '.pool_few.pool.' in k)]
            g0=torch.autograd.grad(real,[v for _,v in names],retain_graph=True,allow_unused=True)
            g1=torch.autograd.grad(loss,[v for _,v in names],retain_graph=True,allow_unused=True);groups={}
            for group,partname in [('main','.pool.pool.'),('few','.pool_few.pool.')]:
                ix=[i for i,(k,_) in enumerate(names) if partname in k]
                a=sum(float(g0[i].square().sum()) for i in ix if g0[i] is not None)**.5
                b=sum(float(g1[i].square().sum()) for i in ix if g1[i] is not None)**.5
                dot=sum(float((g0[i]*g1[i]).sum()) for i in ix if g0[i] is not None and g1[i] is not None)
                groups[group]=dict(real_norm=a,FD_norm=b,ratio=None if a<=1e-12 else b/a,cosine=None if a*b==0 else dot/(a*b))
            total=float(delta.square().sum(1).mean());parallel=float((delta@self.q).square().sum(1).mean())
            self.pending=dict(dataset=self.name,seed=self.seed,method=self.method,epoch=epoch+1,groups=groups,
                parallel_energy=parallel,perpendicular_energy=total-parallel,teacher_fingerprint=self.teacher_hash,Q_sha256=self.q_sha,
                real_reconstruction_residual=float(real.detach())-reconstructed,initial_FD=float(loss.detach()))
            self.before_update={k:v.detach().clone() for k,v in names}
        return real+loss
    def after_step(self,optimizer,args,kwargs):
        if self.pending:
            for group,part in [('main','.pool.pool.'),('few','.pool_few.pool.')]:
                vals=[float((p.detach()-self.before_update[k]).square().sum()) for k,p in self._network.named_parameters() if k in self.before_update and part in k]
                self.pending['groups'][group]['actual_AdamW_update_norm']=sum(vals)**.5
            jsonl(self.r.pub/('gradient_probes.jsonl' if self.r.phase=='formal' else 'engineering_gradient_probes.jsonl'),self.pending)
            self.pending=None;self.before_update={}
        if self.r.phase=='engineering':
            torch.cuda.synchronize()
            self.engineering_step_seconds=getattr(self,'engineering_step_seconds',[])+[time.monotonic()-self.batch_started]
            if self.steps-self.parent_steps>=2:raise EngineeringEnd()
            return
        if not (PUB/'FIRST_FORMAL_UPDATE.json').exists():save('FIRST_FORMAL_UPDATE.json',dict(unix=time.time(),dataset=self.name,seed=self.seed,method=self.method,actual_new_steps=1,pid=os.getpid()))
        if self.steps%10==0:self.r.resources()
    def _v2_epoch_hook(self,*args):
        super()._v2_epoch_hook(*args)
        row=dict(self.records[-1],method=self.method,parent_steps=self.parent_steps,cumulative_new_steps=self.steps-self.parent_steps)
        row['parameter_change_norm']=sum(float((v.detach().cpu()-self.adapter_before[k]).square().sum()) for k,v in self._network.named_parameters() if k in self.adapter_before)**.5
        jsonl(PUB/'epoch_records.jsonl',row)
    def save_new(self,path,resume):
        p=dict(binding=self.binding(),delta={k:v.detach().cpu().clone() for k,v in self._network.state_dict().items() if k in self.nonshared},
            nonshared_keys=sorted(self.nonshared),network_sha256=network_hash(self._network),shared_sha256=self.shared_hash,weight_sha256=WEIGHT_SHA,code_sha256=self.r.code,
            dataset=self.name,seed=self.seed,task=self.task,epoch=self.current_epoch,known=self._known_classes,seen=self._total_classes,order=self.order,
            stats=self.stats,T=self.T,optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict(),rng=self._capture_rng_state(),loader_rng=self.loader_generator.get_state(),
            synthesis_rng=self.synth.get_state(),steps=self.steps,parent_steps=self.parent_steps,records=self.records,teacher_source=self.teacher_source,teacher_hash=self.teacher_hash,
            manifest_sha256=self.r.lock[self.name]['manifest_sha256'],args=dict(self.args,device=['cuda:0']),parent_beta=10.,resume=resume)
        if resume:p.update(anchor=self.last_anchor,anchor_stats=self.anchor_stats,anchor_T=self.anchor_T,epoch_stats=self.epoch_stats,epoch_T=self.epoch_T,last_features=self.last_features,
            anchor_routes=self.anchor_routes,teacher_delta={k:v.detach().cpu().clone() for k,v in self.teacher.state_dict().items() if k in self.nonshared})
        dump(p,path);self.r.resources();return p
    def restore_new(self,path,resume):
        p=load(path);self.validate(p)
        for k,v in [('dataset',self.name),('seed',self.seed),('order',self.order),('weight_sha256',WEIGHT_SHA),('code_sha256',self.r.code),('shared_sha256',self.shared_hash),('nonshared_keys',sorted(self.nonshared)),('manifest_sha256',self.r.lock[self.name]['manifest_sha256']),('args',dict(self.args,device=['cuda:0']))]:assert p[k]==v,('BLOCKED_RESTORE',k)
        state=self._network.state_dict();state.update(p['delta']);self._network.load_state_dict(state,strict=True);assert network_hash(self._network)==p['network_sha256']
        self.task=self._cur_task=p['task'];self.current_epoch=p['epoch'];self._known_classes=p['known'];self._total_classes=p['seen'];self.current=range(p['known'],p['seen'])
        self.stats=p['stats'];self.T=p['T'];self.steps=p['steps'];self.parent_steps=p['parent_steps'];self.records=p['records'];self.teacher_source=p['teacher_source'];self.teacher_hash=p['teacher_hash']
        if self.optimizer is not None:self.optimizer.load_state_dict(p['optimizer']);self.scheduler.load_state_dict(p['scheduler'])
        self.loader_generator.set_state(p['loader_rng']);self.synth.set_state(p['synthesis_rng']);self._restore_rng_state(p['rng'])
        if resume:
            assert p['resume'];self.anchor_routes=p['anchor_routes'];self.last_anchor=p['anchor'];self.anchor_stats=p['anchor_stats'];self.anchor_T=p['anchor_T'];self.epoch_stats=p['epoch_stats'];self.epoch_T=p['epoch_T'];self.last_features=p['last_features']
            st=self.teacher.state_dict();st.update(p['teacher_delta']);self.teacher.load_state_dict(st,strict=True);assert network_hash(self.teacher)==self.teacher_hash
            self._v2_start_epoch=p['epoch'];self.adapter_before={k:st[k].detach().cpu().clone() for k in self.adapter_before}
        return p
class Batches:
    def __init__(self,l):
        self.l=l;d=l.r.dataset(l.name,l.seed,l.current,True)
        self.loader=DataLoader(d,batch_size=48,shuffle=True,num_workers=4,drop_last=False,generator=l.loader_generator)
        self.epoch=l.current_epoch
    def __len__(self):return 2 if self.l.r.phase=='engineering' else len(self.loader)
    def __iter__(self):
        l=self.l;self.epoch+=1
        path=PRIVATE/'batch_checks'/f'{l.name}_{l.seed}_e{self.epoch:02d}.jsonl'
        if l.r.phase=='engineering':
            iterator=iter(self.loader);first=next(iterator);yield first
            yield next(iterator,first)
            return
        expected=[json.loads(x) for x in path.read_text().splitlines()] if l.method=='R' and l.r.phase=='formal' else None
        for batch,(idx,x,y) in enumerate(self.loader):
            v=dict(batch=batch,indices_sha256=digest(idx.numpy()),augmentation_sha256=digest(x.numpy()),labels_sha256=digest(y.numpy()))
            if l.r.phase=='formal':
                if l.method=='D':jsonl(path,v)
                else:assert v==expected[batch],'BLOCKED_D_R_BATCH_MISMATCH'
            yield idx,x,y
        if expected is not None:assert len(expected)==len(self)
def asset_check(r):
    audit=read(ROOT.parent/'public/ASSET_AUDIT.json')['parents'];rows=[]
    for e in CFG['parents']:
        assert sha(e['path'])==e['sha256'];p=load(e['path']);a=next(x for x in audit if (x['dataset'],x['seed'])==(e['dataset'],e['seed']))
        qp=ROOT.parent/'private'/f"{e['dataset']}_{e['seed']}_Q.npz";assert sha(qp)==a['Q_sha256']
        with np.load(qp) as f:
            qd,s,t=old_directions(p['T']['mu']);qr,seed=random_directions(e['dataset'],e['seed'],qd.shape[1])
            np.testing.assert_allclose(f['Q_D']@f['Q_D'].T@qd,qd,atol=1e-10,rtol=1e-10)
            np.testing.assert_array_equal(qr,f['Q_R']);assert int(f['random_seed'])==seed
            for k in ('Q_D','Q_R'):
                q=f[k].astype(np.float32);assert np.max(abs(q.T@q-np.eye(q.shape[1])))<=1e-5
        rows.append(dict(dataset=e['dataset'],seed=e['seed'],parent_sha256=e['sha256'],Q_sha256=a['Q_sha256'],rank=5,expected_steps=a['expected_steps_one_arm']))
    assert sha(r.cfg['weight'])==WEIGHT_SHA
    controls=read(REC/'public/CONTROL_REPRODUCTION_AUDIT.json')['units'];compat=[]
    for name,spec in r.cfg['datasets'].items():
        for split in ('train','val'):assert sha(Path(spec['manifests'])/(split+'.csv'))==spec['manifest_sha256'][split]
    for e in controls:
        path=REC/'private/sealed'/e['file'];assert sha(path)==e['sha256'];order=r.lock[e['dataset']]['orders'][e['seed']][:8]
        spec=r.cfg['datasets'][e['dataset']]
        with (Path(spec['manifests'])/'val.csv').open() as f:rs=sorted([x for x in csv.DictReader(f) if int(x['original_label']) in order],key=lambda x:x['sample_id'])
        with np.load(path) as f:
            for k,v in [('ids',[x['sample_id'] for x in rs]),('component',[x['identity_component'] for x in rs]),('original',[int(x['original_label']) for x in rs]),('order',order)]:np.testing.assert_array_equal(f[k],v)
            np.testing.assert_array_equal(np.array(order)[f['y']],f['original']);assert f['raw'].shape==(len(rs),8) and np.isfinite(f['raw']).all()
        compat.append(dict(dataset=e['dataset'],seed=e['seed'],method=e['method'],sha256=e['sha256'],layout='PASS',status='REBUILT_FIXED_CONTROL',historical_per_sample_equality='UNKNOWN'))
    save('ASSET_LOCK.json',dict(status='PASS',parents=rows,shared_weight_sha256=WEIGHT_SHA,recovery_backups_reused=True))
    save('SUBSPACE_AUDIT.json',dict(status='PASS',units=rows,random_key='DFD-T4-P-v1',unchanged_Q=True))
    save('CONTROL_COMPATIBILITY.json',dict(status='PASS',units=compat,FZ1='DEFERRED_MISSING_TASK4_READOUT'))
def engineering(r,parents):
    rows=read(PUB/'ENGINEERING_PROGRESS.json')['units'] if (PUB/'ENGINEERING_PROGRESS.json').exists() else []
    for e in parents:
        if any((x['dataset'],x['seed'])==(e['dataset'],e['seed']) for x in rows):continue
        assert r.access.get('engineering_optimizer_steps',0)+2<=16,'BLOCKED_ENGINEERING_STEP_LIMIT'
        start=time.monotonic();l,p=r.fork(e,'D');del p
        extract_started=time.monotonic()
        r.phase=r.mode='engineering';l.last_anchor,pt,_,ys,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy();extract_seconds=time.monotonic()-extract_started
        l.last_features=(l.last_anchor,pt,ys);l.epoch_stats=l.anchor_stats;l.epoch_T=l.anchor_T
        rng=l._capture_rng_state();loader_rng=l.loader_generator.get_state().clone()
        # Exact first loader batch including IDs and augmentation under both arms' same parent RNG.
        ds=r.dataset(l.name,l.seed,l.current,True)
        def first():
            dl=DataLoader(ds,batch_size=48,shuffle=True,num_workers=4,drop_last=False,generator=l.loader_generator);return next(iter(dl))
        a=first();l._restore_rng_state(rng);l.loader_generator.set_state(loader_rng);b=first()
        assert all(torch.equal(x,y) for x,y in zip(a,b));l._restore_rng_state(rng);l.loader_generator.set_state(loader_rng)
        began=time.monotonic()
        try:l._init_train(Batches(l),None,l.optimizer,l.scheduler)
        except EngineeringEnd:pass
        else:raise AssertionError('BLOCKED_ENGINEERING_STEP_BOUND')
        step_s=time.monotonic()-began;assert l.steps-l.parent_steps==2
        assert network_hash(l.teacher)==l.teacher_hash and all(v.grad is None and not v.requires_grad for v in l.teacher.parameters())
        norms={}
        for method in ('D','R'):
            l.method=method;l.bind_q();before=context_state(l)
            loss,delta=directional_fd(l._network,l.teacher,l.last_x,l,l.q);same_context(before,context_state(l));assert float(loss)>0
            ns=[(k,v) for k,v in l._network.named_parameters() if v.requires_grad and ('.pool.pool.' in k or '.pool_few.pool.' in k)]
            grads=torch.autograd.grad(loss,[v for k,v in ns],allow_unused=True)
            norms[method]={part:sum(float(g.square().sum()) for (k,_),g in zip(ns,grads) if part in k and g is not None)**.5 for part in ('.pool.pool.','.pool_few.pool.')}
            assert all(v>0 for v in norms[method].values()),'BLOCKED_NO_REAL_FD_GRADIENT'
            old,_=fd(l._network,l.teacher,l.last_x,l)
            torch.testing.assert_close(penalty(delta,l.q[:,:0]),old.detach(),atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(penalty(delta,l.q,10.,10.),10*old.detach(),atol=1e-6,rtol=1e-5)
        l.method='D';l.bind_q();path=PRIVATE/'engineering_resume.pt';l.current_epoch=0
        state=l.save_new(path,True);finger=network_hash(l._network);saved_rng=l._capture_rng_state()
        # Same-lock restore into an independent fresh learner, including optimizer/scheduler/RNG.
        q,_=r.fork(e,'D');q.restore_new(path,True)
        assert network_hash(q._network)==finger and rng_equal(q._capture_rng_state(),saved_rng)
        assert torch.equal(q.loader_generator.get_state(),state['loader_rng']) and torch.equal(q.synth.get_state(),state['synthesis_rng'])
        assert q.scheduler.state_dict()==state['scheduler']
        for k,v in q.optimizer.state_dict()['state'].items():
            for f,x in v.items():
                y=state['optimizer']['state'][k][f];assert torch.equal(x.cpu(),y.cpu()) if torch.is_tensor(x) else x==y
        for field,bad in [('method','X'),('Q_sha256','bad'),('coefficients',[1.,1.]),('worker_sha256','bad'),('protocol_sha256','bad')]:
            trial=dict(state,binding=dict(state['binding']));trial['binding'][field]=bad
            try:q.validate(trial)
            except AssertionError:pass
            else:raise AssertionError('BLOCKED_WRONG_LOCK_ACCEPTED')
        torch.testing.assert_close(features(l._network,l.last_x,l),features(q._network,l.last_x,q),atol=1e-6,rtol=1e-5)
        cp=PRIVATE/'engineering_final.pt.gz';began=time.monotonic();l.save_new(cp,False);save_s=time.monotonic()-began
        size=cp.stat().st_size;ack=r.backup([cp],'engineering_save_transfer');transfer_s=time.monotonic()-began-save_s
        rows.append(dict(dataset=l.name,seed=l.seed,status='PASS',isolated_optimizer_steps=2,FD_adapter_norms=norms,teacher_independent_unchanged=True,
            original_real_loss=True,pointwise_context=True,D_R_batch_augmentation_alignment=True,roundtrip_same_lock_resume=True,wrong_lock_rejected=True,
            seconds_per_step_upper=l.engineering_step_seconds[1],first_probe_step_seconds=l.engineering_step_seconds[0],two_step_elapsed_seconds=step_s,compressed_final_bytes=size,save_seconds=save_s,backup_seconds=transfer_s,case_engineering_seconds=time.monotonic()-start,engineering_probe_seconds=extract_seconds,engineering_probe_rows=len(ys),formal_probe_batches=l.tasks[3]['probe_batches'],val_batches=l.tasks[3]['val_batches']))
        save('ENGINEERING_PROGRESS.json',dict(units=rows));path.unlink();cp.unlink();del l,q,state,a,b,ds;gc.collect();torch.cuda.empty_cache()
        r.resources();print('ENGINEERING_CASE_PASS',e['dataset'],e['seed'],flush=True)
    assert 12<=r.access['engineering_optimizer_steps']<=16
    # 30% throughput reserve plus 45 minutes for extraction, restore, saves and val.
    weighted=sum(x['seconds_per_step_upper']*next(v['expected_steps'] for v in read(PUB/'ASSET_LOCK.json')['parents'] if (v['dataset'],v['seed'])==(x['dataset'],x['seed']))*2 for x in rows)
    probe_seconds=sum((x['engineering_probe_seconds']/int(np.ceil(x['engineering_probe_rows']/48)))*(x['formal_probe_batches']+x['val_batches'])*2 for x in rows)
    seconds=weighted*1.15+probe_seconds*1.15+sum(x['first_probe_step_seconds'] for x in rows)*20+600+r.resources()['GPU_process_residence_seconds']
    storage=sum(x['compressed_final_bytes'] for x in rows)*4*1.15+384*1024**2
    admission=dict(projected_GPU_seconds=seconds,measured_training_projection_seconds=weighted,measured_probe_projection_seconds=probe_seconds,projected_persistent_including_independent_bytes=storage,GPU_limit=r.gpu_limit,persistent_limit=3*1024**3)
    assert seconds<r.gpu_limit and storage<3*1024**3,('BLOCKED_RESOURCE_ADMISSION',admission)
    save('ENGINEERING_GATE.json',dict(status='P0_M1_PASS_FZ1_DEFERRED',units=rows,isolated_optimizer_steps=r.access['engineering_optimizer_steps'],admission=admission))
def train(r,parents):
    assert read(PUB/'ENGINEERING_GATE.json')['status']=='P0_M1_PASS_FZ1_DEFERRED'
    r.phase=r.mode='formal'
    save('TRAINING_STARTED.json',dict(status='STARTING_FORMAL_WORKER',pid=os.getpid(),unix=time.time()))
    for e in parents:
        for method in ('D','R'):
            stem=f"{e['dataset']}_{e['seed']}_{method}"
            receipt=PRIVATE/'units'/(stem+'.json')
            if receipt.exists():
                entry=read(receipt);assert sha(PRIVATE/'finals'/entry['file'])==entry['sha256'];r.lineage.append(entry);continue
            l,p=r.fork(e,method);del p
            active=PRIVATE/'active_resume.pt'
            if active.exists():
                x=load(active);assert (x['dataset'],x['seed'],x['binding']['method'])==(e['dataset'],e['seed'],method)
                l.restore_new(active,True);del x
            else:
                l.last_anchor,_,_,_,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
            l._init_train(Batches(l),None,l.optimizer,l.scheduler)
            assert l.current_epoch==10 and l.steps-l.parent_steps==l.tasks[3]['steps']
            raw,_,y=l.last_features;z=joint(raw)
            l.stats=append(l.epoch_stats,z,y,l.current,4);l.T=append(l.epoch_T,z,y,l.current,4)
            W,diag=ridge(l.T);res=np.linalg.norm((l.T['S']@W+l.T['S'].T@W)/2+.008*W-l.T['mu'])/np.linalg.norm(l.T['mu']);assert res<1e-10
            wp=PRIVATE/'banks'/(stem+'.npz');np.savez_compressed(wp,W=W)
            fp=PRIVATE/'finals'/(stem+'.pt.gz');l.save_new(fp,False)
            entry=dict(dataset=l.name,seed=l.seed,method=method,task=4,known=6,seen=8,file=fp.name,sha256=sha(fp),bytes=fp.stat().st_size,
                W_file=wp.name,W_sha256=sha(wp),ridge_residual=float(res),network_sha256=network_hash(l._network),parent_sha256=e['sha256'],inherited_steps=l.parent_steps,new_steps=l.steps-l.parent_steps,epochs=10)
            write_json(receipt,entry);r.backup([fp,wp,receipt]+list(PUB.glob('*.json'))+list(PUB.glob('*.jsonl')),'completed_'+stem)
            r.lineage.append(entry);save('CHECKPOINT_MANIFEST.json',dict(units=r.lineage))
            active.unlink();del l,raw,z,W;gc.collect();torch.cuda.empty_cache();r.resources()
    assert len(r.lineage)==12 and r.access['formal_optimizer_steps']==5500 and r.access['formal_task_epochs']==120
    save('STATE_W_LOCK.json',dict(status='LOCKED',units=r.lineage,epochs=120,new_optimizer_steps=5500))
def evaluate(r,parents):
    units=read(PUB/'STATE_W_LOCK.json')['units'];assert len(units)==12
    r.phase=r.mode='evaluate';out=[]
    for e in units:
        path=PRIVATE/'sealed'/f"{e['dataset']}_{e['seed']}_{e['method']}_t04.npz"
        receipt=PRIVATE/'units'/(path.stem+'_val.json')
        if receipt.exists():
            row=read(receipt);assert sha(path)==row['sha256'];out.append(row);continue
        parent=next(p for p in parents if (p['dataset'],p['seed'])==(e['dataset'],e['seed']))
        l,_=r.fork(parent,e['method']);fp=PRIVATE/'finals'/e['file'];assert sha(fp)==e['sha256'];l.restore_new(fp,False)
        raw,_,_,y,rows=r.extract(l,range(8),'val');wp=PRIVATE/'banks'/e['W_file'];assert sha(wp)==e['W_sha256']
        with np.load(wp) as f:scores=joint(raw)@f['W']
        assert np.isfinite(scores).all();order=np.array(l.order[:8]);np.savez_compressed(path,raw=scores,y=y,original=order[y],order=order,ids=np.array([x['sample_id'] for x in rows]),component=np.array([x['identity_component'] for x in rows]))
        row=dict(e,file=path.name,sha256=sha(path),method=e['method']+'_T');write_json(receipt,row);r.backup([path,receipt],'sealed_'+path.stem);out.append(row)
        del l,raw,scores;gc.collect();torch.cuda.empty_cache();r.resources()
    save('PREDICTIONS_LOCK.json',dict(status='LOCKED',units=out));r.resources()
def main():
    for folder in ('finals','banks','sealed','units','batch_checks'):(PRIVATE/folder).mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    r=Run();torch.cuda.reset_peak_memory_stats()
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
        if p.suffix.lower() in ('.jpg','.jpeg','.png'):assert str(p) in r.allowed,'BLOCKED_OLD_FUTURE_IMAGE'
        if any(x.lower() in ('test','reserved','oracle') for x in p.parts):raise AssertionError('BLOCKED_FORBIDDEN_ASSET')
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):
            assert not any(p.is_relative_to(Path(x)) for x in CFG['protected_roots'])
    sys.addaudithook(guard)
    try:
        protocol=read(PUB/'PROTOCOL_LOCK.json');assert protocol['worker_sha256']==r.source_sha
        assert sha(ROOT/'source/report_dfd_m1.py')==protocol['report_sha256']
        assert sha(ROOT/'source/dfd_t4_core.py')==protocol['new_source_sha256']['dfd_t4_core.py']
        asset_check(r)
        parents=sorted(CFG['parents'],key=lambda e:(['ISIC','HK'].index(e['dataset']),e['seed']))
        r.backup(list(PUB.glob('*'))+list((ROOT/'source').glob('*.py')),'initial_locks')
        if not (PUB/'ENGINEERING_GATE.json').exists() or read(PUB/'ENGINEERING_GATE.json')['status']=='BLOCKED_RESOURCE':engineering(r,parents)
        train(r,parents);evaluate(r,parents)
        r.phase='report';started=time.monotonic()
        from report_dfd_m1 import report
        report(r);assert time.monotonic()-started<7200
        save('ACCESS_AUDIT.json',r.resources())
        r.backup(list(PUB.glob('*'))+list((PRIVATE/'batch_checks').glob('*')),'final_reports')
        save('DELIVERY_AUDIT.json',dict(status='PASS_PRIVATE_BACKUP',all12_final_bank_score_SHA_verified=True,public_release='AWAITING_SEPARATE_AUTHORIZATION'))
        save('COMPLETE.json',dict(execution_status='COMPLETE',original_five_condition_plan_complete=False,new_steps=5500,epochs=120,U3='NOT_EVALUABLE_MISSING_FZ1',frozen_superiority='NOT_ESTABLISHED',full_method_promotion=False,NEXT_DECISION='STOP'))
        r.backup([PUB/'COMPLETE.json',PUB/'DELIVERY_AUDIT.json'],'completion_receipt')
    except BaseException as e:
        save('FAILURE_'+str(time.time_ns())+'.json',dict(status='BLOCKED',type=type(e).__name__,reason=str(e),traceback=traceback.format_exc(),formal_steps=r.access.get('formal_optimizer_steps',0)))
        raise
    finally:r.resources(False)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
