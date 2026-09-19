"""Strict Task3 continuation of the qualified FD learner, then locked validation."""
import os,time,gc,copy,functools,shutil,sys
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_ct3p as base
from run_ct3p import Prefix,FDStudent,CountedImages,read,sha,write_json,jsonl,npz,network_hash,joint,ridge
from run_medical_v2 import rng_equal
from resume_ct3p import finish_task

def bounds(tasks,task):
    assert 1<=task<=len(tasks)
    return (0 if task==1 else tasks[task-2]['seen']),tasks[task-1]['seen']

class Full(Prefix):
    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert self.scope is not None and self.scope[:2]==(name,seed)
        known,seen=bounds(self.lock[name]['runs'][str(seed)],self.scope[2]);classes=list(classes)
        assert split in ('train','val') and set(classes)<=set(range(known,seen) if split=='train' else range(seen)),'BLOCKED_STAGE_ACCESS'
        if split=='val':assert (self.pub/'STATE_W_LOCK.json').exists(),'BLOCKED_EARLY_VAL'
        spec=self.cfg['datasets'][name];ds=CountedImages(Path(spec['manifests'])/(split+'.csv'),spec['images'],self.lock[name]['orders'][seed],classes,train,smoke)
        ds.rows.sort(key=lambda r:r['sample_id']);ds.counter=self.counter;ds.split=split;ds.offline=False
        self.allowed={str((Path(spec['images'])/x['relative_path']).resolve()) for x in ds.rows};return ds
    def resources(self,enforce=True):
        old=read(self.pub/'RESOURCE_LEDGER.json') if (self.pub/'RESOURCE_LEDGER.json').exists() else {}
        size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(self.root).free
        x=dict(phase=self.mode,GPU_process_residence_seconds=self.prior_gpu+time.monotonic()-self.started,
            calls=self.access,fit_image_reads=self.prior_fit+self.counter[0],val_image_reads=self.prior_val+self.counter[1],
            old_fit_reads=0,future_fit_reads=0,test_predictions=0,oracle_reads=0,active_bytes=size,free_bytes=free,
            peak_GPU_allocated_bytes=max(old.get('peak_GPU_allocated_bytes',0),torch.cuda.max_memory_allocated() if self.gpu_start else 0),
            CPU_analytic_seconds=sum(v for k,v in self.access.items() if k.endswith('analytic_ms'))/1000,new_checkpoints=len(self.lineage),archive_bytes=self.archived)
        write_json(self.pub/'RESOURCE_LEDGER.json',x)
        if enforce:assert size<2*1024**3 and free>=1024**3 and x['CPU_analytic_seconds']<6600,'BLOCKED_RESOURCE'
        return x

def parent(r,e):
    path=r.private/'parents'/e['name'];assert sha(path)==e['sha256']
    l=FDStudent(r,e['dataset'],e['seed']);p=l.restore_new(path,False)
    assert (p['task'],p['epoch'],p['seen'])==(2,10,6)
    assert network_hash(l._network)==e['network_sha256'] and rng_equal(l._capture_rng_state(),p['rng'])
    assert torch.equal(l.loader_generator.get_state(),p['loader_rng']) and torch.equal(l.synth.get_state(),p['synthesis_rng'])
    for bank,label in [(l.stats,'C2'),(l.T,'C3')]:
        assert bank['mu'].shape==(1536,6) and all(np.isfinite(bank[k]).all() for k in ('S','mu','v'))
        expect=[r.lock[l.name]['train_counts'][c] for c in l.order[:6]];np.testing.assert_array_equal(bank['n'],expect)
        old=Path(r.cfg['ct3p_root'])/'output/private/banks'/f'{l.name}_{l.seed}_{label}_t03.npz'
        with np.load(old) as f:np.testing.assert_allclose(ridge(bank)[0],f['W'],atol=1e-9,rtol=1e-9)
    l.teacher_source=e
    return l,p,path

def qualify(r,parents):
    rows=[]
    for e in parents:
        l,p,_=parent(r,e);l.task_setup(3)
        assert l.teacher_hash==e['network_sha256'] and all(not v.requires_grad for v in l.teacher.parameters())
        assert rng_equal(l._capture_rng_state(),p['rng']) and torch.equal(l.loader_generator.get_state(),p['loader_rng'])
        for a,b in zip(l._network.parameters(),l.teacher.parameters()):assert a.data_ptr()!=b.data_ptr()
        rows.append(dict(dataset=l.name,seed=l.seed,parent_sha256=e['sha256'],strict_restore=True,RNG_loader_synthesis=True,A_T_counts_W=True,teacher_task3_identity=True))
        del l,p;gc.collect();torch.cuda.empty_cache();r.resources()
    assert len(rows)==6;write_json(r.pub/'ENGINEERING_GATE.json',dict(status='PASS',parents=rows,neural_steps=0,boundary_test='PASS'))

def train(r,parents):
    assert read(r.pub/'ENGINEERING_GATE.json')['status']=='PASS'
    assert not (r.pub/'TRAINING_STARTED.json').exists()
    write_json(r.pub/'TRAINING_STARTED.json',dict(status='RUNNING',source_commit=r.cfg['source_commit'],unix=time.time()))
    r.phase='formal'
    for e in parents:
        l,p,path=parent(r,e);del p;path.unlink() # Original verified archive remains; this is the admitted disposable copy.
        for task in range(3,len(l.tasks)):
            r.permit(l.name,l.seed,task+1);l.task_setup(task)
            assert l.teacher_hash==l.teacher_source['network_sha256']
            l.last_anchor,_,_,_,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
            finish_task(r,l,l.tasks[task]['steps'],l.steps)
        del l;gc.collect();torch.cuda.empty_cache()
    assert len(r.lineage)==27 and len(r.entries)==54
    assert r.access['formal_optimizer_steps']==5570 and r.access['formal_task_epochs']==270
    write_json(r.pub/'STATE_W_LOCK.json',dict(status='LOCKED',units=r.entries,checkpoints=r.lineage,source_commit=r.cfg['source_commit'],protocol=read(r.pub/'PROTOCOL_LOCK.json')))

def evaluate(r):
    lock=read(r.pub/'STATE_W_LOCK.json');assert len(lock['units'])==54
    assert not (r.pub/'PREDICTIONS_LOCK.json').exists();units=[]
    old=Path(r.cfg['ct3p_root'])/'output'
    for e in read(old/'public/ONLINE_PREDICTIONS_LOCK.json')['units']:
        if e['method'] not in ('C2','C3'):continue
        src=old/'private/sealed'/e['file'];assert sha(src)==e['sha256'];shutil.copyfile(src,r.private/'sealed'/e['file'])
        units.append(dict(e,known=e['seen']-2,reused_CT3P_prediction=True))
    assert len(units)==36
    for e in lock['checkpoints']:
        r.exchange('GET',e['name'],e['sha256']);path=r.private/'parent.part';assert sha(path)==e['sha256']
        l=FDStudent(r,e['dataset'],e['seed']);p=l.restore_new(path,False);del p
        assert network_hash(l._network)==e['network_sha256'];r.permit(l.name,l.seed,e['task'])
        raw,_,_,y,rows=r.extract(l,range(l._total_classes),'val');z=joint(raw);order=np.array(l.order[:l._total_classes])
        assert network_hash(l._network)==e['network_sha256']
        for method in ('C2','C3'):
            q=next(q for q in lock['units'] if (q['dataset'],q['seed'],q['task'],q['method'])==(l.name,l.seed,e['task'],method));wp=r.private/'banks'/q['W_file'];assert sha(wp)==q['W_sha256']
            with np.load(wp) as f:score=z@f['W']
            assert np.isfinite(score).all();name=f'{l.name}_{l.seed}_{method}_t{e["task"]:02d}.npz'
            npz(r.private/'sealed'/name,raw=score,y=y,original=order[y],order=order,ids=np.array([x['sample_id'] for x in rows]),component=np.array([x['identity_component'] for x in rows]))
            units.append(dict(q,file=name,sha256=sha(r.private/'sealed'/name)))
        path.unlink();del l,raw,z;gc.collect();torch.cuda.empty_cache();r.resources()
    assert len(units)==90;write_json(r.pub/'PREDICTIONS_LOCK.json',dict(status='LOCKED',units=units));write_json(r.pub/'INFERENCE_COMPLETE.json',dict(status='PASS',units=90))

def main():
    cfg=read(os.environ['P25_CONFIG']);mode=os.environ['P25_MODE'];cfg['mode']=mode;root=Path(cfg['root']);pub=root/'output/public'
    lock=read(pub/'PROTOCOL_LOCK.json');assert sha(__file__)==lock['worker_sha256']
    prior=read(pub/'RESOURCE_LEDGER.json') if (pub/'RESOURCE_LEDGER.json').exists() else {}
    r=Full(cfg);r.access=prior.get('calls',{});r.prior_gpu=prior.get('GPU_process_residence_seconds',0);r.prior_fit=prior.get('fit_image_reads',0);r.prior_val=prior.get('val_image_reads',0)
    r.setup();r.archived=prior.get('archive_bytes',0)
    def protect(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
            if flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert not any(p.is_relative_to(Path(cfg[k]).resolve()) for k in ('ct1_root','ct3p_root','ct4f_root','ct5f_root'))
    sys.addaudithook(protect)
    original=base.ridge
    def timed(*args,**kwargs):
        t=time.monotonic()
        try:return original(*args,**kwargs)
        finally:r.count('analytic_ms',int((time.monotonic()-t)*1000))
    base.ridge=timed;torch.save=functools.partial(torch.save,pickle_protocol=4)
    parents=[e for e in read(Path(cfg['ct3p_root'])/'output/public/PARENT_AND_FORK_LINEAGE.json') if e['task']==3]
    try:
        if mode=='qualify':qualify(r,parents)
        elif mode=='train':train(r,parents)
        elif mode=='evaluate':evaluate(r)
        else:raise ValueError(mode)
    except BaseException as e:write_json(pub/f'FAILURE_{mode}.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e)));raise
    finally:r.resources(False)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
