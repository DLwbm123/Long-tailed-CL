"""Full frozen-Task1 analytic reference; existing prefix predictions are reused."""
import gc,os,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_ct3p import Prefix,CountedImages,read,write_json,sha,network_hash,joint,append,ridge,npz
from run_ct1 import CTLearner



def bounds(tasks,task):
    assert 1<=task<=len(tasks)
    return (0 if task==1 else tasks[task-2]['seen']),tasks[task-1]['seen']

class Full(Prefix):
    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert self.scope is not None and self.scope[:2]==(name,seed)
        task=self.scope[2];known,seen=bounds(self.lock[name]['runs'][str(seed)],task)
        assert split in ('train','val') and set(classes)<=set(range(known,seen) if split=='train' else range(seen)),'BLOCKED_STAGE_ACCESS'
        if split=='val':assert (self.pub/'STATE_W_LOCK.json').exists(),'BLOCKED_EARLY_VAL'
        spec=self.cfg['datasets'][name];ds=CountedImages(Path(spec['manifests'])/(split+'.csv'),spec['images'],self.lock[name]['orders'][seed],list(classes),False,False)
        ds.rows.sort(key=lambda r:r['sample_id']);ds.counter=self.counter;ds.split=split;ds.offline=False
        self.allowed={str((Path(spec['images'])/x['relative_path']).resolve()) for x in ds.rows};return ds

def main():
    cfg=read(os.environ['P24_CONFIG']);root=Path(cfg['root']);pub=root/'output/public';private=root/'output/private';began=time.monotonic();cpu=0.;r=None
    lock=read(pub/'PROTOCOL_LOCK.json');assert sha(Path(__file__))==lock['worker_sha256'];assert not (pub/'STATE_W_LOCK.json').exists()
    def protect(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
            if flags and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert not any(p.is_relative_to(Path(cfg[k]).resolve()) for k in ('ct1_root','ct3p_root','ct4f_root'))
    sys.addaudithook(protect)
    def resources():
        size=sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink());free=shutil.disk_usage(root).free
        assert size<1024**3 and free>=1024**3 and cpu<1800
        return dict(wall_seconds=time.monotonic()-began,CPU_analytic_report_seconds=cpu,disk_bytes=size,free_bytes=free,neural_epochs=0,optimizer_steps=0,test_reads=0,old_fit_reads=0,future_fit_reads=0,peak_GPU_allocated_bytes=0 if r is None else torch.cuda.max_memory_allocated(),calls={} if r is None else r.access,fit_image_reads=0 if r is None else r.counter[0],val_image_reads=0 if r is None else r.counter[1],prefix_reconstruction_fit_image_reads=31494)
    try:
        r=Full(cfg);r.setup();states=[];audit=[]
        def no_training(*args,**kwargs):raise AssertionError('BLOCKED_NEURAL_TRAINING')
        CTLearner._init_train=no_training
        for name in ('HK','ISIC'):
            for seed in (1993,1994,1995):
                e=next(e for e in read(r.old/'output/public/MODEL_LINEAGE.json') if (e['dataset'],e['seed'],e['stream'],e['task'])==(name,seed,'U',1))
                path=Path(cfg['ct4f_root'])/'output/private/parents'/e['name'];assert sha(path)==e['sha256'];l=CTLearner(r.parent_view,name,seed,'U');p=l.restore(path);l.r=r
                assert p['task']==0 and p['epoch']==10 and p['seen']==2
                for v in l._network.parameters():v.requires_grad_(False)
                fingerprint=network_hash(l._network);state=p['stats'];del p
                for task in range(1,len(l.tasks)+1):
                    known,seen=bounds(l.tasks,task)
                    r.permit(name,seed,task)
                    if task>1:
                        probe_began=time.monotonic()
                        raw,_,_,y,_=r.extract(l,range(known,seen));state=append(state,joint(raw),y,range(known,seen),task);del raw
                        assert state['n'][known:seen].sum()==l.tasks[task-1]['n_train']
                        np.testing.assert_array_equal(state['n'][known:seen],np.bincount(y,minlength=seen)[known:seen])
                        if name=='HK' and seed==1993 and task==4:write_json(pub/'THROUGHPUT_GATE.json',dict(status='PASS',probe_seconds=time.monotonic()-probe_began,rows=l.tasks[task-1]['n_train'],resources=resources()))
                    t=time.monotonic();W,diag=ridge(state);cpu+=time.monotonic()-t
                    assert network_hash(l._network)==fingerprint
                    if task==1:
                        with np.load(r.old/'output/private/sealed'/f'{name}_{seed}_U_W_t01.npz') as f:np.testing.assert_allclose(W,f['CT-J-CB'],atol=1e-9,rtol=1e-9)
                    if task<=3:
                        with np.load(Path(cfg['ct4f_root'])/'output/private/banks'/f'{name}_{seed}_t{task:02d}.npz') as f:np.testing.assert_allclose(W,f['W'],atol=1e-10,rtol=1e-10)
                    stem=f'{name}_{seed}_t{task:02d}';npz(private/'banks'/(stem+'.npz'),W=W)
                    states.append(dict(dataset=name,seed=seed,task=task,method='F1',seen=seen,known=known,W_file=stem+'.npz',W_sha256=sha(private/'banks'/(stem+'.npz')),network_sha256=fingerprint,parent_sha256=e['sha256']))
                    audit.append(dict(dataset=name,seed=seed,task=task,counts=state['n'].tolist(),network_unchanged=True,**diag));write_json(pub/'FIT_AUDIT.json',audit);write_json(pub/'RESOURCES.json',resources());print('FIT',name,seed,task,flush=True)
                del l,state;gc.collect();torch.cuda.empty_cache()
        assert len(states)==45;write_json(pub/'STATE_W_LOCK.json',dict(status='LOCKED',units=states,protocol=lock))
        units=[];r.mode='evaluate';r.phase='evaluate'
        for name in ('HK','ISIC'):
            for seed in (1993,1994,1995):
                path=Path(cfg['ct4f_root'])/'output/private/parents'/f'{name}_{seed}_U_t01.pt';l=CTLearner(r.parent_view,name,seed,'U');p=l.restore(path);l.r=r;fingerprint=network_hash(l._network);del p
                for v in l._network.parameters():v.requires_grad_(False)
                for task in range(1,len(l.tasks)+1):
                    known,seen=bounds(l.tasks,task)
                    if task<=3:
                        old=next(e for e in read(Path(cfg['ct4f_root'])/'output/public/PREDICTIONS_LOCK.json')['units'] if (e['dataset'],e['seed'],e['task'])==(name,seed,task))
                        src=Path(cfg['ct4f_root'])/'output/private/sealed'/old['file'];assert sha(src)==old['sha256'];shutil.copyfile(src,private/'sealed'/old['file'])
                        e=next(e for e in states if (e['dataset'],e['seed'],e['task'])==(name,seed,task));units.append(dict(e,file=old['file'],sha256=old['sha256'],reused_CT4F_prediction=True));continue
                    r.permit(name,seed,task);raw,_,_,y,rows=r.extract(l,range(seen),'val');e=next(e for e in states if (e['dataset'],e['seed'],e['task'])==(name,seed,task))
                    wp=private/'banks'/e['W_file'];assert sha(wp)==e['W_sha256']
                    with np.load(wp) as f:score=joint(raw)@f['W']
                    order=np.array(l.order[:seen]);ids=np.array([v['sample_id'] for v in rows]);comp=np.array([v['identity_component'] for v in rows])
                    assert network_hash(l._network)==fingerprint
                    filename=f'{name}_{seed}_F1_t{task:02d}.npz';npz(private/'sealed'/filename,raw=score,y=y,original=order[y],order=order,ids=ids,component=comp)
                    units.append(dict(e,file=filename,sha256=sha(private/'sealed'/filename)));del raw
                del l;gc.collect();torch.cuda.empty_cache()
        assert len(units)==45
        write_json(pub/'PREDICTIONS_LOCK.json',dict(status='LOCKED',units=units))
        write_json(pub/'RESOURCES.json',resources());write_json(pub/'INFERENCE_COMPLETE.json',dict(status='PASS',units=45,neural_epochs=0,optimizer_steps=0))
    except BaseException as e:write_json(pub/'FAILURE.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e),elapsed=time.monotonic()-began));raise
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
