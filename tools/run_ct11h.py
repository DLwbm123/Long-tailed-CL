"""Frozen-state val-only neural-head diagnosis; all science restores are inherited."""
import gc,os,sys,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from run_ct6f import Full
from run_ct3p import FDStudent,read,sha,write_json,jsonl,npz,network_hash,joint
from run_ct9p import Student
from resume_ct6f import exchange
from run_medical_v2 import rng_equal

class Run(Full):
    exchange=exchange
    def dataset(self,name,seed,classes,train,split='train',smoke=False):
        assert not train and split=='val' and list(classes)==list(range(8)) and self.scope==(name,seed,4),'BLOCKED_NON_VAL'
        assert (self.pub/'ENGINEERING_GATE.json').exists()
        return super().dataset(name,seed,classes,False,'val',False)
    def resources(self,enforce=True):
        x=super().resources(False)
        assert x['fit_image_reads']==0 and not self.access.get('formal_optimizer_steps',0)
        if enforce:assert x['active_bytes']<1024**3 and x['free_bytes']>=1024**3,'BLOCKED_STORAGE'
        return x
    def fetch(self,e):
        self.exchange('GET',e['name'],e['sha256'],origin=e['origin'])
        p=self.private/'parent.part';assert sha(p)==e['sha256']
        l=(Student if e['origin']=='p28' else FDStudent)(self,e['dataset'],e['seed'])
        saved=l.restore_new(p,False)
        assert (saved['task'],saved['epoch'],saved['known'],saved['seen'],saved['beta'])==(3,10,6,8,e['beta'])
        assert network_hash(l._network)==e['network_sha256'] and rng_equal(l._capture_rng_state(),saved['rng'])
        assert torch.equal(l.loader_generator.get_state(),saved['loader_rng']) and torch.equal(l.synth.get_state(),saved['synthesis_rng'])
        assert not l.probe.backbone.pool.batchwise_prompt and not l.probe.backbone.pool_few.batchwise_prompt
        del saved;p.unlink() # Read-only archive retains the validated original.
        return l

def main():
    cfg=read(os.environ['P30_CONFIG']);cfg['mode']=os.environ['P30_MODE'];r=Run(cfg)
    r.prior_gpu=r.prior_fit=r.prior_val=0
    prior=read(r.pub/'RESOURCE_LEDGER.json') if (r.pub/'RESOURCE_LEDGER.json').exists() else {}
    r.prior_gpu=prior.get('GPU_process_residence_seconds',0);r.prior_val=prior.get('val_image_reads',0);r.access=prior.get('calls',{})
    lock=read(r.pub/'DIAGNOSTIC_LOCK.json');assert sha(__file__)==lock['worker_sha256']
    for path,digest in lock['references'].items():assert sha(path)==digest
    oldroots=[Path(cfg[k]).resolve() for k in ('ct1_root','ct3p_root','ct4f_root','ct5f_root','ct6f_root','ct9p_root')]
    def protect(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve();flags=args[2] if len(args)>2 else 0
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):assert not any(p.is_relative_to(q) for q in oldroots)
        elif p.suffix.lower() in ('.npy','.npz','.pt','.jpg','.jpeg','.png'):
            assert str(p) in lock['allowed_assets'] or p.is_relative_to(r.private.resolve()) or str(p) in r.allowed,('BLOCKED_ASSET',str(p))
    sys.addaudithook(protect);r.setup()
    try:
        if r.mode=='qualify':
            assert not (r.pub/'ENGINEERING_GATE.json').exists()
            for e in lock['checkpoints']:
                l=r.fetch(e)
                jsonl(r.pub/'RESTORE_AUDIT.jsonl',dict(dataset=l.name,seed=l.seed,beta=e['beta'],strict_restore=True,RNG_loader_synthesis=True,checkpoint_sha256=e['sha256'],network_sha256=e['network_sha256'],pointwise_probe=True))
                del l;gc.collect();torch.cuda.empty_cache();r.resources()
            assert r.counter[1]==0
            write_json(r.pub/'ENGINEERING_GATE.json',dict(status='PASS',states=12,optimizer_steps=0,image_reads=0))
            write_json(r.pub/'STATE_W_LOCK.json',dict(status='LOCKED',diagnostic_lock_sha256=sha(r.pub/'DIAGNOSTIC_LOCK.json'),states=12,readouts=24))
        elif r.mode=='evaluate':
            assert read(r.pub/'ENGINEERING_GATE.json')['status']=='PASS' and not (r.pub/'PREDICTIONS_LOCK.json').exists()
            units=[]
            for e in lock['checkpoints']:
                l=r.fetch(e);r.permit(l.name,l.seed,4);r.phase='evaluate';rng=l._capture_rng_state()
                calls=[]
                def blind(mod,args,kwargs):
                    assert len(args)==1 and kwargs==dict(train=False,weight=None),'BLOCKED_LABEL_DEPENDENCY'
                    calls.append(len(args[0]))
                hook=l.probe.register_forward_pre_hook(blind,with_kwargs=True)
                try:raw,_,scores,y,rows=r.extract(l,range(8),'val')
                finally:hook.remove()
                assert scores.shape==(len(rows),8) and np.isfinite(scores).all() and sum(calls)==len(rows)
                assert all(n<=48 for n in calls) and network_hash(l._network)==e['network_sha256'] and rng_equal(l._capture_rng_state(),rng)
                ids=np.array([v['sample_id'] for v in rows]);component=np.array([v['identity_component'] for v in rows]);order=np.array(l.order[:8]);z=joint(raw)
                for q in e['readouts']:
                    assert sha(q['W_path'])==q['W_sha256'] and sha(q['prediction_path'])==q['sha256']
                    with np.load(q['W_path'],allow_pickle=False) as f:actual=z@f['W']
                    with np.load(q['prediction_path'],allow_pickle=False) as f:
                        for key,value in [('ids',ids),('component',component),('order',order),('y',y),('original',order[y])]:np.testing.assert_array_equal(f[key][:8] if key=='order' else f[key],value)
                        np.testing.assert_allclose(actual,f['raw'],atol=1e-6,rtol=1e-6)
                        np.testing.assert_array_equal(actual.argmax(1),f['raw'].argmax(1))
                        delta=float(np.max(np.abs(actual-f['raw'])))
                    jsonl(r.pub/'REPRODUCTION_AUDIT.jsonl',dict(dataset=l.name,seed=l.seed,method=q['method'],max_abs_difference=delta,argmax_identical=True,layout_equal=True,labels_not_forwarded=True,state_RNG_unchanged=True))
                method='N1' if e['beta']==1 else 'N10';name=f'{l.name}_{l.seed}_{method}_t04.npz'
                npz(r.private/'sealed'/name,raw=scores,y=y,original=order[y],order=order,ids=ids,component=component)
                units.append(dict(dataset=l.name,seed=l.seed,task=4,known=6,seen=8,method=method,file=name,sha256=sha(r.private/'sealed'/name),network_sha256=e['network_sha256']))
                jsonl(r.pub/'UNITS_COMPLETED.jsonl',units[-1]);del l,raw,z,scores;gc.collect();torch.cuda.empty_cache();r.resources()
            assert len(units)==12 and r.counter[1]==5464
            write_json(r.pub/'PREDICTIONS_LOCK.json',dict(status='LOCKED',units=units))
        else:raise ValueError(r.mode)
    except BaseException as e:
        write_json(r.pub/f'FAILURE_{r.mode}.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e)));raise
    finally:r.resources(False)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
