"""One explicit task-boundary recovery after the recorded archive timeout."""
import os,time,gc
from pathlib import Path
import numpy as np
import torch
import run_ct6f as w
from run_medical_v2 import rng_equal
from run_hyperkvasir_hk1_m1 import tensor_digest
from ct1_statistics import append


def exchange(self,op,name,digest,**extra):
    began=time.monotonic();key=str(time.time_ns());w.write_json(self.private/'REQUEST.json',dict(id=key,op=op,name=name,sha256=digest,**extra))
    try:
        while time.monotonic()-began<660:
            p=self.private/'ACK.json'
            if p.exists():
                ack=w.read(p)
                if ack['id']==key:
                    assert ack['status']=='PASS',ack
                    return ack
            time.sleep(1)
        raise TimeoutError('BLOCKED_TRANSFER_R1')
    finally:self.count(self.mode+'_transfer_ms',int(1000*(time.monotonic()-began)))


def resume(r,parents):
    assert r.access['formal_task_epochs']==180 and r.access['formal_optimizer_steps']==2350
    assert len(r.lineage)==17 and len(r.entries)==36
    assert not (r.pub/'RESUMED_R1.json').exists()
    r.phase='formal';path=r.private/'HK_1995_F_t05.pt';req=w.read(r.private/'REQUEST_R0.json')
    assert req['name']==path.name and w.sha(path)==req['sha256']
    l=w.FDStudent(r,'HK',1995);p=l.restore_new(path,False)
    assert (p['task'],p['epoch'],p['seen'])==(4,10,10)
    assert p['steps']==sum(x['optimizer_steps'] for x in p['records'])
    assert w.network_hash(l._network)==req['network_sha256'] and tensor_digest(p['delta'].items())==req['delta_sha256']
    assert rng_equal(l._capture_rng_state(),p['rng']) and torch.equal(l.loader_generator.get_state(),p['loader_rng']) and torch.equal(l.synth.get_state(),p['synthesis_rng'])
    roll=torch.load(r.private/'active_resume.pt',map_location='cpu',weights_only=False)
    assert (roll['task'],roll['epoch'],roll['network_sha256'],roll['steps'])==(4,10,p['network_sha256'],p['steps'])
    assert rng_equal(roll['rng'],p['rng']) and torch.equal(roll['loader_rng'],p['loader_rng']) and torch.equal(roll['synthesis_rng'],p['synthesis_rng'])
    raw,_,y=roll['last_features'];z=w.joint(raw)
    for label,bank,key in [('C2',l.stats,'epoch_stats'),('C3',l.T,'epoch_T')]:
        expected=append(roll[key],z,y,range(p['known'],p['seen']),5)
        for k in ('S','mu','v','n'):np.testing.assert_allclose(bank[k],expected[k],atol=1e-10,rtol=1e-10)
        entry=next(e for e in r.entries if (e['dataset'],e['seed'],e['task'],e['method'])==('HK',1995,5,label))
        f=r.private/'banks'/entry['W_file'];assert w.sha(f)==entry['W_sha256']
        with np.load(f) as saved:np.testing.assert_allclose(w.ridge(bank)[0],saved['W'],atol=1e-9,rtol=1e-9)
    del roll,raw,z,expected
    w.write_json(r.pub/'STRICT_RECOVERY_R1.json',dict(status='PASS',checkpoint_sha256=req['sha256'],task=5,epoch=10,network_RNG_loader_synthesis=True,committed_A_T_reconstructed_from_epoch10=True,existing_W_reproduced=True,repeated_optimizer_steps=0))
    ack=r.exchange('PUT',path.name,req['sha256'],bytes=req['bytes'],network_sha256=req['network_sha256'],delta_sha256=req['delta_sha256'])
    assert ack['readable'];r.archived=ack['archive_bytes']
    e=dict(dataset='HK',seed=1995,task=5,name=path.name,sha256=req['sha256'],network_sha256=req['network_sha256'],teacher_source={k:l.teacher_source[k] for k in ('name','sha256','network_sha256','task')},source_commit=r.cfg['source_commit'])
    r.lineage.append(e);w.write_json(r.pub/'PARENT_AND_FORK_LINEAGE.json',r.lineage);l.teacher_source=e
    path.unlink();(r.private/'active_resume.pt').unlink();del p
    w.write_json(r.pub/'RESUMED_R1.json',dict(status='RUNNING',unix=time.time(),next_task='HK1995 Task6',remaining_epochs=90,remaining_steps=3220,remaining_new_checkpoints=9,repeated_optimizer_steps=0))
    for task in range(5,len(l.tasks)):
        r.permit(l.name,l.seed,task+1);l.task_setup(task);assert l.teacher_hash==l.teacher_source['network_sha256']
        l.last_anchor,_,_,_,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
        w.finish_task(r,l,l.tasks[task]['steps'],l.steps)
    del l;gc.collect();torch.cuda.empty_cache()
    for parent in parents:
        if parent['dataset']!='ISIC':continue
        l,p,path=w.parent(r,parent);del p;path.unlink()
        for task in range(3,len(l.tasks)):
            r.permit(l.name,l.seed,task+1);l.task_setup(task);assert l.teacher_hash==l.teacher_source['network_sha256']
            l.last_anchor,_,_,_,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
            w.finish_task(r,l,l.tasks[task]['steps'],l.steps)
        del l;gc.collect();torch.cuda.empty_cache()
    assert len(r.lineage)==27 and len(r.entries)==54
    assert r.access['formal_optimizer_steps']==5570 and r.access['formal_task_epochs']==270
    w.write_json(r.pub/'STATE_W_LOCK.json',dict(status='LOCKED',units=r.entries,checkpoints=r.lineage,source_commit=r.cfg['source_commit'],protocol=w.read(r.pub/'PROTOCOL_LOCK.json'),recovery_lock=w.read(r.pub/'RECOVERY_LOCK_R1.json')))


def main():
    root=Path('/tmp/p25root');lock=w.read(root/'output/public/RECOVERY_LOCK_R1.json')
    assert w.sha(__file__)==lock['worker_sha256']
    w.Full.exchange=exchange;w.train=resume
    w.main()
if __name__=='__main__':
    with w.threadpool_limits(limits=4):main()
