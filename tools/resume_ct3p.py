"""Storage-only repair and explicit same-lock continuation of CT3-P R1."""
import functools,gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from threadpoolctl import threadpool_limits
import run_ct3p as base
from run_medical_v2 import rng_equal
from run_ct3p import Prefix,FDStudent,read,write_json,jsonl,sha,network_hash,append,joint


def strict_resume(r,path):
    p=torch.load(path,map_location='cpu',weights_only=False)
    assert (p['dataset'],p['seed'],p['task'],p['epoch'])==('ISIC',1994,2,2)
    assert p['steps']==sum(x['optimizer_steps'] for x in p['records'])==726
    parent=next(x for x in r.lineage if (x['dataset'],x['seed'],x['task'])==('ISIC',1994,2))
    assert p['teacher_source']['sha256']==parent['sha256'] and p['teacher_hash']==parent['network_sha256']
    l=FDStudent(r,'ISIC',1994);l.task_setup(2);l.restore_new(path,True)
    assert network_hash(l.teacher)==p['teacher_hash']
    assert all(v.grad is None and not v.requires_grad for v in l.teacher.parameters())
    assert rng_equal(l._capture_rng_state(),p['rng'])
    assert torch.equal(l.loader_generator.get_state(),p['loader_rng'])
    assert torch.equal(l.synth.get_state(),p['synthesis_rng'])
    assert l.scheduler.state_dict()==p['scheduler']
    for key,state in l.optimizer.state_dict()['state'].items():
        for field,value in state.items():
            expected=p['optimizer']['state'][key][field]
            assert torch.equal(value.cpu(),expected.cpu()) if torch.is_tensor(value) else value==expected
    # The task-start teacher supplies the actual pre-task adapter fingerprint.
    teacher=l.teacher.state_dict()
    l.adapter_before={k:teacher[k].detach().cpu().clone() for k in l.adapter_before}
    l.fdparts=[];l.epoch_parts=[];l.epoch_gradient_nonzero=0
    return l,p


def qualify(r):
    r.phase='engineering';path=r.private/'active_resume.pt.part';digest=sha(path)
    l,p=strict_resume(r,path)
    assert r.access['formal_optimizer_steps']==1956 and r.access['formal_task_epochs']==92
    # Epoch2 is fully serialized and now strictly restored; retire only the obsolete rolling epoch1.
    previous=r.private/'active_resume.pt'
    retired=dict(bytes=previous.stat().st_size,epoch=1)
    path.replace(previous)
    write_json(r.pub/'STRICT_RECOVERY_R2.json',dict(status='PASS',dataset='ISIC',seed=1994,task=3,epoch=2,
        checkpoint_sha256=digest,network_sha256=p['network_sha256'],teacher_sha256=p['teacher_hash'],optimizer_scheduler_RNG=True,
        formal_steps=1956,formal_epochs=92,additional_optimizer_steps=0,retired_obsolete_rolling=retired,
        scientific_source_commit='d1eaa599a23679d025659a6d1b45a43996291b86',serialization_protocol=4))
    del l,p;gc.collect();torch.cuda.empty_cache();r.resources(False)


def finish_task(r,l,expected_steps,start_steps):
    ds=r.dataset(l.name,l.seed,l.current,True)
    loader=DataLoader(ds,batch_size=48,shuffle=True,num_workers=4,drop_last=False,generator=l.loader_generator)
    l._init_train(loader,None,l.optimizer,l.scheduler)
    assert l.steps-start_steps==expected_steps and l.current_epoch==10
    raw,_,y=l.last_features;z=joint(raw)
    l.stats=append(l.epoch_stats,z,y,l.current,l.task+1);l.T=append(l.epoch_T,z,y,l.current,l.task+1)
    changed=[k for k,v in l._network.named_parameters() if k in l.adapter_before and not torch.equal(v.detach().cpu(),l.adapter_before[k])]
    assert changed and network_hash(l.teacher)==l.teacher_hash
    jsonl(r.pub/'TASK_UPDATE_AUDIT.jsonl',dict(dataset=l.name,seed=l.seed,task=l.task+1,adapter_changed_names=changed,teacher_hash=l.teacher_hash,teacher_source=l.teacher_source['network_sha256']))
    for method,state in [('C2',l.stats),('C3',l.T)]:r.save_candidate(l,method,state,'F')
    e=r.archive(l);l.teacher_source=e
    (r.private/'active_resume.pt').unlink();del l.teacher;l.teacher=None
    del raw,z,loader,ds;gc.collect();torch.cuda.empty_cache();r.resources()


def resume(r):
    assert read(r.pub/'STRICT_RECOVERY_R2.json')['status']=='PASS'
    assert read(r.pub/'RECOVERY_ADMISSION_R2.json')['status']=='PASS'
    assert len(r.lineage)==9 and r.access['formal_task_epochs']==92 and r.access['formal_optimizer_steps']==1956
    assert len(r.entries)==84
    r.phase='formal';r.permit('ISIC',1994,3)
    l,p=strict_resume(r,r.private/'active_resume.pt');start=l.steps;del p
    write_json(r.pub/'RESUMED_R2.json',dict(status='RUNNING',unix=time.time(),dataset='ISIC',seed=1994,task=3,next_epoch=3,repeated_formal_steps=0))
    finish_task(r,l,2304,start)
    del l;gc.collect();torch.cuda.empty_cache()
    path=r.private/'parents/ISIC_1995.pt'
    e=next(x for x in read(r.old/'output/public/MODEL_LINEAGE.json') if (x['dataset'],x['seed'],x['stream'],x['task'])==('ISIC',1995,'U',1))
    assert sha(path)==e['sha256'];l=FDStudent(r,'ISIC',1995);l.fork_from_ct1_task1(path,e);path.unlink()
    write_json(r.pub/'ACTUAL_CONFIG_ISIC_1995.json',dict(l.args,device=['cuda:0'],locked_weight_path='PINNED_SHARED_AUGREG',FD_beta=10.,prefix_tasks=[1,2,3]))
    for task in (1,2):
        r.permit('ISIC',1995,task+1);l.task_setup(task)
        l.last_anchor,_,_,_,_=r.extract(l,l.current);l.anchor_routes=r.last_routes.copy()
        finish_task(r,l,l.tasks[task]['steps'],l.steps)
    del l;gc.collect();torch.cuda.empty_cache()
    assert len(r.entries)==90 and len(r.lineage)==12 and r.access['formal_optimizer_steps']==6620 and r.access['formal_task_epochs']==120
    write_json(r.pub/'ONLINE_STATE_W_LOCK.json',dict(status='LOCKED',source_commit=r.cfg['source_commit'],recovery_lock=read(r.pub/'RECOVERY_LOCK_R2.json'),candidates=r.entries,
        control_features=[json.loads(x) for x in (r.private/'CONTROL_FEATURE_CACHE.jsonl').read_text().splitlines()],checkpoints=r.lineage,epochs=120,optimizer_steps=6620))


def install_resource_limit(root):
    lock=read(root/'output/public/RECOVERY_LOCK_R2.json')
    assert sha(Path(__file__))==lock['recovery_worker_sha256'],'BLOCKED_RECOVERY_CODE_DRIFT'
    limit=lock['GPU_limit_seconds']
    assert (limit is None and lock['budget_authorization']=='USER_APPROVED_CT3P_NO_GPU_HOUR_LIMIT') or limit==14400 or (limit==18000 and lock['budget_authorization']=='USER_APPROVED_CT3P_CUMULATIVE_5H')
    original=Prefix.resources
    def resources(self,enforce=True):
        x=original(self,False)
        if enforce:
            assert (limit is None or x['GPU_process_residence_seconds']<limit) and x['CPU_analytic_seconds']+600<7200,'BLOCKED_RESOURCE'
            assert x['active_bytes']<2*1024**3 and x['persistent_archive_upper_bytes']<3*1024**3 and x['min_free_bytes']>=1024**3,'BLOCKED_RESOURCE'
        return x
    Prefix.resources=resources


def main():
    cfg=read(os.environ['P22_CONFIG']);mode=os.environ['P22_RECOVERY_MODE']
    if mode!='qualify':install_resource_limit(Path(cfg['root']))
    r=Prefix(cfg);r.setup()
    original=base.ridge
    def timed(*args,**kwargs):
        t=time.monotonic()
        try:return original(*args,**kwargs)
        finally:r.count(r.mode+'_analytic_ms',int(1000*(time.monotonic()-t)))
    base.ridge=timed
    # Same objects and exact values; protocol4 avoids protocol2's oversized NumPy encoding.
    torch.save=functools.partial(torch.save,pickle_protocol=4)
    try:
        with threadpool_limits(limits=4):{'qualify':qualify,'resume':resume,'evaluate':base.evaluate,'oracle':base.oracle}[mode](r)
    except BaseException as e:
        write_json(r.pub/'FAILURE_RECOVERY_R2.json',dict(status='BLOCKED',error=type(e).__name__,reason=str(e)));raise
    finally:r.resources(False)
if __name__=='__main__':main()
