import os,sys,copy,time,json,types
from pathlib import Path
sys.path[:0]=['/tmp/p20root/dependencies','/tmp/p22root/code/tools']
import torch,numpy as np
from threadpoolctl import threadpool_limits
from run_ct1 import CTLearner,read,task_lock,sha,network_hash,rng_equal
from run_medical_v2 import write_json
root=Path('/tmp/p22root');cfg=read('/tmp/p20root/runtime_formal.json');counts={}
def count(k,n=1):counts[k]=counts.get(k,0)+n
r=types.SimpleNamespace(cfg=cfg,lock=task_lock(cfg),code=read('/tmp/p20root/output/public/SOURCE_COMMIT_BINDING.json')['qualified_code_sha256'],phase='engineering',count=count)
allowed=set()
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]));rp=p.resolve()
        if p.suffix.lower() in ('.jpg','.jpeg','.png'):assert str(rp) in allowed,'BLOCKED_IMAGE_ACCESS'
        if any(x.startswith(('test','reserved')) for x in p.parts):raise AssertionError('BLOCKED_TEST')
        assert not ('ct2d' in str(rp) and '/private/' in str(rp)),'BLOCKED_ORACLE'
sys.addaudithook(guard)
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
began=time.monotonic();record=dict(status='RUNNING',optimizer_steps=0)
try:
    with threadpool_limits(limits=4):
        entry=next(q for q in read('/tmp/p20root/output/public/MODEL_LINEAGE.json') if (q['dataset'],q['seed'],q['stream'],q['task'])==('HK',1993,'U',1))
        assert sha(root/'output/private/parent.pt')==entry['sha256']
        l=CTLearner(r,'HK',1993,'U');p=l.restore(root/'output/private/parent.pt');teacher=copy.deepcopy(l._network).requires_grad_(False).eval()
        l.task_setup(1);l.smoke=True;l.args['tuned_epoch']=1
        from run_medical_v2 import Images
        s=cfg['datasets']['HK'];ds=Images(Path(s['manifests'])/'train.csv',s['images'],l.order,l.current,True);ds.rows.sort(key=lambda x:x['sample_id'])
        ds.rows=ds.rows[:48];allowed.update(str((Path(s['images'])/q['relative_path']).resolve()) for q in ds.rows)
        batch=[ds[i] for i in range(len(ds))];x=torch.stack([q[1] for q in batch]).cuda();y=torch.tensor([q[2] for q in batch]).cuda();record['current_image_reads']=len(batch)
        def pointwise(net):
            flags=[m.training for m in net.modules()];pools=[net.backbone.pool,net.backbone.pool_few];pw=[p.batchwise_prompt for p in pools]
            attrs=[(m,m.adapt_list) for m in net.modules() if hasattr(m,'adapt_list')];rng=l._capture_rng_state()
            try:
                net.eval()
                for q in pools:q.batchwise_prompt=False
                o=net(x,train=False,weight=None)
                return torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1),o
            finally:
                for m,flag in zip(net.modules(),flags):m.training=flag
                for q,flag in zip(pools,pw):q.batchwise_prompt=flag
                for m,value in attrs:m.adapt_list=value
                l._restore_rng_state(rng)
        l.probe.load_state_dict(l._network.state_dict());l.probe.eval()
        with torch.no_grad():
            native,_=pointwise(l._network);o=l.probe(x,train=False,weight=None);patched=torch.nn.functional.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1)
            record['native_vs_extraction_max_abs']=float((native-patched).abs().max());record['native_vs_extraction_RMS_L2']=float(torch.mean(torch.sum((native-patched)**2,1)).sqrt())
            torch.testing.assert_close(native,patched,atol=1e-5,rtol=1e-5)
            target,_=pointwise(teacher);record['identity_FD']=float(torch.mean(torch.sum((native-target)**2,1)))
        h=network_hash(teacher);t=time.monotonic();l._init_train([(None,x,y)],None,l.optimizer,None);record['optimizer_steps']=1;record['real_step_seconds']=time.monotonic()-t
        rng=l._capture_rng_state();flags=[m.training for m in l._network.modules()];l._network.zero_grad(set_to_none=True)
        t=time.monotonic();z,_=pointwise(l._network);loss=torch.mean(torch.sum((z-target)**2,1));(10*loss).backward();torch.cuda.synchronize()
        record['FD_forward_backward_seconds']=time.monotonic()-t;record['post_update_FD']=float(loss)
        grads={}
        for label,part in [('main','backbone.pool.pool.'),('few','backbone.pool_few.pool.')]:
            vals=[v.grad.square().sum() for k,v in l._network.named_parameters() if part in k and v.grad is not None]
            grads[label]=float(torch.stack(vals).sum().sqrt()) if vals else 0
        assert all(v>0 and np.isfinite(v) for v in grads.values()),grads
        assert all(v.grad is None for v in teacher.parameters()) and network_hash(teacher)==h
        assert flags==[m.training for m in l._network.modules()] and rng_equal(rng,l._capture_rng_state())
        record.update(status='PASS',weighted_FD_adapter_grad_norm=grads,teacher_unchanged=True,flags_RNG_restored=True,
            no_CT2D_private_assets=True,old_future_fit_reads=0,test_reads=0,counts=counts)
except BaseException as e:
    record.update(status='BLOCKED',error=type(e).__name__,reason=str(e));raise
finally:
    record.update(GPU_process_residence_seconds=time.monotonic()-began,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(root/'output/public/FD_EARLY_GATE.json',record);print(json.dumps(record),flush=True)
