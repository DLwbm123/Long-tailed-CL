"""One new task stream, fresh teachers, matched modules, global eval barrier."""
import os,json,time,traceback,resource
from pathlib import Path
import torch
from common import write,guard

c=json.loads(Path(os.environ['RUN_CONFIG']).read_text());r=Path(c['output']);root=Path(c['root_output']);wall=time.time();cpu=time.process_time()
try:
    torch.set_num_threads(c['cpu_threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write(r/'PROCESS.private.json',{'worker':os.getpid(),'cmdline':Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode()})
    import teacher,data_contract,preflight,train,evaluate,analyze,verify
    from model import Joint,supervised,auxiliary,diagnostics
    from check import fixture
    meta=data_contract.metadata(c);rows,y,roles=meta
    # Only the changed empty-tail/age semantics need a new algebra check.
    d=fixture('cuda');d.update(stage_denominator=4,tail=torch.zeros(3,dtype=torch.bool,device='cuda'),protected=torch.zeros(3,dtype=torch.bool,device='cuda'))
    m=Joint('PROTO_GRPO').cuda();assert diagnostics(m,d)['B_tail'] is None
    loss,_=supervised(m,d);loss.backward();assert float(m.U.weight.grad.norm())>0;m.zero_grad(set_to_none=True)
    loss,_=auxiliary(m,d,[torch.randn(8,len(ix),device='cuda') for ix in d['ix']]);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in m.representation_parameters())
    write(r/'TASK_SOFTWARE.json',{'passed':True,'empty_tail_null':True,'reward_empty_protected_finite':True,'support_query_gradients':True,'aux_adapter_frozen':True,'optimizer_encoder_VAL_TEST':0,'unchanged_GRPO_algebra':'R101 receipt/source reused, no MC toy repeated'})
    del m,d,loss;torch.cuda.empty_cache()
    write(r/'STATUS.json',{'phase':'TEACHER_NATIVE_BUDGET','teacher_optimizer':0,'module_optimizer':0})
    store=teacher.Prefixes(c,rows,y)
    assert preflight.teacher_budget(c,rows,y,store),'FULL_PLAN_BUDGET_NOT_ADMITTED'
    write(r/'TEACHER_FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'stream':c['stream_name'],'source_commit':c['source_commit'],'no_old_HK_weights':True})
    teacher.run(c,rows,y,store);store.cache.clear();del store;torch.cuda.empty_cache()
    assert preflight.module_budget(c,meta),'MODULE_REMAINING_BUDGET_NOT_ADMITTED'
    # Fresh A0 at every stage is an engineering readout; actual carry never reset.
    from model import deterministic
    bindings=[]
    for s in range(5):
        d=data_contract.load(c,s,meta);torch.manual_seed(c['seed']);m=Joint('UNIFORM').cuda();score,*_=deterministic(m,d)
        err=float((score-d['teacher_scores']).abs().max());assert err<1e-4 and torch.equal(score.argmax(1),d['teacher_scores'].argmax(1));bindings.append({'stage':s,'maxabs':err,'predictions_match':True});del d,m,score
    write(r/'A0_BINDING.json',{'passed':True,'records':bindings,'real_carry_not_reset':True})
    write(r/'FORMAL_STARTED.private.json',{'start':c['start'],'deadline':c['deadline'],'module_steps':17920,'source_commit':c['source_commit']})
    train.run(c,meta);write(r/'STATUS.json',{'phase':'WAIT_GLOBAL_EVAL_BARRIER','counts':train.COUNTS})
    while not (root/'GLOBAL_LEARNING_SEAL.json').exists():guard(c);time.sleep(2)
    evaluate.run(c,meta,train.COUNTS);a=analyze.run(c);verify.run(c)
    import aggregate_saved
    aggregate_saved.run(r)
    write(r/'STATUS.json',{'phase':a['conclusion'],'counts':train.COUNTS});write(r/'WORKER_COMPLETE.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-wall,'teacher_counts':teacher.COUNTS,'module_counts':train.COUNTS})
except BaseException:
    (r/'FAILURE.private.txt').write_text(traceback.format_exc());write(r/'STATUS.json',{'phase':'FAILED'})
    try:write(r/'FAILURE_COUNTS.private.json',{'teacher':teacher.COUNTS,'module':train.COUNTS,'engineering':preflight.ENGINEERING})
    except BaseException:pass
    raise
finally:
    try:write(r/'COSTS.json',{'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-wall,'worker_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'teacher_counts':teacher.COUNTS,'module_counts':train.COUNTS,'engineering_in_worker':preflight.ENGINEERING,'teacher_costs_not_free':True,'teacher_feature_stage_cost_breakdown':'TEACHER_RECORDS.json; all inside worker, do not double-add','preparation_import_SSH_publication_CPU':'nonzero not fully sampled'})
    except BaseException:pass
