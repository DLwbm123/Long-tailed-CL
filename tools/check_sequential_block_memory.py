"""Current-task-only spectral construction and immutable append under class growth."""
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture,family,losses,grouped,memory_losses,moments_objective,project_ball,constrain_family
from check_spectral_distribution_memory import spectral_fit
from check_block_spherical_memory import decode
from check_sequential_spherical_memory import geometry_error


def construct_current(x,y,start,stop,cfg,lock):
    assert len(x)==len(y) and set(y.tolist())==set(range(start,stop)), 'Current task scope mismatch'
    counts=torch.bincount(y-start,minlength=stop-start);d=x.shape[1];m=lock['nodes_per_class']
    means=torch.stack([x[y==j].mean(0) for j in range(start,stop)])
    second=torch.stack([x[y==j].T@x[y==j]/counts[j-start] for j in range(start,stop)])
    initial=[]
    for j in range(start,stop):
        a=x[y==j];g=torch.Generator().manual_seed(cfg['seed']+1000+j)
        initial.append(a[torch.randint(len(a),(m,),generator=g)])
    initial=torch.stack(initial);zero=torch.zeros(stop-start,m,dtype=DT)
    n=initial.clone().requires_grad_();w=zero.clone().requires_grad_();opt=torch.optim.Adam([n,w],lr=lock['memory_lr']);begun=time.monotonic()
    for _ in range(lock['outer_steps']):
        opt.zero_grad();moments_objective(n,w,means,second).backward();opt.step()
        with torch.no_grad():n.copy_(project_ball(n));w.clamp_(-8,8)
    baseline_seconds=time.monotonic()-begun;begun=time.monotonic()
    mu,basis,ev,tail,trace,diag=spectral_fit(x,y-start,stop-start,m)
    for a in diag:a['class_index']+=start
    return {'random':(initial,zero),'moment':(n.detach(),w.detach()),'distribution':(mu,basis,ev,tail)},dict(baseline_seconds=baseline_seconds,spectral_seconds=time.monotonic()-begun,baseline_updates=lock['outer_steps'],candidate_updates=0),diag


def validate_memory(mem,seen,d,task,source):
    assert mem['schema']=='sequence_block_v1' and mem['frame']=='stable_synthetic_input_v1'
    assert mem['seen']==seen and mem['dimension']==d and mem['task']==task and mem['source_commit']==source
    assert set(mem['modes'])=={'random','moment','distribution'}
    for mode,arrays in mem['modes'].items():
        expected=((seen,d),(seen,d,15),(seen,15),(seen,)) if mode=='distribution' else ((seen,16,d),(seen,16))
        assert len(arrays)==len(expected)
        assert all(a.shape==shape and bool(torch.isfinite(a).all()) for a,shape in zip(arrays,expected))
    assert bool((mem['modes']['distribution'][2]>=0).all()) and bool((mem['modes']['distribution'][3]>=0).all())


def samples(distribution,mode,seed):
    # Class-keyed stateless draws preserve old generated points exactly when new classes append.
    mu,basis,ev,tail=distribution;out=[];d=mu.shape[1]
    for j in range(len(mu)):
        k=int((basis[j].square().sum(0)>.5).sum());trace=ev[j].sum()+(d-k)*tail[j]
        out.append(decode(mu[j:j+1],basis[j:j+1],ev[j:j+1],tail[j:j+1],trace[None],seed+100000*j,mode,16)[0])
    return torch.stack(out)


def evaluate(x,y,memories,old,cfg,lock):
    seen=len(memories['moment'][0]);d=cfg['dimension'];zero=torch.zeros(seen,16,dtype=DT)
    fixed={k:memories[k] for k in ('random','moment')};models=('gaussian','block_spherical')
    params=[family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i) for i in range(4)]
    for i in range(2):
        p=[a.clone().requires_grad_() for a in params[i]];opt=torch.optim.Adam(p,lr=lock['query_lr'])
        for step in range(lock['evaluation_adversary_steps']):
            opt.zero_grad();target=grouped(losses(x,p),y,seen)
            err=[(target-memory_losses(n,w,p)).square().mean() for n,w in fixed.values()]
            for mode in models:err.append((target-memory_losses(samples(memories['distribution'],mode,cfg['seed']+5000+100*i+step),zero,p)).square().mean())
            (-torch.stack(err).mean()).backward();opt.step();constrain_family(p)
        params.append([a.detach() for a in p])
    rows=[];predictions={mode:[] for mode in models}
    with torch.no_grad():
        target=torch.cat([grouped(losses(x,p),y,seen) for p in params],1)
        def summarize(mode,n,w,batch=None,pred=None):
            if pred is None:pred=torch.cat([memory_losses(n,w,p) for p in params],1)
            err=(target-pred).abs()
            return dict(mode=mode,batch=batch,mean_abs_error=float(err.mean()),max_abs_error=float(err.max()),random_max=float(err[:,:32].max()),stress_max=float(err[:,32:].max()),old_mean_abs_error=float(err[:old].mean()) if old else None,old_max_abs_error=float(err[:old].max()) if old else None,new_mean_abs_error=float(err[old:].mean()),new_max_abs_error=float(err[old:].max()),per_class_mean=[float(t) for t in err.mean(1)],per_class_max=[float(t) for t in err.max(1).values])
        for mode,(n,w) in fixed.items():rows.append(summarize(mode,n,w))
        for draw in range(lock['audit_batches']):
            for mode in models:
                n=samples(memories['distribution'],mode,cfg['seed']+7000+draw);pred=torch.cat([memory_losses(n,zero,p) for p in params],1);predictions[mode].append(pred)
                a=summarize(mode,n,zero,draw,pred);a['normalized_second_trace']=[float(t) for t in n.square().sum(2).mean(1)];rows.append(a)
        avg=[summarize(mode+'_eight_batch_average_DIAGNOSTIC_ONLY',None,None,pred=torch.stack(pred).mean(0)) for mode,pred in predictions.items()]
        variance={mode:dict(mean=float(torch.stack(pred).var(0,unbiased=True).mean()),max=float(torch.stack(pred).var(0,unbiased=True).max()),per_class_mean=[float(t) for t in torch.stack(pred).var(0,unbiased=True).mean(1)]) for mode,pred in predictions.items()}
    fixed={a['mode']:a for a in rows[:2]};passed=all(a['max_abs_error']<=.9*fixed[b]['max_abs_error'] and a['mean_abs_error']<=fixed[b]['mean_abs_error'] for a in rows if a['mode']=='block_spherical' for b in fixed)
    return rows,passed,avg,variance


def run_scale(cfg,lock,journal):
    begun=time.monotonic();oracle_x,oracle_y,counts=fixture(cfg);d=cfg['dimension'];g=torch.Generator().manual_seed(cfg['seed']+2000)
    basis=torch.linalg.qr(torch.randn(d,cfg['classes'],dtype=DT,generator=g)).Q;probe=F.normalize(torch.randn(19,d,dtype=DT,generator=g),dim=1)
    memories={};stages=[];old=0
    for task,added in enumerate(cfg['task_sizes'],1):
        seen=old+added;journal(dict(phase='current_task_compression',task=task,seen=seen,completed_stage_results=stages))
        previous={k:tuple(a.clone() for a in t) for k,t in memories.items()};mask=(oracle_y>=old)&(oracle_y<seen)
        current_x,current_y=oracle_x[mask].clone(),oracle_y[mask].clone();local,cost,diag=construct_current(current_x,current_y,old,seen,cfg,lock);current_rows=len(current_x);del current_x,current_y
        for key,t in local.items():memories[key]=tuple(torch.cat((memories[key][i],a)) if old else a for i,a in enumerate(t))
        assert all(torch.equal(memories[k][i][:old],a) for k,t in previous.items() for i,a in enumerate(t))
        max_error=0.;old_draw_exact=True
        with torch.no_grad():
            if old:
                for mode in ('gaussian','block_spherical'):
                    before=samples(previous['distribution'],mode,cfg['seed']+7000);after=samples(memories['distribution'],mode,cfg['seed']+7000)[:old]
                    assert torch.equal(before,after);w=torch.zeros(old,16,dtype=DT)
                    for i in range(4):
                        p=family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i);max_error=max(max_error,float((memory_losses(before,w,p)-memory_losses(after,w,p)).abs().max()))
                for mode in ('random','moment'):
                    for i in range(4):
                        p=family(d,lock['adapter_rank'],lock['queries'],cfg['seed']+100+i);max_error=max(max_error,float((memory_losses(*previous[mode],p)-memory_losses(memories[mode][0][:old],memories[mode][1][:old],p)).abs().max()))
        assert max_error==0.;geo=geometry_error(basis,probe,old,seen);assert geo<1e-12
        checkpoint=dict(schema='sequence_block_v1',frame='stable_synthetic_input_v1',seen=seen,dimension=d,task=task,source_commit=os.environ['Q315_SHA'],modes=memories)
        validate_memory(checkpoint,seen,d,task,os.environ['Q315_SHA']);path=Path(os.environ['Q315_ROOT'])/(cfg['id']+'_checkpoint.private.pt');torch.save(checkpoint,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True);validate_memory(loaded,seen,d,task,os.environ['Q315_SHA']);assert all(torch.equal(a,loaded['modes'][k][i]) for k,t in memories.items() for i,a in enumerate(t))
        for mode in ('gaussian','block_spherical'):assert torch.equal(samples(memories['distribution'],mode,cfg['seed']+7000),samples(loaded['modes']['distribution'],mode,cfg['seed']+7000))
        saved={k:tuple(a.clone() for a in t) for k,t in memories.items()};journal(dict(phase='evaluation_only_oracle',task=task,seen=seen,completed_stage_results=stages));start=time.monotonic();mask=oracle_y<seen
        rows,passed,avg,var=evaluate(oracle_x[mask],oracle_y[mask],memories,old,cfg,lock)
        assert all(torch.equal(a,saved[k][i]) for k,t in memories.items() for i,a in enumerate(t))
        baseline_bytes=8*sum(a.numel() for a in memories['moment']);candidate_bytes=8*sum(a.numel() for a in memories['distribution']);assert baseline_bytes==candidate_bytes
        stages.append(dict(task=task,old=old,seen=seen,current_support_rows=current_rows,construction_old_support_rows=0,old_memory_exact=True,old_draw_exact=old_draw_exact,old_fixed_query_error=max_error,rescaled_ETF_old_logit_error=geo,checkpoint_exact=True,checkpoint_bytes=path.stat().st_size,baseline_memory_bytes=baseline_bytes,candidate_memory_bytes=candidate_bytes,construction_costs=cost,analytic_geometry=diag,rows=rows,averaged_prediction_diagnostic=avg,prediction_variance_diagnostic=var,finite_query_screen_pass=passed,evaluation_seconds=time.monotonic()-start))
        old=seen;journal(dict(phase='stage_complete',task=task,seen=seen,completed_stage_results=stages))
    assert old==cfg['classes']
    return dict(status='COMPLETED',case=cfg,synthetic_counts=counts,stages=stages,finite_query_screen_pass=all(s['finite_query_screen_pass'] for s in stages),oracle_is_synthetic_evaluation_only=True,recompression_performed=False,classification_training_performed=False,CPU_pipeline_seconds=time.monotonic()-begun)


def unit_checks():
    d=4;n=F.normalize(torch.arange(1.,129.,dtype=DT).reshape(2,16,d),dim=2);w=torch.zeros(2,16,dtype=DT)
    distribution=(n[:,0],torch.zeros(2,d,15,dtype=DT),torch.zeros(2,15,dtype=DT),torch.zeros(2,dtype=DT))
    mem=dict(schema='sequence_block_v1',frame='stable_synthetic_input_v1',seen=2,dimension=d,task=1,source_commit='unit',modes={'random':(n,w),'moment':(n,w),'distribution':distribution});validate_memory(mem,2,d,1,'unit');reject=0
    for key,value in [('schema','bad'),('frame','bad'),('seen',3),('dimension',5),('task',2),('source_commit','bad')]:
        bad=dict(mem);bad[key]=value
        try:validate_memory(bad,2,d,1,'unit')
        except AssertionError:reject+=1
    assert reject==6
    try:construct_current(n[:,0],torch.tensor([0,2]),1,3,{}, {})
    except AssertionError:pass
    else:raise AssertionError('Old support accepted')
    before=samples(distribution,'block_spherical',315);extended=tuple(torch.cat((a,a[:1])) for a in distribution)
    assert torch.equal(before,samples(extended,'block_spherical',315)[:2])
    return dict(status='PASS',metadata_rejections=reject,old_support_rejected=True,class_keyed_draw_prefix_exact=True,optimizer_updates=0)


def run(cfg,lock,journal):
    torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    result=unit_checks() if cfg['kind']=='unit' else run_scale(cfg,lock,journal)
    result.update(id=cfg['id'],GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0);assert not result['GPU_initialized'];return result
