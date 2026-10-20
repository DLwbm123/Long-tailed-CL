"""Final-20 offline CE capability reference; no image or query loader in fitting."""
import json,os,time,traceback,signal
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.optimize import minimize
from core import objective,self_check
STATE={}


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(cfg):
    torch.set_num_threads(4);root=Path(cfg['output']);root.mkdir(exist_ok=False);start=cfg['original_start_epoch']
    count={'closure_evaluations':0,'new_image_encodes':0,'encoder_optimizer_updates':0,'old_heads_refitted':0,'cache_rows_loaded':0}
    STATE.update(root=root,count=count)
    def status(phase,**kw):
        write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),counts=count,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60
    status('math_qualification');assert self_check()
    src=Path(cfg['cache_run']);old_cfg=json.loads((src/'config.private.json').read_text())
    assert old_cfg['seed']==cfg['seed'] and old_cfg['order']==cfg['order']
    ix=np.load(src/'train_index.private.npz');y=ix['labels'].astype(np.int64);groups=ix['groups'];mass=ix['weights'];bounds=ix['bounds'];cache=np.load(src/'train768.private.npy',mmap_mode='r')
    assert cache.shape==(68058,1536) and cache.dtype==np.float32 and np.array_equal(np.unique(y),np.arange(20))
    for c in range(20):
        lo,hi=bounds[c:c+2];assert np.all(y[lo:hi]==c)
        counts=Counter(groups[lo:hi].tolist());expected=np.array([1/(len(counts)*counts[g]) for g in groups[lo:hi]])
        assert np.max(np.abs(mass[lo:hi]-expected))<1e-14 and abs(float(mass[lo:hi].sum())-1)<1e-12
    old=torch.load(Path(cfg['control_run'])/'heads_5.private.pt',weights_only=True)
    audit=json.loads((Path(cfg['control_run'])/'FIT_SEAL.json').read_text());alpha=audit['stages'][-1]['matched_alpha']
    assert abs(alpha-1.5373739361267944)<1e-12
    fac=old['factors'].numpy();anchor=old['df_matched_joint'].numpy();assert anchor.shape==(1537,20) and fac.shape==(1537,)
    x=np.empty((68058,1537),dtype=np.float64);diag=np.zeros(1537)
    for lo in range(0,len(y),2048):
        hi=min(lo+2048,len(y));x[lo:hi,:1536]=np.asarray(cache[lo:hi],dtype=np.float64)*fac[:1536];x[lo:hi,-1]=fac[-1]
        assert np.isfinite(x[lo:hi]).all();diag+=(x[lo:hi]**2*mass[lo:hi,None]).sum(0);count['cache_rows_loaded']+=hi-lo
        status('load_readonly_cache')
    w0=anchor-anchor.mean(1,keepdims=True);residual=x@w0;residual+=1/20;residual[np.arange(len(y)),y]-=1
    ridge_grad=x.T@(residual*mass[:,None])+alpha*w0;ridge_error=float(np.max(np.abs(ridge_grad)))
    count['ridge_normal_equation_readout_rows']=len(y);assert ridge_error<1e-6
    precondition=np.sqrt(diag+alpha);assert np.isfinite(precondition).all() and np.min(precondition)>0
    np.save(root/'precondition.private.npy',precondition)
    write(root/'QUALIFICATION.json',dict(state='PASS',math=True,cache_shape=list(cache.shape),patients=len(np.unique(groups)),classes=20,images=len(y),class_counts=np.bincount(y).tolist(),ridge_normal_equation_gradient_max=ridge_error,new_image_encodes=0,alpha=alpha,logit_multiplier=20,offline_reference_only=True))
    initial,ig=objective(w0,x,y,mass,alpha);count['initial_CE_gradient_readout_rows']=len(y)
    STATE['head']=w0.copy();trace_path=root/'optimization.private.jsonl'
    def closure(vector):
        status('fit_offline_CE')
        assert count['closure_evaluations']<cfg['closure_cap']
        w=vector.reshape(1537,20)/precondition[:,None];STATE['head']=w.copy()
        value,grad=objective(w,x,y,mass,alpha)
        assert np.isfinite(value) and np.isfinite(grad).all()
        count['closure_evaluations']+=1
        with trace_path.open('a') as f:f.write(json.dumps(dict(evaluation=count['closure_evaluations'],objective=value,raw_gradient_max=float(np.max(np.abs(grad))),elapsed=time.time()-start))+'\n')
        if count['closure_evaluations']%20==0:np.save(root/'candidate.private.npy',w)
        return value,(grad/precondition[:,None]).ravel()
    result=minimize(closure,(w0*precondition[:,None]).ravel(),jac=True,method='L-BFGS-B',options={k:cfg[k] for k in ['maxiter','maxfun','maxls','maxcor','ftol','gtol']})
    w=result.x.reshape(1537,20)/precondition[:,None];STATE['head']=w.copy()
    final,grad=objective(w,x,y,mass,alpha);count['terminal_CE_gradient_readout_rows']=len(y)
    assert np.isfinite(w).all() and np.isfinite(final) and np.isfinite(grad).all()
    gmax=float(np.max(np.abs(grad)));converged=gmax<=cfg['raw_gradient_tolerance']
    np.save(root/'head.private.npy',w);np.save(root/'terminal_gradient.private.npy',grad)
    optimization=dict(alpha=alpha,classes=20,logit_multiplier=20,initial_objective=initial,final_objective=final,initial_raw_gradient_max=float(np.max(np.abs(ig))),terminal_raw_gradient_max=gmax,converged=converged,scipy_success=bool(result.success),scipy_status=int(result.status),scipy_message=str(result.message),iterations=int(result.nit),nfev=int(result.nfev),head_column_mean_max=float(np.max(np.abs(w.mean(1)))),offline_reference_only=True)
    write(root/'OPTIMIZATION.json',optimization)
    write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,heads=1,offline_final20=True,optimization=optimization))
    count.update(head_optimizer_iterations=int(result.nit),closure_feature_row_evaluations=count['closure_evaluations']*len(y),total_new_CE_heads=1)
    write(root/'costs.json',dict(wall_seconds=time.time()-start,counts=count,new_image_encodes=0,device='CPU',threads=4,attempts=1,cache_file_bytes=(src/'train768.private.npy').stat().st_size,cache_copied=False))
    write(root/'status.private.json',dict(state='FIT_COMPLETE_PENDING_VERIFICATION',updated=time.time(),converged=converged,counts=count))


def stop(signum,frame):raise TimeoutError('Independent guardian requested bounded stop')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        root=Path(cfg['output']);root.mkdir(exist_ok=True)
        if 'head' in STATE:np.save(root/'partial_head.private.npy',STATE['head'])
        write(root/'FAILURE.private.json',dict(updated=time.time(),error=traceback.format_exc(),counts=STATE.get('count',{})))
        raise
