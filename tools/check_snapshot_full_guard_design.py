"""Synthetic CPU snapshots and exact legacy-cap probes, no image/checkpoint loader."""
import io
import time
import torch
import core1_competition as competition
import prototype_coherent as coherent
import module_additions as modules
import edge_competition as edge
import native_pointwise_guard_design as native
import snapshot_full_guard_design as guard
from check_full_fit_guard_design import sample,label_counts,bank
from check_joint_refresh_guard_design import prepare as legacy_prepare
from check_native_pointwise_guard_design import oracle


def snapshot(event):
    return dict(event=event,scope='fit',encoder_eval=True,frame='synthetic_snapshot',support_frame='synthetic_snapshot',
                encoder_version='encoder_0',support_encoder_version='encoder_0')


def full_bank_system(b,r):
    k=len(b['n']);initial=torch.linalg.solve(b['Q'].sum(0)/k+.001*r,b['mu'].T/k)
    a=competition.competition(b,initial)
    fixed,check=competition.bank_head(b,a,regularizer=r)
    state=edge.descriptors(b,fixed,k-2)
    pairs=edge.allocation(a,state,torch.zeros(8,dtype=r.dtype))
    assert torch.equal(a,pairs)
    mass=r.new_full((k,),1/k)
    op,rhs,inv=native.system(b['Q']/k,b['mu']/k,mass,pairs,r,torch.zeros_like(initial),proximal=0.)
    w,audit=inv(rhs);assert torch.allclose(w,fixed,atol=2e-8,rtol=2e-8)
    return op,rhs,inv,w,audit


def prepare_event(c):
    d=c['dimension'];k=c['classes'];old=c['old_classes'];gen=torch.Generator().manual_seed(241+k)
    counts=c['current_train_counts'];fit_counts=label_counts(c['fit_n'],counts)
    all_x=torch.cat([sample(n,d,old+i,gen) for i,n in enumerate(counts)])
    all_y=torch.cat([torch.full((n,),old+i,dtype=torch.long) for i,n in enumerate(counts)])
    fit=torch.cat([torch.arange(sum(counts[:i]),sum(counts[:i])+v) for i,v in enumerate(fit_counts)])
    x,y=all_x[fit],all_y[fit]
    old_samples=[sample(n,d,i,gen) for i,n in enumerate(c['old_train_counts'])]
    b0=bank(old_samples);scale=torch.ones(d,dtype=x.dtype);scale[0]=c.get('affine_first',2.8);scale[1]=.65 if c.get('affine_first',2.8)!=1 else 1.
    moved=bank([z*scale for z in old_samples])
    xx,yy=(x,y) if c['event']=='epoch_init' else (all_x,all_y)
    current=bank([xx[yy==v] for v in range(old,k)])
    for component in current['components']:component['label']+=old
    def joined(b):return dict(mu=torch.cat([b['mu'],current['mu']]),Q=torch.cat([b['Q'],current['Q']]),
        n=b['n']+current['n'],components=b['components']+current['components'])
    off,on=joined(b0),joined(moved)
    local=torch.nn.functional.normalize(torch.randn(k,4,d,generator=gen,dtype=x.dtype),dim=-1)
    arm='local_transport' if c['LT'] else 'base'
    R0=modules.metric(off,arm,c.get('dataset','ISIC'),list(range(k)),local if c['LT'] else None)[0]
    R=modules.metric(on,arm,c.get('dataset','ISIC'),list(range(k)),local if c['LT'] else None)[0]
    o0,b0,s0,ref,_=full_bank_system(off,R0);op,rhs,inv,wu,ua=full_bank_system(on,R)
    return op,rhs,inv,wu,ref,x,y,dict(objective_current_n=len(xx),protected_fit_n=len(x),
        already_captured_total_current_n=len(all_x),meta_used_as_protected_support=False,
        head_proximal=0.,zero_selected_static_PC=True,extra_image_extractions=0,unconstrained=ua)


def legacy(c):
    f=legacy_prepare(c);x=f['x'];y=f['y'];bx=x[f['ids']]
    def build(old,r,iv):
        q,mu,mass=competition.moments(old,bx,f['by'],f['alpha'],f['k'])
        return native.system(q,mu,mass,f['a'],r,f['previous'],inverse=iv,x=bx,alpha=f['alpha'])
    o0,b0,s0=build(f['old'],f['r0'],f['inv0']);ref,_=s0(b0)
    op,rhs,inv=build(f['moved'],f['r'],f['inv']);wu,ua=inv(rhs)
    assert abs(float(native.objective(op,rhs,ref))-c['expected_OFF_objective'])<1e-8
    assert abs(float(native.objective(op,rhs,wu))-c['expected_unconstrained_objective'])<1e-8
    return op,rhs,inv,wu,ref,x,y,dict(head_proximal=.01,exact_previous_batch_fixture=True,unconstrained=ua)


def small_checks():
    checks=[]
    for lt in (False,True):
        cfg=dict(dimension=6,classes=3,old_classes=1,current_train_counts=[1,1],fit_n=2,old_train_counts=[8],LT=lt,event='boundary_final')
        op,rhs,inv,wu,ref,x,y,meta=prepare_event(cfg)
        ref=ref+.3*native.atom(x,y,0,0,3)
        s=snapshot('boundary_final');w,a,cert=guard.solve(op,rhs,inv,wu,ref,x,y,1,s)
        exact,_=oracle(op,rhs,x,y,ref)
        assert a['status']=='CERTIFIED' and torch.allclose(w,exact,atol=2e-6,rtol=2e-6)
        A,At,scale,mask=guard.actions(x,y,3)
        probe=torch.arange(4,dtype=x.dtype)+1
        assert abs(float((probe*A(w)).sum()-(At(probe)*w).sum()))<1e-12
        off=guard.solve(op,rhs,inv,wu,ref,x,y,1,s,enabled=False)[0];assert torch.equal(off,wu)
        for key,value in (('support_frame','stale'),('support_encoder_version','after_step'),('scope','val'),('event','minibatch')):
            invalid=dict(s);invalid[key]=value
            try:guard.solve(op,rhs,inv,wu,ref,x,y,1,invalid)
            except ValueError:pass
            else:raise AssertionError('Illegal frame/time/scope accepted')
        buf=io.BytesIO();torch.save(cert,buf);buf.seek(0);saved=torch.load(buf,weights_only=True)
        residual=guard.verify_stationarity(op,rhs,saved,s)
        p=torch.tensor([0,2,1]);ip=torch.argsort(p)
        def pop(v):return op(v[:,ip])[:,p]
        def pinv(v):
            z,aa=inv(v[:,ip]);return z[:,p],aa
        pw,pa,_=guard.solve(pop,rhs[:,p],pinv,wu[:,p],ref[:,p],x,ip[y],1,s)
        assert pa['status']=='CERTIFIED' and torch.allclose(pw,w[:,p],atol=2e-6,rtol=2e-6)
        common=torch.arange(6,dtype=x.dtype)[:,None]*.01
        gw,ga,_=guard.solve(op,rhs,inv,wu,ref+common,x,y,1,s)
        assert ga['status']=='CERTIFIED' and torch.allclose(gw,w,atol=2e-6,rtol=2e-6)
        checks.append(dict(LT=lt,status='PASS',oracle_error=float((w-exact).norm()),checkpoint_stationarity=residual,
            all_constraints=a['constraint_count'],positive_duals=a['positive_duals'],scope_frame_disabled_adjoint_permutation_gauge=True))
    zero=dict(dimension=6,classes=3,old_classes=1,current_train_counts=[1,1],fit_n=2,old_train_counts=[8],LT=False,event='epoch_init',affine_first=1)
    op,rhs,inv,wu,ref,x,y,_=prepare_event(zero);w,a,_=guard.solve(op,rhs,inv,wu,ref,x,y,1,snapshot('epoch_init'))
    assert a['status']=='CERTIFIED' and a['identity'] and torch.equal(w,wu)
    return dict(status='PASS',checks=checks,zero_drift_exact=True)


def run(c):
    start=time.monotonic();torch.set_num_threads(2)
    if c['kind']=='small':r=small_checks()
    else:
        op,rhs,inv,wu,ref,x,y,meta=legacy(c) if c['kind']=='legacy' else prepare_event(c)
        gram=x@x.T if len(x)<x.shape[1] else x.T@x
        eig=torch.linalg.eigvalsh(gram);rank=int((eig>torch.finfo(x.dtype).eps*max(x.shape)*eig.max()).sum())
        del gram,eig
        meta['observed_support_rank']=rank
        prepared=time.monotonic()-start
        w,a,cert=guard.solve(op,rhs,inv,wu,ref,x,y,c['old_classes'],snapshot(c['event']))
        if a['status']=='CERTIFIED':
            off=(x@ref).argmax(1);on=(x@w).argmax(1)
            a.update(synthetic_rescued=int(((off!=y)&(on==y)).sum()),synthetic_damaged=int(((off==y)&(on!=y)).sum()))
        r=dict(status='COMPLETED_CPU_CASE',case=c,meta=meta,result=a,preparation_seconds=prepared,
               guard_gradient_backpropagated=False,all_constraints_verified=a['status']=='CERTIFIED')
    r.update(CPU_pipeline_seconds=time.monotonic()-start,GPU_processes=0,medical_images_read=0,medical_checkpoints_read=0,
             val_reads=0,test_accessed=False,optimizer_updates=0,production_integrated=False,GPU_ready=False)
    return r
