"""CPU-only joint full-fit guard/reference refresh directional derivative."""
import time
import torch
import core1_competition as competition
import prototype_coherent as coherent
import module_additions as modules
import native_pointwise_guard_design as native
import full_fit_guard_design as guard
from check_full_fit_guard_design import sample,label_counts,bank


def prepare(cfg):
    d=1536;k=cfg['classes'];old_count=cfg['old_classes'];n=cfg['fit_n']
    gen=torch.Generator().manual_seed(227+100*k+cfg['task'])
    counts=label_counts(n,cfg['current_train_counts'])
    x=torch.cat([sample(c,d,old_count+i,gen) for i,c in enumerate(counts)])
    y=torch.cat([torch.full((c,),old_count+i,dtype=torch.long) for i,c in enumerate(counts)])
    old_samples=[sample(c,d,i,gen) for i,c in enumerate(cfg['old_train_counts'])]
    old=bank(old_samples);current=bank([x[y==c] for c in range(old_count,k)])
    for c in current['components']:c['label']+=old_count
    def joined(b):return dict(mu=torch.cat([b['mu'],current['mu']]),Q=torch.cat([b['Q'],current['Q']]),
        n=b['n']+current['n'],components=b['components']+current['components'])
    full=joined(old)
    local=torch.nn.functional.normalize(torch.randn(k,4,d,generator=gen,dtype=x.dtype),dim=-1)
    arm='local_transport' if cfg['LT'] else 'base'
    r0=modules.metric(full,arm,cfg['dataset'],list(range(k)),local if cfg['LT'] else None)[0]
    previous=torch.linalg.solve(full['Q'].sum(0)/k+.001*r0,full['mu'].T/k)
    a=competition.competition(full,previous)
    ids=torch.randperm(n,generator=gen)[:min(64,n)];by=y[ids]
    alpha=n*torch.tensor([1/counts[int(v)-old_count] for v in by],dtype=x.dtype)/(k*len(ids))
    scale=torch.ones(d,dtype=x.dtype);scale[0]=2.8;scale[1]=.65
    moved=bank([z*scale for z in old_samples]);moved_full=joined(moved)
    r=modules.metric(moved_full,arm,cfg['dataset'],list(range(k)),local if cfg['LT'] else None)[0]
    inv0=coherent.proximal_base(old,r0,k)[0];inv=coherent.proximal_base(moved,r,k)[0]
    return dict(x=x,y=y,ids=ids,by=by,alpha=alpha,old=old,moved=moved,a=a,r0=r0,r=r,
                inv0=inv0,inv=inv,previous=previous,k=k,old_count=old_count)


def rotate(x,theta):
    # Every fit point moves, preserving its unit norm; no batch-only protection claim.
    value=x.clone();c=x.new_tensor(theta).cos();s=x.new_tensor(theta).sin()
    value[:,0]=c*x[:,0]+s*x[:,1];value[:,1]=-s*x[:,0]+c*x[:,1]
    return value


def systems(f,theta):
    x=rotate(f['x'],theta);bx=x[f['ids']]
    def build(old,r,inv):
        q,mu,mass=competition.moments(old,bx,f['by'],f['alpha'],f['k'])
        return native.system(q,mu,mass,f['a'],r,f['previous'],inverse=inv,x=bx,alpha=f['alpha'])
    o0,b0,s0=build(f['old'],f['r0'],f['inv0']);ref,ref_audit=s0(b0)
    op,rhs,solve=build(f['moved'],f['r'],f['inv']);wu,ua=solve(rhs)
    trace=[];w,audit,dual=guard.constrained(op,rhs,solve,x,f['y'],ref,wu,f['old_count'],trace,
                                         max_atoms=64,time_limit=240.)
    return dict(op=op,rhs=rhs,solve=solve,o0=o0,b0=b0,s0=s0,ref=ref,wu=wu,w=w,dual=dual,
                audit=audit,trace=trace,value=float(native.objective(op,rhs,w)),ref_audit=ref_audit)


def run(cfg):
    begun=time.monotonic();torch.set_num_threads(2);f=prepare(cfg);preparation=time.monotonic()-begun
    base=systems(f,0.);a=base['audit']
    if a['status']!='FEASIBLE':
        return dict(status='BASE_FULL_FIT_UNRESOLVED',case=cfg,base=a,gradient_verified=False,
                    CPU_pipeline_seconds=time.monotonic()-begun,GPU_processes=0,val_reads=0,test_accessed=False)
    # Native directional derivative of batch moments; all outer statistics stay fixed.
    xdot=torch.zeros_like(f['x']);xdot[:,0]=f['x'][:,1];xdot[:,1]=-f['x'][:,0]
    bx=f['x'][f['ids']];dbx=xdot[f['ids']];lap,targets=competition.laplacians(f['a'])
    dmu=torch.zeros((f['k'],1536),dtype=bx.dtype);dqs=[]
    for c in range(f['old_count'],f['k']):
        ids=f['by']==c;z=bx[ids];dz=dbx[ids];alpha=f['alpha'][ids]
        dq=dz.T@(alpha[:,None]*z)+z.T@(alpha[:,None]*dz)
        dqs.append((c,dq));dmu[c]=(alpha[:,None]*dz).sum(0)
    db=dmu.T+.5*(dmu.T@targets)
    def dh(w):
        return sum((dq@w+.5*(dq@w)@lap[c] for c,dq in dqs),torch.zeros_like(w))
    w=base['w'];ref=base['ref']
    naive=float(.5*(w*dh(w)).sum()-(db*w).sum())
    force=torch.zeros_like(w);support_term=0.
    for i,j,l in base['dual']:
        force+=l*native.atom(f['x'],f['y'],i,j,f['k'])
        delta=(w[:,int(f['y'][i])]-w[:,j])-(ref[:,int(f['y'][i])]-ref[:,j])
        support_term+=l*float(xdot[i]@delta)
    adjoint,adj_audit=base['s0'](force)
    ref_rhs_derivative=db-dh(ref)
    reference_term=float((adjoint*ref_rhs_derivative).sum())
    tangent,tan_audit=base['s0'](ref_rhs_derivative)
    tangent_cross=float((force*tangent).sum())
    assert abs(tangent_cross-reference_term)<1e-8
    frozen=naive-support_term;correct=frozen+reference_term
    results=[]
    for theta in (-1e-5,1e-5):
        v=systems(f,theta)
        results.append(dict(theta=theta,status=v['audit']['status'],value=v['value'],audit=v['audit'],trace=v['trace']))
        del v
    feasible=all(v['status']=='FEASIBLE' for v in results)
    finite=(results[1]['value']-results[0]['value'])/(2e-5) if feasible else None
    error=abs(correct-finite) if feasible else None
    passed=feasible and error<=2e-6+2e-5*abs(finite)
    return dict(status='PASS' if passed else 'PERTURBED_UNRESOLVED' if not feasible else 'GRADIENT_MISMATCH',
        case=cfg,base=a,base_active_growth=base['trace'],perturbed=results,
        all_supports_moved=True,finite_difference=finite,naive_derivative=naive,
        without_reference_refresh_derivative=frozen,correct_implicit_derivative=correct,
        support_derivative_correction=-support_term,reference_derivative_correction=reference_term,
        full_refresh_error=error,missing_reference_derivative_error=abs(frozen-finite) if feasible else None,
        naive_error=abs(naive-finite) if feasible else None,adjoint=adj_audit,reference_tangent=tan_audit,
        tangent_adjoint_error=abs(tangent_cross-reference_term),gradient_verified=passed,
        a_R_previous_old_fixed=True,medical_or_val_gain_claim=False,
        preparation_seconds=preparation,CPU_pipeline_seconds=time.monotonic()-begun,
        GPU_processes=0,medical_images_read=0,medical_checkpoints_read=0,val_reads=0,test_accessed=False,optimizer_updates=0)
