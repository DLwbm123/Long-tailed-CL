"""Synthetic CPU oracle: moments, full competition, tail integrals and new safety."""
import copy,io,json,math,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import concm_identity as method
import core1_competition as core
import prototype_coherent as native
import module_additions as modules

def reject(fn):
    try:fn()
    except (ValueError,ArithmeticError,torch.linalg.LinAlgError):return
    raise AssertionError('Invalid scope or input accepted')

def dense_native(bank,pairs,R):
    k,d=bank['mu'].shape;I=torch.eye(k,dtype=R.dtype);H=torch.kron((bank['Q'].mean(0)+.001*R).contiguous(),I)
    for c in range(k):
        for j in range(k):
            if c==j:continue
            e=R.new_zeros(k);e[c]=1;e[j]=-1
            H+=.5*pairs[c,j]/k*torch.kron(bank['Q'][c].contiguous(),torch.outer(e,e).contiguous())
    return H

def explicit_native(bank,W,pairs,R):
    k=len(bank['n']);value=W.new_zeros(())
    for c in range(k):
        value+=(torch.trace(W.T@bank['Q'][c]@W)-2*bank['mu'][c]@W[:,c]+1)/(2*k)
        for j in range(k):
            if c==j:continue
            v=W[:,c]-W[:,j];value+=.25*pairs[c,j]/k*(v@bank['Q'][c]@v-2*bank['mu'][c]@v+1)
    return value+.0005*(W*(R@W)).sum()

def bad_moments(S,e):
    u,a=e['u'],e['a'];Q=e['ratio']*S[e['c']]+torch.outer(u,u)
    bad=e['p']*Q-e['phi']*(torch.outer(u,a)+torch.outer(a,u))+e['z']*e['phi']*torch.outer(a,a)
    return bad,e['p']*u-e['phi']*a

def run(protocol):
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());method.validate_design(p)
    for key in ('GPU_launch_authorized','medical_data_access_authorized','checkpoint_access_authorized','test_accessed','validation_retuning'):
        invalid=dict(p);invalid[key]=True;reject(lambda:method.validate_design(invalid))
    d,k,old=8,5,3;ns=[3,6,9,18,27];y=torch.arange(k).repeat_interleave(torch.tensor(ns))
    x=torch.randn(sum(ns),d,dtype=torch.float64)*torch.linspace(.15,.9,d)
    x+=1.3*F.one_hot(y,d);group=torch.cat([2*c+torch.arange(n)%2 for c,n in enumerate(ns)])
    weights=x.new_tensor([1/ns[int(c)] for c in y]);bank=native.append(native.empty(d),x,y,group,weights);saved=copy.deepcopy(bank)
    revised,ma=method.borrow(bank);assert ma['covariance_change_norm']>0 and ma['max_within_trace_change']<1e-10
    assert torch.equal(revised['mu'],bank['mu']) and revised['n']==bank['n']
    assert all(torch.equal(a['center'],b['center']) and a['mass']==b['mass'] and a['radius']==b['radius'] for a,b in zip(revised['components'],bank['components']))
    method.covariance_parts(revised)
    # Translating all centers must not change the borrowed covariance increment.
    translated=native.translate(bank,x.new_tensor([.1]*d));bt,_=method.borrow(translated)
    torch.testing.assert_close(bt['Q']-translated['Q'],revised['Q']-bank['Q'],atol=1e-10,rtol=1e-8)
    maximum=0.;cases=[]
    for control in ('base','local_transport'):
        local=torch.randn(k,4,d,dtype=x.dtype) if control=='local_transport' else None
        R,_=modules.metric(bank,control,'HK',list(range(k)),local);init,_=native.head(bank)
        pairs=core.competition(bank,init);W0,qa=core.bank_head(bank,pairs,regularizer=R)
        H0=dense_native(bank,pairs,R);S,edges,bt=method.tail_geometry(bank,pairs,W0,old)
        Bads=[bad_moments(S,e) for e in edges]
        for e,(Qbad,mbad) in zip(edges,Bads):
            Q=e['ratio']*S[e['c']]+torch.outer(e['u'],e['u'])
            assert torch.linalg.eigvalsh((Qbad+Qbad.T)/2).min()>-1e-10
            assert torch.linalg.eigvalsh((Q-Qbad+Q.T-Qbad.T)/2).min()>-1e-10
        results={};hessians={};objectives={};forces={}
        for preset in method.PRESETS:
            use_mpc,use_tail,reg,eta=method.PRESETS[preset]
            op,rhs,pre,V,a=method.systems(bank,pairs,R,W0,old,preset);delta,checked=method.fit(bank,pairs,R,W0,old,preset)
            changed=revised if use_mpc else bank;A0=torch.kron(torch.eye(d,dtype=x.dtype),V).T@H0@torch.kron(torch.eye(d,dtype=x.dtype),V)
            E=torch.zeros_like(W0)
            if eta:
                target,_=method.matching_anchors(bank['mu'][:old],W0[:,:old]-W0[:,:old].mean(1,keepdim=True));E[:,:old]=(target-W0[:,:old])@V[:old]@V[:old].T
            direction=torch.zeros_like(W0)
            if eta:
                for e,(Qbad,mbad) in zip(edges,Bads):
                    b=e['weight']*(Qbad@(E[:,e['c']]-E[:,e['j']]));direction[:,e['c']]+=b;direction[:,e['j']]-=b
            def objective(flat):
                Z=flat.reshape(d,old-1);W=W0+Z@V.T;value=explicit_native(changed,W,pairs,R)
                wv=Z.flatten();value+=.5*reg*(wv@A0@wv)
                if use_tail:
                    for e,(Qbad,mbad) in zip(edges,Bads):
                        v=W[:,e['c']]-W[:,e['j']];value+=.5*e['weight']*(v@Qbad@v-2*e['target']*(mbad@v)+e['target']**2*e['p'])
                return value-eta*((W-W0)*direction).sum()
            zero=x.new_zeros(d*(old-1),requires_grad=True)
            oracle=torch.autograd.functional.hessian(objective,zero);g=torch.autograd.grad(objective(zero),zero)[0]
            direct=torch.linalg.solve(oracle,-g).reshape(d,old-1);torch.testing.assert_close(delta,direct@V.T,atol=1e-8,rtol=1e-7)
            assert not delta[:,old:].any() and (delta[:,:old].sum(1)).abs().max()<1e-9
            assert objective((delta@V).flatten())<=objective(zero)+1e-9
            if preset=='REG':assert not delta.any()
            if preset=='TAIL':assert a['tail_component_edges']>0 and delta.norm()>0
            if preset in ('DSM','MPC_DSM'):assert a['ETF_direction_force_norm']>0
            for count in (0,1):
                dz,za=method.fit(bank,pairs,R,W0,count,preset);assert not dz.any() and za['exact_identity']
            maximum=max(maximum,qa['relative_residual'],checked['relative_residual'])
            results[preset]=delta;hessians[preset]=oracle;objectives[preset]=objective((delta@V).flatten()).item();forces[preset]=a['ETF_direction_force_norm']
            meta=dict(dataset='ISIC',control=control,control_source='fixture',order=list(range(k)),seen=list(range(k)),seed=74002,task=2,source='fixture')
            payload=method.pack(delta,old,meta);stream=io.BytesIO();torch.save(payload,stream);stream.seek(0);back=torch.load(stream,weights_only=False)
            assert torch.equal(method.restore(back,old,meta,delta.shape),delta.float())
            bad=copy.deepcopy(back);bad['version']='wrong';reject(lambda:method.restore(bad,old,meta,delta.shape))
        torch.testing.assert_close(hessians['DSM'],hessians['TAIL_REG'],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(hessians['MPC_DSM'],hessians['MPC_TAIL_REG'],atol=1e-12,rtol=1e-12)
        tailh=hessians['TAIL']-hessians['REG']/1.1
        assert torch.linalg.eigvalsh(tailh).min()>-1e-10
        assert torch.linalg.eigvalsh(hessians['REG']/1.1-tailh).min()>-1e-10
        reject(lambda:method.fit(bank,pairs,R,W0+.1,old,'DSM'))
        reject(lambda:method.fit(bank,pairs,R,W0,k+1,'DSM'))
        cases.append(dict(control=control,all_eight_dense_oracles=True,tail_curvature_bounded_by_native=True,all_objective_minima=True,ETF_same_Hessian=True,ETF_forces=forces,head_delta_norms={v:float(z.norm()) for v,z in results.items()},first_task_exact_identity=True))
    # Independent one-dimensional quadrature of truncated moments.
    errors=[]
    for z in (-2.,0.,2.):
        t=np.linspace(-10.,-z,50001);pdf=np.exp(-t*t/2)/math.sqrt(2*math.pi);dt=np.diff(t);mass=np.sum((pdf[:-1]+pdf[1:])*.5*dt);first=np.sum((t[:-1]*pdf[:-1]+t[1:]*pdf[1:])*.5*dt);second=np.sum((t[:-1]**2*pdf[:-1]+t[1:]**2*pdf[1:])*.5*dt)
        pr=.5*math.erfc(z/math.sqrt(2));phi=math.exp(-z*z/2)/math.sqrt(2*math.pi)
        errors.append(max(abs(mass-pr),abs(first+phi),abs(second-pr-z*phi)))
    assert max(errors)<1e-8
    guard_checks={}
    for dtype in (torch.float32,torch.float64):
        base=torch.randn(1000,23,dtype=dtype);change=torch.randn_like(base)*.4;change[:,21:]=0;base[:3]=0;change[:1]=0
        on=method.guard(base,change,21);assert torch.equal(on[:,21:],base[:,21:]);assert (on[:,:21].max(1).values<=base[:,:21].max(1).values).all()
        before=base.argmax(1);after=on.argmax(1);assert torch.equal(before[before>=21],after[before>=21])
        for c in (21,22):
            def margin(scores):return scores[:,c]-scores[:,torch.arange(23)!=c].max(1).values
            assert (margin(on)>=margin(base)).all()
        assert torch.equal(method.guard(base,change*0,21),base)
        reject(lambda:method.guard(base,change,True));bad=change.clone();bad[:,22]=1;reject(lambda:method.guard(base,bad,21))
        guard_checks[str(dtype)]=dict(new_columns_exact=True,new_correctness_never_harmed=True,new_raw_margin_nondecreasing=True)
    base=torch.tensor([[1.,.8,.95],[.8,.7,1.],[.8,.9,.1]],dtype=torch.float64);change=torch.tensor([[-.1,.1,0.],[.4,-.4,0.],[.2,-.2,0.]],dtype=torch.float64)
    on=method.guard(base,change,2);assert base.argmax(1).tolist()==[0,2,1] and on.argmax(1).tolist()==[2,2,0]
    # New scores can gain false positives on old truth; no old-class guarantee.
    assert on[0].argmax()==2
    assert torch.equal(saved['mu'],bank['mu']) and torch.equal(saved['Q'],bank['Q'])
    corrupted=copy.deepcopy(bank);corrupted['Q'][0]-=10*torch.eye(d,dtype=x.dtype);reject(lambda:method.borrow(corrupted))
    return dict(status='PASS',GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,formal_optimizer_updates=0,production_integrated=False,synthetic_only=True,
        mean_component_identity=True,coherent_covariance_borrowing=True,translation_invariant_borrowing=True,exact_native_reduction=True,REG_only_exact_identity=True,regularization_centered_on_increment=True,full_competition_dense_oracle=True,
        wrong_scope_and_checkpoint_refused=True,truncated_moment_quadrature_max_error=max(errors),tail_Q_PSD_and_complement_PSD=True,guard_checks=guard_checks,
        guard_allows_new_rescue_and_old_internal_correction=True,old_truth_false_positive_risk_explicit=True,cases=cases,max_solver_residual=maximum,pipeline_seconds=time.monotonic()-started)
if __name__=='__main__':
    import os
    print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2))
