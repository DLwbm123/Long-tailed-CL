"""Independent synthetic CPU oracles. No medical images/checkpoints/optimizer."""
import copy,io,json,math,time
from pathlib import Path
import torch
import torch.nn.functional as F
import intervention_gain as method
import prototype_coherent as native
import core1_competition as core
import module_additions as modules
from lca_old_pair_repair import native_operator,contrast_basis

def reject(fn):
    try:fn()
    except (ValueError,ArithmeticError,torch.linalg.LinAlgError):return
    raise AssertionError('Invalid input admitted')

def affine_probability(u,l,W,D,a,b):
    lo=-math.inf;hi=math.inf;after=W+D
    for head,win in ((W,a),(after,b)):
        for j in range(W.shape[1]):
            if j==win:continue
            v=head[:,win]-head[:,j];m=float(u@v);s=float(l@v)
            if abs(s)<1e-15:
                if m<0 or m==0 and j<win:return 0.
            elif s>0:lo=max(lo,-m/s)
            else:hi=min(hi,-m/s)
    if lo>=hi:return 0.
    cdf=lambda t:.5*math.erfc(-t/math.sqrt(2))
    return cdf(hi)-cdf(lo)

def exact_gain(models,W,D,old):
    k=W.shape[1];r=h=0.
    for c,components in enumerate(models):
        for z in components:
            S=z['family'][0];e,U=torch.linalg.eigh(S);l=U[:,-1]*e[-1].clamp_min(0).sqrt();u=z['center'];mass=z['mass']/k
            r+=mass*sum(affine_probability(u,l,W,D,j,c) for j in range(old) if j!=c)
            if c<old:h+=mass*sum(affine_probability(u,l,W,D,c,j) for j in range(k) if j!=c)
    return r,h

def model(center,mass=1.,sigma=1e-5,count=5000):
    u=torch.tensor(center,dtype=torch.float64);l=u.new_tensor([sigma,0.]);S=torch.outer(l,l)
    return dict(center=u,mass=mass,count=count,family=[S],envelope=S,unidentified=False)

def run(protocol):
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());method.validate_design(p)
    for key in ('GPU_launch_authorized','production_integrated','medical_data_access_authorized','checkpoint_access_authorized','test_accessed','validation_retuning'):
        q=dict(p);q[key]=True;reject(lambda:method.validate_design(q))
    W=torch.tensor([[-.1,.1,0.],[0.,0.,-1.]],dtype=torch.float64);D=torch.tensor([[0.,0.,0.],[.002,-.002,0.]],dtype=torch.float64)
    good=[[model([.01,1.])],[model([1.,1.])],[model([0.,-1.])]]
    selected,a=method.select_ray(good,W,D,2);assert selected.any() and a['chosen_scale']>0
    u=method.utility(good,W,selected,2,1.);er,eh=exact_gain(good,W,selected,2)
    assert u['rescue_lower']<=er+1e-10<=u['rescue_upper']+1e-10 and u['harm_lower']<=eh+1e-10<=u['harm_upper']+1e-10 and u['net_lower']>0
    # Same predicted rescue but an opposing old-class harm makes macro utility negative.
    bad=[[model([.01,1.],.4),model([-1.,1.],.6)],[model([.01,1.])],[model([0.,-1.])]]
    safe,b=method.select_ray(bad,W,D,2);unsafe,c=method.select_ray(bad,W,D,2,True);assert not safe.any() and unsafe.any()
    br,bh=exact_gain(bad,W,unsafe,2);assert br>0 and br-bh<0
    singleton=copy.deepcopy(good);singleton[0][0].update(count=1,unidentified=True)
    skipped,sk=method.select_ray(singleton,W,D,2);assert not skipped.any()
    # Exact joint Gaussian transitions via an independent scalar interval oracle.
    bound_cases=0;ambiguity_cases=0;maximum_slack=0.
    for width in (.005,.05,.2):
        stats=[[model([.01,1.],sigma=width,count=50)],[model([.15,1.],sigma=width,count=50)],[model([0.,-1.],sigma=width,count=50)]]
        for scale in method.SCALES:
            candidate=D*scale;bounds=method.utility(stats,W,candidate,2,1.);rr,hh=exact_gain(stats,W,candidate,2)
            assert bounds['rescue_lower']<=rr+1e-9<=bounds['rescue_upper']+1e-9
            assert bounds['harm_lower']<=hh+1e-9<=bounds['harm_upper']+1e-9
            factor=2*math.sqrt(2/49)
            for shift in (-2.,0.,2.):
                for variance_scale in (1-factor,1+factor):
                    stress=copy.deepcopy(stats)
                    for rows in stress:
                        for z in rows:
                            z['center'][0]+=shift*width/math.sqrt(50);z['family'][0]*=variance_scale
                    rr,hh=exact_gain(stress,W,candidate,2)
                    assert bounds['rescue_lower']<=rr+1e-9<=bounds['rescue_upper']+1e-9
                    assert bounds['harm_lower']<=hh+1e-9<=bounds['harm_upper']+1e-9
                    ambiguity_cases+=1
            maximum_slack=max(maximum_slack,bounds['net_upper']-bounds['net_lower']);bound_cases+=1
    # Probability extrema include interior variance and interval-mean maxima.
    for m in (-2.,-.2,.2,2.):
        for var in (.001,.2,2.):
            for eps in (.01,.5,3.):
                value=method.interval_mass(m,var,eps)
                phi=lambda z:.5*math.erfc(-z/math.sqrt(2))
                oracle=phi((eps-m)/math.sqrt(var))-phi(-m/math.sqrt(var));assert abs(value-oracle)<1e-14
    # Guarded argmax is OFF on original-new winners, RAW on original-old winners.
    base=torch.randn(1000,23,dtype=torch.float64);V=contrast_basis(21,base);change=torch.randn(1000,20,dtype=base.dtype)@V.T
    delta=torch.cat((change,torch.zeros(1000,2,dtype=base.dtype)),1);on=method.guard(base,delta,21);off=base.argmax(1);raw=(base+delta).argmax(1)
    expected=torch.where(off>=21,off,raw);assert torch.equal(on.argmax(1),expected)
    assert torch.equal(on[:,21:],base[:,21:])
    mean_old=base[:,:21].mean(1);mask=base[:,21:].max(1).values<=mean_old;assert (on.argmax(1)[mask]<21).all()
    for dtype in (torch.float32,torch.float64):
        bb=base.to(dtype);dd=delta.to(dtype);go=method.guard(bb,dd,21)
        for j in (21,22):
            def margin(z):return z[:,j]-z[:,torch.arange(23)!=j].max(1).values
            assert (margin(go)>=margin(bb)).all()
    reject(lambda:method.utility(good,W,D,True,1.));invalid=D.clone();invalid[:,2]=1;reject(lambda:method.utility(good,W,invalid,2,1.))
    reject(lambda:method.utility(good[:2],W,D,2,1.))
    reject(lambda:method.cone_bounds(W.new_zeros(2),-torch.eye(2,dtype=W.dtype),[W.new_tensor([1.,0.])],[False],torch.eye(2,dtype=W.dtype),50))
    # Independent dense objective for the full native-curvature direction equation.
    d,k,old=6,4,3;y=torch.arange(k).repeat_interleave(16);features=torch.randn(len(y),d,dtype=torch.float64)*.5+F.one_hot(y,d);groups=2*y+torch.arange(len(y))%2
    weights=features.new_full((len(y),),1/16);bank=native.append(native.empty(d),features,y,groups,weights);original=copy.deepcopy(bank);reference,_=native.head(bank);pairs=core.competition(bank,reference)
    weighted=copy.deepcopy(bank);rows=[r for r in weighted['components'] if r['label']==0];rows[0]['count']-=1;rows[1]['count']+=1
    reject(lambda:method.credibility(weighted,reference))
    cases=[];maximum=0.
    for arm in ('base','local_transport'):
        R,_=modules.metric(bank,arm,'HK',list(range(k)),None if arm=='base' else torch.randn(k,4,d,dtype=features.dtype));W0,qa=core.bank_head(bank,pairs,regularizer=R);mods,cred=method.credibility(bank,W0);g=method.proposal(bank,mods,W0,old)
        V=W0.new_zeros(k,old-1);V[:old]=contrast_basis(old,W0);H=lambda Z:native_operator(bank,pairs,R,Z@V.T)@V
        rhs=g@V;factor=torch.linalg.cholesky(bank['Q'].mean(0)+.001*R);pre=lambda Z:torch.cholesky_solve(Z,factor)
        if rhs.any():
            Z,sa=core.pcg(H,rhs,pre,torch.zeros_like(rhs));maximum=max(maximum,sa['relative_residual'])
            def objective(flat):
                t=flat.reshape(d,old-1);return .5*(t*H(t)).sum()-(t*rhs).sum()
            zero=rhs.new_zeros(rhs.numel(),requires_grad=True);A=torch.autograd.functional.hessian(objective,zero);b=-torch.autograd.grad(objective(zero),zero)[0];dense=torch.linalg.solve(A,b).reshape_as(rhs)
            torch.testing.assert_close(Z,dense,atol=1e-8,rtol=1e-7);assert torch.linalg.eigvalsh(A).min()>0
        outcomes={}
        for head in method.HEADS:
            change,audit=method.fit(bank,pairs,R,W0,old,head,W0.float());assert not change[:,old:].any() and change[:,:old].sum(1).abs().max()<1e-8
            maximum=max(maximum,audit['relative_residual']);outcomes[head]=dict(nonzero=bool(change.any()),chosen_scale=audit.get('chosen_scale'),reason=audit.get('reason'))
            meta=dict(dataset='HK',control=arm,control_source='fixture',order=list(range(k)),seen=list(range(k)),seed=74002,task=2,source='fixture')
            state=method.pack(change,old,meta,head);stream=io.BytesIO();torch.save(state,stream);stream.seek(0);back=torch.load(stream,weights_only=False)
            assert torch.equal(method.restore(back,old,meta,change.shape,head),change.float())
            wrong=copy.deepcopy(back);wrong['head']='invalid';reject(lambda:method.restore(wrong,old,meta,change.shape,head))
            wrong=copy.deepcopy(meta);wrong['control_source']='wrong';reject(lambda:method.restore(back,old,wrong,change.shape,head))
            first,_=method.fit(bank,pairs,R,W0,0,head);assert not first.any()
        reject(lambda:method.fit(bank,pairs,R,W0+.1,old,'GAIN'))
        cases.append(dict(control=arm,native_dense_oracle=True,outcomes=outcomes,support_not_augmented=all(z['actual_support']==bank['n'][z['label']] and z['prior_does_not_increase_support'] for z in cred)))
    assert torch.equal(original['Q'],bank['Q']) and torch.equal(original['mu'],bank['mu'])
    # A nonzero neutral control matches the same native energy before the guard.
    neutral,E=method.neutral_match(D,lambda z:z,2);assert neutral.any() and abs(float(neutral.square().sum())-E)<1e-14 and not neutral[:,2:].any()
    large=torch.randn(1536,20,dtype=torch.float64)@contrast_basis(21,W).T;large=torch.cat((large,torch.zeros(1536,2,dtype=torch.float64)),1);matched,E=method.neutral_match(large,lambda z:.1*z,21);assert matched.shape==(1536,23) and abs(float((matched*(.1*matched)).sum())-E)<1e-8
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,medical_val_reads=0,formal_optimizer_updates=0,production_integrated=False,all_design_permissions_refused=True,joint_transition_scalar_Gaussian_oracle_cases=bound_cases,ambiguity_set_scalar_oracle_cases=ambiguity_cases,weighted_support_and_negative_variance_refused=True,largest_model_utility_interval_width=maximum_slack,positive_synthetic_ray=dict(chosen_scale=a['chosen_scale'],net_lower=u['net_lower'],exact_rescue=er,exact_harm=eh,independent_of_native_solver_fixture=True),rescue_only_negative_net_counterexample=dict(exact_rescue=br,exact_harm=bh,exact_net=br-bh,safe_returns_identity=True),unidentified_support_abstains=True,guard_argmax_scope_exact=True,old_mean_floor_verified=True,float32_float64_new_margin_safe=True,full_native_dense_cases=cases,maximum_solver_residual=maximum,bank_R_pairs_unchanged=True,checkpoint_roundtrip_and_wrong_head_source_refused=True,neutral_nonzero_energy_matched=True,shape_1536_by_23_only_not_full_high_dimensional_solve=True,model_confidence_calibrated=False,pipeline_seconds=time.monotonic()-start)

if __name__=='__main__':
    import os
    print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
