"""Design-only joint score integration. No images, medical loaders or GPU entry."""
import math
import numpy as np
import torch
import intervention_gain as frozen
from lca_old_pair_repair import contrast_basis,native_operator
import core1_competition as core

VERSION='gain_joint_predictive_v2'
STEPS=(.125,.25,.5,1.)
NODES=65536
BATCH=4096
ALPHA=.01
EVENTS=432
SEED=1074005

def validate_design(p):
    exact=dict(version=VERSION,status='CPU_DESIGN_ONLY',primary_control='local_transport',production_integrated=False,GPU_launch_authorized=False,medical_data_access_authorized=False,checkpoint_access_authorized=False,test_accessed=False,training_feedback=False,validation_retuning=False,automatic_retry=False,optimizer_updates=0,policy_updates=0,seed=74002,direction_source_commit='d258fff32b804717ee17b81d4985f0920fbf843b',steps=list(STEPS),zero_anchor=True,direction_changed=False,guard='old_max_nonincrease_new_logits_fixed',score_integration_nodes=NODES,score_integration_batch=BATCH,numerical_alpha=ALPHA,numerical_family_events=EVENTS,numerical_seed=SEED,borrow_scenario_fractions=[0.,.5,1.],main_selection='JOINT_PRED_SELECTED',selection_epsilon=1e-12,singleton_data_identified=False,zero_empirical_scatter_identified=False,predictive_distribution='shared_scale_student_t_df_n_minus_1',clinical_confidence_calibrated=False,old_feature_draws=0)
    if any(p.get(k)!=v for k,v in exact.items()):raise ValueError('Independent design-only scope required')

@torch.no_grad()
def frozen_ray(bank,pairs,R,W0,old,readout):
    frozen.finite(bank['mu'],bank['Q'],pairs,R,W0);k,d=bank['mu'].shape;W=readout.double()
    if type(old)!=int or not 0<=old<=k or W.shape!=W0.shape or W0.shape!=(d,k) or pairs.shape!=(k,k) or R.shape!=(d,d) or not torch.allclose(R,R.T,atol=1e-10,rtol=1e-10) or pairs.diag().any() or (pairs<0).any() or not torch.allclose(pairs.sum(1),pairs.new_ones(k),atol=1e-8,rtol=1e-8):raise ValueError('Native geometry differs')
    frozen.finite(W);_,targets=core.laplacians(pairs);H=lambda Z:native_operator(bank,pairs,R,Z)
    rhs=bank['mu'].T/k+.5*(bank['mu'].T/k)@targets
    if float((H(W0)-rhs).norm()/rhs.norm().clamp_min(1e-30))>1e-8:raise ValueError('Native reference differs')
    if old<2:return torch.zeros_like(W0),dict(relative_residual=0.,exact_identity=True)
    models,_=frozen.credibility(bank,W,True);g=frozen.proposal(bank,models,W,old)
    V=W0.new_zeros(k,old-1);V[:old]=contrast_basis(old,W0);b=g@V
    if not b.any():return torch.zeros_like(W0),dict(relative_residual=0.,exact_identity=True)
    A=lambda Z:H(Z@V.T)@V;factor=torch.linalg.cholesky(bank['Q'].mean(0)+.001*R);pre=lambda Z:torch.cholesky_solve(Z,factor)
    Z,qa=core.pcg(A,b,pre,torch.zeros_like(b));ray=Z@V.T;reference=(W0@V)@V.T
    cap=float((reference*H(reference)).sum());energy=float((ray*H(ray)).sum());factor=min(1.,math.sqrt(cap/energy)) if energy else 0.;ray*=factor
    return ray,dict(qa,proposal_norm=float(g.norm()),ray_norm=float(ray.norm()),native_ray_scale=factor,native_energy_cap=cap,direction_changed=False)

def models_from_bank(bank,W):
    models,rows=frozen.credibility(bank,W,True);out=[];audit=[]
    for c,parts in enumerate(models):
        fresh=[]
        for r in parts:
            trace=float(r['family'][0].trace());tol=64*torch.finfo(torch.float64).eps*max(1.,float(r['center'].square().sum()))
            identified=r['count']>=2 and trace>tol
            fresh.append(dict(r,data_identified=identified,empirical_trace=trace,scatter_tolerance=tol,prior_available=float(r['family'][-1].trace())>tol))
        out.append(fresh)
        audit.append(dict(rows[c],data_identified_components=sum(z['data_identified'] for z in fresh),unidentified_mass=sum(z['mass'] for z in fresh if not z['data_identified']),singleton_components=sum(z['count']==1 for z in fresh),prior_does_not_identify_data=True))
    return out,audit

def guard_scores(before,change,old):
    after=before+change
    if old:
        raw=after[:,:old];rm=raw.max(1,keepdims=True);bm=before[:,:old].max(1,keepdims=True)
        after[:,:old]=np.where(rm>bm,bm-(rm-raw),raw)
    return after

def integrate(mean,cov,old,truth,df,stream):
    """One coupled distribution for OFF, ON, rescue, harm and net."""
    k=len(mean)//2
    if k<1 or type(old)!=int or type(truth)!=int or mean.shape!=(2*k,) or cov.shape!=(2*k,2*k) or not 0<=old<=k or not 0<=truth<k or df is not None and (type(df)!=int or df<1):raise ValueError('Invalid score model')
    covariance_scale=max(np.finfo(float).tiny,float(np.abs(cov).max()))
    if not np.isfinite(mean).all() or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=64*np.finfo(float).eps*covariance_scale,rtol=1e-10):raise ValueError('Invalid covariance')
    if np.any(mean[k+old:]) or np.any(cov[k+old:]):raise ValueError('New score correction is not fixed')
    e,u=np.linalg.eigh((cov+cov.T)/2);tol=64*np.finfo(float).eps*len(e)*max(np.finfo(float).tiny,float(np.max(np.abs(e))))
    if e.min() < -tol:raise ValueError('Non-PSD joint covariance')
    discarded=float(np.clip(e[e<tol],0,None).sum());active=e>=tol;root=u[:,active]*np.sqrt(e[active]);random=bool(active.any())
    # Exact duplicate covariance rows represent the same noise, including score ties.
    duplicates=[]
    for j in range(len(mean)):
        for i in range(j):
            if np.array_equal(cov[j],cov[i]):duplicates.append((j,i));break
    rng=np.random.default_rng(np.random.SeedSequence([SEED,stream[0],stream[1]]));rescued=harmed=0;total=NODES if random else 1
    for start in range(0,total,BATCH):
        size=min(BATCH,total-start)
        noise=rng.standard_normal((size,root.shape[1]))@root.T if random else np.zeros((size,len(mean)))
        for j,i in duplicates:noise[:,j]=noise[:,i]
        if df is not None and random:noise/=np.sqrt(rng.chisquare(df,size=size)/df)[:,None]
        scores=mean+noise;before=scores[:,:k];change=scores[:,k:];change[:,old:]=0.
        after=guard_scores(before,change,old);po=before.argmax(1);pn=after.argmax(1)
        rescued+=int(((po!=truth)&(pn==truth)).sum());harmed+=int(((po==truth)&(pn!=truth)).sum())
        if truth>=old and ((po==truth)&(pn!=truth)).any():raise ArithmeticError('New-class guard failed')
    return dict(rescue=rescued/total,harm=harmed/total,net=(rescued-harmed)/total,random=random,numeric_score_nodes=total,discarded_covariance_trace=discarded)

@torch.no_grad()
def evaluate(models,W,delta,old,predictive):
    frozen.finite(W,delta)
    if W.device.type!='cpu' or type(predictive)!=bool or type(old)!=int or not 0<=old<=W.shape[1] or delta.shape!=W.shape or delta[:,old:].any() or len(models)!=W.shape[1]:raise ValueError('CPU old-only scope required')
    bound=1e-8+torch.finfo(torch.float32).eps*delta[:,:old].abs().sum(1)
    if (delta[:,:old].sum(1).abs()>bound).any():raise ValueError('Common component exceeds rounding bound')
    k=W.shape[1]
    if not delta.any():return dict(scenarios=[dict(scenario=j,known_net=0.,net_lower=0.,net_upper=0.,numerical_error=0.,unknown_mass=0.,classes=[]) for j in range(3)],robust_lower=0.,robust_upper=0.,exact_identity=True,numeric_score_nodes=0)
    projection=torch.cat((W,delta),1);scenarios=[];nodes=0
    for scenario in range(3):
        classes=[];known_net=0.;unknown_lower=unknown_upper=unknown_mass=variance_proxy=0.
        for c,parts in enumerate(models):
            if not parts or abs(sum(z['mass'] for z in parts)-1)>1e-8 or any(type(z['count'])!=int or z['count']<1 or not math.isfinite(z['mass']) or z['mass']<=0 for z in parts):raise ValueError('Invalid component supports')
            cr=ch=cn=ul=uu=um=0.;component_rows=[]
            for index,r in enumerate(parts):
                mass=r['mass'];identified=r['count']>=2 and r['data_identified']
                if not identified:
                    ul-=mass*int(c<old);uu+=mass;um+=mass;component_rows.append(dict(count=r['count'],mass=mass,data_identified=False,numeric_score_nodes=0));continue
                S=r['family'][min(scenario,len(r['family'])-1)];frozen.finite(S,r['center'])
                if S.shape!=(W.shape[0],W.shape[0]) or r['center'].shape!=(W.shape[0],):raise ValueError('Component dimensions differ')
                mean=(r['center']@projection).numpy();cov=(projection.T@S@projection).numpy();df=None
                # shortcut: shared-shape predictive approximation; calibrate independently before claiming statistical coverage.
                if predictive:df=r['count']-1;cov=cov*((r['count']+1)/(r['count']-1))
                z=integrate(mean,cov,old,c,df,(c,index));nodes+=z['numeric_score_nodes'];cr+=mass*z['rescue'];ch+=mass*z['harm'];cn+=mass*z['net'];component_rows.append(dict(z,count=r['count'],mass=mass,data_identified=True,predictive_df=df))
                if z['random']:variance_proxy+=(mass/k)**2/NODES
            classes.append(dict(label=c,known_rescue=cr,known_harm=ch,known_net=cn,unknown_mass=um,unknown_lower=ul,unknown_upper=uu,components=component_rows))
            known_net+=cn/k;unknown_lower+=ul/k;unknown_upper+=uu/k;unknown_mass+=um/k
        error=math.sqrt(2*math.log(2*EVENTS/ALPHA)*variance_proxy)
        scenarios.append(dict(scenario=scenario,known_net=known_net,net_lower=max(-1.,known_net-error+unknown_lower),net_upper=min(1.,known_net+error+unknown_upper),numerical_error=error,unknown_mass=unknown_mass,classes=classes))
    return dict(scenarios=scenarios,robust_lower=min(z['net_lower'] for z in scenarios),robust_upper=max(z['net_upper'] for z in scenarios),worst_model_net_upper=min(z['net_upper'] for z in scenarios),exact_identity=False,numeric_score_nodes=nodes,predictive=predictive,model_conditional_not_clinical_confidence=True)

@torch.no_grad()
def select(models,W,ray,old,predictive):
    selected=torch.zeros_like(ray);chosen=0.;best=1e-12;rows=[]
    for scale in STEPS:
        delta=(scale*ray).float().double();r=evaluate(models,W,delta,old,predictive);rows.append(dict(scale=scale,**r))
        if r['robust_lower']>best:best=r['robust_lower'];chosen=scale;selected=delta
    return selected,dict(version=VERSION,chosen_scale=chosen,candidates=rows,selection='JOINT_PRED_SELECTED' if predictive else 'JOINT_POINT_SELECTED',direction_changed=False,clinical_confidence_calibrated=False)
