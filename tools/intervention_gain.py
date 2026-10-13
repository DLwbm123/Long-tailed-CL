"""Design-only guarded macro-gain bounds from class moments. No loaders or launcher."""
import math
import torch
from concm_identity import covariance_parts,psd,finite,guard
from lca_old_pair_repair import contrast_basis,native_operator,checkpoint
import core1_competition as core

VERSION='guarded_class_gain_v1'
HEADS=('GAIN','GAIN_NO_BORROW','GAIN_RESCUE_ONLY','NEUTRAL_MATCHED')
SCALES=(0.,.125,.25,.5,1.)

def validate_design(p):
    exact=dict(version=VERSION,status='CPU_DESIGN_ONLY',primary_control='local_transport',production_integrated=False,GPU_launch_authorized=False,medical_data_access_authorized=False,checkpoint_access_authorized=False,test_accessed=False,training_feedback=False,validation_retuning=False,automatic_retry=False,optimizer_updates=0,seed=74002,future_gpus=[0,1],heads=list(HEADS),future_source_stage_checkpoints=18,future_sidecars=72,support_pseudocount=16.,borrow_max_weight=.4,projected_shape_temperature=.1,ambiguity_multiplier=2.,proposal_smoothing=.1,native_relative_energy_radius=1.,step_scales=list(SCALES),gain_acceptance_epsilon=1e-12,control_boundary_weighting='uniform_zero_action_static_pc',guard='old_max_nonincrease_new_logits_fixed',certificate='model_and_envelope_only_not_clinical_confidence',ETF_enabled=False)
    if any(p.get(k)!=v for k,v in exact.items()):raise ValueError('Independent design-only scope required')

def cdf(m,var,strict=False):
    if not math.isfinite(m) or not math.isfinite(var) or var < -1e-10:raise ValueError('Invalid projected moment')
    if var<=1e-24:return float(m>0 if strict else m>=0)
    return .5*math.erfc(-m/math.sqrt(2*var))

def interval_mass(m,var,upper):
    if upper<0:return 0.
    if var<=1e-24:return float(0<=m<=upper)
    return max(0.,cdf(m,var)-cdf(m-upper,var))

def projected_variance(cov,D):
    values=(D*(cov@D)).sum(0)
    if not torch.isfinite(values).all() or (values < -1e-10).any():raise ValueError('Invalid projected covariance')
    return values.clamp_min(0)

def cone_bounds(mean,cov,columns,strict,envelope,n):
    """Bonferroni lower / marginal upper; simultaneous rectangular ambiguity."""
    if not columns:return 1.,1.
    D=torch.stack(columns,1);m=(mean@D).tolist();v=projected_variance(cov,D).tolist();e=projected_variance(envelope,D).tolist()
    factor=2*math.sqrt(2/max(n-1,1));lows=[];highs=[]
    for mu,var,err,hard in zip(m,v,e,strict):
        dm=2*math.sqrt(err/n);vl=var*max(0.,1-factor);vh=var*(1+factor)
        values=[cdf(a,b,hard) for a in (mu-dm,mu+dm) for b in (vl,vh)]
        lows.append(min(values));highs.append(max(values))
    return max(0.,1-sum(1-q for q in lows)),min(highs)

def crossing_upper(mean,cov,v,change,envelope,n):
    """A switch needs 0<=old gap<=epsilon or a negative increment beyond epsilon."""
    D=torch.stack((v,change),1);s,q=projected_variance(cov,D).tolist();ev,ec=projected_variance(envelope,D).tolist();m=float(mean@v);t=float(mean@change)
    em=2*math.sqrt(ev/n);et=2*math.sqrt(ec/n)
    if not change.any():return 0.
    factor=2*math.sqrt(2/max(n-1,1));sl=max(0.,s)*max(0.,1-factor);sh=max(0.,s)*(1+factor);ql=max(0.,q)*max(0.,1-factor);qh=max(0.,q)*(1+factor)
    if qh<=1e-24 and t-et>=0:return 0.
    scale=abs(t)+et+math.sqrt(qh)
    candidates=[0.,.25*scale,.5*scale,scale,2*scale,4*scale,8*scale]
    bound=1.
    for eps in candidates:
        # Maximum interval probability over a mean interval may occur at eps/2.
        means=[m-em,m+em,min(max(eps/2,m-em),m+em)]
        # For fixed mean, interval mass can have an interior variance maximum.
        variances=[sl,sh]
        for mu in means:
            if mu*(mu-eps)>0 and abs(mu)!=abs(mu-eps) and eps:
                a,b=abs(mu),abs(mu-eps);critical=(a*a-b*b)/(2*math.log(a/b)) if a and b else 0.
                if sl<critical<sh:variances.append(critical)
        mass=max(interval_mass(mu,var,eps) for mu in means for var in variances)
        # t<=-eps; closed event is conservative for continuous crossings.
        tail=max(cdf(-eps-mu,var) for mu in (t-et,t+et) for var in (ql,qh))
        bound=min(bound,mass+tail)
    return min(1.,max(0.,bound))

def score_maps(k,scale,like):
    C=contrast_basis(k,like);z=like.new_zeros(k-1)
    before=[torch.cat((C[i],z)) for i in range(k)]
    after=[torch.cat((C[i],scale*C[i])) for i in range(k)]
    return before,after

def transition(mean,cov,envelope,n,before,after,a,b):
    if a==b:raise ValueError('Only off-diagonal transitions')
    k=len(before);D=[before[a]-before[j] for j in range(k) if j!=a]+[after[b]-after[j] for j in range(k) if j!=b]
    hard=[j<a for j in range(k) if j!=a]+[j<b for j in range(k) if j!=b]
    lo,hi=cone_bounds(mean,cov,D,hard,envelope,n)
    v=before[a]-before[b];change=after[a]-after[b]-v
    hi=min(hi,crossing_upper(mean,cov,v,change,envelope,n))
    if lo>hi+1e-10:raise ArithmeticError('Transition bounds inconsistent')
    return min(lo,hi),hi

def credibility(bank,W,borrow=True):
    parts,S=covariance_parts(bank);mu=bank['mu'];k,d=mu.shape;C=contrast_basis(k,W)
    if any(abs(float(r['mass'])-r['count']/bank['n'][c])>1e-8 for c in range(k) for r in parts[c]):raise ValueError('Uniform component support required')
    B,sv,_=torch.linalg.svd(W@C,full_matrices=False);B=B[:,sv>torch.finfo(W.dtype).eps*max(W.shape)*sv.max()]
    totals=torch.stack([psd(bank['Q'][c]-torch.outer(mu[c],mu[c])) for c in range(k)])
    projected=B.T@totals@B;tr=projected.diagonal(dim1=1,dim2=2).sum(1);shapes=projected/tr.clamp_min(1e-30)[:,None,None]
    trace=totals.diagonal(dim1=1,dim2=2).sum(1);positive=trace[trace>1e-12];fallback=float(positive.median()) if len(positive) else 0.
    models=[];rows=[]
    for c in range(k):
        donors=[j for j in range(k) if j!=c and bank['n'][j]>=2 and float(tr[j])>1e-12]
        weights=W.new_empty(0);prior=torch.zeros_like(S[c])
        if donors and borrow:
            distance=(shapes[donors]-shapes[c]).square().sum((1,2));reliability=W.new_tensor([bank['n'][j]/(bank['n'][j]+16.) for j in donors])
            weights=reliability*torch.exp(-distance/.1);weights=weights/weights.sum()
            prior=sum(w*totals[j]/trace[j] for w,j in zip(weights,donors))
        eta=.4*16/(bank['n'][c]+16.) if len(weights) else 0.
        class_models=[]
        for r in parts[c]:
            empirical=S[c]*(float(r['radius'])/float(S[c].trace())) if float(S[c].trace())>1e-12 else torch.zeros_like(S[c])
            prior_cov=prior*(float(r['radius']) if float(r['radius'])>1e-12 else fallback)
            family=[empirical] if not eta else [empirical,(1-eta/2)*empirical+eta/2*prior_cov,(1-eta)*empirical+eta*prior_cov]
            # Sum dominates each PSD family member; never treat a borrowed prior as extra observations.
            envelope=sum(family)
            unidentified=r['count']<2 and not float(envelope.trace())
            class_models.append(dict(center=r['center'],mass=float(r['mass']),count=r['count'],family=family,envelope=envelope,unidentified=unidentified))
        models.append(class_models)
        rows.append(dict(label=c,actual_support=bank['n'][c],model_prior_weight=eta,donor_effective_count=None if not len(weights) else float(1/weights.square().sum()),donor_labels=donors,donor_weights=weights.tolist(),prior_does_not_increase_support=True,component_counts=[r['count'] for r in parts[c]],unidentified_components=sum(z['unidentified'] for z in class_models)))
    return models,rows

def utility(models,W,delta,old,scale):
    finite(W,delta)
    if type(old)!=int or not 0<=old<=W.shape[1] or not math.isfinite(scale) or scale<0 or delta.shape!=W.shape or delta[:,old:].any() or (delta[:,:old].sum(1)).abs().max()>1e-8:raise ValueError('Old contrast only')
    k=W.shape[1]
    if len(models)!=k:raise ValueError('Incomplete class distributions')
    if not delta.any() or not scale:return dict(rescue_lower=0.,rescue_upper=0.,harm_lower=0.,harm_upper=0.,net_lower=0.,net_upper=0.,classes=[dict(label=c,rescue_lower=0.,rescue_upper=0.,harm_lower=0.,harm_upper=0.) for c in range(k)])
    before,after=score_maps(k,scale,W);C=contrast_basis(k,W);projection=torch.cat((W@C,delta@C),1);classes=[]
    for c,components in enumerate(models):
        if not components or abs(sum(r['mass'] for r in components)-1)>1e-8 or any(type(r['count'])!=int or r['count']<1 or r['mass']<=0 for r in components):raise ValueError('Invalid class distribution')
        rl=ru=hl=hu=0.
        for r in components:
            if r['unidentified']:
                a,b,e,f=0.,1.,0.,float(c<old)
            else:
                mean=r['center']@projection;envelope=projection.T@r['envelope']@projection;bounds=[]
                for S in r['family']:
                    cov=projection.T@S@projection
                    rescues=[transition(mean,cov,envelope,r['count'],before,after,j,c) for j in range(old) if j!=c]
                    harms=[transition(mean,cov,envelope,r['count'],before,after,c,j) for j in range(k) if j!=c] if c<old else []
                    bounds.append((min(1.,sum(z[0] for z in rescues)),min(1.,sum(z[1] for z in rescues)),min(1.,sum(z[0] for z in harms)),min(1.,sum(z[1] for z in harms))))
                a=min(z[0] for z in bounds);b=max(z[1] for z in bounds);e=min(z[2] for z in bounds);f=max(z[3] for z in bounds)
            rl+=r['mass']*a;ru+=r['mass']*b;hl+=r['mass']*e;hu+=r['mass']*f
        classes.append(dict(label=c,rescue_lower=rl,rescue_upper=ru,harm_lower=hl,harm_upper=hu))
    res={name:sum(row[name] for row in classes)/k for name in ('rescue_lower','rescue_upper','harm_lower','harm_upper')}
    return dict(res,net_lower=res['rescue_lower']-res['harm_upper'],net_upper=res['rescue_upper']-res['harm_lower'],classes=classes)

def proposal(bank,models,W,old):
    """Smoothed winner-facet direction proposal; acceptance uses rescue-minus-harm bounds."""
    k=W.shape[1];C=contrast_basis(k,W);scale=math.sqrt(max(0.,float(torch.trace(C.T@W.T@bank['Q'].mean(0)@W@C)))/max(1,k-1));noise=(.1*scale)**2;g=torch.zeros_like(W)
    for c in range(old):
        for r in models[c]:
            u=r['center'];S=sum(r['family'])/len(r['family']);score=u@W;G=W.T@S@W+noise*torch.eye(k,dtype=W.dtype,device=W.device)
            for j in range(old):
                if j==c:continue
                v=W[:,c]-W[:,j];m=float(score[c]-score[j]);var=float(G[c,c]+G[j,j]-2*G[c,j])
                if var<=1e-24:continue
                sd=math.sqrt(var);density=math.exp(-m*m/(2*var))/math.sqrt(2*math.pi*var)
                mean_x=u-(S@v)*m/var
                gap=G[:,c]-G[:,j];conditional=G-torch.outer(gap,gap)/var;conditional_mean=score-gap*m/var
                probabilities=[]
                for q in range(k):
                    if q in (c,j):continue
                    pm=float(conditional_mean[c]-conditional_mean[q]);pv=float(conditional[c,c]+conditional[q,q]-2*conditional[c,q]);probabilities.append(cdf(pm,max(0.,pv)))
                # The midpoint of valid winner-probability bounds is a proposal heuristic, not a true gradient.
                lower=max(0.,1-sum(1-p for p in probabilities));upper=min(probabilities) if probabilities else 1.
                force=(r['mass']/k)*density*(lower+upper)/2*mean_x;g[:,c]+=force;g[:,j]-=force
    return g

def select_ray(models,W,delta,old,rescue_only=False):
    rows=[];chosen=0.;best=1e-12;accepted=torch.zeros_like(delta)
    for scale in SCALES:
        candidate=(scale*delta).float().double();u=utility(models,W,candidate,old,1.);objective=u['rescue_lower'] if rescue_only else u['net_lower'];rows.append(dict(scale=scale,selection_score=objective,**u))
        if objective>best:best=objective;chosen=scale;accepted=candidate
    return accepted,dict(chosen_scale=chosen,selection='rescue_only' if rescue_only else 'macro_rescue_lower_minus_harm_upper',candidates=rows,model_only=True,clinical_or_distribution_free_guarantee=False)

def neutral_match(accepted,H,old):
    d,k=accepted.shape;V=accepted.new_zeros(k,max(0,old-1));V[:old]=contrast_basis(old,accepted)
    energy=float((accepted*H(accepted)).sum());rng=torch.Generator(device=accepted.device).manual_seed(1074005)
    delta=torch.randn((d,max(0,old-1)),generator=rng,dtype=accepted.dtype,device=accepted.device)@V.T
    de=float((delta*H(delta)).sum());delta*=math.sqrt(energy/de) if energy else 0.
    return delta,energy

@torch.no_grad()
def fit(bank,pairs,R,W0,old,head,readout=None):
    if head not in HEADS or type(old)!=int or not 0<=old<=W0.shape[1]:raise ValueError('Invalid preset/old prefix')
    finite(bank['mu'],bank['Q'],pairs,R,W0);k,d=bank['mu'].shape
    if W0.shape!=(d,k) or pairs.shape!=(k,k) or R.shape!=(d,d) or not torch.allclose(R,R.T,atol=1e-10,rtol=1e-10) or pairs.diag().any() or (pairs<0).any() or not torch.allclose(pairs.sum(1),pairs.new_ones(k),atol=1e-8,rtol=1e-8):raise ValueError('Invalid native geometry')
    Wref=W0 if readout is None else readout.double()
    if Wref.shape!=W0.shape:raise ValueError('Saved readout differs')
    finite(Wref,W0)
    _,targets=core.laplacians(pairs);rhs=bank['mu'].T/k+.5*(bank['mu'].T/k)@targets;H=lambda Z:native_operator(bank,pairs,R,Z)
    error=float((H(W0)-rhs).norm()/rhs.norm().clamp_min(1e-30))
    if error>1e-8:raise ValueError('Native reference is not optimum')
    if old<2:return torch.zeros_like(W0),dict(version=VERSION,head=head,exact_identity=True,reason='fewer_than_two_old_classes',relative_residual=0.)
    if head=='NEUTRAL_MATCHED':
        accepted,a=fit(bank,pairs,R,W0,old,'GAIN',readout);delta,energy=neutral_match(accepted,H,old)
        return delta,dict(version=VERSION,head=head,matched_native_energy=energy,reference_chosen_scale=a.get('chosen_scale'),risk_gate_used=False,exact_identity=not bool(delta.any()),relative_residual=0.)
    models,cred=credibility(bank,Wref,head!='GAIN_NO_BORROW');g=proposal(bank,models,Wref,old);V=W0.new_zeros(k,old-1);V[:old]=contrast_basis(old,W0);b=g@V
    if not b.any():return torch.zeros_like(W0),dict(version=VERSION,head=head,exact_identity=True,reason='zero_proposal',relative_residual=0.,credibility=cred)
    A=lambda Z:H(Z@V.T)@V;factor=torch.linalg.cholesky(bank['Q'].mean(0)+.001*R);pre=lambda Z:torch.cholesky_solve(Z,factor)
    Z,qa=core.pcg(A,b,pre,torch.zeros_like(b));delta=Z@V.T;reference=(W0@V)@V.T
    cap=float((reference*H(reference)).sum());energy=float((delta*H(delta)).sum());ray_scale=min(1.,math.sqrt(cap/energy)) if energy else 0.;delta*=ray_scale
    accepted,a=select_ray(models,Wref,delta,old,head=='GAIN_RESCUE_ONLY')
    return accepted,dict(a,version=VERSION,head=head,credibility=cred,relative_residual=qa['relative_residual'],iterations=qa['iterations'],proposal_norm=float(g.norm()),preselection_delta_norm=float(delta.norm()),native_energy_cap=cap,native_ray_scale=ray_scale,accepted_native_energy=float((accepted*H(accepted)).sum()),exact_identity=not bool(accepted.any()),unchanged_native_Q_R_pairs=True,new_columns_fixed=True,common_old_fixed=True)

def pack(delta,old,meta,head='GAIN'):
    if head not in HEADS:raise ValueError('Wrong head')
    s=checkpoint(delta,old,meta);s.update(version=VERSION,head=head);return s

def restore(s,old,meta,shape,head='GAIN'):
    if s.get('version')!=VERSION or s.get('head')!=head or s.get('metadata')!=meta or s.get('old_count')!=old or s['delta'].shape!=shape:raise ValueError('Provenance differs')
    return pack(s['delta'],old,meta,head)['delta'].clone()
