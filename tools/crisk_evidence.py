"""Projected training-only evidence and conditional model integration."""
import numpy as np
import gain_conditional_risk as risk
from gain_joint import guard_scores
from gain_risk import event_bounds
from gain_calibration_plan import paired_score_moments

CANDIDATES=('C0125','C025','C05','C1')
MODEL_NODES=65536
MODEL_BATCH=4096


def training_evidence(x,before,labels,head,deltas,bank,seen,old,task,task_sizes):
    ray=deltas['C1'];k=len(seen)
    for name,step in zip(CANDIDATES,risk.STEPS):
        if not np.array_equal(deltas[name],(step*ray).astype(np.float32)):
            raise ValueError('Binary-scale fixed sidecar differs; no approximation')
    P=np.concatenate((head,ray),1).astype(float);z=x.astype(float)@P
    off=x@head;po=off.argmax(1);y=np.array([seen.index(int(v)) for v in labels])
    shift=np.stack([(x-before)[labels==v].astype(float).mean(0) for v in seen[old:]]).mean(0) if old else None
    distributions=[];counts={};edges={};drift=[]
    for i,label in enumerate(seen):
        m=labels==label;zs=z[m];mean=zs.mean(0);cov=np.cov(zs,rowvar=False,bias=True) if m.sum()>1 else np.zeros((2*k,2*k))
        pair=paired_score_moments(before[m],x[m],P,shift) if i<old else None
        proxy=bank['mu'][i].numpy().astype(float)@P
        distributions.append(dict(label=label,n=int(m.sum()),mean=mean,covariance=cov,
                                  paired_second=None if pair is None else pair['second']))
        introduced=next(t for t in range(1,task+1) if label in seen[:sum(task_sizes[:t])])
        drift.append(dict(label=label,n=int(m.sum()),transport_age=task-introduced,
                          paired_second_trace=None if pair is None else float(np.trace(pair['second'])),
                          cumulative_bank_score_residual_norm=float(np.linalg.norm(mean-proxy)),
                          paired_mean_norm=None if pair is None else float(np.linalg.norm(pair['mean'])),
                          current_model_not_bank_rewrite=True))
    for name,step in zip(CANDIDATES,risk.STEPS):
        on=guard_scores(off,x@deltas[name],old);pn=on.argmax(1);counts[name]=[];edges[name]=[]
        for i,label in enumerate(seen):
            m=y==i
            counts[name].append(dict(label=label,n=int(m.sum()),correct_off=int((m&(po==i)).sum()),
                                     rescued=int((m&(po!=i)&(pn==i)).sum()),harmed=int((m&(po==i)&(pn!=i)).sum())))
            edges[name].append(risk.boundary_summary(off,on,y,int(i),old))
    return dict(task=task,old=old,seen=seen,distributions=distributions,counts=counts,edges=edges,drift=drift)


def winner_bounds(before,radius,truth):
    lo,hi=before-radius,before+radius;others=[i for i in range(before.shape[1]) if i!=truth]
    strict=np.asarray(others)<truth
    sure=np.all(np.where(strict,lo[:,truth,None]>hi[:,others],lo[:,truth,None]>=hi[:,others]),1)
    possible=np.all(np.where(strict,hi[:,truth,None]>lo[:,others],hi[:,truth,None]>=lo[:,others]),1)
    return sure,possible


def integrated(mean,covariance,n,old,truth,stream,radius=None):
    """One shared score draw for all binary fixed steps; guarded coupled events."""
    k=len(mean)//2;G=(covariance+covariance.T)/2
    eigen,U=np.linalg.eigh(G);tol=64*np.finfo(float).eps*len(G)*max(np.finfo(float).tiny,float(np.abs(eigen).max()))
    if eigen.min() < -tol or n<2:raise ValueError('Unidentified/non-PSD current training distribution')
    active=eigen>tol;root=U[:,active]*np.sqrt(eigen[active]*(n+1)/(n-1))
    duplicates=[(j,next(i for i in range(j) if np.array_equal(G[j],G[i]))) for j in range(1,2*k) if any(np.array_equal(G[j],G[i]) for i in range(j))]
    rng=np.random.default_rng(np.random.SeedSequence([risk.SEED,*stream]));total=MODEL_NODES if active.any() else 1
    sums={name:dict(wrong=0,rescued=0,harmed=0,qlo=0.,qhi=0.,rl=0.,ru=0.,hl=0.,hu=0.) for name in CANDIDATES}
    for start in range(0,total,MODEL_BATCH):
        size=min(MODEL_BATCH,total-start)
        noise=rng.standard_normal((size,root.shape[1]))@root.T
        for j,i in duplicates:noise[:,j]=noise[:,i]
        if active.any():noise/=np.sqrt(rng.chisquare(n-1,size=size)/(n-1))[:,None]
        scores=mean+noise;off=scores[:,:k];change=scores[:,k:];change[:,old:]=0.;po=off.argmax(1)
        if radius is not None:
            sure,possible=winner_bounds(off,radius[:k],truth);qlo=float((~possible).sum());qhi=float((~sure).sum())
        for name,step in zip(CANDIDATES,risk.STEPS):
            on=guard_scores(off,step*change,old);pn=on.argmax(1);a=sums[name]
            a['wrong']+=int((po!=truth).sum());a['rescued']+=int(((po!=truth)&(pn==truth)).sum());a['harmed']+=int(((po==truth)&(pn!=truth)).sum())
            if truth>=old and a['harmed']:raise ValueError('New class model guard failed')
            if radius is not None:
                rad=radius.copy();rad[k:]*=step;rad[k+old:]=0.
                b=event_bounds(off,step*change,rad,old,truth);a['qlo']+=qlo;a['qhi']+=qhi
                for key,field in [('rl','rescue_lower'),('ru','rescue_upper'),('hl','harm_lower'),('hu','harm_upper')]:a[key]+=size*b[field]
    out={}
    for name,a in sums.items():
        q=a['wrong']/total;correct=total-a['wrong'];r=a['rescued']/a['wrong'] if a['wrong'] else None;h=a['harmed']/correct if correct else None
        model=dict(off_error=q,rescue_given_wrong=.5 if r is None else r,harm_given_correct=.5 if h is None else h,
                   rescue_prior_available=r is not None and bool(active.any()),harm_prior_available=h is not None and bool(active.any()))
        sensitivity=dict(available=True,off_error_sensitivity=0.,rescue_sensitivity=0.,harm_sensitivity=0.)
        if radius is not None:
            ql,qh=a['qlo']/total,a['qhi']/total
            rl=(a['rl']/total)/qh if qh else 0.;ru=min(1.,(a['ru']/total)/ql) if ql else 1.
            hl=(a['hl']/total)/(1-ql) if ql<1 else 0.;hu=min(1.,(a['hu']/total)/(1-qh)) if qh<1 else 1.
            sensitivity.update(off_error_sensitivity=min(1.,max(abs(q-ql),abs(qh-q))),
                               rescue_sensitivity=1. if r is None else min(1.,max(abs(r-rl),abs(ru-r))),
                               harm_sensitivity=1. if h is None else min(1.,max(abs(h-hl),abs(hu-h))))
        out[name]=dict(model=model,drift=sensitivity,numeric_node_counts=dict(total=total,wrong=a['wrong'],rescued=a['rescued'],harmed=a['harmed']),
                       numeric_standard_error_q=0. if not active.any() else float(np.sqrt(q*(1-q)/total)),
                       conditional_model_rescue=r,conditional_model_harm=h,model_not_empirical_confidence=True,
                       distribution_scatter_identified=bool(active.any()))
    return out


def build_selections(evidence,metadata,budget=lambda:None,progress=lambda **kw:None):
    k=len(evidence['seen']);old=evidence['old'];task=evidence['task'];models=evidence['distributions']
    if old==0:
        return dict(selections={mode:dict(chosen_scale=0.,status='MODEL_CONDITIONAL_SELECTION',exact_identity=True,candidates=[]) for mode in risk.HEADS},class_audit=[],drift=evidence['drift'])
    covs=[m['covariance'] for m in models];traces=[max(float(np.trace(S)),np.finfo(float).tiny) for S in covs]
    decisions={mode:[[] for _ in CANDIDATES] for mode in risk.HEADS};audit=[]
    for c,m in enumerate(models):
        budget();progress(task=task,label=m['label'],phase='CONDITIONAL_MODEL_INTEGRATION')
        radius=np.sqrt(np.maximum(0.,np.diag(m['paired_second']))) if c<old else None
        local=integrated(m['mean'],m['covariance'],m['n'],old,c,(task,c,0),radius)
        competitors=[j for j in range(k) if j!=c];B=np.zeros((2*(k-1),2*k))
        for j,t in enumerate(competitors):B[j,c]=1.;B[j,t]=-1.;B[j+k-1,k+c]=1.;B[j+k-1,k+t]=-1.
        target=B@covs[c]@B.T;target_trace=max(float(np.trace(target)),np.finfo(float).tiny);shape=target/target_trace
        weights=[];loan=[]
        for donor in range(k):
            if donor==c:continue
            if traces[c]<=np.finfo(float).tiny or traces[donor]<=np.finfo(float).tiny:continue
            budget();Sd=covs[donor]*traces[c]/traces[donor];gd=B@Sd@B.T;dt=max(float(np.trace(gd)),np.finfo(float).tiny)
            mismatch=float(np.linalg.norm(gd/dt-shape));age=evidence['drift'][donor]['transport_age']
            D=models[donor]['paired_second'];available=D is not None or donor>=old
            drift_z=0. if D is None else float(np.sqrt(max(0.,np.trace(B@D@B.T))/target_trace))
            proposal=models[donor]['n']/(models[donor]['n']+16.)*np.exp(-mismatch*mismatch/.1)
            sampled=integrated(m['mean'],Sd,m['n'],old,c,(task,c,donor+1))
            loan.append(dict(sampled=sampled,weight=proposal,support=models[donor]['n'],age=age,boundary_mismatch_z=mismatch,drift_z=drift_z,evidence_available=available,donor_label=models[donor]['label']))
        for ci,name in enumerate(CANDIDATES):
            counts=evidence['counts'][name][c];meta=dict(metadata,task=task,label=m['label'],support=m['n'],candidate=risk.STEPS[ci])
            obs=dict(counts,metadata=meta,basis='training_inner',classifier_independent=False,val_accessed=False,test_accessed=False)
            donors=[dict({key:d[key] for key in d if key!='sampled'},**d['sampled'][name]['model']) for d in loan]
            for mode in risk.HEADS:
                z=risk.estimate_class(obs,local[name]['model'],donors,local[name]['drift'],evidence['edges'][name][c],meta,c<old,mode)
                if not z['ready']:raise ValueError('Production evidence NOT_READY; no automatic fallback')
                decisions[mode][ci].append(z)
                audit.append(dict(mode=mode,candidate=name,label=m['label'],**{key:value for key,value in z.items() if key!='net_samples'},
                                  training_counts=counts,edge_statistics=evidence['edges'][name][c],
                                  conditional_net_mean=float(z['net_samples'].mean()),conditional_net_q10=float(np.quantile(z['net_samples'],.1)),
                                  numeric_node_counts=local[name]['numeric_node_counts'],model_q=local[name]['model']['off_error'],
                                  numeric_standard_error_q=local[name]['numeric_standard_error_q'],distribution_scatter_identified=local[name]['distribution_scatter_identified'],
                                  model_r=local[name]['conditional_model_rescue'],model_h=local[name]['conditional_model_harm']))
    return dict(selections={mode:risk.select(decisions[mode],old,mode) for mode in risk.HEADS},class_audit=audit,drift=evidence['drift'])
