"""Synthetic CPU oracles; no medical inputs, GPU, optimizer or production entry."""
import copy,json,math,os,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import gain_joint as joint
import intervention_gain as legacy
import intervention_gain_production as production
import prototype_coherent as native
import core1_competition as core
import module_additions as modules
from check_intervention_gain import reject,exact_gain

def component(center,S,n=5000,mass=1.):
    return dict(center=center,count=n,mass=mass,family=[S],data_identified=True,envelope=S,unidentified=False)

def run(protocol):
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());joint.validate_design(p)
    for key in ('GPU_launch_authorized','production_integrated','medical_data_access_authorized','checkpoint_access_authorized','test_accessed','validation_retuning'):
        q=dict(p);q[key]=True;reject(lambda:joint.validate_design(q))
    q=dict(p,steps=[.1,.25,.5,1.]);reject(lambda:joint.validate_design(q))
    W=torch.tensor([[-.1,.1,0.],[0.,0.,-1.]],dtype=torch.float64);D=torch.tensor([[0.,0.,0.],[.1,-.1,0.]],dtype=torch.float64)
    S=torch.diag(W.new_tensor([.25,0.]));stats=[[component(W.new_tensor([.1,1.]),S)],[component(W.new_tensor([3.,1.]),S)],[component(W.new_tensor([0.,-1.]),S)]]
    good=joint.evaluate(stats,W,D,2,False);rr,hh=exact_gain(stats,W,D,2)
    assert good['robust_lower']<=rr-hh<=good['robust_upper'] and good['robust_lower']>0
    scaled=joint.evaluate(stats,W*1e-6,D*1e-6,2,False);assert abs(scaled['scenarios'][0]['known_net']-good['scenarios'][0]['known_net'])<1e-10
    chosen,a=joint.select(stats,W,D,2,True);assert chosen.any() and a['chosen_scale'] in joint.STEPS
    bad=copy.deepcopy(stats);bad[0][0]['center'][0]=-3.;bad[1][0]['center'][0]=.1
    negative=joint.evaluate(bad,W,D,2,False);nr,nh=exact_gain(bad,W,D,2)
    assert negative['robust_lower']<=nr-nh<=negative['robust_upper'] and negative['robust_upper']<0
    none,b=joint.select(bad,W,D,2,True);assert not none.any()
    unidentified=copy.deepcopy(stats);unidentified[0][0].update(count=1,data_identified=True)
    unknown=joint.evaluate(unidentified,W,D,2,True);assert unknown['robust_lower']<0 and abs(unknown['scenarios'][0]['unknown_mass']-1/3)<1e-14
    anchor=joint.evaluate(unidentified,W,torch.zeros_like(D),2,True);assert anchor['robust_lower']==anchor['robust_upper']==0
    for n in (2,3):
        tiny=copy.deepcopy(stats);tiny[0][0]['count']=n;z=joint.evaluate(tiny,W,D,2,True)
        assert all(math.isfinite(t['known_net']) for t in z['scenarios'])
    # Twenty-one identical competing new heads: scalar oracle retains their correlation.
    k=23;WW=torch.cat((torch.eye(k,dtype=torch.float64),torch.zeros(1,k,dtype=torch.float64)));DD=torch.zeros_like(WW);DD[-1,:2]=WW.new_tensor([.1,-.1]);v=WW.new_tensor([-.1]+[.1]*(k-1)+[0.]);CC=torch.outer(v,v)
    multi=[]
    for c in range(k):
        u=WW.new_full((k+1,),-10.);u[-1]=1.;u[c]=10.
        if c==0:u[:k]=0.
        multi.append([component(u,CC if c==0 else torch.zeros_like(CC))])
    correlated=joint.evaluate(multi,WW,DD,2,False);exact=(.5*math.erfc(-.5/math.sqrt(2))-.5)/k
    assert correlated['robust_lower']<=exact<=correlated['robust_upper'] and correlated['robust_lower']>0
    old=legacy.utility(multi,WW,DD,2,1.);assert old['rescue_lower']==0
    # Full 1536x23 score projection interface, not a full-dimensional native solver claim.
    wideW=torch.zeros(1536,23,dtype=torch.float64);wideW[:24]=WW;wideD=torch.zeros_like(wideW);wideD[:24]=DD
    c=multi[0][0];u=torch.zeros(1536,dtype=torch.float64);u[:24]=c['center'];cov=torch.zeros(1536,1536,dtype=torch.float64);cov[:24,:24]=CC
    proj=torch.cat((wideW,wideD),1);torch.testing.assert_close(u@proj,c['center']@torch.cat((WW,DD),1));torch.testing.assert_close(proj.T@cov@proj,torch.cat((WW,DD),1).T@CC@torch.cat((WW,DD),1))
    d,k,old_count=6,4,3;y=torch.arange(k).repeat_interleave(16);features=torch.randn(len(y),d,dtype=torch.float64)*.5+F.one_hot(y,d);groups=2*y+torch.arange(len(y))%2
    bank=native.append(native.empty(d),features,y,groups,features.new_full((len(y),),1/16));snapshot=copy.deepcopy(bank);reference,_=native.head(bank);pairs=core.competition(bank,reference);ray_cases=[]
    for arm in ('base','local_transport'):
        R,_=modules.metric(bank,arm,'HK',list(range(k)),None if arm=='base' else torch.randn(k,4,d,dtype=features.dtype));W0,_=core.bank_head(bank,pairs,regularizer=R)
        captured=[];original=legacy.select_ray
        def spy(models,readout,ray,old,rescue_only=False):
            captured.append(ray.clone());return torch.zeros_like(ray),dict(chosen_scale=0.,candidates=[])
        legacy.select_ray=spy
        try:production.fit_all(bank,pairs,R,W0,old_count,W0.float())
        finally:legacy.select_ray=original
        ray,qa=joint.frozen_ray(bank,pairs,R,W0,old_count,W0.float());assert torch.equal(ray,captured[0])
        for scale in joint.STEPS:assert torch.equal((scale*ray).float(),(scale*captured[0]).float())
        fixed,cred=joint.models_from_bank(bank,W0);assert all(r['count']==snapshot['components'][i]['count'] for i,r in enumerate([z for parts in fixed for z in parts]))
        reject(lambda:joint.frozen_ray(bank,pairs,R,W0+.1,old_count,W0.float()))
        ray_cases.append(dict(control=arm,original_ray_bit_exact=True,four_candidate_bits_exact=True,solver_residual=qa['relative_residual']))
    assert torch.equal(bank['Q'],snapshot['Q']) and torch.equal(bank['mu'],snapshot['mu'])
    # Singletons stay unidentified even when rounding leaves nonzero scatter and borrowing exists.
    yy=torch.tensor([0]+[1]*16+[2]*16);xx=torch.randn(len(yy),d,dtype=torch.float64);bb=native.append(native.empty(d),xx,yy,yy,xx.new_tensor([1.]+[1/16]*32));head,_=native.head(bb)
    for residual in (0.,1e-20):
        part=bb['components'][0];part['radius']=part['radius'].new_tensor(residual)
        mm,aa=joint.models_from_bank(bb,head);assert not mm[0][0]['data_identified'] and aa[0]['unidentified_mass']==1
    # Guard preserves all original-new winners and their margins, including tied old maxima.
    before=np.random.default_rng(12).normal(size=(1000,23));change=np.zeros_like(before);change[:,:21]=np.random.default_rng(13).normal(size=(1000,21));change[:,:21]-=change[:,:21].mean(1,keepdims=True)
    after=joint.guard_scores(before,change,21);off=before.argmax(1);assert np.array_equal(after[:,21:],before[:,21:]) and np.array_equal(after.argmax(1)[off>=21],off[off>=21])
    invalid=D.clone();invalid[:,2]=1.;reject(lambda:joint.evaluate(stats,W,invalid,2,False));reject(lambda:joint.evaluate(stats,W,D,True,False))
    mean=np.zeros(6);covariance=np.eye(6);reject(lambda:joint.integrate(mean,covariance,2,0,None,(0,0)))
    reject(lambda:joint.integrate(np.zeros(4),-np.eye(4),2,0,None,(0,0)))
    return dict(status='PASS',synthetic_only=True,GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,medical_val_reads=0,optimizer_updates=0,policy_updates=0,production_integrated=False,scope_rejected=True,native_compatibility=ray_cases,bank_unchanged=True,positive_model=dict(exact_net=rr-hh,lower=good['robust_lower'],upper=good['robust_upper'],predictive_chosen_scale=a['chosen_scale']),negative_model=dict(exact_net=nr-nh,lower=negative['robust_lower'],upper=negative['robust_upper'],predictive_identity=True),correlated_23_class_model=dict(exact_net=exact,joint_lower=correlated['robust_lower'],joint_upper=correlated['robust_upper'],legacy_rescue_lower=old['rescue_lower']),positive_score_rescaling_invariant=True,exact_covariance_duplicate_ties_preserved=True,singleton_always_unidentified=True,unknown_old_mass_retained=True,df_1_2_finite_categorical_probabilities=True,new_guard_preserved=True,interface_1536_by_23_and_score_46=True,full_high_dimensional_native_solve=False,clinical_confidence_calibrated=False,pipeline_seconds=time.monotonic()-start)

if __name__=='__main__':print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
