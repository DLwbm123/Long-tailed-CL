"""CPU-only synthetic compatibility and intervention-risk oracles."""
import copy
import io
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import gain_conditional_risk as risk
import gain_joint as joint
import prototype_coherent as native
import core1_competition as core
import module_additions as modules


def reject(f):
    try:f()
    except ValueError:return
    raise AssertionError('Illegal risk input accepted')


def fixture(n=400, correct=200, rescued=100, harmed=1, old=True, erosion=0., mode='CRISK', drift_size=0.):
    meta=dict(source='synthetic', control='base', task=2, label=0, support=n, candidate=.125)
    evidence=dict(metadata=meta, basis='synthetic', classifier_independent=False,
                  val_accessed=False, test_accessed=False, n=n, correct_off=correct, rescued=rescued, harmed=harmed)
    model=dict(off_error=.9, rescue_given_wrong=.9, harm_given_correct=.001)
    drift=dict(available=True, off_error_sensitivity=drift_size, rescue_sensitivity=drift_size, harm_sensitivity=drift_size)
    edge=dict(correct_support=correct, erosion_sum=correct*erosion, erosion_square_sum=correct*erosion**2)
    return risk.estimate_class(evidence, model, [], drift, edge, meta, old, mode)


def run(protocol):
    start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(74002)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    p=json.loads(Path(protocol).read_text());risk.validate_design(p)
    for key in ('GPU_launch_authorized','medical_data_access_authorized','checkpoint_access_authorized','validation_retuning','test_accessed','production_integrated'):
        reject(lambda key=key:risk.validate_design(dict(p,**{key:True})))
    reject(lambda:risk.validate_design(dict(p,steps=[.1,.25,.5,1.])))
    for n,correct in [(1,1),(3,3),(23,23),(25,21),(34,31),(400,200)]:
        a=fixture(n,correct,0,0);assert np.isfinite(a['net_samples']).all()
        assert a['rescue_mean'] <= a['off_error_mean'] and a['harm_mean'] <= 1-a['off_error_mean']
        assert a['model_prior_pseudocount'] <= risk.PRIOR_CAP and a['target_actual_n']==n
    positive=fixture();new=fixture(harmed=0,old=False)
    assert risk.select([[positive,positive,new]]*4,2)['chosen_scale']==.125
    negative=fixture(rescued=1,harmed=100)
    assert risk.select([[negative,negative,new]]*4,2)['chosen_scale']==0.
    fragile=fixture(erosion=.6)
    assert fragile['edge_failed'] and risk.select([[fragile,positive,new]]*4,2)['chosen_scale']==0.
    assert risk.select([[fragile,positive,new]]*4,2,'NO_EDGE_GUARD')['chosen_scale']==.125
    a=fixture();b=fixture(mode='NO_RESERVOIR');assert abs(b['off_error_mean']-.9)<1e-14 and a['off_error_mean']<.51
    assert risk.select([[new,new,new]]*4,0)['chosen_scale']==0.
    uncertain=fixture(drift_size=.1);assert np.all(uncertain['net_samples']<=a['net_samples'])
    assert np.array_equal(fixture(mode='NO_DRIFT',drift_size=.1)['net_samples'],a['net_samples'])
    target=dict(off_error=.1,rescue_given_wrong=.1,harm_given_correct=.2)
    donor=dict(off_error=.9,rescue_given_wrong=.99,harm_given_correct=0.,weight=1.,support=100000,age=0,boundary_mismatch_z=10.,drift_z=10.,evidence_available=True)
    trusted,audit=risk.donor_model(target,[donor],'CRISK');untrusted,_=risk.donor_model(target,[donor],'NO_DONOR_TRUST')
    assert trusted['off_error']==target['off_error'] and audit[0]['credibility_score']<.005
    assert trusted['rescue_given_wrong']<untrusted['rescue_given_wrong']
    missing,_=risk.donor_model(target,[dict(donor,evidence_available=False)],'CRISK');assert missing==target
    # Exposure lineage, exact metadata, reservoirs and missing-edge evidence.
    meta=dict(source='synthetic',control='base',task=2,label=0,support=25,candidate=.125)
    e=dict(metadata=meta,basis='training_inner',classifier_independent=False,val_accessed=False,test_accessed=False,n=25,correct_off=21,rescued=3,harmed=0)
    risk.provenance(e,meta)
    reject(lambda:risk.provenance(dict(e,classifier_independent=True),meta))
    reject(lambda:risk.provenance(dict(e,val_accessed=True),meta))
    reject(lambda:risk.provenance(dict(e,metadata=dict(meta,control='local_transport')),meta))
    reject(lambda:risk.estimate_class(dict(e,rescued=5),target,[],dict(available=True),None,meta,True))
    nr=risk.estimate_class(e,target,[],dict(available=False),None,meta,True)
    assert not nr['ready'] and risk.select([[nr,nr,new]]*4,2)['status']=='NOT_READY'
    reject(lambda:risk.selected_delta(np.zeros((4,3)),risk.select([[nr,nr,new]]*4,2),2))
    assert not risk.estimate_class(e,target,[],dict(available=True),None,meta,True)['ready']
    before=np.array([[1.,0.,-1.],[2.,0.,-1.],[0.,1.,-1.]])
    after=joint.guard_scores(before,np.array([[-.6,.6,0.],[-.1,.1,0.],[.2,-.2,0.]]),2)
    labels=np.array([0,0,1]);edge=risk.boundary_summary(before,after,labels,0,2)
    assert edge['correct_support']==2 and edge['erosion_sum']>1.
    scaled=risk.boundary_summary(before*1e-6,after*1e-6,labels,0,2)
    assert abs(scaled['erosion_sum']-edge['erosion_sum'])<1e-12
    tie=risk.boundary_summary(np.zeros((3,3)),np.zeros((3,3)),np.zeros(3,dtype=int),0,2)
    assert tie['erosion_sum']==0. and tie['tied_correct']==3
    # Full native head/ray under both control metrics; risk changes no native state.
    d,k,old=1536,23,21;mu=torch.eye(d,dtype=torch.float64)[:k]*.2;mu[:,-1]=1.
    S=torch.eye(d,dtype=mu.dtype)*.05;counts=[2048]*k;counts[1]=3
    bank=dict(mu=mu,Q=torch.stack([torch.outer(m,m)+S for m in mu]),n=counts,
              components=[dict(label=c,center=mu[c],count=counts[c],mass=mu.new_tensor(1.),radius=S.trace()) for c in range(k)])
    snapshot=copy.deepcopy(bank);initial,_=native.head(bank);pairs=core.competition(bank,initial);cases=[]
    for arm in ('base','local_transport'):
        R,_=modules.metric(bank,arm,'HK',list(range(k)),None if arm=='base' else torch.randn(k,4,d,dtype=mu.dtype))
        head,qa=core.bank_head(bank,pairs,regularizer=R);ray,ra=joint.frozen_ray(bank,pairs,R,head,old,head.float())
        assert ray.any() and qa['relative_residual']<=1e-8 and ra['relative_residual']<=1e-8
        for step in risk.STEPS:
            delta=risk.selected_delta(ray.numpy(),dict(chosen_scale=step),old)
            assert np.array_equal(delta,(step*ray).float().double().numpy())
            buf=io.BytesIO();torch.save(torch.from_numpy(delta),buf);buf.seek(0)
            assert torch.equal(torch.load(buf,weights_only=True),torch.from_numpy(delta))
        X=np.random.default_rng(74002).normal(size=(256,d));off=X@head.float().double().numpy()
        delta=risk.selected_delta(ray.numpy(),dict(chosen_scale=.125),old);on=joint.guard_scores(off,X@delta,old)
        y=off.argmax(1)
        assert not ((y>=old)&(on.argmax(1)!=y)).any()
        for c in range(k):risk.boundary_summary(off,on,y,int(c),old)
        assert np.array_equal(risk.selected_delta(ray.numpy(),dict(chosen_scale=0.),old),np.zeros((d,k)))
        assert torch.equal(bank['Q'],snapshot['Q']) and torch.equal(bank['mu'],snapshot['mu']) and bank['n']==snapshot['n']
        cases.append(dict(control=arm,dimensions=[d,k],native_residual=qa['relative_residual'],ray_residual=ra['relative_residual'],candidate_bits_exact=True,checkpoint_roundtrip=True,new_guard_preserved=True,statistics_unchanged=True))
    return dict(status='PASS',version=risk.VERSION,synthetic_only=True,production_integrated=False,
                independent_calibration_ready=False,GPU_processes=0,medical_images_read=0,
                medical_checkpoint_reads=0,medical_val_reads=0,test_accessed=False,optimizer_updates=0,
                policy_updates=0,highdimensional_checks=cases,conditional_reservoir_oracles=True,
                zero_observed_harm_not_certified_safe=True,rare_edge_positive_macro_rejected=True,
                missing_evidence_NOT_READY=True,all_five_modes_checked=True,
                donor_support_not_target_support=True,drift_monotone_adverse=True,
                scale_and_tie_checks=True,positive_and_negative_admission_checked=True,
                pipeline_seconds=time.monotonic()-start)


if __name__=='__main__':print(json.dumps(run(os.environ['Q128_PROTOCOL']),indent=2,allow_nan=False))
