"""CPU-only known-map and non-identifiability fixtures; no model/data loading."""
import copy
import io
import json
import os
import time

import torch
from torch.nn import functional as F

import core1_competition as competition
import module_additions as modules
import paired_transport_design as design
import prototype_coherent as native


def bank(x,y,k):
    weights=x.new_zeros(len(x))
    for c in range(k):weights[y==c]=1/int((y==c).sum())
    return native.append(native.empty(x.shape[1]),x,y,y,weights)


def fit(x,y,labels,space='global'):
    ids=torch.arange(len(x));return design.fit(x,y,labels,ids,ids,'start','end',space)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(1074005);dtype=torch.float64;begun=time.monotonic()
    d=8;n=24;x=torch.randn(n,d,dtype=dtype);labels=torch.arange(n)%2
    zero=fit(x,x,labels);b=design.framed(bank(x,labels,2),'start');bz=design.push_bank(b,zero)
    assert torch.equal(bz['mu'],b['mu']) and torch.equal(bz['Q'],b['Q'])
    shift=torch.randn(d,dtype=dtype)*.02;translation=fit(x,x+shift,labels)
    bt=design.push_bank(b,translation);legacy=native.translate(b,translation['offset'])
    assert translation['rank']==0 and torch.allclose(bt['mu'],legacy['mu'],atol=1e-12) and torch.allclose(bt['Q'],legacy['Q'],atol=1e-12)
    for a,c in zip(bt['components'],b['components']):assert torch.equal(a['radius'],c['radius'])
    # Non-isotropic paired map: raw moment pushforward matches virtual samples exactly.
    M=torch.eye(d,dtype=dtype);M[0,0]=1.2;M[0,1]=.2;M[2,3]=-.1
    state=fit(x,x@M+shift,labels);mapped=design.push_bank(b,state)
    virtual=bank(design.predict(x,state),labels,2)
    moment_error=max(float((mapped[key]-virtual[key]).abs().max()) for key in ('mu','Q'))
    assert moment_error<1e-11
    for c,v in zip(mapped['components'],virtual['components']):
        assert c['radius'] is None and c['radius_bounds'][0]-1e-12<=v['radius']<=c['radius_bounds'][1]+1e-12
    # Known low-dimensional rotation; remaining input directions are not learned.
    plane=torch.zeros(12,d,dtype=dtype);plane[:,:2]=torch.randn(12,2,dtype=dtype)
    rotation=torch.eye(d,dtype=dtype);rotation[:2,:2]=torch.tensor([[0.,-1.],[1.,0.]],dtype=dtype)
    known=fit(plane,plane@rotation,torch.arange(12)%2)
    rotation_error=float((design.predict(plane,known)-plane@rotation).norm()/plane.norm())
    assert rotation_error<.005 and known['rank']==2
    hidden=torch.zeros(d,dtype=dtype);hidden[4]=1
    assert torch.allclose(design.predict(hidden,known)-design.predict(torch.zeros_like(hidden),known),hidden,atol=1e-12)
    hidden_rotation=torch.eye(d,dtype=dtype);hidden_rotation[4:6,4:6]=torch.tensor([[0.,-1.],[1.,0.]],dtype=dtype)
    invisible=fit(plane,plane@hidden_rotation,torch.arange(12)%2)
    unseen_error=float((design.predict(hidden,invisible)-hidden@hidden_rotation).norm());assert unseen_error>1
    # Numerical weak input directions are ridge-shrunk rather than amplified.
    weak=x.clone();weak[:,1]*=1e-10;noisy=weak.clone();noisy[:,1]+=.01*torch.randn(n,dtype=dtype)
    stable=fit(weak,noisy,labels);assert float(stable['right'].norm())<1 and torch.isfinite(stable['right']).all()
    # Re-estimate start->epoch2 from the same immutable snapshot, never epoch1 state.
    first=design.push_bank(b,translation)
    ids=torch.arange(n);secondmap=design.fit(x,x+2*shift,labels,ids,ids,'end','third','global')
    composed=design.push_bank(first,secondmap)
    direct=native.translate(b,translation['offset']+secondmap['offset'])
    assert torch.allclose(composed['Q'],direct['Q'],atol=1e-11)
    rejected=[]
    for case in ('double_transport','wrong_region_space','pair_order','current_frame'):
        try:
            if case=='double_transport':design.push_bank(first,translation)
            elif case=='wrong_region_space':design.push_regions(dict(values=x[:2,None,:].expand(-1,4,-1),frame='start'),[translation]*4)
            elif case=='pair_order':design.fit(x,x,labels,ids,ids.flip(0),'start','end','global')
            else:design.append_current(b,x,labels,labels,x.new_ones(n), 'end')
        except ValueError:rejected.append(case)
    assert len(rejected)==4
    # Exact stored n/mu/Q on the unit circle still does not determine renormalized moments.
    axes=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],dtype=dtype)
    diagonal=torch.tensor([[1.,1.],[-1.,1.],[1.,-1.],[-1.,-1.]],dtype=dtype)/2**.5
    assert torch.allclose(axes.mean(0),diagonal.mean(0)) and torch.allclose(axes.T@axes/4,diagonal.T@diagonal/4)
    t=torch.tensor([.3,.1],dtype=dtype)
    renorm_error=float((F.normalize(axes+t,dim=1).mean(0)-F.normalize(diagonal+t,dim=1).mean(0)).norm())
    assert renorm_error>1e-3
    # Same global CLS, changing patch geometry: a global identity map cannot update regions.
    tokens=[torch.randn(8,5,4,dtype=dtype),torch.randn(8,5,4,dtype=dtype)]
    changed=[v.clone() for v in tokens];changed[0][:,1:,0]+=.8
    region_before=modules.local_parts(tokens);region_after=modules.local_parts(changed)
    cls_before=F.normalize(torch.cat([tokens[1][:,0],tokens[0][:,0]],1),dim=1)
    cls_after=F.normalize(torch.cat([changed[1][:,0],changed[0][:,0]],1),dim=1)
    assert torch.equal(cls_before,cls_after)
    regional_mismatch=float((region_after-region_before).norm());assert regional_mismatch>.1
    # Global and regional maps have separate domains but one shared checkpoint frame.
    local=region_before[:2];maps=[fit(region_before[:,i],region_after[:,i],torch.arange(8)%2,f'region_{i}') for i in range(4)]
    local_moved=design.push_regions(dict(values=local,frame='start'),maps)
    assert all(torch.allclose(local_moved['values'][:,i],design.predict(local[:,i],maps[i])) for i in range(4))
    high=[];d,k=1536,23
    basis=torch.linalg.qr(torch.randn(d,8,dtype=dtype)).Q
    count=[3,6,6,18]+[8]*17;old_y=torch.cat([torch.full((n,),c,dtype=torch.long) for c,n in enumerate(count)])
    old_x=F.normalize(torch.randn(len(old_y),8,dtype=dtype)@basis.T,dim=1)
    current=torch.randn(32,8,dtype=dtype)@basis.T;cy=torch.arange(32)%2+21
    change=torch.eye(8,dtype=dtype);change[0,0]=1.1;change[2,3]=.15
    after=current+current@basis@(change-torch.eye(8,dtype=dtype))@basis.T+.001*basis[:,0]
    state=fit(current,after,cy);old=design.framed(bank(old_x,old_y,21),'start')
    transported=design.push_bank(old,state)
    weights=current.new_full((len(current),),1/16)
    combined=design.append_current(transported,after,cy,cy,weights,'end')
    allx=torch.cat([design.predict(old_x,state),after]);ally=torch.cat([old_y,cy]);virtual=bank(allx,ally,k)
    high_moment=max(float((combined[key]-virtual[key]).abs().max()) for key in ('mu','Q'));assert high_moment<1e-10
    assert torch.allclose(combined['mu'][21:],torch.stack([after[cy==c].mean(0) for c in (21,22)]),atol=1e-12)
    local=torch.randn(k,4,d,dtype=dtype)*.01
    for arm in ('base','local_transport'):
        R,a=modules.metric(combined,arm,'HK',list(range(k)),local if arm=='local_transport' else None)
        Rv,_=modules.metric(virtual,arm,'HK',list(range(k)),local if arm=='local_transport' else None)
        reference,_=native.head(virtual);pairs=competition.competition(virtual,reference)
        w,wa=competition.bank_head(combined,pairs,regularizer=R)
        v,va=competition.bank_head(virtual,pairs,regularizer=Rv)
        error=float((w-v).norm()/v.norm());assert error<1e-8
        high.append(dict(control=arm,dimension=[d,k],moment_max_error=high_moment,head_relative_error=error,
                         solve_residual=wa['relative_residual'],legacy_component_radius_not_consumed=True,
                         current_new_statistics_not_transported=True))
    stream=io.BytesIO();torch.save(dict(bank=combined,map=state),stream);stream.seek(0)
    saved=torch.load(stream,weights_only=False);design.validate(saved['map'],current,'global')
    assert torch.equal(saved['bank']['Q'],combined['Q']) and saved['bank']['_frame']=='end'
    assert all(c['radius'] is None for c in combined['components'][:21])
    return dict(status='CPU_DESIGN_PASS_NOT_PRODUCTION_READY',version=design.VERSION,synthetic_only=True,
                GPU_processes=0,medical_images_read=0,medical_checkpoint_reads=0,val_reads=0,test_accessed=False,
                optimizer_updates=0,production_integrated=False,GPU_ready=False,
                zero_drift_exact=True,translation_reduces_to_native=True,virtual_sample_moment_error=moment_error,
                known_observed_rotation_relative_error=rotation_error,relative_ridge=design.RELATIVE_RIDGE,
                numerical_rank_only=True,weak_direction_regularization_checked=True,
                unobserved_true_rotation_error=unseen_error,unobserved_drift_not_identified=True,
                renormalized_moments_nonidentifiability=renorm_error,
                unchanged_CLS_changed_region_error=regional_mismatch,separate_regional_maps_required=True,
                component_radius_only_bounded=True,composition_once_checked=True,invalid_inputs_refused=rejected,
                highdimensional_checks=high,checkpoint_roundtrip_exact=True,
                independent_calibration=False,new_class_accuracy_guarantee=False,
                pipeline_seconds=time.monotonic()-begun)


if __name__=='__main__':print(json.dumps(run(),allow_nan=False),flush=True)
