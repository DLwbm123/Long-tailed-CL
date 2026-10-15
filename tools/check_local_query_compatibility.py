"""Necessary common-query domain audit, not a classifier performance experiment."""
import math
import os
import time
import torch
import torch.nn.functional as F
from check_progressive_decision_memory import DT,fixture

RADIUS=1/8


def geometry(anchors):
    c=len(anchors);pairs=[]
    for i in range(c):
        for j in range(i+1,c):
            distance=float((anchors[i]-anchors[j]).norm())
            pairs.append(dict(class_i=i,class_j=j,anchor_distance=distance,common_radius_lower_bound=distance/2,excess_over_two_r=distance-2*RADIUS,incompatible=distance>2*RADIUS+1e-12))
    center=anchors.mean(0);upper=float((anchors-center).norm(dim=1).max());lower=max((p['common_radius_lower_bound'] for p in pairs),default=0.)
    incompatible=sum(p['incompatible'] for p in pairs)
    status='EMPTY_INTERSECTION_CERTIFIED' if incompatible else ('NONEMPTY_WITNESS' if upper<=RADIUS+1e-12 else 'UNRESOLVED_BY_PAIRWISE_TEST')
    return dict(seen_classes=c,status=status,incompatible_pairs=incompatible,total_pairs=len(pairs),necessary_common_radius_lower_bound=lower,mean_center_radius_upper_bound=upper,original_local_radius=RADIUS),pairs


def unit_checks():
    same=torch.tensor([[2.,0.],[2.,0.]],dtype=DT);a,_=geometry(same);assert a['status']=='NONEMPTY_WITNESS'
    boundary=torch.tensor([[-RADIUS,0.],[RADIUS,0.]],dtype=DT);b,_=geometry(boundary);assert b['status']=='NONEMPTY_WITNESS'
    opposite=torch.tensor([[2.,0.],[0.,2.]],dtype=DT);c,_=geometry(opposite);assert c['status']=='EMPTY_INTERSECTION_CERTIFIED'
    side=.24;tri=torch.tensor([[0.,0.],[side,0.],[side/2,math.sqrt(3)*side/2]],dtype=DT);t,_=geometry(tri)
    assert t['incompatible_pairs']==0 and t['status']=='UNRESOLVED_BY_PAIRWISE_TEST'
    circumradius=side/math.sqrt(3);assert circumradius>RADIUS
    assert abs(t['mean_center_radius_upper_bound']-circumradius)<1e-12
    perm,_=geometry(opposite.flip(0));assert abs(perm['necessary_common_radius_lower_bound']-c['necessary_common_radius_lower_bound'])<1e-12
    return dict(status='PASS',identical_anchor_feasible=True,boundary_tangent_feasible=True,separated_pair_infeasible=True,pairwise_nonviolation_not_sufficient=True,equilateral_circumradius=circumradius,permutation_distance_invariant=True)


def run(cfg,lock,journal):
    started=time.monotonic();torch.set_num_threads(lock['threads']);torch.set_default_dtype(DT);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if cfg['kind']=='unit':out=unit_checks()
    else:
        x,y,counts=fixture(cfg);means=torch.stack([x[y==j].mean(0) for j in range(cfg['classes'])]);assert bool((means.norm(dim=1)>1e-12).all());anchors=2*F.normalize(means,dim=1)
        stages=[];seen=0
        for task,step in enumerate(cfg['task_sizes'],1):
            seen+=step;journal(dict(phase='shared_query_domain_geometry',task=task,seen_classes=seen));row,_=geometry(anchors[:seen]);row['task']=task;stages.append(row)
        assert seen==cfg['classes'];final,pairs=geometry(anchors)
        shared,_=geometry(anchors[:1].repeat(len(anchors),1));assert shared['status']=='NONEMPTY_WITNESS'
        out=dict(status='COMPLETED',case=cfg,synthetic_counts=counts,stages=stages,pairs=pairs,shared_anchor_algebraic_control=shared,loss_or_classifier_evaluated=False,old_fitted_memory_read=False,query_optimization_updates=0,memory_optimization_updates=0)
    out.update(id=cfg['id'],CPU_pipeline_seconds=time.monotonic()-started,GPU_initialized=torch.cuda.is_initialized(),medical_images=0,medical_checkpoints=0,val_accessed=False,test_accessed=False,formal_updates=0)
    assert not out['GPU_initialized'];return out
