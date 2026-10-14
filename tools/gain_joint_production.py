"""Same-state fixed rays, four sidecars and two pre-val selection rules."""
import torch
import gain_joint as joint
CANDIDATES=('C0125','C025','C05','C1')
HEADS=CANDIDATES+('LEGACY_SELECTED','JOINT_POINT_SELECTED','JOINT_PRED_SELECTED')

@torch.no_grad()
def fit_all(bank,pairs,R,W,old,readout,legacy):
    ray,qa=joint.frozen_ray(bank,pairs,R,W,old,readout)
    if old and abs(float(ray.norm())-legacy['preselection_delta_norm'])>1e-6*max(1.,legacy['preselection_delta_norm']):raise ValueError('Frozen direction norm differs from public legacy')
    deltas={v:(scale*ray).float().double() for v,scale in zip(CANDIDATES,joint.STEPS)}
    models,cred=joint.models_from_bank(bank,readout.double())
    selections={};point,pa=joint.select(models,readout.double(),ray,old,False);pred,ta=joint.select(models,readout.double(),ray,old,True)
    selections['LEGACY_SELECTED']=dict(chosen_scale=legacy.get('chosen_scale',0.),public_reuse=True)
    selections['JOINT_POINT_SELECTED']=dict(chosen_scale=pa['chosen_scale'],selection=pa['selection'])
    selections['JOINT_PRED_SELECTED']=dict(chosen_scale=ta['chosen_scale'],selection=ta['selection'])
    for z in selections.values():
        if z['chosen_scale'] not in (0.,)+joint.STEPS:raise ValueError('Frozen selection differs')
    audits={}
    for i,v in enumerate(CANDIDATES):
        delta=deltas[v];bound=1e-8+torch.finfo(torch.float32).eps*delta[:,:old].abs().sum(1)
        if delta[:,old:].any() or (delta[:,:old].sum(1).abs()>bound).any() or not torch.isfinite(delta).all():raise ValueError('Quantized old-only ray differs')
        audits[v]=dict(qa,scale=joint.STEPS[i],exact_identity=not bool(delta.any()),point=pa['candidates'][i],predictive=ta['candidates'][i],legacy=next((r for r in legacy.get('candidates',[]) if r['scale']==joint.STEPS[i]),None),full_delta_norm=float(delta.norm()),discriminative_delta_norm=float((delta-delta.mean(1,keepdim=True)).norm()))
    return deltas,dict(candidates=audits,selections=selections,credibility=cred,direction=qa,direction_changed=False,new_columns_fixed=True,legacy_public_reuse=True)

def expanded(deltas,selections):
    zero=torch.zeros_like(next(iter(deltas.values()))) if isinstance(next(iter(deltas.values())),torch.Tensor) else __import__('numpy').zeros_like(next(iter(deltas.values())))
    result=dict(deltas)
    for name,z in selections.items():result[name]=zero if not z['chosen_scale'] else deltas[CANDIDATES[joint.STEPS.index(z['chosen_scale'])]]
    if set(result)!=set(HEADS):raise ValueError('Seven non-OFF readouts required')
    return result
