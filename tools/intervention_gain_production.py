"""Shared four-head preparation; same frozen mathematics, no image access."""
import copy
import math
import torch
import intervention_gain as method
from lca_old_pair_repair import contrast_basis,native_operator
import core1_competition as core

def choose(rows,delta,rescue_only):
    rows=copy.deepcopy(rows);chosen=0.;best=1e-12;accepted=torch.zeros_like(delta)
    for row in rows:
        value=row['rescue_lower'] if rescue_only else row['net_lower'];row['selection_score']=value
        if value>best:best=value;chosen=row['scale'];accepted=(chosen*delta).float().double()
    return accepted,dict(chosen_scale=chosen,selection='rescue_only' if rescue_only else 'macro_rescue_lower_minus_harm_upper',candidates=rows,model_only=True,clinical_or_distribution_free_guarantee=False)

@torch.no_grad()
def fit_all(bank,pairs,R,W0,old,readout):
    method.finite(bank['mu'],bank['Q'],pairs,R,W0);k,d=bank['mu'].shape;Wref=readout.double()
    if type(old)!=int or not 0<=old<=k or W0.shape!=(d,k) or Wref.shape!=W0.shape or pairs.shape!=(k,k) or R.shape!=(d,d) or not torch.allclose(R,R.T,atol=1e-10,rtol=1e-10) or pairs.diag().any() or (pairs<0).any() or not torch.allclose(pairs.sum(1),pairs.new_ones(k),atol=1e-8,rtol=1e-8):raise ValueError('Invalid native geometry')
    method.finite(Wref);_,targets=core.laplacians(pairs);rhs=bank['mu'].T/k+.5*(bank['mu'].T/k)@targets;H=lambda Z:native_operator(bank,pairs,R,Z)
    error=float((H(W0)-rhs).norm()/rhs.norm().clamp_min(1e-30))
    if error>1e-8:raise ValueError('Native reference is not optimum')
    if old<2:
        return {v:(torch.zeros_like(W0),dict(version=method.VERSION,head=v,exact_identity=True,reason='fewer_than_two_old_classes',relative_residual=0.)) for v in method.HEADS}
    V=W0.new_zeros(k,old-1);V[:old]=contrast_basis(old,W0);A=lambda Z:H(Z@V.T)@V
    factor=torch.linalg.cholesky(bank['Q'].mean(0)+.001*R);pre=lambda Z:torch.cholesky_solve(Z,factor)
    reference=(W0@V)@V.T;cap=float((reference*H(reference)).sum());result={};main_models=None
    for borrow,heads in [(True,('GAIN','GAIN_RESCUE_ONLY')),(False,('GAIN_NO_BORROW',))]:
        models,cred=method.credibility(bank,Wref,borrow);g=method.proposal(bank,models,Wref,old);b=g@V
        if borrow:main_models=models
        if not b.any():
            for v in heads:result[v]=(torch.zeros_like(W0),dict(version=method.VERSION,head=v,exact_identity=True,reason='zero_proposal',relative_residual=0.,credibility=cred))
            continue
        Z,qa=core.pcg(A,b,pre,torch.zeros_like(b));delta=Z@V.T;energy=float((delta*H(delta)).sum());ray_scale=min(1.,math.sqrt(cap/energy)) if energy else 0.;delta*=ray_scale
        accepted,a=method.select_ray(models,Wref,delta,old)
        for v in heads:
            selected,selection=(accepted,a) if v=='GAIN' or v=='GAIN_NO_BORROW' else choose(a['candidates'],delta,True)
            result[v]=(selected,dict(selection,version=method.VERSION,head=v,credibility=cred,relative_residual=qa['relative_residual'],iterations=qa['iterations'],proposal_norm=float(g.norm()),preselection_delta_norm=float(delta.norm()),native_energy_cap=cap,native_ray_scale=ray_scale,accepted_native_energy=float((selected*H(selected)).sum()),exact_identity=not bool(selected.any()),unchanged_native_Q_R_pairs=True,new_columns_fixed=True,common_old_fixed=True))
    neutral,energy=method.neutral_match(result['GAIN'][0],H,old);actual=neutral.float().double()
    result['NEUTRAL_MATCHED']=(neutral,dict(version=method.VERSION,head='NEUTRAL_MATCHED',matched_native_energy=energy,rounded_native_energy=float((actual*H(actual)).sum()),reference_chosen_scale=result['GAIN'][1].get('chosen_scale'),risk_gate_used=False,exact_identity=not bool(neutral.any()),relative_residual=0.,model_utility=method.utility(main_models,Wref,actual,old,1.)))
    for delta,audit in result.values():
        actual=delta.float().double();bound=1e-8+torch.finfo(torch.float32).eps*actual[:,:old].abs().sum(1)
        if (actual[:,:old].sum(1).abs()>bound).any():raise ValueError('Quantized common component exceeds rounding bound')
        audit['quantized_old_sum_max_abs']=float(actual[:,:old].sum(1).abs().max());audit['float32_rounding_bound_max']=float(bound.max())
        audit['ideal_common_old_fixed']=True;audit['common_old_fixed']=bool(not actual[:,:old].sum(1).any())
    return result
