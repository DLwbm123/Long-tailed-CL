"""Visual memory completion and maximum-matching analytic head; nonofficial."""
import math
import torch
import torch.nn.functional as F
import core1_competition as competition

VERSION = 'visual_support_completion_etf_head_v1'
VARIANTS = ('MPC', 'DSM', 'MPC_DSM')


def validate_parameters(p):
    if (p['version'] != VERSION or p['variants'] != list(VARIANTS)
            or p['support_pseudocount'] != 16. or p['completion_max_weight'] != .4
            or p['visual_temperature'] != 16. or p['structure_relative_curvature'] != .1
            or p['optimizer_updates'] != 0 or p['test_accessed'] or p['validation_retuning']
            or p['train_images_allowed'] or p['gpus'] != [0, 1]
            or p['primary_control'] != 'local_transport' or p['automatic_retry']
            or p['seed'] != 74002 or p['batch_size'] != 64 or p['workers'] != 0
            or p['status'] != 'FROZEN_EVALUATION_ONLY' or p['training_feedback']):
        raise ValueError('Frozen statistical design differs')


def complete(bank):
    """Counts are a support heuristic, never a measured old-drift confidence."""
    mu, Q = bank['mu'], bank['Q']; k, d = mu.shape
    if mu.dtype != torch.float64 or Q.dtype != torch.float64 or Q.shape != (k,d,d):
        raise ValueError('Float64 coherent moments required')
    counts = mu.new_tensor(bank['n'])
    if len(counts) != k or (counts <= 0).any() or not torch.isfinite(Q).all():
        raise ValueError('Invalid class counts or moments')
    confidence = counts/(counts+16.)
    similarity = F.normalize(mu,dim=1) @ F.normalize(mu,dim=1).T
    logits = 16.*similarity + confidence.log()[None,:]
    logits.fill_diagonal_(-torch.inf)
    if k < 2: raise ValueError('At least two observed classes required')
    weights = logits.softmax(1)
    eta = .4*16./(counts+16.)
    donor = weights @ mu
    shifted = mu + eta[:,None]*(donor-mu)
    delta = shifted-mu
    # Exact second-moment update for the assumed classwise translation.
    second = Q + mu[:,:,None]*delta[:,None,:] + delta[:,:,None]*mu[:,None,:] + delta[:,:,None]*delta[:,None,:]
    out=dict(mu=shifted,Q=second,n=list(bank['n']),components=[dict(c,center=c['center']+delta[c['label']]) for c in bank['components']])
    audit=dict(eta=eta.tolist(),shift_norm=delta.norm(dim=1).tolist(),
        donor_effective_count=(1/weights.square().sum(1)).tolist(),
        covariance_preserved_max_abs=float(((second-shifted[:,:,None]*shifted[:,None,:])-(Q-mu[:,:,None]*mu[:,None,:])).abs().max()),
        measured_old_drift=False,semantic_attributes=False,meta_learning=False)
    if audit['covariance_preserved_max_abs'] > 1e-9 or not torch.isfinite(second).all():
        raise ArithmeticError('Classwise covariance consistency failed')
    return out,audit


def matching_anchors(mu, reference):
    """Centered polar factor implements maximum matching to simplex geometry."""
    k,d=mu.shape
    if not 1 < k < d: raise ValueError('Simplex dimensions invalid')
    C=torch.eye(k,dtype=mu.dtype,device=mu.device)-mu.new_full((k,k),1/k)
    initial=F.normalize(mu,dim=1).T
    U,s,Vh=torch.linalg.svd(initial @ C,full_matrices=False)
    tolerance=torch.finfo(mu.dtype).eps*max(d,k)*float(s.max())
    rank=int((s>tolerance).sum())
    if rank < k-1:
        return None,dict(rank=rank,skipped=True,reason='insufficient_centered_rank')
    anchors=math.sqrt(k/(k-1))*(U @ Vh @ C)
    wanted=k/(k-1)*C
    error=float((anchors.T@anchors-wanted).abs().max())
    if error > 1e-9: raise ArithmeticError('ETF geometry failed')
    scale=float(reference.norm()/anchors.norm().clamp_min(1e-30))
    return scale*anchors,dict(rank=rank,skipped=False,ETF_gram_max_abs_error=error,
        matching_inner_product=float((initial*anchors).sum()),target_scale=scale)


def fit(bank,pairs,R,reference,variant):
    if variant not in VARIANTS:raise ValueError('Unknown variant')
    if not all(torch.isfinite(v).all() for v in (bank['mu'],bank['Q'],pairs,R,reference)):
        raise ValueError('Nonfinite source statistics')
    k,d=bank['mu'].shape
    revised,ca=complete(bank) if variant!='DSM' else (bank,dict(enabled=False))
    gamma=0.;target=None;sa=dict(enabled=False)
    if variant!='MPC':
        target,sa=matching_anchors(revised['mu'],reference)
        if target is not None:
            base=revised['Q'].sum(0)/k+.001*R
            gamma=.1*float(torch.trace(base)/d)
    mass=reference.new_full((k,),1/k)
    W,solve=competition.solve(revised['Q']/k,revised['mu']/k,mass,pairs,
        regularizer=R,previous=target,proximal=gamma)
    if solve['relative_residual']>1e-8:raise ArithmeticError('Analytic solve failed')
    return W,dict(version=VERSION,variant=variant,completion=ca,structure=sa,
        structure_coefficient=gamma,solver=solve,head_delta_norm=float((W-reference).norm()),
        prior_and_pair_weights_reused=True,optimizer_updates=0,hard_new_class_guard=False)
