"""Fixed final-state diagnosis, with no fitted hyperparameter or data loader."""
import torch
import torch.nn.functional as F
import concm_statistical as native
import core1_competition as competition
VERSION='concm_final_mechanism_diagnostic_v1'
VARIANTS=('MPC','REG_ZERO','IDENTITY','ETF_FULL','ETF_DISC','MPC_DSM')
def validate_parameters(p):
    native.validate_parameters(p)
    if p['diagnostic_version']!=VERSION or p['diagnostic_heads']!=list(VARIANTS) or p['source_stage_checkpoints']!=4 or p['normal_GPU_processes']!=8 or p['diagnostic_train_images']:
        raise ValueError('Diagnostic protocol differs')
def norms(w):
    common=w.mean(1,keepdim=True).expand_as(w);disc=w-common
    total=float(w.square().sum());share=float(common.square().sum())/max(total,1e-30)
    return dict(full_norm=float(w.norm()),discriminative_norm=float(disc.norm()),common_norm=float(common.norm()),common_energy_fraction=share,orthogonal_energy_error=abs(total-float(common.square().sum()+disc.square().sum()))/max(total,1e-30))
def donors(bank,seen):
    mu=bank['mu'];n=mu.new_tensor(bank['n']);q=n/(n+16.)
    sim=F.normalize(mu,dim=1)@F.normalize(mu,dim=1).T
    logits=16*sim+q.log()[None,:];logits.fill_diagonal_(-torch.inf);w=logits.softmax(1)
    delta=(.4*16/(n+16.))[:,None]*(w@mu-mu)
    rows=[]
    for c in range(len(seen)):
        for j in range(len(seen)):
            if c==j:continue
            direction=mu[j]-mu[c];denom=float(delta[c].norm()*direction.norm())
            rows.append(dict(label=seen[c],donor=seen[j],donor_n=int(n[j]),weight=float(w[c,j]),cosine=float(sim[c,j]),shift_alignment=None if denom==0 else float(delta[c]@direction)/denom))
    return rows

def fit(bank,pairs,R,reference,variant):
    if variant not in VARIANTS:raise ValueError('Unknown diagnosis')
    gamma=.1*float(torch.trace(bank['Q'].mean(0)+.001*R)/reference.shape[0])
    if variant in ('MPC','MPC_DSM'):
        w,a=native.fit(bank,pairs,R,reference,variant)
    else:
        target,structure=native.matching_anchors(bank['mu'],reference)
        if target is None:raise ValueError('ETF rank insufficient; no fallback')
        if variant=='REG_ZERO':target=torch.zeros_like(reference)
        elif variant=='IDENTITY':target=reference
        elif variant=='ETF_DISC':target=target*(reference-reference.mean(1,keepdim=True)).norm()/reference.norm()
        mass=reference.new_full((len(bank['n']),),1/len(bank['n']))
        w,solve=competition.solve(bank['Q']/len(bank['n']),bank['mu']/len(bank['n']),mass,pairs,regularizer=R,previous=target,proximal=gamma)
        a=dict(solver=solve,structure=structure,structure_coefficient=gamma,target_norm=float(target.norm()))
        if variant=='IDENTITY' and float((w-reference).norm()/reference.norm())>1e-7:raise ArithmeticError('Identity target changed optimum')
    if a['solver']['relative_residual']>1e-8:raise ArithmeticError('Invalid solve')
    a.update(version=VERSION,variant=variant,norms=norms(w),reference_norms=norms(reference),relative_head_change=float((w-reference).norm()/reference.norm()))
    return w,a
