"""CPU verifier: independent metric, class competition and explicit-difference CG."""
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import LinearOperator,cg
from common import settings,write,report


def solve(q,mu,components):
    k,d=mu.shape;P=np.stack([np.asarray(c['center'])/max(np.linalg.norm(c['center']),1e-12)*float(c['mass'])**.5 for c in components],axis=1)
    # Same mathematical R, independent NumPy implementation.
    R=np.eye(d)-P@np.linalg.solve(np.eye(P.shape[1])+P.T@P,P.T);R=(R+R.T)/2
    base=q.sum(0)/k+.001*R;factor=cho_factor(base);ref=cho_solve(factor,mu.T/k)
    hard=np.zeros((k,k))
    for c in components:
        s=np.asarray(c['center'])@ref;j=int(c['label']);hard[j]+=float(c['mass'])*np.maximum(1-s[j]+s,0)**2
    np.fill_diagonal(hard,0);uniform=(1-np.eye(k))/(k-1);s=hard.sum(1,keepdims=True)
    a=.5*uniform+.5*np.divide(hard,s,out=uniform.copy(),where=s>1e-12)
    rhs=mu.T/k
    for c in range(k):
        for j in range(k):
            if a[c,j]:rhs[:,c]+=.5*a[c,j]*mu[c]/k;rhs[:,j]-=.5*a[c,j]*mu[c]/k
    def action(flat):
        w=flat.reshape(d,k);out=base@w
        for c in range(k):
            qw=q[c]@w/k;dif=qw[:,c,None]-qw
            out[:,c]+=.5*(dif*a[c]).sum(1)
            out-=.5*dif*a[c]
        return out.ravel()
    op=LinearOperator((d*k,d*k),matvec=action,dtype=np.float64)
    pre=LinearOperator(op.shape,matvec=lambda v:cho_solve(factor,v.reshape(d,k)).ravel(),dtype=np.float64)
    w,info=cg(op,rhs.ravel(),M=pre,x0=ref.ravel(),rtol=1e-10,atol=0,maxiter=256)
    residual=np.linalg.norm(action(w)-rhs.ravel())/max(np.linalg.norm(rhs),1e-30)
    assert info==0 and residual<=1e-8,(info,residual)
    return w.reshape(d,k),residual


def self_check():
    # Direct independent sample quadratic solves the same signed-margin target.
    rng=np.random.default_rng(35);x=rng.normal(size=(12,3));y=np.repeat(np.arange(3),4)
    mu=np.stack([x[y==c].mean(0) for c in range(3)]);q=np.stack([x[y==c].T@x[y==c]/4 for c in range(3)])
    comp=[dict(label=c,center=mu[c],mass=1.) for c in range(3)]
    w,res=solve(q,mu,comp)
    assert np.isfinite(w).all() and res<=1e-8
    # Check the operator's explicit class difference against a dense construction via finite basis.
    P=(mu/np.maximum(np.linalg.norm(mu,axis=1,keepdims=True),1e-12)).T;R=np.eye(3)-P@np.linalg.solve(np.eye(3)+P.T@P,P.T)
    ref=np.linalg.solve(q.sum(0)/3+.001*R,mu.T/3);hard=np.stack([np.maximum(1-mu[c]@ref[:,c]+mu[c]@ref,0)**2 for c in range(3)])
    np.fill_diagonal(hard,0);a=.5*(1-np.eye(3))/2+.5*hard/hard.sum(1,keepdims=True)
    H=np.kron(q.sum(0)/3+.001*R,np.eye(3));rhs=mu.T/3
    for c in range(3):
        for j in range(3):
            e=np.eye(3)[c]-np.eye(3)[j];H+=.5*a[c,j]*np.kron(q[c]/3,np.outer(e,e));rhs+=.5*a[c,j]*np.outer(mu[c]/3,e)
    direct=np.linalg.solve(H,rhs.ravel()).reshape(3,3)
    error=float(abs(direct-w).max());assert error<1e-8
    return error


def main():
    start=time.time();cfg,out=settings();torch.set_num_threads(4);test=self_check()
    bank=torch.load(out/'bank.private.pt',map_location='cpu',weights_only=False)
    heads=np.load(out/'heads.private.npz');query=np.load(out/'query.private.npz');seal=json.loads((out/'FIT_SEAL.json').read_text())
    assert seal['validation_encodes']==0 and seal['encoder_unchanged'] and seal['old_bank_unchanged']
    mu=bank['mu'].numpy();q=bank['Q'].numpy();records=[];saved=[]
    for task,k in enumerate([10,12,14,16,18,20],1):
        assert time.time()<cfg['hard_deadline_epoch']-60
        components=[{key:(v.numpy() if torch.is_tensor(v) else v) for key,v in c.items()} for c in bank['components'] if c['label']<k]
        w,res=solve(q[:k],mu[:k],components);stored=heads[f'stage{task}'];rel=float(np.linalg.norm(w-stored)/np.linalg.norm(stored));score=float(abs(query['x']@(w-stored)).max())
        assert rel<=2e-5 and score<=2e-5,(task,rel,score)
        saved.append(stored);records.append(dict(task=task,relative_head_error=rel,max_double_score_error=score,residual=res))
    actual=report(query['x'],query['labels'],saved,cfg['order'])
    parity=json.loads((out/'TASK1_PARITY.json').read_text());parity_pass=parity['actual']==parity['expected']
    if (out/'RESULTS.json').exists():assert actual==json.loads((out/'RESULTS.json').read_text())
    else:
        assert not parity_pass and json.loads((out/'status.private.json').read_text())['error']=='Task1 validation parity failed'
        write(out/'RESULTS_UNQUALIFIED.json',actual)
    # Manifests are metadata: independent coverage and patient separation check.
    import csv
    sets=[]
    for split in ['train','val']:
        with (Path(cfg['manifest'])/(split+'.csv')).open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==(68058 if split=='train' else 300)
        sets.append({r['identity_component'] for r in rows})
        if split=='train':assert bank['n']==[sum(int(r['original_label'])==c for r in rows) for c in cfg['order']]
    assert not sets[0]&sets[1]
    costs=json.loads((out/'COSTS.json').read_text());assert costs['qualification_training_encodes']==64 and costs['scientific_training_encodes']==7929 and costs['validation_encodes']==300 and costs['encoder_optimizer_updates']==0
    write(out/'VERIFICATION.json',dict(status='PASS_NUMERICAL_ONLY_TASK1_PARITY_FAILED' if not parity_pass else 'PASS',scientific_acceptance=parity_pass,task1_parity_pass=parity_pass,stages=records,synthetic_dense_error=test,patient_intersection=0,query_reencodes=0,verification_seconds=time.time()-start,scope='NIH_A_single_F1_control'))

if __name__=='__main__':main()
