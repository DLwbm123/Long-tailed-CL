"""Bounded support-only indicator feasibility and independent witness validation."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from run_offline_information_ab import validate
from run_full_support_linear_ce import check_head
from run_real_query_node import load, save


def model(z, y, quotas, seconds, logfile=None):
    from pyscipopt import Model, quicksum
    m = Model('support_feasibility')
    m.setRealParam('limits/time', seconds)
    m.setRealParam('limits/memory', 20000.)
    m.setIntParam('parallel/maxnthreads', 2)
    m.setRealParam('numerics/feastol', 1e-9)
    m.setIntParam('randomization/randomseedshift', 0)
    if logfile:
        m.setLogfile(str(logfile))
    else:
        m.hideOutput()
    n, d = z.shape
    k = len(quotas)
    w = [[m.addVar(lb=None, ub=None, name=f'w{c}_{h}') for h in range(d)] for c in range(k-1)]
    a = [m.addVar(vtype='B', name=f'a{i}') for i in range(n)]
    for c in range(k):
        m.addCons(quicksum(a[i] for i in range(n) if y[i] == c) >= int(quotas[c]))
    for i in range(n):
        scores = [quicksum(float(z[i,h])*w[c][h] for h in range(d)) for c in range(k-1)] + [0.]
        for j in range(k):
            if j != y[i]:
                # Smaller-index rivals require strict superiority; scale to unit margin.
                margin = 1. if j < y[i] else 0.
                m.addConsIndicator(scores[j]-scores[y[i]] <= -margin, binvar=a[i])
        if logfile and (i+1) % 250 == 0:
            print(json.dumps(dict(stage='BUILDING',rows=i+1,total=n)), flush=True)
    m.setObjective(0.)
    return m, w


def unit():
    z = np.ones((2,1)); y = np.array([0,1])
    m,w = model(z,y,[1,1],10.)
    m.optimize(); assert str(m.getStatus()) == 'infeasible'
    m,w = model(z,y,[1,0],10.)
    m.optimize(); assert m.getNSols() > 0
    # A weak tie is correct only for the smaller class index.
    assert np.array([[0.,0.],[0.,0.]]).argmax(1).tolist() == [0,0]
    return dict(contradictory_labels_rejected=True,optional_correct_set_allowed=True,tie_rule_checked=True)


def inputs(root,cfg):
    bases = {k:Path(v).resolve() for k,v in cfg['inputs'].items()}
    allowed = {'prior':{'ISIC.S_old.features.private.pt','ISIC.S_new.features.private.pt'},
               'full':{'ISIC.ce.private.pt'},'ce':{'ISIC.ce.private.pt'}}
    def guard(event,args):
        if event != 'open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root.parent) and not p.is_relative_to(root):
            assert any(p.parent == b and p.name in allowed[k] for k,b in bases.items()), 'Input outside frozen support/head allowlist'
    sys.addaudithook(guard)
    for filename in ('ISIC.Qfit.features.private.pt','ISIC.Qgate.features.private.pt','ISIC.E.features.private.pt'):
        try:
            (bases['prior']/filename).open('rb')
        except AssertionError:
            pass
        else:
            raise AssertionError('Input guard failed')
    heads={k:load(bases[k]/'ISIC.ce.private.pt') for k in ('full','ce')}
    for h in heads.values():
        check_head(h,6)
        assert all(torch.equal(v,b) for v,b in zip(h['model'],heads['full']['model']))
    z0,y0=validate(load(bases['prior']/'ISIC.S_old.features.private.pt'),[375,23,208,2630],0,heads['full']['model'])
    z1,y1=validate(load(bases['prior']/'ISIC.S_new.features.private.pt'),[2254,36],4,heads['full']['model'])
    z=np.concatenate((z0,z1));y=np.concatenate((y0,y1))
    assert z.shape==(5526,1536) and np.isfinite(z).all()
    expected={'ce':[136,22,66,1498,2171,36],'full':[329,23,202,2297,1810,36]}
    for k,h in heads.items():
        pred=(8*torch.from_numpy(z)@h['frame']).argmax(1).numpy()
        assert [int(((pred==y)&(y==c)).sum()) for c in range(6)]==expected[k]
    return z,y


def verify(root,z,y,quotas):
    w=np.load(root/'candidate.private.npy',allow_pickle=False)
    assert w.shape==(1536,6) and np.isfinite(w).all()
    scores=8*torch.from_numpy(z)@torch.from_numpy(w)
    pred=scores.argmax(1).numpy()
    # Independent extended-precision implementation; no indicator variables are loaded.
    ld=(z.astype(np.longdouble)@w.astype(np.longdouble))*8
    pred2=ld.argmax(1)
    correct=[int(((pred==y)&(y==c)).sum()) for c in range(6)]
    correct2=[int(((pred2==y)&(y==c)).sum()) for c in range(6)]
    stable=np.ones(len(y),dtype=bool)
    for j in range(6):
        gaps=ld[np.arange(len(y)),y]-ld[:,j]
        bound=np.longdouble(1e-10)*(1+np.abs(ld[np.arange(len(y)),y])+np.abs(ld[:,j]))
        stable &= (y==j)|(gaps>bound)
    stable_counts=[int((stable&(y==c)).sum()) for c in range(6)]
    ok=all(a>=b for a,b in zip(correct,quotas)) and all(a>=b for a,b in zip(correct2,quotas))
    robust=all(a>=b for a,b in zip(stable_counts,quotas))
    return dict(status='FEASIBLE_WITNESS_VERIFIED' if ok and robust else 'NUMERICALLY_FRAGILE_CANDIDATE' if ok else 'CANDIDATE_REJECTED',
                correct=correct,extended_precision_correct=correct2,stable_correct=stable_counts,
                prediction_disagreements=int((pred!=pred2).sum()),quotas=quotas,
                raw_argmax_quotas_met=ok,robust_quotas_met=robust,
                boundary='Support-only numerical witness; no query/generalization or memory/multistage claim.')


def main():
    root=Path(os.environ['Q356_ROOT']).resolve();cfg=json.loads((root/'CONFIG.private.json').read_text())
    lock=json.loads((Path(os.environ['Q356_SOURCE'])/os.environ['Q356_PROTOCOL']).read_text())
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    role=json.loads(os.environ['Q356_JOB'])['role'];quotas=lock['quotas']
    if role=='unit':
        result=unit()
    else:
        z,y=inputs(root,cfg)
        if role=='verify':
            if not (root/'candidate.private.npy').exists():
                result=dict(status='UNRESOLVED_NO_WITNESS',reason='No candidate; no infeasibility certificate')
            else:
                result=verify(root,z,y,quotas)
        else:
            begin=time.monotonic()
            print(json.dumps(dict(stage='INPUTS_VERIFIED',samples=len(y))),flush=True)
            m,w=model(z,y,quotas,lock['solver_seconds'],root/'solver.private.log')
            remaining=min(lock['solver_seconds'],cfg['deadline_epoch']-time.time()-300)
            assert remaining>0
            m.setRealParam('limits/time',remaining)
            print(json.dumps(dict(stage='SOLVING',build_seconds=time.monotonic()-begin,remaining_seconds=remaining)),flush=True)
            m.optimize()
            result=dict(status='CANDIDATE_FOUND' if m.getNSols()>0 else 'UNRESOLVED',solver_status=str(m.getStatus()),
                        solver_seconds=m.getSolvingTime(),nodes=m.getNNodes(),solutions=m.getNSols(),
                        infeasibility_certified=False,build_and_solve_seconds=time.monotonic()-begin)
            if m.getNSols()>0:
                sol=m.getBestSol();ww=np.zeros((1536,6))
                for c in range(5):
                    for h in range(1536):ww[h,c]=m.getSolVal(sol,w[c][h])
                np.save(root/'candidate.private.npy',ww,allow_pickle=False)
    result.update(source_commit=os.environ['Q356_SHA'],GPU_used=False,query_access=False,E_access=False,role=role)
    save(root/(json.loads(os.environ['Q356_JOB'])['id']+'.json'),result)


if __name__=='__main__':
    main()
