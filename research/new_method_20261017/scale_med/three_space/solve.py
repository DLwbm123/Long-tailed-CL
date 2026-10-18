"""Bounded sufficient-witness LP followed by unrestricted quota indicators."""
import os,json,time,traceback,resource,warnings
from pathlib import Path
import numpy as np
from verify import verify

def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)

def main(cfg):
    start=time.monotonic();deadline=start+300;root=Path(cfg['output']);space=os.environ['SPACE'];seal=json.loads((root/'QUOTA_SEAL.json').read_text())
    assert (root/'DISPATCH_SEAL.json').exists() and not (root/(space+'.result.json')).exists()
    X=np.load(root/(space+'.features.private.npy'));y=np.load(root/'labels.private.npy');selected=np.load(root/'selected.private.npy');k=seal['quotas'];d=X.shape[1];history={}
    def progress(stage,**kw):write(root/(space+'.progress.json'),dict(stage=stage,elapsed_seconds=time.monotonic()-start,**kw))
    def finish(state,**kw):
        result=dict(space=space,classification=state,history=history,wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,full_family_infeasible_proven=False,**kw)
        write(root/(space+'.result.json'),result);return result
    def witness(W,origin):
        p=root/(space+'.witness.private.npy');np.save(p,W)
        # Reload full original matrices; selected masks and solver a_i are not inputs.
        checked=verify(np.load(root/(space+'.features.private.npy')),np.load(root/'labels.private.npy'),np.load(p),k)
        history[origin+'_verification']=checked
        if checked['meets_all']:return finish('FEASIBLE_WITNESS',witness_origin=origin,verification=checked)
    progress('selected_example_LP_build')
    from scipy import sparse
    from scipy.optimize import linprog,OptimizeWarning
    lp_deadline=min(deadline,start+90);blocks=[];bounds=[]
    for c in range(6):
        ids=np.flatnonzero(selected&(y==c));xc=sparse.csr_matrix(X[ids]);zero=sparse.csr_matrix((len(ids),d))
        for j in range(6):
            if j==c:continue
            if time.monotonic()>lp_deadline:break
            cols=[zero]*5
            if c>0:cols[c-1]=-xc
            if j>0:cols[j-1]=xc
            blocks.append(sparse.hstack(cols,format='csr'));bounds.append(-np.ones(len(ids)))
        if time.monotonic()>lp_deadline:break
    if time.monotonic()<lp_deadline:
        A=sparse.vstack(blocks,format='csc');b=np.concatenate(bounds);del blocks,bounds
        progress('selected_example_LP_solve',constraints=A.shape[0],nonzeros=A.nnz)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
            sol=linprog(np.zeros(5*d),A_ub=A,b_ub=b,bounds=(None,None),method='highs',options=dict(time_limit=max(.01,lp_deadline-time.monotonic()),threads=4,primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8))
        history['LP']=dict(status=int(sol.status),success=bool(sol.success),elapsed_seconds=time.monotonic()-start,selected_examples=int(selected.sum()),negative_result_is_not_full_infeasibility=True)
        del A,b
        if sol.x is not None:
            W=np.zeros((d,6));W[:,1:]=sol.x.reshape(5,d).T
            found=witness(W,'LP')
            if found:return found
        del sol
    else:
        history['LP']=dict(status='CONSTRUCTION_TIME_LIMIT',elapsed_seconds=time.monotonic()-start,negative_result_is_not_full_infeasibility=True);del blocks,bounds
    if time.monotonic()>=deadline:return finish('UNRESOLVED_TIME_LIMIT')
    progress('full_quota_indicator_build')
    from pyscipopt import Model,quicksum
    model=Model();model.hideOutput();model.setLogfile(str(root/(space+'.solver.private.log')))
    model.setIntParam('parallel/maxnthreads',4);model.setRealParam('limits/memory',16384.);model.setIntParam('randomization/randomseedshift',74012);model.setRealParam('numerics/feastol',1e-8)
    Wvars={0:[0.]*d}
    for c in range(1,6):Wvars[c]=[model.addVar(vtype='C',lb=None,ub=None) for _ in range(d)]
    flags=[model.addVar(vtype='B') for _ in y]
    for c in range(6):model.addCons(quicksum(flags[i] for i in np.flatnonzero(y==c))>=int(k[c]))
    for i,row in enumerate(X):
        if time.monotonic()>=deadline:
            history['MIP']=dict(status='CONSTRUCTION_TIME_LIMIT',built_rows=i,total_rows=len(y));return finish('UNRESOLVED_TIME_LIMIT')
        c=int(y[i]);nonzero=np.flatnonzero(row)
        for j in range(6):
            if j==c:continue
            expr=quicksum(float(row[q])*(Wvars[j][q]-Wvars[c][q]) for q in nonzero)
            model.addConsIndicator(expr<=(-1. if j<c else 0.),binvar=flags[i])
        if i%100==0:progress('full_quota_indicator_build',built_rows=i,total_rows=len(y))
    model.setObjective(0);model.setRealParam('limits/time',max(.01,deadline-time.monotonic()));progress('full_quota_indicator_solve',variables=model.getNVars(),constraints=model.getNConss())
    model.optimize();status=str(model.getStatus());history['MIP']=dict(status=status,solutions=model.getNSols(),nodes=model.getNNodes(),solve_seconds=model.getSolvingTime())
    if model.getNSols():
        sol=model.getBestSol();W=np.zeros((d,6))
        for c in range(1,6):W[:,c]=[model.getSolVal(sol,v) for v in Wvars[c]]
        found=witness(W,'MIP')
        if found:return found
        return finish('UNRESOLVED_NUMERICAL_WITNESS')
    if status=='infeasible':return finish('UNRESOLVED_UNCERTIFIED_INFEASIBLE')
    return finish('UNRESOLVED_TIME_LIMIT' if status=='timelimit' else 'UNRESOLVED_NO_WITNESS')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:print(json.dumps(main(cfg)),flush=True)
    except Exception:
        root=Path(cfg['output']);space=os.environ['SPACE'];write(root/(space+'.failure.private.json'),dict(traceback=traceback.format_exc()));write(root/(space+'.result.json'),dict(space=space,classification='UNRESOLVED_ENGINEERING_FAILURE',full_family_infeasible_proven=False));raise
