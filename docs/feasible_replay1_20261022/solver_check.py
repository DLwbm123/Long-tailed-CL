"""Bounded numerical repair tests: analytic degeneracy and saved real training gradients.
Does not re-create the missing failed QP; zero images/encoding/optimization/validation.
"""
import os,json,time,itertools,resource
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import frozen_core
old_project=frozen_core.project
import core

def enumerate_projection(g,G):
    best=None
    for k in range(len(G)+1):
        for active in itertools.combinations(range(len(G)),k):
            H=G[list(active)]
            lam=np.linalg.lstsq(H@H.T,-H@g,rcond=None)[0] if k else np.array([])
            d=g+H.T@lam if k else g.copy()
            if (k and lam.min()<-1e-8) or np.min(G@d)<-1e-8:continue
            value=float(np.dot(d-g,d-g))
            if best is None or value<best[0]:best=(value,d)
    assert best is not None
    return best[1]

def main():
    start=time.time();threadpool_limits(4);torch.set_num_threads(4)
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']).parent
    tests=[('orthant',np.array([-2.,-1.,3.]),np.array([[1.,0.,0.],[0.,1.,0.]])),
           ('duplicate',np.array([-2.,1.]),np.array([[1.,0.],[1.,0.]])),
           ('nearly_opposed',np.array([1.,-1.]),np.array([[1.,0.],[-1.,.01]]))]
    receipts=[]
    for name,g,G in tests:
        ref=enumerate_projection(g,G)
        d,r=core.project(torch.tensor(g),torch.tensor(G),cfg)
        err=float(np.max(abs(d.numpy()-ref)));assert err<2e-7
        old_failed=False
        try:old_project(torch.tensor(g),torch.tensor(G),cfg)
        except AssertionError:old_failed=True
        receipts.append(dict(case=name,independent_active_subset_error=err,old_frozen_solver_failed=old_failed,repair=r))
    assert receipts[-1]['old_frozen_solver_failed']
    pack=torch.load(Path(cfg['resume_source'])/'stage_1.private.pt',map_location='cpu',weights_only=False)
    witness=pack['transition_probe'];g=witness['gradient'];G=witness['constraint_gradients']
    old,oldreceipt=old_project(g,G,cfg);new,receipt=core.project(g,G,cfg)
    delta=float((old-new).abs().max());assert delta<=2e-5
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    dg,gr=core.project(g.cuda(),G.cuda(),cfg);assert float((dg.cpu()-new).abs().max())<2e-5
    normalized=G.cuda()/G.cuda().norm(dim=1,keepdim=True)
    active=-normalized[0]
    projected,ar=core.project(active,G.cuda(),cfg)
    assert float(projected.norm())<2e-6 and ar['correction_norm']>.9
    out=dict(state='PASS',analytic_cases=receipts,saved_real_stage1_gradient_QP=dict(old_new_direction_max_error=delta,old_receipt=oldreceipt,new_receipt=receipt,GPU_receipt=gr,active_GPU_case=ar),failed_original_QP_reconstructed=False,new_encoding=0,optimization=0,validation_scoring=0,synthetic_projection_calls=6,saved_real_projection_calls=4,seconds=time.time()-start,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (root/'SOLVER_CHECK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
