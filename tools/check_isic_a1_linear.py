"""Engineering-only batch-preserving linear lowering; no classifier scores.

Evaluate XW^T+b via batched GEMM at B>1, retaining the per-image token
matrix dimensions. B=1 remains the untouched native reference. No loop
splits the batch, no parameter/dtype/routing changes, no epsilon changes.
The original hard parity gate decides release, without looking at scores.
"""
import gc
import os
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diagnose_isic_a1_parity import parent
from isic_a1_attribution import Run, ROOT, FIXED
from report_locked_holdout_r1 import read, write


def install_batched_linear(net):
    for module in net.modules():
        if not isinstance(module,torch.nn.Linear):continue
        native=module.forward
        def forward(x,module=module,native=native):
            if x.ndim!=3 or x.shape[0]==1:return native(x)
            assert x.dtype==module.weight.dtype==torch.float32
            B,N,_=x.shape;weight=module.weight.t().expand(B,-1,-1)
            if module.bias is None:return torch.bmm(x,weight)
            return torch.baddbmm(module.bias.reshape(1,1,-1).expand(B,N,-1),x,weight)
        module.forward=forward


def main():
    assert os.environ.get('DISABLE_ADDMM_CUDA_LT') is None
    cfg=read(os.environ['P17_CONFIG']);r=Run(cfg)
    previous=r.pub
    assert read(previous/'ASSET_AUDIT.json')['parent_file_hashes_verified']==3
    r.out=r.root/'oversight'/'linear_lowering_01';r.pub=r.out/'public';r.private=r.out/'private'
    assert not r.out.exists(),'CONTROL_ALREADY_ATTEMPTED'
    r.pub.mkdir(parents=True);r.private.mkdir()
    r1=read(cfg['r1_runtime']);r.v2=read(r1['v2_runtime']);r.images=Path(r1['images']);r.cache=Path(r1['v3_complete_output'])/'p1/A'
    r.entries=[e for e in r1['entries'] if e['method']=='C' and e['session']==0]
    r.expected={x['seed']:x for x in read(ROOT/'docs/isic_locked_holdout_r1/MODEL_LINEAGE.json')['models'] if x['method']=='C' and x['stage']==0}
    r.verified_parents={e['seed'] for e in r.entries}
    for split in ('train','val'):
        with np.load(r.cache/(split+'.npz'),allow_pickle=False) as z:
            r.data[split]=(z['z'].copy(),z['y'].copy(),read(r.cache/(split+'_associations.json')))
    original_restore=r.restore
    def restore(e):
        net,line=original_restore(e)
        install_batched_linear(net)
        return net,line
    r.restore=restore
    results=[]
    with threadpool_limits(limits=4):
        r.setup_torch()
        try:
            for e in r.entries:
                result=parent(r,e);results.append(result);gc.collect();torch.cuda.empty_cache()
                write(r.pub/'RESULT.json',dict(status='ENGINEERING_PASS' if len(results)==3 and all(x['gate_pass'] for x in results) else 'NOT_RELEASED',
                    backend_control={'linear_3D_B_gt_1':'baddbmm; native B1 unchanged'},torch_git_version=torch.version.git_version,parents=results,
                    threshold_changed=False,precision_changed=False,batch_changed=False,S_candidate_scores_generated=0))
                if not result['gate_pass']:break
        finally:
            write(r.pub/'ACCESS_AUDIT.json',dict(**FIXED,**r.access,encoder_forward_calls=sum(r.access['module_batch_calls'].values()),
                encoder_image_forward_rows=sum(r.access['module_image_rows'].values()),formal_feature_rows=0))
            write(r.pub/'RESOURCE_REPORT.json',r.resource_check(enforce=False))
    print('CONTROL_COMPLETE',[(x['parent_id'],x['gate_pass']) for x in results],flush=True)


if __name__=='__main__':main()
