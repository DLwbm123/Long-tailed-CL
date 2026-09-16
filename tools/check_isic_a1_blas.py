"""One bounded FP32 backend control after A1's batch parity failure.

Uses the original fixed probe and unchanged gate; never scores a classifier.
DISABLE_ADDMM_CUDA_LT is a process-local PyTorch backend switch, not a dtype,
batch, routing, weight, or model formula change. A failed parent stops this
control immediately, since all three must pass before any formal release.
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


def main():
    assert os.environ.get('DISABLE_ADDMM_CUDA_LT') == '1'
    cfg=read(os.environ['P17_CONFIG']);r=Run(cfg)
    previous=r.pub
    assert read(previous/'ASSET_AUDIT.json')['parent_file_hashes_verified']==3
    r.out=r.root/'oversight'/'blas_control_01';r.pub=r.out/'public';r.private=r.out/'private'
    assert not r.out.exists(),'CONTROL_ALREADY_ATTEMPTED'
    r.pub.mkdir(parents=True);r.private.mkdir()
    r1=read(cfg['r1_runtime']);r.v2=read(r1['v2_runtime']);r.images=Path(r1['images']);r.cache=Path(r1['v3_complete_output'])/'p1/A'
    r.entries=[e for e in r1['entries'] if e['method']=='C' and e['session']==0]
    r.expected={x['seed']:x for x in read(ROOT/'docs/isic_locked_holdout_r1/MODEL_LINEAGE.json')['models'] if x['method']=='C' and x['stage']==0}
    r.verified_parents={e['seed'] for e in r.entries}
    for split in ('train','val'):
        with np.load(r.cache/(split+'.npz'),allow_pickle=False) as z:
            r.data[split]=(z['z'].copy(),z['y'].copy(),read(r.cache/(split+'_associations.json')))
    results=[]
    with threadpool_limits(limits=4):
        r.setup_torch()
        try:
            for e in r.entries:
                result=parent(r,e);results.append(result);gc.collect();torch.cuda.empty_cache()
                write(r.pub/'RESULT.json',dict(status='ENGINEERING_PASS' if len(results)==3 and all(x['gate_pass'] for x in results) else 'NOT_RELEASED',
                    backend_control={'DISABLE_ADDMM_CUDA_LT':'1'},torch_git_version=torch.version.git_version,parents=results,
                    threshold_changed=False,precision_changed=False,batch_changed=False,S_candidate_scores_generated=0))
                if not result['gate_pass']:break
        finally:
            write(r.pub/'ACCESS_AUDIT.json',dict(**FIXED,**r.access,encoder_forward_calls=sum(r.access['module_batch_calls'].values()),
                encoder_image_forward_rows=sum(r.access['module_image_rows'].values()),formal_feature_rows=0))
            write(r.pub/'RESOURCE_REPORT.json',r.resource_check(enforce=False))
    print('CONTROL_COMPLETE',[(x['parent_id'],x['gate_pass']) for x in results],flush=True)


if __name__=='__main__':main()
