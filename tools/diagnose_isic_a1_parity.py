"""Bounded A1 engineering forensics on the predeclared 32 images per parent.

No classifier scoring, precision changes, new feature cache, or gate relaxation.
Called only after the original parity gate failed; initial evidence is preserved.
"""
import gc
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from isic_a1_attribution import Run,ROOT,FIXED
from report_locked_holdout_r1 import read,write


def comparison(actual,reference):
    difference=np.abs(actual-reference);limit=1e-5+1e-5*np.abs(reference)
    return dict(pass_tolerance=bool(np.all(difference<=limit)),max_abs=float(difference.max()),
                relative_l2=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),np.finfo(float).tiny)),
                outside_tolerance_elements=int(np.sum(difference>limit)),elements=actual.size,
                max_error_to_allowed_ratio=float(np.max(difference/limit)))


def parent(r,e):
    from run_medical_v2 import network_hash
    net,line=r.restore(e);seed=e['seed'];base=r.orders[seed][:4];xs=[];ixs={};capture={};context={'branch':None};hooks=[]
    for split in ('train','val'):
        ix=np.flatnonzero(np.isin(r.data[split][1],base))[:16];ixs[split]=ix;xs.append(r.images_at(split,ix))
    x=torch.cat(xs);cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state();np_rng=np.random.get_state();py_rng=random.getstate()
    counted=net.backbone.forward_features
    def tagged(x,*args,**kwargs):
        context['branch']='few' if kwargs.get('few',0)==1 else 'main'
        return counted(x,*args,**kwargs)
    net.backbone.forward_features=tagged
    def hook(name,branch=None):
        def collect(module,args,value):
            if isinstance(value,dict):value=value['pre_logits']
            value=value[:,0] if value.ndim==3 else value
            capture[(branch or context['branch'])+'/'+name]=value.detach().cpu().numpy()
        return collect
    for name,mod in [('patch_embed',net.original_backbone.model.patch_embed)]+[(f'block_{i:02}',m) for i,m in enumerate(net.original_backbone.model.blocks)]+[('norm',net.original_backbone.model.norm)]:
        hooks.append(mod.register_forward_hook(hook(name,'original')))
    for name,mod in [('patch_embed',net.backbone.patch_embed)]+[(f'block_{i:02}',m) for i,m in enumerate(net.backbone.blocks)]+[('norm',net.backbone.norm)]:
        hooks.append(mod.register_forward_hook(hook(name)))
    singles=[];layers=[]
    for i in range(32):
        capture.clear();singles.append(r.forward(net,x[i:i+1]));layers.append(dict(capture))
    reference={k:np.concatenate([o[k] for o in singles]) for k in singles[0]}
    reference_layers={k:np.concatenate([o[k] for o in layers]) for k in layers[0]}
    r.pointwise(net,True);capture.clear();point=r.forward(net,x);point_layers=dict(capture)
    layer_rows=[dict(layer=k,**comparison(point_layers[k],reference_layers[k])) for k in reference_layers]
    cases=[]
    layouts=[('batch32',np.arange(32)),('reversed32',np.arange(31,-1,-1)),
             ('train_peers32',np.r_[np.arange(16),np.arange(16)]),('val_peers32',np.r_[np.arange(16,32),np.arange(16,32)]),
             ('batch48',np.r_[np.arange(32),np.arange(16)]),('pointwise_B1',np.array([0]))]
    for name,layout in layouts:
        o=r.forward(net,x[layout]);cases.append(dict(case=name,batch=len(layout),
            route_ID_mismatches={k:int(np.sum(o[k]!=reference[k][layout])) for k in ('prompt_idx','prompt_idx_few')},
            features={k:comparison(o[k],reference[k][layout]) for k in ('pre_logits','pre_logits_few')}))
    repeat=r.forward(net,x)
    exact_repeat=all(np.array_equal(point[k],repeat[k]) for k in point)
    r.pointwise(net,False);native=r.forward(net,x[layouts[4][1]])
    native_mismatch={k:int(np.sum(native[k]!=reference[k][layouts[4][1]])) for k in ('prompt_idx','prompt_idx_few')}
    r.pointwise(net,True)
    with torch.inference_mode():a=torch.nn.functional.normalize(net.original_backbone(xs[0])['pre_logits'],dim=1).cpu().numpy()
    cache_comparison=comparison(a,r.data['train'][0][ixs['train']])
    unchanged=network_hash(net)==line['network_sha256'];assert unchanged
    rng_unchanged=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()) and all(np.array_equal(a,b) for a,b in zip(np_rng,np.random.get_state())) and py_rng==random.getstate()
    assert rng_unchanged and all(not m.training for m in net.modules()) and not any(p.requires_grad for p in net.parameters())
    assert all(p.dtype==torch.float32 for p in net.parameters())
    for h in hooks:h.remove()
    net.backbone.forward_features=counted
    result=dict(parent_id=seed,parent_lock=line,cases=cases,layerwise_CLS_comparison=layer_rows,
                repeated_pointwise_batch_bitwise_equal=exact_repeat,native_batch48_route_mismatches=native_mismatch,
                original_backbone_A_cache=cache_comparison,parameters_buffers_unchanged=unchanged,rng_unchanged=rng_unchanged,
                precision=dict(parameter_dtype='float32',autocast_enabled=torch.is_autocast_enabled(),TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
                               float32_matmul_precision=torch.get_float32_matmul_precision()),
                gate_pass=all(c['features'][k]['pass_tolerance'] and all(v==0 for v in c['route_ID_mismatches'].values()) for c in cases for k in ('pre_logits','pre_logits_few')))
    r.resource_check();return result


def main():
    cfg=read(os.environ['P17_CONFIG']);r=Run(cfg)
    assert read(r.pub/'ASSET_AUDIT.json')['parent_file_hashes_verified']==3
    # Reuse the completed file verification of this same protected asset set.
    # Full network tensor hashes are still checked on every restoration.
    r1=read(cfg['r1_runtime']);r.v2=read(r1['v2_runtime']);r.images=Path(r1['images']);r.cache=Path(r1['v3_complete_output'])/'p1/A'
    r.entries=[e for e in r1['entries'] if e['method']=='C' and e['session']==0]
    r.expected={x['seed']:x for x in read(ROOT/'docs/isic_locked_holdout_r1/MODEL_LINEAGE.json')['models'] if x['method']=='C' and x['stage']==0}
    r.verified_parents={e['seed'] for e in r.entries}
    for split in ('train','val'):
        with np.load(r.cache/(split+'.npz'),allow_pickle=False) as z:r.data[split]=(z['z'].copy(),z['y'].copy(),read(r.cache/(split+'_associations.json')))
    results=[]
    with threadpool_limits(limits=4):
        r.setup_torch()
        for e in r.entries:
            results.append(parent(r,e));gc.collect();torch.cuda.empty_cache()
            write(r.pub/'ROUTING_PARITY_FORENSICS.json',dict(status='DIAGNOSIS_ONLY',parents=results,thresholds_changed=False,extra_formal_feature_rows=0,new_S_candidate_scores=0))
        write(r.pub/'FORENSICS_ACCESS_AUDIT.json',dict(**FIXED,**r.access,encoder_forward_calls=sum(r.access['module_batch_calls'].values()),
              source_file_hash_verification='P0 ASSET_AUDIT; same protected parents',formal_feature_rows=0))
        write(r.pub/'FORENSICS_RESOURCE_REPORT.json',r.resource_check())
    print('FORENSICS_COMPLETE',[(p['parent_id'],p['gate_pass']) for p in results],flush=True)


if __name__=='__main__':main()
