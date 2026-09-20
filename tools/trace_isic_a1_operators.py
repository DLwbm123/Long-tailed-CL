"""Locate FP32 batch effects using identical first-block operator inputs.

Only the locked parent-1993 engineering probe is read. No source operator,
backend flag, precision, routing rule, checkpoint or feature cache is changed.
"""
import os
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from diagnose_isic_a1_parity import comparison
from isic_a1_attribution import Run, ROOT, FIXED
from report_locked_holdout_r1 import read, write


def main():
    cfg=read(os.environ['P17_CONFIG']);r=Run(cfg);prior=r.pub
    assert read(prior/'ASSET_AUDIT.json')['parent_file_hashes_verified']==3
    r.out=r.root/'oversight'/'operators_01';r.pub=r.out/'public';r.private=r.out/'private'
    assert not r.out.exists(),'CONTROL_ALREADY_ATTEMPTED'
    r.pub.mkdir(parents=True);r.private.mkdir()
    r1=read(cfg['r1_runtime']);r.v2=read(r1['v2_runtime']);r.images=Path(r1['images']);r.cache=Path(r1['v3_complete_output'])/'p1/A'
    e=next(e for e in r1['entries'] if e['method']=='C' and e['session']==0 and e['seed']==1993)
    r.expected={x['seed']:x for x in read(ROOT/'docs/isic_locked_holdout_r1/MODEL_LINEAGE.json')['models'] if x['method']=='C' and x['stage']==0}
    r.verified_parents={1993}
    for split in ('train','val'):
        with np.load(r.cache/(split+'.npz'),allow_pickle=False) as z:r.data[split]=(z['z'].copy(),z['y'].copy(),read(r.cache/(split+'_associations.json')))
    records=[];handles=[];busy=False;branch='original';operator_calls=0
    def compare(name,actual,reference):
        records.append(dict(operator=name,shape=list(actual.shape),**comparison(actual.detach().cpu().numpy(),reference.detach().cpu().numpy())))
    def check(name):
        def hook(module,args,value):
            nonlocal busy,operator_calls
            if busy:return
            busy=True
            try:
                x=args[0];assert len(x)==32
                with torch.inference_mode():
                    single=torch.cat([module(x[i:i+1]) for i in range(len(x))]);operator_calls+=len(x)
                    compare(branch+'/'+name,value,single)
                    if name=='attn' and branch!='original':
                        B,N,_=x.shape;h=module.num_heads;d=module.head_dim
                        q=module._shape(module.q_proj(x),N,B).view(B*h,N,d)
                        k=module._shape(module.k_proj(x),N,B).view(B*h,N,d)
                        v=module._shape(module.v_proj(x),N,B).view(B*h,N,d)
                        scores=torch.bmm(q,k.transpose(1,2));operator_calls+=4
                        one=torch.cat([torch.bmm(q[i*h:(i+1)*h],k[i*h:(i+1)*h].transpose(1,2)) for i in range(B)]);operator_calls+=B
                        compare(branch+'/same_input_qk_bmm',scores,one)
                        scaled=scores*module.scale;prob=torch.softmax(scaled,dim=-1)
                        one=torch.cat([torch.softmax(scaled[i*h:(i+1)*h],dim=-1) for i in range(B)]);operator_calls+=B+1
                        compare(branch+'/same_input_softmax',prob,one)
                        out=torch.bmm(prob,v);one=torch.cat([torch.bmm(prob[i*h:(i+1)*h],v[i*h:(i+1)*h]) for i in range(B)]);operator_calls+=B+1
                        compare(branch+'/same_input_pv_bmm',out,one)
            finally:busy=False
        return hook
    with threadpool_limits(limits=4):
        r.setup_torch();net,line=r.restore(e);r.pointwise(net,True)
        native=net.backbone.forward_features
        def tag(x,*a,**kw):
            nonlocal branch
            branch='few' if kw.get('few',0)==1 else 'main'
            return native(x,*a,**kw)
        net.backbone.forward_features=tag
        for mod in (net.original_backbone.model,net.backbone):
            handles.append(mod.patch_embed.register_forward_hook(check('patch_embed_all_tokens')))
            for name,child in mod.blocks[0].named_modules():
                if name and (isinstance(child,(torch.nn.Linear,torch.nn.LayerNorm,torch.nn.GELU)) or name=='attn'):
                    handles.append(child.register_forward_hook(check(name)))
        xs=[]
        for split in ('train','val'):
            ix=np.flatnonzero(np.isin(r.data[split][1],r.orders[1993][:4]))[:16];xs.append(r.images_at(split,ix))
        r.forward(net,torch.cat(xs))
        for h in handles:h.remove()
        from run_medical_v2 import network_hash
        assert network_hash(net)==line['network_sha256']
        write(r.pub/'RESULT.json',dict(status='DIAGNOSIS_ONLY',parent_id=1993,backend_changed=False,operators=records,
              isolated_operator_calls=operator_calls,operator_inputs_identical=True,parameters_buffers_unchanged=True,S_candidate_scores_generated=0))
        write(r.pub/'ACCESS_AUDIT.json',dict(**FIXED,**r.access,encoder_forward_calls=sum(r.access['module_batch_calls'].values()),
              encoder_image_forward_rows=sum(r.access['module_image_rows'].values()),isolated_operator_calls=operator_calls))
        write(r.pub/'RESOURCE_REPORT.json',r.resource_check())
    print('OPERATOR_TRACE_COMPLETE',[(x['operator'],x['max_abs']) for x in records if x['max_abs']>0],flush=True)


if __name__=='__main__':main()
