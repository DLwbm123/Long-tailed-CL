"""Independent objective derivatives and two-stage complete-function qualification."""
import copy,time
import numpy as np
import torch
from torch.nn import functional as F
from core import Growing,Upper,np_scores
from training import state,optimizer,update,counter,cached,forward,restore,ARMS
from foundation import write,save


def math_check(cfg,p):
    from types import SimpleNamespace
    from timm.models.vision_transformer import Block
    from functools import partial
    torch.manual_seed(cfg['seed']);norm=partial(torch.nn.LayerNorm,eps=1e-6)
    b=SimpleNamespace(blocks=torch.nn.Sequential(*([torch.nn.Identity() for _ in range(10)]+[Block(24,3,mlp_ratio=2,qkv_bias=True,norm_layer=norm) for _ in range(2)])),norm=norm(24))
    u=Upper(b).double();x=torch.randn(2,3,24,dtype=torch.float64);y=torch.tensor([0,2]);m=Growing(True)
    m.append(u,torch.randn(2,48,dtype=torch.float64),[7,8]);m.append(u,torch.randn(1,48,dtype=torch.float64),[9]);m.double()
    loss=F.cross_entropy(m(x),y);loss.backward();ss=[state(e) for e in m.experts]
    def objective(st):
        v=np_scores(x.numpy(),[{k:t.numpy() for k,t in e.items()} for e in st],heads=3);v-=v.max(1,keepdims=True)
        return float(np.mean(np.log(np.exp(v).sum(1))-v[np.arange(2),y.numpy()]))
    err=abs(objective(ss)-float(loss.detach()));de=[]
    for n,pos in [('head',(0,1)),('upper.blocks.0.attn.qkv.weight',(1,2)),('upper.blocks.1.mlp.fc1.weight',(0,1))]:
        plus=copy.deepcopy(ss);minus=copy.deepcopy(ss);plus[-1][n][pos]+=1e-5;minus[-1][n][pos]-=1e-5
        de.append(abs((objective(plus)-objective(minus))/2e-5-float(dict(m.experts[-1].named_parameters())[n].grad[pos])))
    r=dict(objective_error=err,derivative_error=max(de),Torch_forward_rows=2,Torch_backward_rows=2,NumPy_objective_rows=14,NIH_images=0,optimizer_updates=0)
    write(p/'MATH_QUALIFICATION.json',r)
    assert err<cfg['math_objective_tolerance'] and max(de)<cfg['math_derivative_tolerance']
    assert all(v.grad is None for v in m.experts[0].parameters())
    return r


def qualify(template,tokens,means,cfg,p):
    receipts={};paired=[0.,0.]
    for arm in ARMS[1:]:
        m=Growing(arm=='FULL_SCORE_FROZEN');m.append(template,means[:8],list(range(8)));m.cuda();cc=counter()
        opt=optimizer(m,cfg);ix=torch.arange(64)%16;yy=torch.arange(8).repeat_interleave(2)[ix]
        native=state(m.experts[0]);audit={};torch.cuda.synchronize();tick=time.time()
        timing=update(m,opt,tokens[ix],yy,cfg,cc,audit=audit);torch.cuda.synchronize();audit['residence_seconds']=time.time()-tick;paired[0]+=timing['canonical_step_seconds']
        write(p/'QUALIFICATION_PROGRESS.json',dict(discarded_updates=2*len(receipts)+1,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()));write(p/f'OPERATOR_{arm}_0.json',audit);assert audit['AdamW_normalized_error']<=1
        old=state(m.experts[0]);m.append(template,means[8:].cuda(),[8,9]);m.cuda();opt=optimizer(m,cfg)
        before=forward(m.experts[0],tokens[:2]);features=cached(m,tokens)
        ix=torch.arange(64)%20;yy=torch.arange(10).repeat_interleave(2)[ix]
        with torch.no_grad():direct=m(tokens[:2].cuda().float()).cpu();cache=m(tokens[:2].cuda().float(),[features[0][:2].cuda()]).cpu()
        cache_err=float((cache-direct).abs().max());audit={};torch.cuda.synchronize();tick=time.time()
        timing=update(m,opt,tokens[ix],yy,cfg,cc,[v[ix] for v in features],audit);torch.cuda.synchronize();audit['residence_seconds']=time.time()-tick;paired[1]+=timing['canonical_step_seconds']
        write(p/'QUALIFICATION_PROGRESS.json',dict(discarded_updates=2*len(receipts)+2,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved()));write(p/f'OPERATOR_{arm}_1.json',audit);assert audit['AdamW_normalized_error']<=1
        now=state(m.experts[0]);old_head_delta=float((old['head']-now['head']).abs().max());old_score_delta=float((before-forward(m.experts[0],tokens[:2])).abs().max())
        old_upper_delta=max(float((v-now[k]).abs().max()) for k,v in old.items() if k.startswith('upper.'))
        new_state=state(m.experts[1]);new_delta=max(float((v-new_state['upper.'+k]).abs().max()) for k,v in state(template).items())
        seals=[state(e) for e in m.experts];save(p/f'qualification_{arm}.private.pt',seals)
        loaded=torch.load(p/f'qualification_{arm}.private.pt',map_location='cpu',weights_only=False)
        cpu=restore(template,[list(range(8)),[8,9]],loaded,arm=='FULL_SCORE_FROZEN').double()
        fp64=forward(cpu,tokens[:2],1).numpy();npv=np_scores(tokens[:2].numpy(),[{k:v.numpy() for k,v in s.items()} for s in loaded]);gpu=forward(m,tokens[:2],1).numpy()
        r=dict(FP64_NumPy_error=float(abs(fp64-npv).max()),FP32_FP64_error=float(abs(gpu-fp64).max()),cache_error=cache_err,old_head_delta=old_head_delta,old_score_delta=old_score_delta,old_upper_delta=old_upper_delta,new_upper_delta=new_delta,discarded_updates=cc['updates'],operator_checks=2,parameters=sum(v.numel() for v in m.parameters()))
        write(p/f'QUALIFICATION_{arm}.json',r)
        assert r['FP64_NumPy_error']<=cfg['score_absolute_tolerance'] and r['FP32_FP64_error']<=cfg['FP32_probe_tolerance'] and cache_err<cfg['FP32_probe_tolerance'] and old_upper_delta==0 and new_delta>0
        assert (old_head_delta==0 and old_score_delta==0) if arm=='FULL_SCORE_FROZEN' else old_head_delta>0
        assert any(float((native[k]-old[k]).abs().max())>0 for k in native if k.startswith('upper.'))
        receipts[arm]=r
        del m,opt,cpu,loaded,seals,features;torch.cuda.empty_cache()
    return dict(arms=receipts,paired_stage_step_seconds=paired,paired_step_seconds=max(paired),discarded_updates=4)
