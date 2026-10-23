import copy,time
import numpy as np
import torch
from torch.nn import functional as F
from core import Model,ARMS,np_scores,update
from foundation import save,write


def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def optimizer(m,cfg):return torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
def counter():return dict(updates=0,forward_rows=0,backward_rows=0)

def math_check():
    torch.manual_seed(74002);err=0.;fd=0.;x=torch.randn(3,5,4,dtype=torch.float64);labels=torch.tensor([0,1,0])
    for arm in ARMS[1:]:
        m=Model(torch.randn(2,8),torch.randn(2,4),arm).double()
        with torch.no_grad():m.local_head.copy_(torch.randn_like(m.local_head)*.1)
        loss=F.cross_entropy(m(x),labels);loss.backward();s=state(m)
        def objective(ss):
            v=np_scores(x.numpy(),{k:t.numpy() for k,t in ss.items()},arm);v-=v.max(1,keepdims=True)
            return np.mean(np.log(np.exp(v).sum(1))-v[np.arange(3),labels.numpy()])
        err=max(err,abs(objective(s)-float(loss.detach())))
        for name in list(s):
            pos=(0,1);plus={k:v.clone() for k,v in s.items()};minus={k:v.clone() for k,v in s.items()};plus[name][pos]+=1e-5;minus[name][pos]-=1e-5
            fd=max(fd,abs((objective(plus)-objective(minus))/2e-5-float(dict(m.named_parameters())[name].grad[pos])))
        assert torch.allclose(m(x),m(torch.cat([x[:,:1],x[:,1:].flip(1)],1)),atol=1e-12,rtol=0)
    assert err<1e-9 and fd<2e-6
    return dict(state='PASS',objective_error=err,finite_difference_error=fd,synthetic_dimension=4,NIH_images=0,optimization=0,synthetic_Torch_forward_rows=27,synthetic_Torch_backward_rows=9,synthetic_NumPy_objective_rows=57)


def qualify(tokens,global_means,patch_means,cfg,p):
    receipts={};clock=time.time();timed=0.
    for arm in ARMS[1:]:
        m=Model(global_means[:8],patch_means[:8],arm).cuda();before=state(m);opt=optimizer(m,cfg);cc=counter()
        ix=torch.arange(64)%16;labels=torch.arange(8).repeat_interleave(2)[ix]
        torch.cuda.synchronize();tick=time.time();update(m,opt,tokens[ix],labels,cfg,cc);torch.cuda.synchronize();timed+=time.time()-tick
        old=state(m);m.expand(global_means[8:].cuda(),patch_means[8:].cuda());assert torch.equal(old['global_head'],state(m)['global_head'][:8]) and torch.equal(old['local_head'],state(m)['local_head'][:8])
        if arm=='CLASS_ATTN':assert torch.equal(old['query'],state(m)['query'][:8])
        if arm=='SHARED_ATTN':assert torch.equal(old['query'],state(m)['query'])
        opt=optimizer(m,cfg);ix=torch.arange(64)%20;labels=torch.arange(10).repeat_interleave(2)[ix]
        torch.cuda.synchronize();tick=time.time();update(m,opt,tokens[ix],labels,cfg,cc);torch.cuda.synchronize();timed+=time.time()-tick
        seal=state(m);save(p/f'qualification_{arm}.private.pt',seal);loaded=torch.load(p/f'qualification_{arm}.private.pt',map_location='cpu',weights_only=False)
        from core import cpu_scores
        cpu=cpu_scores(tokens[:2],loaded,arm).numpy();reference=np_scores(tokens[:2].numpy(),{k:v.numpy() for k,v in loaded.items()},arm)
        with torch.no_grad():gpu=m(tokens[:2].cuda().float()).cpu().numpy()
        error=float(abs(cpu-reference).max());gpu_error=float(abs(cpu-gpu).max());write(p/f'READOUT_PARITY_{arm}.json',dict(arm=arm,FP64_NumPy_error=error,GPU_FP32_FP64_error=gpu_error,qualification_updates=cc['updates']));assert error<=cfg['score_absolute_tolerance'] and gpu_error<=cfg['FP32_probe_tolerance']
        assert float(seal['global_head'].abs().sum())>0 and float(seal['local_head'].abs().sum())>0
        if arm!='MEAN':assert float((seal['query'][:len(before['query'])]-before['query']).abs().max())>0
        # Full64 vs micro8 checks coupled clipping and the actual update, not just output values.
        a=copy.deepcopy(m);b=copy.deepcopy(m);oa=optimizer(a,cfg);ob=optimizer(b,cfg);oa.zero_grad()
        loss=F.cross_entropy(a(tokens[ix].cuda().float()),labels.cuda());loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),cfg['gradient_clip']);gr={k:v.grad.detach().clone() for k,v in a.named_parameters()};oa.step()
        extra=counter();update(b,ob,tokens[ix],labels,cfg,extra)
        ge=max(float((v.grad-gr[k]).abs().max()) for k,v in b.named_parameters());ue=max(float((a.state_dict()[k]-v).abs().max()) for k,v in b.state_dict().items());write(p/f'GRADIENT_PARITY_{arm}.json',dict(arm=arm,gradient_error=ge,AdamW_error=ue,gradient_tolerance=cfg['gradient_tolerance'],AdamW_tolerance=cfg['AdamW_tolerance'],discarded_gradient_check_updates=2));assert ge<cfg['gradient_tolerance'] and ue<cfg['AdamW_tolerance']
        receipts[arm]=dict(score_error=error,GPU_FP32_error=gpu_error,gradient_error=ge,AdamW_error=ue,qualification_updates=2,gradient_check_discarded_updates=2,qualification_forward_rows=128,qualification_backward_rows=128,gradient_check_forward_rows=128,gradient_check_backward_rows=128,parameter_count=sum(v.numel() for v in m.parameters()))
        del m,a,b,opt,oa,ob
    return dict(state='PASS',arms=receipts,paired_step_seconds=timed/2,resident_after_means_seconds=time.time()-clock,discarded_updates=12)
