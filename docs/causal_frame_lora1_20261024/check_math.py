"""First small core qualification only; zero NIH/GPU/optimization, receipt before assert."""
import copy,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import ArrivalFrame,Model,merged_body,ARMS
from reference import np_tokens


def main(root):
    from functools import partial
    from types import SimpleNamespace
    from timm.models.vision_transformer import Block
    from threadpoolctl import threadpool_limits
    threadpool_limits(2);torch.set_num_threads(2);torch.manual_seed(74004)
    receipts=[]
    def write(name,value):
        t=root/(name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(root/name)
    def record(value):receipts.append(value);write('PROGRESS.json',dict(receipts=receipts,NIH_images=0,GPU_calls=0,optimizer_updates=0))
    means=torch.randn(20,48,dtype=torch.float64);frame=ArrivalFrame(record);w=frame.append(means[:10],list(range(10)));old=w.clone()
    for i in range(10,20,2):
        w=frame.append(means[i:i+2],list(range(i,i+2)));r=frame.receipt();r['old_rows_max_delta']=float((old-w[:len(old)]).abs().max());record(r)
        assert r['Gram_error']<1e-10 and r['old_rows_max_delta']==0;old=w.clone()
    norm=partial(torch.nn.LayerNorm,eps=1e-6)
    b=SimpleNamespace(blocks=torch.nn.Sequential(*([torch.nn.Identity() for _ in range(8)]+[Block(24,3,mlp_ratio=2,qkv_bias=True,norm_layer=norm).double() for _ in range(4)])),norm=norm(24).double())
    for block in b.blocks[8:]:block.attn.fused_attn=False
    native={}
    for i in range(4):native.update({f'blocks.{i}.'+k:v.detach().numpy() for k,v in b.blocks[8+i].state_dict().items()})
    native.update({'norm.'+k:v.detach().numpy() for k,v in b.norm.state_dict().items()})
    x=torch.randn(2,5,24,dtype=torch.float64);yy=torch.tensor([0,2]);models={}
    for arm in ARMS[1:]:
        torch.manual_seed(74004);m=Model(b,arm,rank=3).double();m.arrive(means[:3],w[:3],[0,1,2]);models[arm]=m
        with torch.no_grad():
            raw=b.norm(b.blocks[8:](x));z=F.normalize(torch.cat([raw[:,0],raw[:,1:].mean(1)],1),dim=1);bypass=float(abs(z-m.features(x)).max())
        record(dict(arm=arm,zero_B_bypass_error=bypass));assert bypass==0
        assert all(not v.requires_grad for k,v in m.named_parameters() if '.base.' in k or k.startswith('norm.'))
        # Nonzero synthetic B makes both A and B derivatives auditable, no optimizer step.
        with torch.no_grad():
            for k,v in m.named_parameters():
                if k.endswith(('.qB','.vB')):v.copy_(torch.randn_like(v)*.03)
        loss=m.loss(x,yy).mean();loss.backward();s=m.adapter_state()
        def objective(ss):
            st=merged_body(native,{k:v.numpy() for k,v in ss.items()});tokens=np_tokens(x.numpy(),st,heads=3)
            zz=np.concatenate([tokens[:,0],tokens[:,1:].mean(1)],1);zz/=np.linalg.norm(zz,axis=1,keepdims=True);h=ss['head'].numpy();h=h/np.linalg.norm(h,axis=1,keepdims=True)
            if arm=='FRAME_DR':return float(np.mean(.5*(np.sum(zz*h[yy.numpy()],1)-1)**2))
            sc=25*zz@h.T;sc-=sc.max(1,keepdims=True);return float(np.mean(np.log(np.exp(sc).sum(1))-sc[np.arange(2),yy.numpy()]))
        error=abs(objective(s)-float(loss.detach()));derivative=0.
        keys=['blocks.0.attn.qkv.qB','blocks.3.attn.qkv.vA']+(['head'] if arm=='LEARNED_CE' else [])
        for key in keys:
            pos=(0,1);plus=copy.deepcopy(s);minus=copy.deepcopy(s);plus[key][pos]+=1e-5;minus[key][pos]-=1e-5
            finite=(objective(plus)-objective(minus))/2e-5;actual=float(dict(m.named_parameters())[key].grad[pos]);de=abs(finite-actual);derivative=max(derivative,de)
            record(dict(arm=arm,parameter=key,finite_difference=finite,Torch_gradient=actual,error=de));assert de<2e-6
        record(dict(arm=arm,objective_error=error,derivative_max_error=derivative));assert error<1e-9
    assert torch.equal(models['FRAME_CE'].head,models['FRAME_DR'].head)
    write('MATH_QUALIFICATION.json',dict(state='PASS',receipts=receipts,scope='Small24-dimensional native4-block LoRA and48-dimensional causal20-row geometry only; not full1536/native768/GPU/NIH qualification.',NIH_images=0,GPU_calls=0,optimizer_updates=0,Torch_objective_forward_rows=6,Torch_objective_backward_rows=6,Torch_zero_B_probe_rows=6,Torch_native_baseline_rows=6,NumPy_objective_rows=34,finished=time.time()))

if __name__=='__main__':
    root=Path(os.environ['RUN_OUTPUT']);root.mkdir(exist_ok=False)
    try:main(root)
    except BaseException:
        (root/'FAILURE.json').write_text(json.dumps(dict(at=time.time(),traceback=traceback.format_exc(),NIH_images=0,GPU_calls=0,optimizer_updates=0),indent=2)+'\n');raise
