"""Pinned lifecycle, explicit duplicated labels, gradients, old text and RNG."""
import os,json,torch
from pathlib import Path
from compat import load_upstream,install_feature_model
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());inc,mod,geo=load_upstream(cfg['official']);torch.manual_seed(57)
install_feature_model(inc,torch.randn(16,512),[str(i) for i in range(16)],1.)
args=dict(curvature=1.,nullspace_energy_threshold=.95,init_cls=10,increment=2)
net=inc.hyper_net(args);net.update_fc(10);net.update_task();net.to('cuda')
x=torch.randn(5,512,device='cuda');ids=torch.tensor([0,9,0,10,11],device='cuda');old=net.Text_encode(x[:3],ids[:3]).detach().clone()
param_count=sum(p.numel() for p in net.parameters());old_state={k:v.clone() for k,v in net.Text_Adapter[0].state_dict().items()}
net.update_fc(12);net.update_task();net.to('cuda');current=net.Text_Adapter[-1]
assert all(len(a.fc)==1 for a in net.Image_Adapter) and all(len(a.fc)==1 for a in net.Text_Adapter)
assert not any(p.requires_grad for p in net.Text_Adapter[0].parameters()) and all(p.requires_grad for p in current.parameters())
cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all();z=net.Text_encode(x,ids)
assert torch.equal(z[:3],old)
expected=torch.stack([a(x[3:]) for a in net.Text_Adapter]).sum(0);assert torch.equal(z[3:],expected)
assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
probe=x[:3].detach().requires_grad_(True);zz=net.Text_encode(probe,ids[:3]);gg=torch.autograd.grad(zz.square().mean(),list(current.parameters()),allow_unused=True);assert all(g is None or torch.count_nonzero(g)==0 for g in gg)
net.zero_grad();z[3:].square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in current.parameters()) and sum(float(p.grad.square().sum()) for p in current.parameters())>0
opt=torch.optim.AdamW(current.parameters(),lr=.001);opt.step();assert torch.equal(net.Text_encode(x[:3],ids[:3]),old)
assert all(torch.equal(v,net.Text_Adapter[0].state_dict()[k]) for k,v in old_state.items())
# No router parameters or state keys; only the original task expansion adds parameters.
assert not any('route' in k for k in net.state_dict())
try:net.Text_encode(x[:1],torch.tensor([12],device='cuda'))
except AssertionError:pass
else:raise AssertionError('FUTURE_CLASS_ROUTING_ALLOWED')
print('PINNED_BIRTH_TEXT_OLD_EXACT_NEW_GRAD_RNG_NO_EXTRA_PARAMS_PASS')
