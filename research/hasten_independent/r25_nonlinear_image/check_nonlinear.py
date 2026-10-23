import os,json,copy,torch
from compat import load_upstream,install_feature_model
from pathlib import Path
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());inc,mod,geom=load_upstream(cfg['official'])
torch.manual_seed(42);a=inc.AMLP_Adapter(512,512);b=copy.deepcopy(a)
count=sum(p.numel() for p in a.parameters());rng=torch.get_rng_state();b.fc.add_module('1',torch.nn.GELU(approximate='none'))
assert torch.equal(rng,torch.get_rng_state()) and sum(p.numel() for p in b.parameters())==count
for name,p in a.named_parameters():assert torch.equal(p,dict(b.named_parameters())[name])
x=torch.randn(8,512);y=torch.randn(8,512)
assert torch.allclose(a((x+y)/2),(a(x)+a(y))/2,atol=1e-7)
error=(b((x+y)/2)-(b(x)+b(y))/2).abs().max();assert error>1e-5
b(x).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())
install_feature_model(inc,torch.randn(4,512),['a','b','c','d'],1.)
net=inc.hyper_net(dict(curvature=1.,nullspace_energy_threshold=.95));net.update_task();net.update_task()
assert all(len(v.fc)==2 and isinstance(v.fc[1],torch.nn.GELU) for v in net.Image_Adapter)
assert all(len(v.fc)==1 for v in net.Text_Adapter)
assert not any(p.requires_grad for p in net.Image_Adapter[0].parameters()) and not any(p.requires_grad for p in net.Text_Adapter[0].parameters())
assert all(p.requires_grad for p in net.Image_Adapter[1].parameters())
print(dict(passed=True,added_parameters=0,RNG_added_by_activation=False,non_affine_midpoint_error=float(error.detach()),finite_gradients=True,text_path_original=True,old_adapter_frozen=True))
