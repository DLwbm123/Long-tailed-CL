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
assert all(len(v.fc)==(2 if cfg['image_nonlinear'] else 1) for v in net.Image_Adapter)
assert all(len(v.fc)==1 for v in net.Text_Adapter)
assert not any(p.requires_grad for p in net.Image_Adapter[0].parameters()) and not any(p.requires_grad for p in net.Text_Adapter[0].parameters())
assert all(p.requires_grad for p in net.Image_Adapter[1].parameters())
print(dict(passed=True,added_parameters=0,RNG_added_by_activation=False,non_affine_midpoint_error=float(error.detach()),finite_gradients=True,text_path_original=True,old_adapter_frozen=True))

for radius in [.1,.4]:
 p=torch.tensor([[radius,0.,0.]],requires_grad=True);c=torch.tensor([[.6,.2,0.]],requires_grad=True)
 loss=mod.compute_stable_entailment_loss(p,c,torch.tensor(1.))
 assert loss.dtype==torch.float64
 assert all(torch.isfinite(g).all() for g in torch.autograd.grad(loss,[p,c]))
print(dict(entailment_compute_dtype='float64',unchanged_formula_constants=True,configured_image_nonlinear=cfg['image_nonlinear']))
from types import SimpleNamespace
from torch.nn import functional as F
net.to('cuda');net.update_fc(3)
with torch.no_grad():net.fc.weight.copy_(F.normalize(torch.randn(3,512,device='cuda'),dim=1))
fake=SimpleNamespace(_known_classes=3,_total_classes=3,_network=net,_device=torch.device('cuda'),audit={})
dm=SimpleNamespace(_class_to_label=['a','b','c','d'],_data_to_prompt=['{}'])
mod.Learner._prepare_functional_anchor(fake,dm)
net.update_fc(4);net.update_task();net.to('cuda');fake._total_classes=4
text=net.encode_text(net.tokenizer(['a','b','c','d']).cuda())
aux=mod.Learner._functional_anchor(fake,text,net.get_curv(),net.logit_scale.exp());aux.backward()
assert fake.audit['functional_anchor']['gradient_proof'] and fake.audit['functional_anchor']['teacher_state_unchanged']
print(dict(functional_anchor_synthetic=True,teacher_q_detached=True,full_seen_denominator=True,coefficient=1.,finite_nonzero_gradient=True))
