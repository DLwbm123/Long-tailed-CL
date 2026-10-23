"""Pinned network: known opposing/aligned gradients, original space, no RNG."""
import os,json,torch
from types import SimpleNamespace
from pathlib import Path
from compat import load_upstream,install_feature_model,apply_delta_projection
from priority import prepare_priority,apply_priority,MAPPING
cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());inc,mod,G=load_upstream(cfg['official']);torch.manual_seed(60)
install_feature_model(inc,torch.randn(6,512),[str(i) for i in range(6)],1.)
net=inc.hyper_net(dict(curvature=1.,init_cls=4,increment=2,nullspace_energy_threshold=.95));net.update_fc(4);net.update_task();net.to('cuda');net.update_fc(6);net.update_task();net.to('cuda')
net.learned_lorentz_exp.null_space_basis=torch.eye(512,device='cuda')[:,1:];x=torch.randn(7,512,device='cuda');y=torch.tensor([0,1,2,3,4,5,0],device='cuda');raw=net.model.text_bank;fc=net.fc.weight.detach().clone()
for factor in [.5,1.5]:
 net.zero_grad(set_to_none=True);a=SimpleNamespace(_network=net,_cur_task=1,audit={})
 im=net.trans_hyper(net.Image_encode(x));tx=net.trans_hyper(net.Text_encode(raw));logits=-G.pairwise_dist(im,tx[y],net.get_curv());contrast=torch.nn.functional.cross_entropy(net.logit_scale.exp()*logits,torch.arange(len(x),device='cuda'))
 prepare_priority(a,contrast);(factor*contrast).backward();w=net.learned_lorentz_exp.exp_map[0].weight;w.grad=net.learned_lorentz_exp.project_gradient(w.grad)
 before=[None if p.grad is None else p.grad.clone() for n,p in a._priority_named];cached=[g.clone() for g in a._priority_contrast];named=list(a._priority_named);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all();apply_priority(a)
 assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(u,v) for u,v in zip(gpu,torch.cuda.get_rng_state_all()))
 assert net.fc.weight.grad is None and torch.equal(fc,net.fc.weight)
 if factor<1:
  assert a.audit['geometry_priority']['conflict_batches']==1
  assert all(p.grad is None or torch.allclose(p.grad,c,atol=2e-6,rtol=2e-5) for (n,p),c in zip(named,cached))
 else:
  assert a.audit['geometry_priority']['conflict_batches']==0
  assert all((p.grad is None and b is None) or torch.equal(p.grad,b) for (n,p),b in zip(named,before))
 original=w.detach().clone();opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001);opt.step();apply_delta_projection(w,original,net.learned_lorentz_exp.null_space_basis);assert torch.equal(fc,net.fc.weight)
# Zero contrast preserves every original gradient and does not enable unused-parameter decay.
net.zero_grad(set_to_none=True);a=SimpleNamespace(_network=net,_cur_task=1,audit={});contrast=net.Text_Adapter[-1].fc[0].weight.sum()*0;prepare_priority(a,contrast);net.Text_Adapter[-1].fc[0].weight.square().mean().backward();before=[None if p.grad is None else p.grad.clone() for n,p in a._priority_named];named=list(a._priority_named);apply_priority(a);assert a.audit['geometry_priority']['zero_contrast_batches']==1;assert all((p.grad is None and b is None) or torch.equal(p.grad,b) for (n,p),b in zip(named,before))
print('PINNED_NATIVE_PRIORITY_CONFLICT_ALIGNED_EXACT_ZERO_NONE_RNG_SPACE_PASS')
