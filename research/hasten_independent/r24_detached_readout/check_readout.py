import torch
from readout import Head,expand,balanced_ce

torch.manual_seed(42)
x=torch.randn(7,512);y=torch.tensor([0,0,0,1,1,2,2]);core=torch.nn.Linear(512,3)
rng=torch.get_rng_state();h=Head(3,'cpu');assert torch.equal(rng,torch.get_rng_state())
z=core(x);assert torch.equal(z.detach()+h(x),z.detach())
loss=balanced_ce(z.detach()+h(x),y)
g=torch.autograd.grad(loss,list(core.parameters()),allow_unused=True,retain_graph=True);assert all(v is None for v in g)
before=[p.detach().clone() for p in core.parameters()]
opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.001);loss.backward();opt.step()
assert h.weight.norm()>0 and all(torch.equal(p,b) for p,b in zip(core.parameters(),before))
new=expand(h,5,'cpu');assert torch.equal(new.weight[:3],h.weight) and new.weight[3:].count_nonzero()==0 and new.bias[3:].count_nonzero()==0
scale=torch.tensor(100.)
assert torch.allclose(scale*(z.detach()+h(x)/scale),scale*z.detach()+h(x))
idx=torch.tensor([0,1,2,0,1,2,3,4,5,6]);assert torch.allclose(balanced_ce(z,y),balanced_ce(z[idx],y[idx]))
print('zero-head parity, no core gradients/updates, head learning, exact expansion, RNG preservation and score units passed')
