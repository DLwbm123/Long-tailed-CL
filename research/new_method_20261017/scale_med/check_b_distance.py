import torch
from core import gaussian,logpdf
from run_b_distance import offsets

torch.set_default_dtype(torch.float64)
stats=[dict(groups=2,local=(torch.tensor([0.,1.]),torch.tensor([[2.,.3],[.3,1.]]))),dict(groups=9,local=(torch.tensor([1.,0.]),torch.tensor([[1.,-.2],[-.2,3.]])))]
ref=(torch.zeros(2),torch.eye(2),.001);o=dict(stats=stats,reference={'local':ref})
p=torch.tensor([[[0.,1.],[2.,3.]],[[1.,2.],[0.,0.]]]);add=offsets(o,20)
for i,s in enumerate(stats):
    beta=s['groups']/(s['groups']+20);cov=beta*s['local'][1]+(1-beta)*ref[1]
    g=gaussian(s['local'][0],cov,ref[2]);raw=logpdf(p,g).mean(1)/2
    centered=p-s['local'][0];mahal=torch.einsum('bpd,de,bpe->bp',centered,torch.linalg.inv(cov+.001*torch.eye(2)),centered).mean(1)
    expected=-.5*mahal/2-.5*torch.log(torch.tensor(2*torch.pi))
    assert torch.allclose(raw+add[i],expected,atol=1e-12,rtol=0)
print('PASS: volume removal equals mean Mahalanobis score plus common constant for every class')
