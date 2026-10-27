"""Complete empirical angular-kernel class scores, no k/bandwidth selection."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def make_support(features,labels,nclass):
 features=features.detach();labels=labels.detach();counts=torch.bincount(labels,minlength=nclass);assert bool((counts>0).all())
 sums=torch.zeros(nclass,features.shape[1],device=features.device,dtype=torch.double);sums.index_add_(0,labels,features.double());prototypes=F.normalize((sums/counts[:,None]).float(),dim=1).to(features.dtype)
 return dict(prototypes=prototypes.detach(),features=features,labels=labels,counts=counts.detach())

def support_scores(z,support,arm):
 if not arm.startswith('distribution_'):return 20*z@support['prototypes'].T
 features=support['features'].detach();labels=support['labels'];n=support['prototypes'].shape[0];s=20*z@features.T
 return torch.stack([torch.logsumexp(s[:,labels==c],dim=1)-torch.log((labels==c).sum().to(s.dtype)) for c in range(n)],dim=1)
