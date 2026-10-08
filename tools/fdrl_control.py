"""On-policy, discounted FD scheduling; only aggregated training-meta feedback."""
import math
import torch
from torch import nn
import prototype_coherent as method

ACTIONS = (5., 10., 20.)


def meta_append(bank, x, y):
    """Uniform class moments only: never retain meta prototypes or examples."""
    classes = sorted(y.unique().tolist()); old = len(bank['n'])
    if classes != list(range(old, old+len(classes))):
        raise ValueError('Noncontiguous meta labels')
    means = [x[y == c].mean(0) for c in classes]
    seconds = [x[y == c].T @ x[y == c] / int((y == c).sum()) for c in classes]
    return dict(mu=torch.cat([bank['mu'], torch.stack(means)]),
        Q=torch.cat([bank['Q'], torch.stack(seconds)]),
        n=bank['n']+[int((y == c).sum()) for c in classes], components=[])


def risks(meta, head, x, y):
    old = .5*method.old_square_losses(meta, head)
    target = torch.nn.functional.one_hot(y, head.shape[1]).to(x)
    error = .5*(x @ head-target).square().sum(1)
    new = torch.stack([error[y == c].mean() for c in sorted(y.unique().tolist())])
    result = torch.cat([old, new])
    if not torch.isfinite(result).all(): raise ValueError('Nonfinite meta risk')
    return result.detach()


def reward(before, after, old, tail):
    gain = before-after
    penalty = gain[:old].clamp_max(0).mean() if old else gain.new_zeros(())
    penalty += gain[old:].clamp_max(0).mean()
    if tail: penalty += gain[tail].clamp_max(0).mean()
    return float(100*(gain.mean()+penalty))


def features(risk, previous, old, tail, gradient, shift, meta, progress):
    def mean(x): return float(x.mean()) if len(x) else 0.
    delta = previous-risk
    uncertainty = float((meta['Q'].diagonal(dim1=1,dim2=2).sum(1)-meta['mu'].square().sum(1)).clamp_min(0).mean()) if old else 0.
    ratio = gradient['weighted_fd_norm']/max(gradient['task_norm'],1e-12)
    values = list(progress)+[old/len(risk), gradient['cosine'], math.tanh(math.log1p(ratio)/3),
        math.tanh(mean(risk[:old])), math.tanh(mean(risk[old:])),
        math.tanh(mean(risk[tail])) if tail else 0., math.tanh(float(risk.max())),
        math.tanh(10*mean(delta[:old])), math.tanh(10*mean(delta[old:])),
        math.tanh(10*mean(delta[tail])) if tail else 0., math.tanh(shift),
        math.tanh(uncertainty*5), math.tanh(math.log(max(meta['n'])/min(meta['n']))/5) if old else 0.]
    result = torch.tensor(values,dtype=torch.float64)
    if result.shape != (16,) or not torch.isfinite(result).all(): raise ValueError('Invalid policy state')
    return result


def discounted(rewards, gamma=.95):
    value=0.; out=[]
    for r in reversed(rewards):
        value=float(r)+gamma*value;out.append(value)
    return torch.tensor(out[::-1],dtype=torch.float64)


class Policy:
    def __init__(self, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+531)
            self.hidden=nn.Sequential(nn.Linear(16,32),nn.Tanh()).double()
            self.actor=nn.Linear(32,3).double();self.critic=nn.Linear(32,1).double()
            for m in (self.actor,self.critic): nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
        self.parameters=list(self.hidden.parameters())+list(self.actor.parameters())+list(self.critic.parameters())
        self.optimizer=torch.optim.Adam(self.parameters,lr=.003)
        self.generator=torch.Generator().manual_seed(seed+83117);self.updates=0

    def forward(self, x):
        hidden=self.hidden(x)
        return self.actor(hidden).softmax(-1),self.critic(hidden).squeeze(-1)

    def sample(self, x):
        p,_=self.forward(x)
        return int(torch.multinomial(p.detach(),1,generator=self.generator)),p.detach()

    def update(self, trajectory):
        x=torch.stack([r['policy_state'] for r in trajectory]); actions=torch.tensor([r['action'] for r in trajectory])
        target=discounted([r['reward'] for r in trajectory]);p,v=self.forward(x)
        behavior=torch.stack([r['behavior'] for r in trajectory])
        if not torch.allclose(p.detach(),behavior,atol=1e-12,rtol=1e-12):
            raise ValueError('Off-policy update or mutated behavior policy')
        advantage=target-v.detach()
        entropy=-(p*p.clamp_min(1e-30).log()).sum(1)
        loss=-(p[torch.arange(len(actions)),actions].log()*advantage).mean()+.5*(v-target).square().mean()-.01*entropy.mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite actor critic update')
        self.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        return dict(returns=target.tolist(),advantage=advantage.tolist(),entropy=entropy.tolist(),loss=float(loss.detach()))

    def saved(self):
        return dict(hidden=self.hidden.state_dict(),actor=self.actor.state_dict(),critic=self.critic.state_dict(),updates=self.updates)
