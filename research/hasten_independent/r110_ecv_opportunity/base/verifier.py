"""Shared class tokens and invariant pooling; no trainable class IDs."""
import torch
from torch import nn
from .state_features import CLASS_DIM, ACTION_DIM


class Verifier(nn.Module):
    def __init__(self, dynamics=True):
        super().__init__(); self.use_dynamics=dynamics
        self.encoder=nn.Sequential(nn.Linear(CLASS_DIM,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.class_action=nn.Sequential(nn.Linear(64+ACTION_DIM,64),nn.ReLU())
        self.condition=nn.Sequential(nn.Linear(128+ACTION_DIM,64),nn.ReLU())
        self.dynamic=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
        # DIRECT_RETURN has exactly the same parameter count; dynamic features
        # remain latent but their targets are never supervised in that arm.
        self.return_head=nn.Sequential(nn.Linear(70,64),nn.ReLU(),nn.Linear(64,6))
        self.register_buffer('class_mean',torch.zeros(CLASS_DIM));self.register_buffer('class_std',torch.ones(CLASS_DIM))
        self.register_buffer('action_mean',torch.zeros(ACTION_DIM));self.register_buffer('action_std',torch.ones(ACTION_DIM))
        self.register_buffer('return_scale',torch.ones(6));self.register_buffer('dynamic_scale',torch.ones(3))

    def fit_scales(self,records):
        assert records and all(r['split']=='fit' for r in records)
        with torch.no_grad():
            for name,rows in [('class',torch.cat([r['classes'] for r in records])),('action',torch.cat([r['actions'].reshape(-1,ACTION_DIM) for r in records]))]:
                getattr(self,name+'_mean').copy_(rows.mean(0));getattr(self,name+'_std').copy_(rows.std(0,unbiased=False).clamp_min(.02))
            for j in range(6):
                vals=torch.cat([r['targets'][:,j][r['target_mask'][:,j]] for r in records])
                self.return_scale[j]=vals.std(unbiased=False).clamp_min(.02) if len(vals) else .02
            for j in range(3):
                vals=torch.cat([r['dynamic'][:,:,j][r['dynamic_mask'][:,:,j]] for r in records])
                self.dynamic_scale[j]=vals.std(unbiased=False).clamp_min(.02) if len(vals) else .02

    def forward(self, classes, actions):
        tokens=self.encoder((classes-self.class_mean)/self.class_std)
        actions=(actions-self.action_mean)/self.action_std
        if actions.ndim==2:actions=actions[:,None].expand(-1,len(tokens),-1)
        assert actions.shape[1]==len(tokens)
        conditioned=self.class_action(torch.cat([tokens[None].expand(len(actions),-1,-1),actions],-1))
        pooled=torch.cat([conditioned.mean(1),conditioned.max(1).values],-1)
        context=self.condition(torch.cat([pooled,actions.mean(1)],-1))
        dynamic=self.dynamic(torch.cat([conditioned,context[:,None].expand(-1,len(tokens),-1)],-1))
        values=self.return_head(torch.cat([context,dynamic.mean(1),dynamic.max(1).values],-1))
        return values,dynamic


class FitNormalizer:
    def fit(self, rows, split):
        assert split=='fit', 'NORMALIZER_FIT_ONLY'
        self.mean=rows.mean(0).detach(); self.std=rows.std(0,unbiased=False).clamp_min(.02).detach()
        return self

    def __call__(self, rows):
        assert hasattr(self,'mean'), 'NORMALIZER_NOT_FITTED'
        return (rows-self.mean.to(rows))/self.std.to(rows)

