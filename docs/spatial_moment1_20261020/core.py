"""Signed power pooling; p is frozen before seeing validation outcomes."""
import torch


def signed_pool(tokens, p=3):
    x = tokens.float()
    if p == 1:
        # Preserve the historical float32 reduction exactly at the identity endpoint.
        return x.mean(1)
    scale = x.abs().amax(1, keepdim=True)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    u = x / safe
    positive = u.clamp_min(0).pow(p).mean(1).pow(1 / p)
    negative = (-u).clamp_min(0).pow(p).mean(1).pow(1 / p)
    return (positive - negative) * scale.squeeze(1)


def concentration(tokens):
    x = tokens.float()
    scale = x.abs().amax(1, keepdim=True).clamp_min(1e-30)
    out = {}
    for name, u in [('positive', x / scale), ('negative', -x / scale)]:
        mass = u.clamp_min(0).pow(3)
        total = mass.sum(1)
        valid = total > 0
        effective = total.square() / mass.square().sum(1).clamp_min(1e-30)
        out[name] = dict(valid_channels=int(valid.sum()),
                         effective_sum=float(effective[valid].sum()))
    return out


def self_check():
    gen = torch.Generator().manual_seed(74002)
    x = torch.randn(4, 196, 17, generator=gen)
    assert torch.allclose(signed_pool(x, 1), x.mean(1), atol=2e-6, rtol=2e-6)
    assert torch.equal(signed_pool(-x), -signed_pool(x))
    assert torch.allclose(signed_pool(x[:, torch.arange(195, -1, -1)]), signed_pool(x), atol=2e-6)
    for value in [0., 1., -1., 1e20, -1e20]:
        z = torch.full((2, 196, 3), value)
        assert torch.isfinite(signed_pool(z)).all()
        assert torch.allclose(signed_pool(z), z.mean(1), rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    self_check()
    print('PASS signed-pool algebra')
