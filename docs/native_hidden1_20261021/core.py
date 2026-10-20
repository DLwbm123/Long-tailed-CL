"""Fixed base-only hidden-coordinate selection and read-only native MLP hooks."""
import torch


def choose_hidden(base_grams, cls_dim=768, post_dim=768, hidden_dim=3072, keep=768):
    """Class masses must each be one; no future-class or query inputs."""
    offset = cls_dim + post_dim
    means, seconds = [], []
    for g in base_grams:
        assert g.dtype == torch.float64
        assert g.shape == (offset + hidden_dim + 1,) * 2
        assert abs(float(g[-1, -1]) - 1.) < 1e-9
        means.append(g[offset:-1, -1])
        seconds.append(g.diagonal()[offset:-1])
    means, seconds = torch.stack(means), torch.stack(seconds)
    assert torch.isfinite(means).all() and torch.isfinite(seconds).all()
    within_each = seconds - means.square()
    tolerance = 1e-10 * seconds.abs().clamp_min(torch.finfo(torch.float64).tiny)
    assert (within_each >= -tolerance).all(), 'Negative variance exceeds rounding allowance'
    within = within_each.clamp_min(0).mean(0)
    between = means.var(0, unbiased=False)
    score = torch.zeros_like(within)
    positive = within > 0
    score[positive] = between[positive] / within[positive]
    score[(~positive) & (between > 0)] = float('inf')
    order = torch.argsort(score, descending=True, stable=True)
    return order[:keep] + offset


def feature_indices(hidden_indices, cls_dim=768, post_dim=768, hidden_dim=3072):
    bias = cls_dim + post_dim + hidden_dim
    pre = torch.cat([torch.arange(cls_dim), hidden_indices.cpu(), torch.tensor([bias])])
    post = torch.cat([torch.arange(cls_dim + post_dim), torch.tensor([bias])])
    return {'native_pre': pre, 'native_post': post}


class NativeCapture:
    """Capture only CLS-position activations, without modifying forward outputs."""
    def __init__(self, encoder):
        assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
        assert len(encoder.blocks) == 12
        mlp = encoder.blocks[11].mlp
        assert mlp.fc1.out_features == 3072 and mlp.fc2.out_features == 768
        assert isinstance(mlp.norm, torch.nn.Identity)
        self.values = {}
        self.handles = [mlp.act.register_forward_hook(self._hook('hidden', 3072)),
                        mlp.fc2.register_forward_hook(self._hook('post', 768))]

    def _hook(self, name, dim):
        def capture(module, args, output):
            assert name not in self.values and output.ndim == 3 and output.shape[-1] == dim
            self.values[name] = output[:, 0].detach().clone()
        return capture

    def take(self, final_tokens):
        assert final_tokens.shape[1:] == (197, 768)
        hidden, post = self.values.pop('hidden'), self.values.pop('post')
        raw = torch.cat([final_tokens[:, 0].double(), post.double(), hidden.double()], 1)
        assert torch.isfinite(raw).all()
        return raw

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.values.clear()
