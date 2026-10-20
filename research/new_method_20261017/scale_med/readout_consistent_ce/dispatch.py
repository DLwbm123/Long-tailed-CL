import json
import os
import sys
from pathlib import Path


def main(cfg):
    ref, matched = Path(cfg['reference']).resolve(), Path(cfg['matched']).resolve()
    bc = json.loads(Path(cfg['benchmark_config']).read_text())
    testroot = Path(bc['output']).resolve()
    raw = [Path(bc['datasets']['ISIC'][k]).resolve() for k in ['images', 'development_images']]
    previous = Path(cfg['previous']).resolve()
    anchor = Path(cfg['teacher']).resolve().parents[1]
    allowed = {Path(cfg['teacher']).resolve(), Path(cfg['historical_predictions']).resolve()}

    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        assert not any(p.is_relative_to(v) for v in raw), 'raw images forbidden'
        assert not p.is_relative_to(testroot), 'test forbidden'
        assert not p.is_relative_to(previous), 'old learned group-out checkpoints forbidden'
        if p.is_relative_to(anchor):
            assert p in allowed, 'old learned anchor encoders/other private assets forbidden'
        if p.is_relative_to(ref/'inputs'):
            assert p == ref/'inputs/0.private.pt', 'future support forbidden'
        if p.is_relative_to(matched):
            assert p.is_relative_to(matched/'pixels_0'), 'future/test pixels forbidden'

    sys.addaudithook(guard)
    for bad in [ref/'inputs/1.private.pt', matched/'pixels_3/INDEX.json',
                previous/'train/O.encoder.private.pt', anchor/'train/D.encoder.private.pt']:
        try:
            guard('open', (str(bad), 'rb'))
        except AssertionError:
            pass
        else:
            raise AssertionError('access guard selfcheck')
    if cfg['mode'] == 'INJECT_FAILURE':
        raise RuntimeError('planned injected failure')
    if cfg['mode'] == 'SYNTHETIC':
        import torch
        from evaluation import synthetic
        from sampling import write
        gen = torch.Generator().manual_seed(74270)
        x = torch.randn(16, 7, dtype=torch.float64, generator=gen, requires_grad=True)
        w = torch.randn(8, 4, dtype=x.dtype, generator=gen)
        y = torch.arange(16) % 4
        fixed = torch.nn.functional.cross_entropy(x@w[:-1]+w[-1], y)
        gx = torch.autograd.grad(fixed, x)[0]
        live = w.clone().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(x@live[:-1]+live[-1], y)
        a, b = torch.autograd.grad(loss, [x, live])
        assert torch.equal(gx, a) and b.norm() > 0 and not w.requires_grad
        shift = torch.randn(7, dtype=x.dtype, generator=gen)
        assert torch.allclose(x@w[:-1]+w[-1], (x-shift)@w[:-1]+w[-1]+shift@w[:-1], atol=1e-12)
        write(Path(cfg['output'])/'aggregate.json', dict(pass_all=True, same_initial_encoder_gradient=True,
              fixed_head_no_gradient=True, trainable_head_gradient=True, translation_intercept_invariant=True,
              **synthetic()))
    elif cfg['mode'] in ['PREFLIGHT', 'TRAIN']:
        from train import run
        run(cfg)
    else:
        assert cfg['mode'] == 'FIT'
        from pipeline import run
        run(cfg)
