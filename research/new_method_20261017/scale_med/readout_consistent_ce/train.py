import gc
import json
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint as recompute

from binding import checkpoint, native
from core_depth import PERM
from sampling import load_training, schedule, write
from solver import transform
from space import check_space
from train_base import finite, opt, raw, setup


def run(cfg):
    out = Path(cfg['output'])
    pixels, y, groups, cached = load_training(cfg)
    orders = schedule(y, groups)
    teacher = torch.load(Path(cfg['teacher']), map_location='cpu', weights_only=False)
    mp_cpu = teacher['mapper']
    mp = {k: tuple(t.cuda() for t in v) if isinstance(v, tuple) else v.cuda() for k, v in mp_cpu.items()}
    initial_w = teacher['teacher_W'].cuda().detach().clone()
    assert list(initial_w.shape) == [1536, 4]
    m, params, bkeys, ckeys, heads, pointwise = setup(cfg)
    initial_bias = {k: params[k].detach().clone() for k in bkeys}
    initial_native_heads = [p.detach().clone() for p in heads]
    counts = dict(logical_forward_images=0, forward_calls=0)

    def reset():
        with torch.no_grad():
            for k, v in initial_bias.items():
                params[k].copy_(v)
        for k in bkeys:
            params[k].requires_grad_(True)
        m.zero_grad(set_to_none=True)

    def encode(x):
        counts['logical_forward_images'] += len(x)
        counts['forward_calls'] += 1
        o = raw(m, pointwise, x)
        return torch.nn.functional.normalize(torch.nn.functional.normalize(
            torch.cat([o['pre_logits'], o['pre_logits_few']], 1), dim=1).double(), dim=1)

    def task(ids, w, checkpointed=True, full=False):
        values = []
        for part in ids.split(len(ids) if full else 8):
            values.append(recompute(encode, pixels[part].cuda(), use_reentrant=False)
                          if checkpointed else encode(pixels[part].cuda()))
        g = torch.cat(values)
        z = transform(g, mp)
        return torch.nn.functional.cross_entropy(z@w[:-1]+w[-1], y[ids].cuda())

    reset()
    ps = [params[k] for k in bkeys]
    assert sum(p.numel() for p in ps) == 102912
    if cfg['mode'] == 'PREFLIGHT':
        ids = orders[0]
        with torch.no_grad():
            g = encode(pixels[ids].cuda()).cpu()
            micro = torch.cat([encode(pixels[z].cuda()).cpu() for z in ids.split(8)])
        replay = float((g-cached[ids]).abs().max())
        assert replay <= 1e-10
        wp = initial_w.detach().clone()
        wc = initial_w.detach().clone().requires_grad_(True)
        lp = task(ids, wp, False, True)
        gp = torch.autograd.grad(lp, ps)
        lc = task(ids, wc, False, True)
        gradients = torch.autograd.grad(lc, ps+[wc])
        assert torch.allclose(lp, lc, atol=1e-12, rtol=0)
        assert all(torch.allclose(a, b, atol=1e-7, rtol=1e-5) for a, b in zip(gp, gradients[:-1]))
        assert gradients[-1].norm() > 0 and torch.isfinite(gradients[-1]).all()
        errors = []
        for w, target in [(wp, ps), (wc, ps+[wc])]:
            fresh = torch.autograd.grad(task(ids, w, False, True), target)
            micrograd = torch.autograd.grad(task(ids, w), target)
            errors.append(max(float((a-b).abs().max()) for a, b in zip(fresh, micrograd)))
            assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-3) for a, b in zip(fresh, micrograd))
        for arm in ['P', 'C']:
            reset()
            w = initial_w.detach().clone().requires_grad_(arm == 'C')
            target = ps+([w] if arm == 'C' else [])
            optimizer = opt(target)
            optimizer.zero_grad(set_to_none=True)
            loss = task(ids, w)
            loss.backward()
            finite(target)
            optimizer.step()
            finite(target)
            assert torch.equal(w, initial_w) if arm == 'P' else not torch.equal(w, initial_w)
            assert all(torch.equal(p, v) for p, v in zip(heads, initial_native_heads))
            over = checkpoint({k: params[k].detach().cpu() for k in bkeys}, ckeys)
            assert all(torch.equal(native(over)[k], params[k].detach().cpu()) for k in bkeys)
        reset()
        assert all(torch.equal(params[k], v) for k, v in initial_bias.items())
        n = 1536
        full_w = torch.cat([initial_w.cpu(), torch.zeros(n, 2, dtype=torch.float64)], 1)
        template = dict(A=torch.ones(n*(n+1)//2, dtype=torch.float64), B=torch.zeros(n, 6, dtype=torch.float64),
                        W=full_w, mapper=mp_cpu, PERM=PERM, bias_overrides=over, classes=4,
                        head_origin='fixed_reference_or_CE_not_ridge_of_stored_A_B')
        torch.save(template, out/'capacity.private.pt')
        size = (out/'capacity.private.pt').stat().st_size
        assert size <= 19058800
        back = torch.load(out/'capacity.private.pt', weights_only=False)
        assert torch.equal(back['W'], full_w)
        assert all(torch.equal(back['bias_overrides'][k], v) for k, v in over.items())
        write(out/'aggregate.json', dict(pass_all=True, original_G_replay_error=replay,
              micro_feature_roundoff=float((g-micro).abs().max()), checkpoint_gradient_max_errors=errors,
              same_initial_bias_gradients=True, C_head_gradient_nonzero=True,
              P_head_fixed_exact=True, native_heads_unchanged=True, both_one_steps_finite=True,
              override_roundtrip_restore_exact=True, discarded_probe_parameters=True,
              capacity_template_bytes=size, training_rows_only=True, forward_counts=counts))
        return
    assert json.loads(Path(cfg['preflight']).read_text())['pass_all']
    torch.save(orders, out/'sampling.private.pt')
    receipts = []
    for arm in ['P', 'C']:
        reset()
        w = initial_w.detach().clone().requires_grad_(arm == 'C')
        target = ps+([w] if arm == 'C' else [])
        optimizer = opt(target)
        counts.update(logical_forward_images=0, forward_calls=0)
        losses = []
        start = time.monotonic()
        for step, ids in enumerate(orders):
            optimizer.zero_grad(set_to_none=True)
            loss = task(ids, w)
            assert torch.isfinite(loss)
            loss.backward()
            finite(target)
            optimizer.step()
            finite(target)
            losses.append(float(loss.detach()))
            if step % 10 == 0:
                check_space(cfg['window_root'])
                write(out/'status.json', dict(state='RUNNING', arm=arm, step=step+1, total_steps=800,
                      elapsed_seconds=time.monotonic()-start, CE=losses[-1], forward_counts=counts))
            if (step+1) % 100 == 0:
                check_space(cfg['window_root'], reserve=24*2**20)
                torch.save(dict(arm=arm, step=step+1, bias_overrides=checkpoint(
                    {k: params[k].detach().cpu() for k in bkeys}, ckeys), head=w.detach().cpu(),
                    optimizer=optimizer.state_dict()), out/(arm+'.resume.private.pt'))
        assert torch.equal(w, initial_w) if arm == 'P' else not torch.equal(w, initial_w)
        assert all(torch.equal(p, v) for p, v in zip(heads, initial_native_heads))
        over = checkpoint({k: params[k].detach().cpu() for k in bkeys}, ckeys)
        torch.save(dict(bias_overrides=over, head=w.detach().cpu(), mapper=mp_cpu,
                        source_commit=cfg['source_commit'], head_fixed=arm == 'P'), out/(arm+'.encoder.private.pt'))
        receipts.append(dict(arm=arm, steps=800, supervised_slots=25600, mean_first100=sum(losses[:100])/100,
                             mean_last100=sum(losses[-100:])/100, forward_counts=dict(counts),
                             trainable_bias_parameters=102912, extra_head_parameters=6144 if arm == 'C' else 0))
        write(out/'partial.json', dict(arms=receipts))
        del w, optimizer
        gc.collect()
    write(out/'ENCODER_SEAL.json', dict(arms=['P', 'C'], steps_each=800, P_head_fixed_exact=True,
          future_frozen=True, test_used=False, initial_training_only=True, source_commit=cfg['source_commit']))
    write(out/'aggregate.json', dict(decision='TWO_READOUT_CONSISTENCY_TRAJECTORIES_COMPLETE', arms=receipts,
          method_success=False, test_used=False))
    write(out/'status.json', dict(state='COMPLETE', phase='ENCODERS_SEALED'))
