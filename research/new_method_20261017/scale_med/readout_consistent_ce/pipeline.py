import gc
import json
from pathlib import Path

import torch
from binding import native
from core_attribution import weights
from core_depth import PERM
from evaluation import evaluate, points
from sampling import write
from solver import mapper, transform
from space import check_space
from train_base import setup


def solve(a, b, alpha):
    h = a/4
    h = h.clone()
    h.diagonal()[:-1].add_(alpha)
    w = torch.linalg.solve(h, b/4)
    residual = float((h@w-b/4).abs().max())
    assert residual <= 1e-7 and torch.isfinite(w).all()
    return w, residual


def run(cfg):
    out, training = Path(cfg['output']), Path(cfg['training_output'])
    assert json.loads((training/'ENCODER_SEAL.json').read_text())['P_head_fixed_exact']
    obj = torch.load(Path(cfg['reference'])/'inputs/0.private.pt', weights_only=False)
    tr, va, y, groups = ~obj['valid'], obj['valid'], obj['y'], obj['groups']
    train_groups = [groups[i] for i in torch.where(tr)[0].tolist()]
    devgroups = [groups[i] for i in torch.where(va)[0].tolist()]
    assert not set(train_groups) & set(devgroups)
    assert [int(tr.sum()), int(va.sum())] == [2555, 681]
    q = weights(y[tr], train_groups)
    assert abs(float(q.sum())-4) < 1e-10
    teacher = torch.load(cfg['teacher'], map_location='cpu', weights_only=False)
    mpf, wf = teacher['mapper'], teacher['teacher_W']
    old = torch.load(cfg['historical_predictions'], map_location='cpu', weights_only=False)
    assert torch.equal(old['y'], y[va]) and old['groups'] == devgroups
    fz = transform(obj['g'].double(), mpf)
    pred = dict(F=(fz[va]@wf[:-1]+wf[-1]).argmax(1), L=old['pred']['L'])
    assert torch.equal(pred['F'], old['pred']['F'])
    assert points(pred['F'], y[va])['correct'] == [55, 2, 22, 430]
    assert points(pred['L'], y[va])['correct'] == [56, 2, 21, 474]
    fits, sizes, rows, selected = [], {}, {}, {}

    def stats(g, mp):
        z = transform(g, mp)
        x = torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)
        a = x[tr].T@(x[tr]*q[:, None])
        a = (a+a.T)/2
        b = x[tr].T@(torch.nn.functional.one_hot(y[tr], 6).double()*q[:, None])
        return x, a, b

    def save(label, a, b, w, mp, over, origin, alpha=None):
        ix = torch.triu_indices(len(a), len(a))
        artifact = dict(A=a[ix.unbind()].contiguous(), B=b.contiguous(), W=w.contiguous(),
                        mapper=mp, PERM=PERM, bias_overrides=over, bias_representation='absolute_overrides',
                        head_origin=origin, alpha=alpha, classes=4, source_commit=cfg['source_commit'])
        file = out/(label+'.state.private.pt')
        check_space(cfg['window_root'], reserve=24*2**20)
        torch.save(artifact, file)
        sizes[label] = file.stat().st_size
        assert sizes[label] <= 19058800
        back = torch.load(file, weights_only=False)
        assert torch.equal(back['W'], w) and all(torch.equal(back['bias_overrides'][k], v) for k, v in over.items())
        probe = transform(g[:32], mp)@w[:-1, :4]+w[-1, :4]
        replay_score = transform(g[:32], back['mapper'])@back['W'][:-1, :4]+back['W'][-1, :4]
        assert torch.equal(probe, replay_score)
        rec = torch.zeros_like(a)
        rec[ix.unbind()] = back['A']
        rec[ix[1], ix[0]] = back['A']
        assert torch.equal(rec, a)
        if alpha is not None:
            replay, _ = solve(rec, back['B'], alpha)
            assert (replay-w).abs().max() <= 1e-10

    for arm, ridge_label in [('P', 'Q'), ('C', 'R')]:
        endpoint = torch.load(training/(arm+'.encoder.private.pt'), weights_only=False)
        over, head = endpoint['bias_overrides'], endpoint['head']
        if arm == 'P':
            assert torch.equal(head, wf)
        started, seal, cache = [out/(arm+s) for s in ['.ENCODING_STARTED.private.json', '.FEATURE_SEAL.json', '.g.private.pt']]
        assert not started.exists() and not seal.exists(), 'no implicit repeated endpoint encoding'
        write(started, dict(arm=arm, quota=3236))
        m, params, keys, _, heads, _ = setup(cfg)
        values = []
        with torch.no_grad():
            bound = native(over)
            for k in keys:
                params[k].copy_(bound[k].cuda())
            index = json.loads((Path(cfg['matched'])/'pixels_0/INDEX.json').read_text())
            assert index['rows'] == 3236
            done = 0
            for chunk in index['chunks']:
                xx = torch.load(Path(cfg['matched'])/'pixels_0'/chunk['file'], weights_only=False)
                for x in xx.split(32):
                    values.append(torch.nn.functional.normalize(m(x.cuda()).cpu().double(), dim=1))
                    done += len(x)
                    check_space(cfg['window_root'])
                    write(out/'status.json', dict(state='RUNNING', arm=arm, phase='ENDPOINT_ENCODING', rows=done, total_rows=3236))
        g = torch.cat(values)
        assert list(g.shape) == [3236, 1536] and torch.isfinite(g).all()
        check_space(cfg['window_root'], reserve=g.numel()*g.element_size()*3)
        torch.save(g, cache)
        write(seal, dict(arm=arm, rows=3236, one_pass=True, test_used=False))
        rows[arm] = len(g)
        del m, params, heads, values
        gc.collect()
        torch.cuda.empty_cache()
        x, a, b = stats(g, mpf)
        fullhead = torch.cat([head, torch.zeros(len(head), 2, dtype=head.dtype)], 1)
        pred[arm] = (x[va]@fullhead[:, :4]).argmax(1)
        save(arm, a, b, fullhead, mpf, over, 'fixed_F_reference' if arm == 'P' else 'joint_CE_endpoint')
        write(out/(arm+'.HEAD_SEAL.json'), dict(head_origin='fixed_F_reference' if arm == 'P' else 'joint_CE_endpoint', test_used=False))
        ownmp = mapper(g[tr], q)
        x, a, b = stats(g, ownmp)
        options = []
        for alpha in [.0001, .01, 1.]:
            w, residual = solve(a, b, alpha)
            ba = points((x[va]@w[:, :4]).argmax(1), y[va])['BA']
            fits.append(dict(arm=ridge_label, alpha=alpha, status='CERTIFIED', residual=residual, development_BA=ba))
            options.append((ba, alpha, w))
        _, alpha, w = max(options, key=lambda v: (v[0], v[1]))
        pred[ridge_label] = (x[va]@w[:, :4]).argmax(1)
        selected[ridge_label] = alpha
        save(ridge_label, a, b, w, ownmp, over, 'own_mapper_refitted_ridge', alpha)
        write(out/(ridge_label+'.SELECTION_SEAL.json'), dict(alpha=alpha, grid=[.0001, .01, 1.], own_initial_development_only=True))
        write(out/'partial.json', dict(fits=fits, artifact_bytes=sizes, feature_rows=rows, alphas=selected))
        del g, x, a, b
    assert len(fits) == 6 and len(sizes) == 4
    write(out/'MODEL_SEAL.json', dict(endpoints=['P', 'C', 'Q', 'R'], inherited=['F', 'L'], total_fits=6,
          new_artifacts=4, alphas=selected, test_used=False, future_used=False))
    result = evaluate(pred, y[va], devgroups)
    result.update(fits=fits, artifact_bytes=sizes, feature_rows=rows, alphas=selected,
                  new_parameter_updates_during_fit=0, source_commit=cfg['source_commit'])
    torch.save(dict(pred=pred, y=y[va], groups=devgroups), out/'development_predictions.private.pt')
    write(out/'aggregate.json', result)
    write(out/'status.json', dict(state='COMPLETE', decision=result['decision']))
