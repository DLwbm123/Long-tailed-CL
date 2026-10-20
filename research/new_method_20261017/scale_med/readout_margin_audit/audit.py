"""CPU-only descriptive audit of sealed features and frozen readouts."""
import json
import os
import resource
import time
from pathlib import Path

import torch

torch.set_num_threads(4)
ARMS = ['F', 'E', 'L', 'R', 'D']


def transform(g, mp):
    return (((g-mp['gn'][0])/mp['gn'][1])[:, mp['ids']]-mp['on'][0])/mp['on'][1]


def raw_readout(mp, w, dimension):
    scale = mp['gn'][1]*mp['on'][1]
    coefficient = torch.zeros(dimension, 4, dtype=w.dtype)
    coefficient[mp['ids']] = w[:-1, :4]/scale
    offset = (-mp['gn'][0][mp['ids']]/mp['gn'][1]-mp['on'][0])/mp['on'][1]
    intercept = w[-1, :4]+offset@w[:-1, :4]
    return coefficient, intercept, float(scale)


def points(scores, y):
    pred = scores.argmax(1)
    counts = [int((y == c).sum()) for c in range(4)]
    correct = [int(((pred == c) & (y == c)).sum()) for c in range(4)]
    recall = [v/n for v, n in zip(correct, counts)]
    return dict(BA=sum(recall)/4, correct=correct, counts=counts, recall=recall)


def stats(v):
    if not v.numel():
        return dict(count=0, mean=None, quantiles=None)
    assert torch.isfinite(v).all()
    return dict(count=v.numel(), mean=float(v.mean()), quantiles=torch.quantile(
        v, torch.tensor([0., .1, .5, .9, 1.], dtype=v.dtype)).tolist())


def synthetic():
    gen = torch.Generator().manual_seed(74260)
    x = torch.randn(40, 7, dtype=torch.float64, generator=gen)
    mp = dict(gn=(torch.randn(7, dtype=x.dtype, generator=gen), torch.tensor(2., dtype=x.dtype)),
              on=(torch.randn(5, dtype=x.dtype, generator=gen), torch.tensor(.7, dtype=x.dtype)),
              ids=torch.tensor([6, 1, 4, 2, 0]))
    w = torch.randn(6, 4, dtype=x.dtype, generator=gen)
    v, b, scale = raw_readout(mp, w, 7)
    err = float((transform(x, mp)@w[:-1]+w[-1]-(x@v+b)).abs().max())
    assert err < 1e-10
    z = transform(x, mp)
    target = torch.randn(40, 4, dtype=x.dtype, generator=gen)

    def fit(z, alpha):
        xx = torch.cat([z, torch.ones(len(z), 1, dtype=z.dtype)], 1)
        h = xx.T@xx/len(z)
        h.diagonal()[:-1].add_(alpha)
        return xx@torch.linalg.solve(h, xx.T@target/len(z))

    centered = x[:, mp['ids']]
    ridge_err = float((fit(z, .01)-fit(centered, .01*scale**2)).abs().max())
    assert ridge_err < 1e-10
    q = torch.linalg.qr(w[:-1]-w[:-1].mean(1, keepdim=True), mode='reduced')[0][:, :3]
    delta = torch.randn(40, 5, dtype=x.dtype, generator=gen)
    for c in range(4):
        for k in range(4):
            diff = w[:-1, c]-w[:-1, k]
            assert ((delta@diff).abs() <= (delta@q).norm(dim=1)*diff.norm()+1e-10).all()
    return dict(raw_score_error=err, effective_alpha_ridge_error=ridge_err, pair_bound=True)


def main(cfg):
    start = time.monotonic()
    tests = synthetic()
    old = Path(cfg['sealed_root'])
    fitcfg = json.loads((old/'fit.config.private.json').read_text())
    obj = torch.load(Path(fitcfg['reference'])/'inputs/0.private.pt', map_location='cpu', weights_only=False)
    y, valid = obj['y'], obj['valid']
    assert [int((~valid).sum()), int(valid.sum())] == [2555, 681]
    teacher = torch.load(old/'train/protection.private.pt', map_location='cpu', weights_only=False)
    modelseal = json.loads((old/'fit/MODEL_SEAL.json').read_text())
    expected = json.loads((old/'fit/aggregate.json').read_text())
    assert modelseal['total_fits'] == 15 and not modelseal['test_used'] and not modelseal['future_used']
    features = {'F': obj['g'].double()}
    readouts = {'F': (teacher['mapper'], teacher['teacher_W'], .01)}
    for arm in ARMS[1:]:
        seal = json.loads((old/'fit'/(arm+'.FEATURE_SEAL.json')).read_text())
        assert seal['one_pass'] and seal['rows'] == 3236
        features[arm] = torch.load(old/'fit'/(arm+'.g.private.pt'), map_location='cpu', weights_only=False).double()
        artifact = torch.load(old/'fit'/(arm+'.state.private.pt'), map_location='cpu', weights_only=False)
        readouts[arm] = artifact['mapper'], artifact['W'][:, :4], artifact['alpha']
    scores, cells, scales = {}, {}, {}
    maximum_error = 0.
    for arm in ARMS:
        mp, w, alpha = readouts[arm]
        assert torch.equal(mp['ids'], readouts['F'][0]['ids'])
        coef, bias, scale = raw_readout(mp, w, 1536)
        scales[arm] = dict(scale=scale, alpha=alpha, effective_raw_alpha=alpha*scale**2)
        for feature in (ARMS if arm == 'F' else ['F', arm]):
            g = features[feature]
            direct = transform(g, mp)@w[:-1, :4]+w[-1, :4]
            raw = g@coef+bias
            maximum_error = max(maximum_error, float((direct-raw).abs().max()))
            assert torch.isfinite(direct).all() and maximum_error <= 1e-8
            key = arm+'_on_'+feature
            scores[key] = direct
            cells[key] = {name: points(direct[mask], y[mask]) for name, mask in [('train', ~valid), ('development', valid)]}
        assert cells[arm+'_on_'+arm]['development']['correct'] == expected['summary'][arm]['correct']
    assert cells['F_on_F']['development']['correct'] == [55, 2, 22, 430]
    pairs = {}
    fscore = scores['F_on_F']
    fcorrect = fscore.argmax(1) == y
    mpf, wf, _ = readouts['F']
    fz = transform(features['F'], mpf)
    decomposition_error = 0.
    for arm in ARMS[1:]:
        mid, final = scores['F_on_'+arm], scores[arm+'_on_'+arm]
        midok, finalok = mid.argmax(1) == y, final.argmax(1) == y
        dz = transform(features[arm], mpf)-fz
        projection = (dz@teacher['D']).norm(dim=1)
        rows = {}
        for split, mask in [('train', ~valid), ('development', valid)]:
            classes = []
            for c in range(4):
                cc = mask & (y == c)
                original = cc & fcorrect
                harmed = original & ~finalok
                certificate = torch.ones(len(y), dtype=torch.bool)
                rival = []
                for k in range(4):
                    if k == c:
                        continue
                    fm = fscore[:, c]-fscore[:, k]
                    im = mid[:, c]-mid[:, k]
                    am = final[:, c]-final[:, k]
                    representation, replacement = im-fm, am-im
                    bound = projection*(wf[:-1, c]-wf[:-1, k]).norm()
                    assert (representation[cc].abs() <= bound[cc]+1e-8).all()
                    certificate &= fm > bound
                    decomposition_error = max(decomposition_error, float(((am-fm)-representation-replacement)[cc].abs().max()))
                    subsets = {}
                    for name, take in [('all', cc), ('F_correct', original), ('own_new_harm', harmed)]:
                        subsets[name] = {n: stats(v[take]) for n, v in [('F_margin', fm), ('representation_term', representation), ('readout_term', replacement), ('final_margin', am), ('bound', bound)]}
                    rival.append(dict(competitor=k, subsets=subsets))
                certified = original & certificate
                assert midok[certified].all()
                categories = {
                    'reference_wrong_final_wrong': int((original & ~midok & ~finalok).sum()),
                    'reference_correct_final_wrong': int((original & midok & ~finalok).sum()),
                    'reference_wrong_final_correct': int((original & ~midok & finalok).sum()),
                    'reference_correct_final_correct': int((original & midok & finalok).sum())}
                assert sum(categories.values()) == int(original.sum())
                classes.append(dict(class_index=c, count=int(cc.sum()), F_correct_count=int(original.sum()),
                                    own_new_harm_count=int(harmed.sum()), categories=categories,
                                    fixed_F_actual_retained=int((original & midok).sum()),
                                    sufficient_bound_certified=int(certified.sum()),
                                    projection_norm=stats(projection[cc]), rivals=rival))
            rows[split] = classes
        pairs[arm] = rows
    assert decomposition_error <= 1e-8 and len(cells) == 13
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(decision='DESCRIPTIVE_READOUT_AUDIT_COMPLETE', method_success=False,
                  not_causal_identification=True, development_exposed=True, test_used=False, future_used=False,
                  encoder_forwards=0, parameter_updates=0, ridge_fits=0, cells=cells,
                  scales=scales, pairs=pairs, synthetic=tests, raw_score_max_error=maximum_error,
                  margin_decomposition_max_error=decomposition_error, verified_own_endpoint_parity=True,
                  resource=dict(wall_seconds=time.monotonic()-start, CPU_seconds=usage.ru_utime+usage.ru_stime,
                                peak_RSS_bytes=usage.ru_maxrss*1024))
    out = Path(cfg['output'])/'aggregate.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
