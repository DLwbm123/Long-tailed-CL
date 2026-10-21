"""One guarded experiment: base-only meta-training, append-only class states."""
import copy
import json
import os
import signal
import time
import traceback
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader
from core import Comparator, ARMS, patient_pools, episode, episode_risk, objective, math_check
from data import read_rows, Images


def write(p, value):
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def main(cfg):
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(cfg['output'])
    root.mkdir(exist_ok=False)
    cost = dict(qualification_encodes=0, science_train_encodes=0, validation_encodes=0,
                qualification_updates=0, meta_updates={a: 0 for a in ARMS},
                embedding_rows={a: 0 for a in ARMS}, support_view_rows={a: 0 for a in ARMS},
                query_rows={a: 0 for a in ARMS}, state_embedding_rows={a: 0 for a in ARMS},
                validation_embedding_rows={a: 0 for a in ARMS}, encoder_seconds={})
    live_models = {}

    def status(phase, **kwargs):
        cost.update(wall_from_first=time.time() - cfg['original_start_epoch'], GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(root / 'COSTS.private.json', cost)
        write(root / 'status.private.json', dict(state='RUNNING', phase=phase, updated=time.time(), cost=cost, **kwargs))
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        assert cost['GPU_peak_reserved_bytes'] <= cfg['gpu_reserved_budget_bytes']

    def stop(sig, frame):
        write(root / 'INTERRUPTED.private.json', dict(at=time.time(), cost=cost))
        for a, model in live_models.items():
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, root / ('interrupted_' + a + '.private.pt'))
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, stop)
    status('math_qualification')
    maths = math_check()
    train, val = read_rows(cfg, 'train'), read_rows(cfg, 'val')
    assert len(train) == 68058 and len(val) == 300
    assert not {r['group'] for r in train} & {r['group'] for r in val}
    base_rows = [r for r in train if r['label'] in cfg['order'][:10]]
    assert len(base_rows) == 60129
    support, query = patient_pools(base_rows, cfg['order'][:10], cfg['seed'])
    sp = sorted(set().union(*(set(v) for v in support.values())), key=int)
    qp = sorted(set().union(*(set(v) for v in query.values())), key=int)
    write(root / 'PATIENT_POOLS.private.json', dict(support_patients=sp, query_patients=qp))
    coverage = dict(train_counts=dict(Counter(str(r['label']) for r in train)), patient_overlap=0,
                    base_support_patients=len(sp), base_query_patients=len(qp),
                    per_class_pool_counts={str(c): dict(support_patients=len(support[c]), query_patients=len(query[c])) for c in support},
                    support_query_patient_overlap=0, future_classes_in_meta=0)
    write(root / 'COVERAGE.json', coverage)
    status('encoder_loading')
    import timm
    from safetensors.torch import load_file
    encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    encoder.load_state_dict({k: v for k, v in load_file(cfg['weight']).items() if not k.startswith('head.')}, strict=True)
    encoder.eval().requires_grad_(False).cuda()

    def extract(rows, field):
        features = torch.empty(len(rows), 1536, dtype=torch.float32)
        tick = time.time()
        for images, idx in DataLoader(Images(rows, cfg), batch_size=32, num_workers=0, shuffle=False):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                raw = encoder.forward_features(images.cuda())
            assert raw.shape[1:] == (197, 768) and torch.isfinite(raw).all()
            # Mean is FP32; no patch quantization beyond the frozen FP16 encoder.
            raw = raw.float()
            features[idx] = torch.cat((raw[:, 0], raw[:, 1:].mean(1)), 1).cpu()
            torch.cuda.synchronize()
            cost['encoder_seconds'].setdefault(field, 0.)
            cost['encoder_seconds'][field] += time.time() - tick
            cost[field] += len(idx)
            status(field, encoded_in_current_set=int(idx[-1]) + 1)
            tick = time.time()
        return features

    # Exactly one support and one query patient per base class, 20 unique rows.
    qe = episode(support, query, cfg['order'][:10], cfg, 0, qualification=True)
    qids = list(dict.fromkeys([j for ids in qe['support'] for j in ids] + qe['query']))
    assert len(qids) == 20
    qrows = [base_rows[j] for j in qids]
    qualification = extract(qrows, 'qualification_encodes')
    index = {j: i for i, j in enumerate(qids)}
    mapped_ep = dict(qe, support=[[index[j] for j in ids] for ids in qe['support']],
                     views=[[[index[j] for j in ids] for ids in view] for view in qe['views']], query=[index[j] for j in qe['query']])
    from verify import embed as manual_embed, scores as manual_scores, risk_objectives
    import numpy as np
    qreceipt = {}
    for arm in ARMS:
        torch.manual_seed(cfg['seed'])
        model = Comparator().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        risk, _, _, _ = episode_risk(model, qualification, mapped_ep)
        reductions = risk_objectives(risk.detach().double().cpu().numpy())
        loss = objective(risk, arm)
        assert abs(float(loss.detach()) - reductions[arm]) < 2e-6
        opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        assert norm > 0
        opt.step()
        cost['qualification_updates'] += 1
        clone = copy.deepcopy(model).double().eval()
        state = {k: v.detach().double().cpu().numpy() for k, v in clone.state_dict().items()}
        raw = qualification.double().cuda()
        with torch.no_grad():
            z = clone.embed(raw)
            means = z[[ids[0] for ids in mapped_ep['support']]]
            var = torch.zeros_like(means)
            score = clone.scores(z[mapped_ep['query']], means, var).cpu().numpy()
        iz = manual_embed(qualification.numpy(), state)
        independent = manual_scores(iz[mapped_ep['query']], iz[[ids[0] for ids in mapped_ep['support']]], np.zeros((10, 64)), state)
        error = float(np.max(np.abs(score - independent)))
        assert error <= cfg['score_absolute_tolerance']
        qreceipt[arm] = dict(loss=float(loss.detach()), gradient_norm=float(norm), independent_score_error=error)
        del model, opt, clone, raw, z, means, var, loss, risk
    assert cost['qualification_encodes'] == 20 and cost['qualification_updates'] == 3
    write(root / 'QUALIFICATION.json', dict(state='PASS', math=maths, real_encodes=20, engineering_updates=3, validation_encodes=0, arms=qreceipt))
    del qualification, qrows
    torch.cuda.empty_cache()
    # All currently available base data may be held only until Task1 finishes.
    features = extract(base_rows, 'science_train_encodes')
    models, opts = {}, {}
    for arm in ARMS:
        torch.manual_seed(cfg['seed'])
        models[arm] = Comparator().cuda()
        opts[arm] = torch.optim.AdamW(models[arm].parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        live_models[arm] = models[arm]
    for arm in ['WORST', 'PAIRED']:
        assert all(torch.equal(v, models[arm].state_dict()[k]) for k, v in models['META'].state_dict().items())
    curves = []
    for step in range(1, cfg['meta_episodes'] + 1):
        ep = episode(support, query, cfg['order'][:10], cfg, step)
        support_groups = {base_rows[j]['group'] for ids in ep['support'] for j in ids}
        query_groups = {base_rows[j]['group'] for j in ep['query']}
        assert not support_groups & query_groups
        assert all(base_rows[j]['label'] == ep['classes'][label] for j, label in zip(ep['query'], ep['labels']))
        rec = dict(step=step, arms={})
        for arm in ARMS:
            opt = opts[arm]
            opt.zero_grad()
            risk, n, ns, nq = episode_risk(models[arm], features, ep)
            loss = objective(risk, arm)
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(models[arm].parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            cost['meta_updates'][arm] += 1
            cost['embedding_rows'][arm] += n
            cost['support_view_rows'][arm] += ns
            cost['query_rows'][arm] += nq
            if step == 1 or step % 25 == 0:
                rec['arms'][arm] = dict(loss=float(loss.detach()), risk_grid=risk.detach().double().cpu().tolist(), mean_risk=float(risk.detach().mean()), worst_class_mean=float(risk.detach().mean(0).max()), worst_pair=float(risk.detach().max()), gradient_norm=float(grad), worst_view=int(risk.detach().max(1).values.argmax()), worst_class=int(risk.detach().max(0).values.argmax()))
        if rec['arms']:
            curves.append(rec)
            write(root / 'TRAINING_CURVES.json', curves)
        if step == 1 or step % 5 == 0:
            status('base_meta_training', step=step, total=cfg['meta_episodes'])
    del opts, loss, risk
    for arm in ARMS:
        models[arm].eval().requires_grad_(False).double()
        torch.save({k: v.cpu() for k, v in models[arm].state_dict().items()}, root / (arm + '.private.pt'))
    write(root / 'MODEL_SEAL.json', dict(at=time.time(), steps_per_arm=3000, base_train_encodes=cost['science_train_encodes'], future_class_encodes=0, validation_encodes=0, parameters_per_arm=sum(p.numel() for p in models['META'].parameters())))
    states = {setting: {a: {} for a in ARMS} for setting in cfg['evaluation_support_settings']}
    seals = []

    def accumulate(rows, raw):
        for c in sorted({r['label'] for r in rows}, key=cfg['order'].index):
            ids = [i for i, row in enumerate(rows) if row['label'] == c]
            groups = sorted({rows[i]['group'] for i in ids}, key=int)
            lookup = {g: j for j, g in enumerate(groups)}
            codes = torch.tensor([lookup[rows[i]['group']] for i in ids])
            counts = torch.bincount(codes, minlength=len(groups))
            for arm in ARMS:
                sums = torch.zeros(len(groups), 64, dtype=torch.float64)
                sumsq = torch.zeros_like(sums)
                for off in range(0, len(ids), 256):
                    with torch.no_grad():
                        z = models[arm].embed(raw[ids[off:off + 256]].double().cuda()).cpu()
                    sums.index_add_(0, codes[off:off + len(z)], z)
                    sumsq.index_add_(0, codes[off:off + len(z)], z.square())
                    cost['state_embedding_rows'][arm] += len(z)
                ranked = sorted(range(len(groups)), key=lambda j: ((int(groups[j]) * 1103515245 + cfg['seed']) % 2147483648, int(groups[j])))
                for setting in cfg['evaluation_support_settings']:
                    chosen = list(range(len(groups))) if setting == 'all' else ranked[:1 if setting == 'one' else 4]
                    mean = (sums[chosen] / counts[chosen, None]).mean(0)
                    second = (sumsq[chosen] / counts[chosen, None]).mean(0)
                    states[setting][arm][str(c)] = dict(mean=mean, variance=(second - mean.square()).clamp_min(0))
                torch.save(dict(groups=groups, counts=counts, sum=sums, sumsq=sumsq), root / ('audit_' + arm + '_' + str(c) + '.private.pt'))
            status('class_state_complete', class_id=c)

    for task in range(6):
        if task == 0:
            current_rows, raw = base_rows, features
        else:
            new = cfg['order'][sum(cfg['task_sizes'][:task]):sum(cfg['task_sizes'][:task + 1])]
            current_rows = [r for r in train if r['label'] in new]
            raw = extract(current_rows, 'science_train_encodes')
        accumulate(current_rows, raw)
        seen = cfg['order'][:sum(cfg['task_sizes'][:task + 1])]
        for arm in ARMS:
            assert all(set(states[s][arm]) == {str(c) for c in seen} for s in states)
            torch.save({s: states[s][arm] for s in states}, root / ('stage_' + str(task) + '_' + arm + '.private.pt'))
        seals.append(dict(task=task, at=time.time(), classes=seen, validation_encodes=0,
                          training_encodes=cost['science_train_encodes'], meta_updates=dict(cost['meta_updates'])))
        write(root / 'STAGE_SEALS.json', seals)
        del raw, current_rows
        if task == 0:
            del features
        torch.cuda.empty_cache()
    assert cost['science_train_encodes'] == 68058
    assert all(v == 68058 for v in cost['state_embedding_rows'].values())
    for arm in ARMS:
        torch.save({s: states[s][arm] for s in states}, root / (arm + '_moments.private.pt'))
    write(root / 'FIT_SEAL.json', dict(at=time.time(), stages=6, query_opened=False, states_per_arm=20))
    write(root / 'QUERY_OPEN.private.json', dict(at=time.time(), validation_encodes=0))
    ordered_val = [r for c in cfg['order'] for r in val if r['label'] == c]
    assert all(sum(r['label'] == c for r in ordered_val) == 15 for c in cfg['order'])
    raw = extract(ordered_val, 'validation_encodes')
    labels = torch.tensor([r['label'] for r in ordered_val])
    torch.save(dict(features=raw, labels=labels), root / 'query.private.pt')
    saved = {}
    for arm in ARMS:
        saved[arm] = {}
        for setting in states:
            saved[arm][setting] = []
            for task in range(6):
                seen = seals[task]['classes']
                mean = torch.stack([states[setting][arm][str(c)]['mean'] for c in seen]).cuda()
                var = torch.stack([states[setting][arm][str(c)]['variance'] for c in seen]).cuda()
                chunks = []
                for batch in raw.split(16):
                    with torch.no_grad():
                        chunks.append(models[arm].scores(models[arm].embed(batch.double().cuda()), mean, var).cpu())
                    cost['validation_embedding_rows'][arm] += len(batch)
                saved[arm][setting].append(torch.cat(chunks))
                status('validation_scoring', arm=arm, setting=setting, task=task)
    torch.save(dict(scores=saved, labels=labels, order=cfg['order']), root / 'scores.private.pt')
    write(root / 'COSTS.json', cost)
    status('COMPLETE_PENDING_VERIFICATION')


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:
        main(cfg)
    except BaseException:
        root = Path(cfg['output'])
        root.mkdir(exist_ok=True)
        write(root / 'FAILURE.private.json', dict(at=time.time(), error=traceback.format_exc()))
        raise
