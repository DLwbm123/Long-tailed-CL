"""Equal growing architecture: freeze old functions versus train every branch."""
import copy
import json
import os
import signal
import time
import traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS, Model, objective, memory_indices
from data import read_rows, Images
from verify import numpy_forward, numpy_loss, summarize


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def mathematical_check(cfg):
    torch.manual_seed(cfg['seed'])
    means = torch.randn(3, 8, dtype=torch.float64)
    model = Model(8, 3, means).double()
    with torch.no_grad():
        model.branches[0].up.weight.normal_(0, .1)
    x = torch.randn(12, 8, dtype=torch.float64)
    y = torch.tensor([0, 0, 1, 1, 2, 2] * 2)
    ids = torch.tensor(list(range(6)) * 2)
    receipts = {}
    for arm in ARMS:
        model.zero_grad()
        loss, _, _ = objective(model, x, y, ids, arm, cfg)
        loss.backward()
        s = {k: v.numpy().astype(np.float64) for k, v in state(model).items()}
        value = numpy_loss(x.numpy(), y.numpy(), ids.numpy(), s, arm, cfg)
        objective_error = abs(float(loss.detach()) - value)
        gradient_errors = []
        for name, position in [('head', (0, 0)), ('branches.0.up.weight', (2, 1)), ('branches.0.down.bias', (0,))]:
            plus = {k: v.copy() for k, v in s.items()}
            minus = {k: v.copy() for k, v in s.items()}
            plus[name][position] += 1e-5
            minus[name][position] -= 1e-5
            delta = (numpy_loss(x.numpy(), y.numpy(), ids.numpy(), plus, arm, cfg) -
                     numpy_loss(x.numpy(), y.numpy(), ids.numpy(), minus, arm, cfg)) / 2e-5
            gradient_errors.append(abs(delta - float(dict(model.named_parameters())[name].grad[position])))
        assert objective_error < 1e-9 and max(gradient_errors) < 2e-6
        perm = torch.tensor([5, 3, 1, 0, 4, 2, 11, 9, 7, 6, 10, 8])
        permuted = objective(model, x[perm], y[perm], ids[perm], arm, cfg)[0]
        assert abs(float(permuted.detach()) - float(loss.detach())) < 1e-10
        receipts[arm] = dict(objective_error=objective_error, finite_difference_gradient_error=max(gradient_errors),
                             permutation_invariant=True, synthetic_torch_objectives=2, synthetic_numpy_objectives=7)
    with torch.no_grad():
        score_before = model(x, cfg['logit_scale'])[1]
        extended = copy.deepcopy(model)
        extended.append_branch(True)
        score_after = extended(x, cfg['logit_scale'])[1]
    expansion_error = float((score_after - score_before / (2 ** .5)).abs().max())
    assert expansion_error < 1e-9 and torch.equal(score_before.argmax(1), score_after.argmax(1))
    for receipt in receipts.values():
        receipt['old_only_zero_padded_head_extension_error'] = expansion_error
        receipt['extension_torch_model_rows'] = 24
    return receipts


def main(cfg):
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(cfg['output'])
    root.mkdir(exist_ok=False)
    cost = dict(qualification_encodes=0, science_train_encodes=0, validation_encodes=0,
                qualification_updates=0, qualification_model_rows={a: 0 for a in ARMS},
                qualification_init_embedding_rows={a: 0 for a in ARMS},
                qualification_freeze_probe_rows={a: 0 for a in ARMS},
                head_initialization_embedding_rows={a: 0 for a in ARMS},
                head_initialization_branch_rows={a: 0 for a in ARMS}, current_class_input_mean_rows=0,
                training_branch_rows={a: 0 for a in ARMS}, gradient_branch_rows={a: 0 for a in ARMS},
                validation_branch_rows={a: 0 for a in ARMS},
                qualification_FP64_model_rows={a: 0 for a in ARMS}, qualification_numpy_rows={a: 0 for a in ARMS},
                updates={a: 0 for a in ARMS}, sampled_image_rows={a: 0 for a in ARMS},
                training_view_rows={a: 0 for a in ARMS}, validation_model_rows={a: 0 for a in ARMS},
                encoder_parameter_updates=0, encoder_seconds={}, contrastive_similarity_pairs={a: 0 for a in ARMS})
    models, optimizers, curves, seals = {}, {}, [], []

    def status(phase, **kwargs):
        cost.update(wall_from_first=time.time()-cfg['original_start_epoch'],
                    GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(root / 'COSTS.private.json', cost)
        write(root / 'status.private.json', dict(state='RUNNING', phase=phase, updated=time.time(), cost=cost, **kwargs))
        assert time.time() < cfg['hard_deadline_epoch'] - 60
        assert cost['GPU_peak_reserved_bytes'] < cfg['gpu_reserved_budget_bytes']

    def interrupt(sig, frame):
        save(root / 'interrupted.private.pt', dict(models={a: state(m) for a, m in models.items()},
             optimizers={a: o.state_dict() for a, o in optimizers.items()}, cost=cost))
        write(root / 'INTERRUPTED.private.json', dict(at=time.time(), cost=cost))
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, interrupt)
    status('mathematical_qualification')
    math_receipt = mathematical_check(cfg)
    write(root / 'MATH_QUALIFICATION.json', math_receipt)
    train, val = read_rows(cfg, 'train'), read_rows(cfg, 'val')
    for i, row in enumerate(train):
        row['id'] = i
    assert len(train) == 68058 and len(val) == 300
    assert not {r['group'] for r in train} & {r['group'] for r in val}
    write(root / 'COVERAGE.json', dict(train_counts=dict(Counter(str(r['label']) for r in train)),
          patient_overlap=0, base_images=sum(r['label'] in cfg['order'][:10] for r in train),
          total_train=68058, total_val=300, new_feature_memory_permission=True,
          historical_image_replay=False, reconstruction_of_deleted_cache=False))
    import timm
    from safetensors.torch import load_file
    status('encoder_loading')
    encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    encoder.load_state_dict({k: v for k, v in load_file(cfg['weight']).items() if not k.startswith('head.')}, strict=True)
    encoder.eval().requires_grad_(False).cuda()

    def extract(rows, field):
        out = torch.empty(len(rows), cfg['input_dimension'], dtype=torch.float32)
        tick = time.time()
        for images, idx in DataLoader(Images(rows, cfg), batch_size=32, num_workers=0, shuffle=False):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                raw = encoder.forward_features(images.cuda())
            assert raw.shape[1:] == (197, 768) and torch.isfinite(raw).all()
            raw = raw.float()
            out[idx] = F.normalize(torch.cat([raw[:, 0], raw[:, 1:].mean(1)], 1), dim=1).cpu()
            torch.cuda.synchronize()
            cost[field] += len(idx)
            cost['encoder_seconds'][field] = cost['encoder_seconds'].get(field, 0) + time.time() - tick
            status(field, encoded_in_current_set=int(idx[-1])+1, current_count=len(rows))
            tick = time.time()
        return out

    qrows = []
    for c in cfg['order'][:10]:
        groups = set()
        for r in train:
            if r['label'] == c and r['group'] not in groups:
                qrows.append(r)
                groups.add(r['group'])
                if len(groups) == 2:
                    break
        assert len(groups) == 2
    q = extract(qrows, 'qualification_encodes')
    torch.manual_seed(cfg['seed'])
    qmeans = torch.stack([q[2*i:2*i+2].mean(0) for i in range(10)])
    template = Model(cfg['input_dimension'], cfg['adapter_rank'], qmeans[:8])
    views = q.repeat(2, 1)
    gen = torch.Generator().manual_seed(cfg['seed']+2)
    views *= (torch.rand(views.shape, generator=gen) >= cfg['input_dropout']) / (1-cfg['input_dropout'])
    y = torch.arange(10).repeat_interleave(2).repeat(2)
    identities = torch.arange(20).repeat(2)
    qualification = {}
    for arm in ARMS:
        model = copy.deepcopy(template)
        model.append_branch(arm == 'FROZEN_PAST')
        with torch.no_grad():
            new_z = model.embedding(q[16:])
            new_means = torch.stack([new_z[0:2].mean(0), new_z[2:4].mean(0)])
            before = model.branches[0](q).clone()
        model.append_class_rows(new_means)
        cost['qualification_init_embedding_rows'][arm] += 4
        cost['qualification_freeze_probe_rows'][arm] += 20
        before_state = {k: v.clone() for k, v in model.branches[0].state_dict().items()}
        loss, ce, aux = objective(model, views, y, identities, arm, cfg)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert sum(float(p.grad.square().sum()) for p in model.parameters() if p.requires_grad) > 0
        cost['qualification_model_rows'][arm] += len(views)
        s = {k: v.numpy().astype(np.float64) for k, v in state(model).items()}
        ref = numpy_forward(views.numpy(), s, cfg['logit_scale'])[1]
        with torch.no_grad():
            fp64 = copy.deepcopy(model).double()(views.double(), cfg['logit_scale'])[1]
        error = float(np.max(abs(ref-fp64.numpy())))
        assert error < cfg['score_absolute_tolerance']
        cost['qualification_FP64_model_rows'][arm] += len(views)
        cost['qualification_numpy_rows'][arm] += len(views)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        opt.step()
        cost['qualification_updates'] += 1
        with torch.no_grad():
            after = model.branches[0](q)
        cost['qualification_freeze_probe_rows'][arm] += 20
        change = float((before-after).abs().max())
        fixed = all(torch.equal(v, model.branches[0].state_dict()[k]) for k, v in before_state.items())
        if arm == 'FROZEN_PAST':
            assert fixed and change == 0 and all(p.grad is None for p in model.branches[0].parameters())
        else:
            assert not fixed and change > 0
        qualification[arm] = dict(loss=float(loss.detach()), CE=float(ce.detach()), auxiliary=float(aux.detach()),
                                  score_error=error, engineering_updates=1, old_branch_parameters_fixed=fixed, old_branch_probe_max_change=change)
    write(root / 'QUALIFICATION.json', dict(state='PASS', math=math_receipt, real=qualification,
          real_training_encodes=20, engineering_updates=2, validation_encodes=0,
          timing_is_not_formal_throughput=True))
    del model, template, q, qrows, qmeans, opt, views, loss, ce, aux, fp64
    torch.cuda.empty_cache()
    memory_features = torch.empty(0, cfg['input_dimension'])
    memory_rows = []
    for task in range(6):
        seen = cfg['order'][:sum(cfg['task_sizes'][:task+1])]
        new = seen[sum(cfg['task_sizes'][:task]):]
        rows = [r for r in train if r['label'] in new]
        features = extract(rows, 'science_train_encodes')
        means = []
        for c in new:
            patients = {}
            for i, r in enumerate(rows):
                if r['label'] == c:
                    patients.setdefault(r['group'], []).append(i)
            means.append(torch.stack([features[ix].double().mean(0) for ix in patients.values()]).mean(0).float())
        means = torch.stack(means)
        cost['current_class_input_mean_rows'] += len(features)
        initialization_record = {}
        if task == 0:
            torch.manual_seed(cfg['seed'])
            template = Model(cfg['input_dimension'], cfg['adapter_rank'], means)
            models = {a: copy.deepcopy(template).cuda() for a in ARMS}
            initialization_record = {a: means.clone() for a in ARMS}
            del template
        else:
            for a, m in models.items():
                m.append_branch(a == 'FROZEN_PAST')
                embedded = []
                with torch.no_grad():
                    for block in features.split(256):
                        embedded.append(m.embedding(block.cuda()).cpu())
                        cost['head_initialization_embedding_rows'][a] += len(block)
                        cost['head_initialization_branch_rows'][a] += len(block) * (task+1)
                embedded = torch.cat(embedded)
                embedded_means = []
                for c in new:
                    patients = {}
                    for i, r in enumerate(rows):
                        if r['label'] == c:
                            patients.setdefault(r['group'], []).append(i)
                    embedded_means.append(torch.stack([embedded[ix].double().mean(0) for ix in patients.values()]).mean(0).float())
                initialization_record[a] = torch.stack(embedded_means)
                m.append_class_rows(initialization_record[a].cuda())
                del embedded, embedded_means
        optimizers = {a: torch.optim.AdamW(m.parameters(), lr=cfg['learning_rate'],
                      weight_decay=cfg['weight_decay']) for a, m in models.items()}
        # Old data consist only of explicit bounded memory, current data of this arrival.
        source = torch.cat([memory_features, features])
        source_rows = memory_rows + rows
        pools = {}
        for i, r in enumerate(source_rows):
            pools.setdefault(r['label'], {}).setdefault(r['group'], []).append(i)
        assert set(pools) == set(seen)
        rng = np.random.default_rng(cfg['seed'] + 1000 + task)
        view_rng = torch.Generator().manual_seed(cfg['seed'] + 2000 + task)
        steps = cfg['base_updates'] if task == 0 else cfg['later_updates']
        step_start = time.time()
        for step in range(1, steps+1):
            chosen_classes = rng.choice(seen, size=cfg['classes_per_batch'], replace=False).tolist()
            indices = []
            for c in chosen_classes:
                gs = list(pools[c])
                for _ in range(cfg['draws_per_class']):
                    g = gs[int(rng.integers(len(gs)))]
                    values = pools[c][g]
                    indices.append(values[int(rng.integers(len(values)))])
            x = source[indices].repeat(2, 1)
            x *= (torch.rand(x.shape, generator=view_rng) >= cfg['input_dropout']) / (1-cfg['input_dropout'])
            x = x.cuda()
            y = torch.tensor([seen.index(source_rows[i]['label']) for i in indices]*2, device='cuda')
            identity = torch.tensor([source_rows[i]['id'] for i in indices]*2, device='cuda')
            details = {}
            for a in ARMS:
                opt = optimizers[a]
                opt.zero_grad()
                loss, ce, aux = objective(models[a], x, y, identity, a, cfg)
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(models[a].parameters(), cfg['gradient_clip'])
                assert torch.isfinite(norm)
                opt.step()
                cost['updates'][a] += 1
                cost['sampled_image_rows'][a] += len(indices)
                cost['training_view_rows'][a] += len(x)
                cost['training_branch_rows'][a] += len(x) * (task+1)
                cost['gradient_branch_rows'][a] += len(x) * ((task+1) if a == 'ALL_TRAINABLE' else 1)
                # Both arms use CE only; no contrastive matrix or teacher.
                details[a] = dict(loss=float(loss.detach()), CE=float(ce.detach()), auxiliary=float(aux.detach()),
                                  gradient_norm_before_clip=float(norm))
            if step == 1 or step % 100 == 0 or step == steps:
                torch.cuda.synchronize()
                curves.append(dict(task=task, step=step, arms=details, at=time.time(),
                                   elapsed_stage_training=time.time()-step_start))
                write(root / 'TRAINING_CURVES.json', curves)
                status('actual_stage_training', task=task, step=step, steps=steps, seen=seen)
        selected = memory_indices(rows, cfg['memory_per_class'], cfg['seed'])
        memory_features = torch.cat([memory_features, features[selected]])
        memory_rows.extend(rows[i] for i in selected)
        assert len(memory_rows) <= len(seen) * cfg['memory_per_class']
        assert len(memory_rows) == len({(r['label'], r['group']) for r in memory_rows})
        save(root / f'stage_{task}.private.pt', dict(seen=seen, models={a: state(m) for a, m in models.items()},
             memory_features=memory_features, memory_rows=memory_rows, new_head_initialization_means=initialization_record))
        seals.append(dict(task=task, at=time.time(), seen=seen, validation_encodes=0,
                          science_train_encodes=cost['science_train_encodes'], updates=dict(cost['updates']),
                          memory_images=len(memory_rows), parameters_per_arm=sum(p.numel() for p in models['ALL_TRAINABLE'].parameters()),
                          trainable_parameters={a: sum(p.numel() for p in m.parameters() if p.requires_grad) for a, m in models.items()}))
        write(root / 'STAGE_SEALS.json', seals)
        del source, source_rows, features, rows, pools, x, y, loss, ce, aux
        torch.cuda.empty_cache()
    assert cost['science_train_encodes'] == 68058
    write(root / 'FIT_SEAL.json', dict(at=time.time(), models=12, stages=6, validation_encodes=0))
    write(root / 'QUERY_OPEN.private.json', dict(at=time.time(), validation_encodes=0))
    query = extract(val, 'validation_encodes')
    labels = torch.tensor([r['label'] for r in val])
    save(root / 'query.private.pt', dict(features=query, labels=labels))
    stages = []
    for task in range(6):
        pack = torch.load(root / f'stage_{task}.private.pt', map_location='cpu', weights_only=False)
        seen = pack['seen']
        mask = torch.isin(labels, torch.tensor(seen))
        metrics = {}
        for a in ARMS:
            m = Model(cfg['input_dimension'], cfg['adapter_rank'], torch.zeros(len(seen), cfg['input_dimension']))
            for _ in range(task):
                m.append_branch(False)
            m = m.double()
            m.load_state_dict({k: v.double() for k, v in pack['models'][a].items()})
            with torch.inference_mode():
                _, score = m(query[mask].double(), cfg['logit_scale'])
            cost['validation_model_rows'][a] += int(mask.sum())
            cost['validation_branch_rows'][a] += int(mask.sum()) * (task+1)
            save(root / f'scores_{task}_{a}.private.pt', score)
            pred = torch.tensor(seen)[score.argmax(1)]
            correct = {str(c): int(((pred == c) & (labels[mask] == c)).sum()) for c in seen}
            metrics[a] = dict(correct=correct, total_correct=sum(correct.values()),
                              BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=task, seen=seen, metrics=metrics))
        status('sealed_model_scoring', task=task)
    result, decision = summarize(stages, cfg, Counter(r['label'] for r in train))
    write(root / 'RESULTS.json', result)
    write(root / 'DECISION.json', decision)
    status('scientific_complete')
    write(root / 'COSTS.json', cost)
    write(root / 'status.private.json', dict(state='COMPLETE_PENDING_VERIFY', updated=time.time(), cost=cost))


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:
        main(cfg)
    except BaseException as error:
        root = Path(cfg['output'])
        if root.exists():
            write(root / 'FAILURE.private.json', dict(at=time.time(), error=repr(error), traceback=traceback.format_exc()))
        raise
