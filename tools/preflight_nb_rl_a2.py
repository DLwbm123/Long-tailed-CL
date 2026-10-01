"""Bounded current-Task1 probe for the new conditions and native recovery."""
import copy
import hashlib
import json
import os
import time
import numpy as np
import torch
import run_nb_rl_a1 as r
if r.CFG.get('experiment') == 'NB-RL-A4':
    from nb_rl_a4_core import CONDITIONS, selfcheck
else:
    from nb_rl_a2_core import CONDITIONS, selfcheck


def check(run):
    run.phase = 'engineering'
    job = json.loads(os.environ['N78_JOB']); method = job['method']; seed = job.get('seed',1993)
    assert r.CFG['conditions'] == CONDITIONS
    math_result = selfcheck()
    model = r.Model(seed); model.method = method
    initial_hash = r.delta_hash(model.delta())
    z, y, rows, extract_seconds = run.extract(model, seed, 1, bounded=True)
    model.bank = r.append(r.empty(1536), z, y, range(2), 1)
    W2, _ = r.ridge(model.bank)
    rng = np.random.default_rng(84001)
    W = torch.tensor(np.column_stack([W2, rng.normal(size=(1536, 6)) * .01]), device='cuda', dtype=torch.float32)
    ds = r.BatchDataset(run, seed, 1, epoch=1, train=True, bounded=True)
    batches = [(x.cuda(), yy.cuda()) for _, x, yy in r.loader(ds, True)]
    weights = torch.ones(8, device='cuda')
    anchor = copy.deepcopy(model.net).requires_grad_(False).eval()
    # The expanded head is synthetic; no Task2 or validation images are opened.
    model.task = 2; model.epoch = 1; model.optimizer(3)
    assert model.opt.param_groups[0]['lr'] == CONDITIONS[method]['lr']
    durations = []; diagnostics = None
    for index in range(6):
        x, yy = batches[index % len(batches)]
        torch.cuda.synchronize(); start = time.monotonic()
        _, gd, _ = r.step(run, model, anchor, x, yy, W, weights, method, 2, diagnose=index == 1)
        torch.cuda.synchronize(); durations.append(time.monotonic() - start)
        if gd is not None: diagnostics = gd
    assert diagnostics['terms']['FD']['weighted_norm'] > 0
    if method == 'D': assert diagnostics['terms']['Exact']['weighted_norm'] > 0
    # Restore and repeat the next distinct real batch, including all optimizer/RNG state.
    model.batch = 1
    state = copy.deepcopy(model.snapshot())
    path = r.OUT / 'cursor_resume.pt'; r.dump(path, state)
    x, yy = batches[1]
    d1, _, a1 = r.step(run, model, anchor, x, yy, W, weights, method, 2)
    after = copy.deepcopy(model.snapshot())
    model.restore(r.load(path))
    d2, _, a2 = r.step(run, model, anchor, x, yy, W, weights, method, 2)
    assert a1 is a2 is None and d1 == d2 and r.tensors_equal(after, model.snapshot())
    assert len(run.allowed) == 96 and run.forbidden == 0
    r.save('PARALLEL_PROBE.json', dict(status='PASS', method=method, steps=8,
        seed=seed, **r.seed_fields(seed), initial_delta_sha256=initial_hash,
        first_batch_sha256=hashlib.sha256(batches[0][0].cpu().numpy().tobytes()+batches[0][1].cpu().numpy().tobytes()).hexdigest(),
        GPU=os.environ['CUDA_VISIBLE_DEVICES'], core_seconds=float(np.median(durations[2:])),
        diagnostic_seconds=durations[1], bounded_extract_seconds=extract_seconds,
        validation_images=0, future_fit_images=0, mathematical_checks=math_result,
        next_distinct_real_batch_exact=True, optimizer_scheduler_sigma_rng_exact=True,
        gradient=diagnostics, native_checkpoint_bytes=path.stat().st_size))
    run.resources()
    print('PREFLIGHT_PASS', method, flush=True)
