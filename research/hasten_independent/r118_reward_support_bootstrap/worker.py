"""Fixed shared balanced support epoch, then matched incremental RL controls."""
import csv
import gc
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .core import write

ARMS = ['SUPPORT_TARS_FULL_IS', 'SUPPORT_GRPO_FULL_IS', 'SUPPORT_CE_FULL_IS']


def dependency(cfg):
    root = Path(cfg['allocation_source'])
    assert json.loads((root / 'CONFIG.private.json').read_text())['source_commit'] == '289407261df7cdad58a41bc4151d96dcc4d5b80a'
    assert json.loads((root / 'CONTROL_EXIT.json').read_text())['exit_code'] == 0
    name = 'q117_dependency'
    spec = importlib.util.spec_from_file_location(name, root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module;spec.loader.exec_module(module)
    return __import__(name + '.worker', fromlist=['make_experiment'])


def coefficients(e, arm):
    return e.coefficients('ALLOC_GRPO_FULL_IS' if arm == 'SUPPORT_GRPO_FULL_IS' else 'ALLOC_TARS_FULL_IS')


def audited_rl(e, features, labels, opt, pg, kl, seed):
    """Observe the unchanged sampler return; never add/filter/reweight responses."""
    generate = e.engine.generate
    captured = []
    def capture(*args, **kwargs):
        result = generate(*args, **kwargs)
        captured.append((result[3], result[4].detach().cpu()))
        return result
    e.engine.generate = capture
    try:
        record = e.rl_corrected(features, labels, opt, pg, kl, seed)
    finally:
        e.engine.generate = generate
    assert len(captured) == 1
    text, done = captured[0]
    normalize = lambda s: re.sub(r'[\s_\-]+', '', s.lower().strip())
    target = [normalize(e.names[int(c)]) for c in labels for _ in range(8)]
    rewards = torch.tensor([int(t == y and bool(d)) for t, y, d in zip(text, target, done)]).reshape(-1, 8)
    sums = rewards.sum(-1)
    record['allwrong_groups_by_class'] = torch.bincount(labels[sums == 0], minlength=23).tolist()
    record['allright_groups_by_class'] = torch.bincount(labels[sums == 8], minlength=23).tolist()
    record['mixed_groups_by_class'] = torch.bincount(labels[(sums > 0) & (sums < 8)], minlength=23).tolist()
    assert sum(record['allwrong_groups_by_class']) + sum(record['allright_groups_by_class']) == record['constant_groups']
    assert torch.bincount(labels, weights=sums.float(), minlength=23).tolist() == record['correct_by_class']
    return record


def native(e, dep):
    e.status('NATIVE_ENGINEERING')
    fixture = dep.objective_check()
    features, _ = e.engine.encode(e.images(e.train, [0, 1]));e.counts['TRAIN_encoder_rows'] += 2
    timings = {};details = []
    for arm in ['SUPPORT_STAGE'] + ARMS:
        e.engine.restore(e.sft);e.engine.freeze_reference()
        opt = e.optimizer(5e-6 if arm == 'SUPPORT_STAGE' else 2e-5)
        start = time.time()
        pg, kl = coefficients(e, arm)
        if arm in ['SUPPORT_STAGE', 'SUPPORT_CE_FULL_IS']:
            loss, norm = e.ce(features, e.labels[:2], opt, torch.ones(23, device='cuda') if arm == 'SUPPORT_STAGE' else pg)
            rec = {'loss': loss, 'gradient_norm': norm}
        else:
            rec = audited_rl(e, features, e.labels[:2], opt, pg, kl, 118000)
        e.counts['engineering_optimizer_updates'] += 1
        rec['parameter_delta_L2'] = e.last_parameter_delta_L2
        assert rec['parameter_delta_L2'] >= 0
        if rec['gradient_norm'] == 0:
            assert rec['parameter_delta_L2'] == 0  # Fresh Adam and no weight decay: a legal zero-signal step.
        torch.cuda.synchronize();timings[arm] = time.time() - start;details.append({'arm': arm, 'result': rec})
        del opt
    e.engine.restore(e.sft);e.engine.freeze_reference()
    assert all(torch.equal(a.a, a.ref_a) and torch.equal(a.b, a.ref_b) for a in e.engine.adapters)
    forecast = 1.5 * (6821 / 2 * timings['SUPPORT_STAGE'] + 4 * 256 * sum(timings[a] for a in ARMS)) + 1800
    passed = forecast < e.cfg['deadline'] - time.time()
    write(e.out / 'NATIVE_ADMISSION.json', {'passed': passed, 'objective_fixture': fixture,
          'two_query_seconds': timings, 'paths': details, 'engineering_only': True,
          'TRAIN_images': 2, 'VAL_TEST': 0, 'original_SFT_and_reference_restored': True,
          'full_forecast_seconds_with_margin_and_reserve': forecast, 'remaining_seconds': e.cfg['deadline'] - time.time()})
    assert passed, 'FULL_MATRIX_NOT_TIME_ADMITTED'


def science(e):
    cfg = e.cfg
    data = np.memmap(Path(cfg['base_source']) / 'TRAIN_FEATURES.private.bin', mode='r', dtype=np.uint16, shape=(6821, 64, 2048))
    generator = torch.Generator().manual_seed(11874005)
    classes = torch.randint(23, (6821,), generator=generator)
    pools = {c: torch.where(e.labels == c)[0] for c in range(23)}
    tape = torch.stack([pools[int(c)][torch.randint(len(pools[int(c)]), (), generator=generator)] for c in classes])
    torch.save(tape, e.out / 'SUPPORT_QUERY_TAPE.private.pt')
    exposure = [{'class': c, 'queries': int((classes == c).sum()),
                 'unique_TRAIN_components': len(set(tape[classes == c].tolist()))} for c in range(23)]
    assert sum(r['queries'] for r in exposure) == 6821
    write(e.out / 'SUPPORT_EXPOSURE.json', {'rows': exposure, 'new_independent_data': 0, 'same_shared_support_for_all_arms': True})
    e.records = [];opt = e.optimizer(5e-6);ones = torch.ones(23, device='cuda')
    for k, lo in enumerate(range(0, 6821, 8)):
        ids = tape[lo:lo+8];e.status('SHARED_SUPPORT', update=k, total=853);start = time.time()
        loss, norm = e.ce(e.features(data, ids), e.labels[ids], opt, ones)
        e.counts['SFT_updates'] += 1
        e.records.append({'phase': 'SHARED_SUPPORT', 'arm': 'SUPPORT_SFT', 'update': k, 'queries': len(ids),
                          'loss': loss, 'gradient_norm': norm, 'parameter_delta_L2': e.last_parameter_delta_L2,
                          'seconds': time.time() - start})
        if k % 8 == 0:write(e.out / 'TRAINING_CURVES.json', e.records)
    common = e.engine.snapshot();e.engine.freeze_reference();del opt
    torch.save(common, e.out / 'SUPPORT_SFT.private.pt')
    write(e.out / 'SUPPORT_SFT_SEALED.json', {'updates': 853, 'queries': 6821, 'epoch': time.time(), 'VAL_TEST': 0})
    post_tape = torch.load(Path(cfg['allocation_source']) / 'QUERY_TAPE.private.pt', weights_only=True, map_location='cpu')
    assert post_tape.shape == (256, 8)
    write(e.out / 'ALLOCATION.json', json.loads((Path(cfg['allocation_source']) / 'ALLOCATION.json').read_text()))
    for arm in ARMS:
        e.engine.restore(common)
        assert all(torch.equal(a.a, a.ref_a) and torch.equal(a.b, a.ref_b) for a in e.engine.adapters)
        opt = e.optimizer(2e-5);pg, kl = coefficients(e, arm)
        for k, ids in enumerate(post_tape):
            e.status('POSTTRAINING', arm=arm, update=k, total=256);start = time.time()
            f, y = e.features(data, ids), e.labels[ids]
            if arm == 'SUPPORT_CE_FULL_IS':
                loss, norm = e.ce(f, y, opt, pg);rec = {'loss': loss, 'gradient_norm': norm, 'groups': 0}
            else:rec = audited_rl(e, f, y, opt, pg, kl, 11480000+k)
            e.counts['posttraining_optimizer_updates'] += 1
            rec.update(phase='POSTTRAINING', arm=arm, update=k, parameter_delta_L2=e.last_parameter_delta_L2,
                       seconds=time.time()-start);e.records.append(rec)
            if k % 8 == 0:write(e.out / 'TRAINING_CURVES.json', e.records)
        torch.save(e.engine.snapshot(), e.out / (arm + '.private.pt'))
        write(e.out / (arm + '_SEALED.json'), {'updates': 256, 'epoch': time.time(), 'common_initialization_and_reference_verified': True, 'VAL_TEST': 0})
        del opt;gc.collect();torch.cuda.empty_cache()
    write(e.out / 'TRAINING_CURVES.json', e.records)
    write(e.out / 'ALL_BRANCHES_SEALED.json', {'arms': ['SUPPORT_SFT'] + ARMS, 'epoch': time.time(), 'new_VAL_TEST_so_far': 0})
    rows = list(csv.DictReader((Path(cfg['manifest']) / 'val.csv').open()))
    assert len(rows) == 1698 and not {r['identity_component'] for r in rows} & {r['identity_component'] for r in e.train}
    val = np.memmap(Path(cfg['base_source']) / 'VAL_FEATURES.private.bin', mode='r', dtype=np.uint16, shape=(1698, 64, 2048))
    results = []
    for arm in ['SUPPORT_SFT'] + ARMS:
        e.engine.restore(torch.load(e.out / (arm + '.private.pt'), weights_only=True, map_location='cpu'))
        results.append(e.evaluate(val, rows, arm))
    support, main, ordinary, ce = results
    gates = {'macro_over_support_ge_1pp': main['macro']-support['macro'] >= .01,
             'tail_over_support_ge_2pp': main['tail']-support['tail'] >= .02,
             'head_over_support_ge_minus1pp': main['head']-support['head'] >= -.01,
             'macro_over_GRPO_positive': main['macro'] > ordinary['macro'],
             'tail_over_GRPO_positive': main['tail'] > ordinary['tail'],
             'macro_over_CE_positive': main['macro'] > ce['macro'], 'tail_over_CE_positive': main['tail'] > ce['tail']}
    reward_support = {arm: {str(c): int(sum(rec['correct_by_class'][c] for rec in e.records if rec['arm'] == arm))
                           for c in [5, 6, 17, 19]} for arm in ARMS[:2]}
    historical = json.loads((Path(cfg['allocation_source']) / 'FINAL_RESULTS.json').read_text())
    original = historical['historical_SFT']
    gates.update(end_to_end_macro_over_original_ge_1pp=main['macro']-original['macro'] >= .01,
                 end_to_end_tail_over_original_ge_2pp=main['tail']-original['tail'] >= .02,
                 end_to_end_head_over_original_ge_minus1pp=main['head']-original['head'] >= -.01)
    write(e.out / 'FINAL_RESULTS.json', {'results': results, 'gates': gates, 'all_gates_pass': all(gates.values()),
          'successful_responses_formerly_zero_classes': reward_support,
          'any_formerly_zero_class_discovered_by_main': any(reward_support[ARMS[0]].values()),
          'historical_original_SFT': historical['historical_SFT'], 'historical_R117_primary': historical['results'][0],
          'seed': 74005, 'TEST': 0, 'independent_confirmation': False, 'patient_independence_verified': False,
          'scope': 'static_shared_support_and_incremental_RL_diagnostic'})
    e.status('COMPLETE_READY_FOR_SAVED_VERIFICATION', all_gates_pass=all(gates.values()))


def run(cfg):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction((56*2**30-128*2**20)/torch.cuda.get_device_properties(0).total_memory)
    e = None
    try:
        dep = dependency(cfg);e = dep.make_experiment(cfg)
        step = e.step
        def measured_step(opt):
            before = [p.detach().clone() for p in e.engine.parameters]
            norm = step(opt)
            e.last_parameter_delta_L2 = sum(float((p.detach()-old).double().square().sum())
                                             for p, old in zip(e.engine.parameters, before)) ** .5
            return norm
        e.step = measured_step
        native(e, dep);science(e)
    finally:
        if e is not None:write(e.out / 'COSTS.json', {'counts': e.counts, 'CPU_seconds': time.process_time()-e.cpu,
              'wall_seconds': time.time()-e.begin, 'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'shared_support_queries': sum(r.get('queries', 0) for r in e.records if r['phase'] == 'SHARED_SUPPORT'),
              'historical_training_not_newly_executed': True})
