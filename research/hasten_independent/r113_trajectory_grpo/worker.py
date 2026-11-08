"""Native admission then actual GRPO updates and five-stage own-state policies."""
import copy
import csv
import gc
import importlib
import json
import os
import time
import traceback
from pathlib import Path
import torch
from . import trajectory as T

common = importlib.import_module('r110_ecv_opportunity.common')
encoding = importlib.import_module('r110_ecv_opportunity.encoding')
native112 = importlib.import_module('r112_evidence_memory.native')
write = common.write
ARMS = ['ZERO', 'BAL', 'MYOPIC_GRPO', 'LONG_GRPO', 'SUP_SHARED']


def actors(seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 113000)
        reference = T.Policy().to(device).eval().requires_grad_(False)
    policies = {a: copy.deepcopy(reference).train().requires_grad_(True) for a in ARMS[2:]}
    optimizers = {a: torch.optim.AdamW(p.parameters(), lr=1e-4, weight_decay=.001)
                  for a, p in policies.items()}
    return reference, policies, optimizers


def native(cfg, encoder, rows, first, count):
    start = time.time()
    count.phase = 'native'
    roles = importlib.import_module('r112_evidence_memory.supply').birth_roles(first, rows)
    ids = [i for i in first if roles[i] == 'S'][:30]
    times = []
    for _ in range(2):
        t = time.time()
        generic = encoder.encode(cfg, rows, ids, count, 'native', first)
        torch.cuda.synchronize()
        times.append(time.time() - t)
    pool = native112.fixture(generic, 3331, 0, torch.cat([torch.arange(20), torch.full((3311,), 20)]))
    future = [native112.fixture(generic, 64, 10000 + 1000 * h, torch.full((64,), 21 + h), 1 + h)
              for h in range(2)]
    m = T.old.learner.Learner(113).cuda()
    m.forbidden = {int(i) for x in [pool] + future for role in [1, 2] for i in x.role(role).parents}
    m.arrive(pool.role(0), 21)
    opt = T.old.learner.optimizer(m)
    T.old.learner.learn(m, opt, pool.role(0), 200, 113, count, None, 64, 'native_prefix')
    ref, policies, opts = actors(113, 'cuda')
    t = time.time()
    packed = T.prepare(pool, None, m, opt, 0, 113, count)
    torch.cuda.synchronize()
    proposal = time.time() - t
    t = time.time()
    batch = T.collect(pool, future, m, opt, None, policies['LONG_GRPO'], packed,
                      set(range(20)), 113, 0, count)
    history = T.update(policies['LONG_GRPO'], ref, opts['LONG_GRPO'], batch)
    sup = T.supervise(policies['SUP_SHARED'], opts['SUP_SHARED'], batch)
    torch.cuda.synchronize()
    group_seconds = time.time() - t + proposal
    # Full protocol, with duplicated proposal/update cost and a two-hour reserve.
    unit = group_seconds / 3500
    projected = 2 * (40 * group_seconds + 71000 * unit + 30 * proposal + 6821 * times[-1] / 30) + 7200
    result = {'passed': projected < cfg['deadline'] - time.time(),
              'engineering_only': True, 'repeated_generic_features_not_scientific_data': True,
              'wall_seconds': time.time() - start, 'encode30_seconds': times,
              'full_G4_H3_group_seconds': group_seconds, 'proposal_seconds': proposal,
              'complete_science_projected_seconds': projected, 'groups_per_worker': 40,
              'max_science_classifier_updates_per_worker': 212000,
              'native_actor_updates': len(history), 'native_supervised_updates': len(sup),
              'counts': dict(count)}
    write(Path(cfg['output']) / 'NATIVE_ADMISSION.json', result)
    del generic, pool, future, m, opt, batch, packed, policies, opts, ref
    gc.collect()
    torch.cuda.empty_cache()
    return result


def science(cfg, encoder, rows, labels, count):
    out = Path(cfg['output'])
    count.phase = 'science'
    states = {}
    for seed in [74003, 74004]:
        ref, policies, popts = actors(seed, 'cuda')
        arms = {}
        for arm in ARMS:
            model = T.old.learner.Learner(seed).cuda()
            arms[arm] = {'model': model, 'opt': T.old.learner.optimizer(model), 'memory': None}
        states[seed] = {'reference': ref, 'policies': policies, 'popts': popts, 'arms': arms}
    stages, group_records, policy_updates = [], [], {a: 0 for a in ARMS[2:]}
    for stage, classes in enumerate(cfg['stages']):
        common.guard(cfg)
        ids = [i for i in range(len(rows)) if int(labels[i]) in classes]
        write(out / 'STATUS.json', {'phase': 'SCIENCE_ENCODING_CURRENT_ARRIVAL', 'stage': stage,
                                   'actor_updates': policy_updates, 'counts': dict(count)})
        z = encoder.encode(cfg, rows, ids, count, 'science_TRAIN', ids)
        full = T.E.mem.from_views(z, torch.tensor(ids, device='cuda'), labels[ids].cuda())
        current = T.E.born(full, rows, stage)
        del z, full
        stage_results = []
        for seed in [74003, 74004]:
            state = states[seed]
            # On-policy LONG and MYOPIC batches; SUP receives the exact LONG data.
            for mode in ['LONG_GRPO', 'MYOPIC_GRPO']:
                for scenario in [0, 1]:
                    common.guard(cfg)
                    write(out / 'STATUS.json', {'phase': 'REAL_POLICY_TRAINING', 'stage': stage,
                                               'seed': seed, 'mode': mode, 'scenario': scenario,
                                               'groups_completed': len(group_records),
                                               'actor_updates': policy_updates, 'counts': dict(count)})
                    previous = state['arms'][mode]['memory']
                    pool, prev, future, tail, meta, m, opt, teacher = T.old.root(
                        previous, current, cfg, stage, scenario, count, True)
                    packed = T.prepare(pool, prev, m, opt, stage, meta['seed'], count, teacher)
                    tape = 11331001 + 100000 * (seed - 74003) + stage * 100 + scenario
                    batch = T.collect(pool, future, m, opt, teacher, state['policies'][mode],
                                      packed, tail, tape, stage, count)
                    history = T.update(state['policies'][mode], state['reference'], state['popts'][mode],
                                       batch, immediate=mode == 'MYOPIC_GRPO')
                    policy_updates[mode] += len(history)
                    supervised = []
                    if mode == 'LONG_GRPO':
                        supervised = T.supervise(state['policies']['SUP_SHARED'], state['popts']['SUP_SHARED'], batch)
                        policy_updates['SUP_SHARED'] += len(supervised)
                    record = {'stage': stage, 'seed': seed, 'mode': mode, 'scenario': scenario,
                              'real_prefix_only': True, 'pseudo_H3_not_real_future': True,
                              'updates': history, 'supervised_updates': supervised, **T.receipt(batch)}
                    group_records.append(record)
                    write(out / f'GROUP_{seed}_{stage}_{scenario}_{mode}.json', record)
                    del pool, prev, future, m, opt, teacher, packed, batch
                    gc.collect()
                    torch.cuda.empty_cache()
            for arm in ARMS:
                a = state['arms'][arm]
                m, opt, previous = a['model'], a['opt'], a['memory']
                pool = T.E.join(previous, current)
                m.forbidden = {int(i) for role in [1, 2] for i in pool.role(role).parents}
                teacher = T.old.learner.frozen(m) if m.C else None
                m.arrive(current.role(0), pool.C)
                event = seed + stage * 1000
                T.old.branch.paired_learn(m, opt, pool.role(0), 200, event, count, teacher, 64)
                if arm in ['ZERO', 'BAL']:
                    zbank, bbank, _ = T.E.anchors(pool, event + 1)
                    bank, selected, n = (zbank if arm == 'ZERO' else bbank), (0 if arm == 'ZERO' else 1), 2
                    del zbank, bbank
                else:
                    packed = T.prepare(pool, previous, m, opt, stage, event + 1, count, teacher)
                    with torch.no_grad():
                        lp = state['policies'][arm].log_probs(packed[1], packed[2], packed[3])
                        selected = int(lp.argmax())
                    bank, n = packed[0][selected].memory, len(packed[0])
                kept = pool.with_support(bank)
                T.old.branch.paired_learn(m, opt, kept.role(0), 100, event + 2, count, teacher, 64)
                # Isolated, descriptive outer score; never returned to actor training.
                audit = T.old.paired_labels.evaluate(m, kept.role(2), count)
                tail = [c for c in range(kept.C) if kept.stats[c]['birth'] < stage and kept.stats[c]['birth_total'] < 250]
                evidence_file = out / f'{seed}_{arm}_CURRENT.private.bin'
                size = kept.serialize(evidence_file)
                a['memory'] = T.E.Evidence.restore(evidence_file, 'cuda')
                checkpoint = {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'C': m.C}
                if arm in state['policies']:
                    checkpoint.update(policy=state['policies'][arm].state_dict(),
                                      policy_optimizer=state['popts'][arm].state_dict())
                torch.save(checkpoint, out / f'{seed}_{arm}_CURRENT.private.pt')
                result = {'stream': cfg['setting'], 'seed': seed, 'stage': stage, 'arm': arm,
                          'correct': audit.correct.tolist(), 'denominator': audit.denominator.tolist(),
                          'macro_BA': audit.ba(), 'oldtail_BA': audit.ba(set(tail)), 'oldtail_classes': tail,
                          'action': selected, 'candidate_count': n, 'evidence_bytes': size,
                          'S_by_class': torch.bincount(kept.role(0).y, minlength=kept.C).tolist()}
                stage_results.append(result)
                del pool, previous, teacher, kept, bank, checkpoint, audit
                if arm not in ['ZERO', 'BAL']:
                    del packed
            # No training records or feature decoder persists after this real stage.
        stages.extend(stage_results)
        write(out / f'STAGE_{stage}.json', stage_results)
        write(out / 'PROGRESS.json', {'stages_completed': stage + 1, 'policy_updates': policy_updates,
                                      'groups_completed': len(group_records), 'counts': dict(count)})
        del current
        gc.collect()
        torch.cuda.empty_cache()
    assert len(stages) == 50 and len(group_records) == 40
    updates = sum(v for k, v in count.items() if k.startswith('science_')
                  and ('classifier_optimizer' in k or k.endswith('paired_classifier_updates')))
    assert updates <= 212000, ('SCIENCE_UPDATE_BOUND', updates)
    write(out / 'RESULTS.json', {'stream': cfg['setting'], 'stages': stages,
                                'policy_updates': policy_updates, 'groups': len(group_records),
                                'science_classifier_updates': updates, 'counts': dict(count),
                                'independent_population_confirmation': False, 'VAL_TEST': 0})
    write(out / 'STATUS.json', {'phase': 'COMPLETE_REAL_POLICY_EXPERIMENT', 'policy_updates': policy_updates,
                               'groups': len(group_records), 'real_stage_rows': len(stages)})


def run(cfg):
    out, parent = Path(cfg['output']), Path(cfg['root_output'])
    start, cpu = time.time(), time.process_time()
    count = common.Counts()
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((12 * 2**30 - 64 * 2**20) / total, 0)
        rows = list(csv.DictReader((Path(cfg['manifest']) / 'train.csv').open()))
        assert len(rows) == len({r['identity_component'] for r in rows}) == 6821
        labels = torch.tensor([cfg['order'].index(int(r['original_label'])) for r in rows])
        assert torch.bincount(labels, minlength=23).tolist() == cfg['train_counts']
        encoder = encoding.Encoder(cfg)
        write(out / 'STATUS.json', {'phase': 'FULL_SHAPE_NATIVE_ADMISSION'})
        native(cfg, encoder, rows, [i for i in range(len(rows)) if int(labels[i]) < 5], count)
        while not (parent / 'ADMISSION.json').exists():
            if (parent / 'ADMISSION_FAILED.json').exists():
                write(out / 'STATUS.json', {'phase': 'NOT_RESOURCE_ADMITTED'})
                return
            common.guard(cfg)
            time.sleep(2)
        write(out / 'SCIENCE_STARTED.json', {'epoch': time.time(), 'original_start': cfg['start']})
        science(cfg, encoder, rows, labels, count)
    except BaseException:
        (out / 'FAILURE.private.txt').write_text(traceback.format_exc())
        write(out / 'STATUS.json', {'phase': 'FAILED_PRESERVED', 'counts': dict(count)})
        raise
    finally:
        write(out / 'COSTS.json', {'CPU_seconds': time.process_time() - cpu,
                                 'wall_seconds': time.time() - start, 'counts': dict(count),
                                 'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0})


if __name__ == '__main__':
    run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
