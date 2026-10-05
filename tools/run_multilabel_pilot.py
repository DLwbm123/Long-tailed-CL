"""Prepare or run the bounded NIH engineering pilot; configuration via stdin."""
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from ml_data import NIH_LABELS, read_manifest, validate_plan


def save(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(config):
    root = Path(config['suite_root'])
    root.mkdir(parents=True, exist_ok=False)
    rows = read_manifest(Path(config['data_root']) / 'prepared/manifest.csv')
    pool = {n: r for n, r in rows.items() if r['official_split'] == 'train_val'}
    rng = random.Random(74002)
    patients = sorted({r['patient_id'] for r in pool.values()})
    rng.shuffle(patients)
    nval = round(len(patients) * .1)
    val_patients = set(patients[:nval])
    remaining = patients[nval:]
    partitions = [set(remaining[::2]), set(remaining[1::2])]
    train_pool = [r for r in pool.values() if r['patient_id'] not in val_patients]
    counts = {c: sum(r['target'][i] for r in train_pool) for i, c in enumerate(NIH_LABELS)}
    ranked = sorted(NIH_LABELS, key=lambda c: (-counts[c], c))
    # Balance frequency ranks, without searching task orders or using validation labels.
    first = [c for i, c in enumerate(ranked) if i % 4 in (0, 3)]
    second = [c for c in ranked if c not in first]
    assert len(first) == len(second) == 7
    def sample(patient_set, size):
        return sorted(rng.sample(sorted(n for n, r in pool.items() if r['patient_id'] in patient_set), size))
    plan = dict(tasks=[dict(train_images=sample(partitions[0], 128), visible_labels=first),
                       dict(train_images=sample(partitions[1], 128), visible_labels=list(NIH_LABELS))],
                validation_images=sample(val_patients, 64),
                metric_groups=dict(head=ranked[:7], tail=ranked[7:], introduced_stage1=first,
                                   introduced_stage2=second))
    audit = validate_plan(rows, plan)
    def support(names):
        return {c: int(sum(rows[n]['target'][i] for n in names)) for i, c in enumerate(NIH_LABELS)}
    protocol = dict(scope='Engineering pilot only; not an efficacy or rare-label confirmation experiment',
                    seed=74002, conditions=['A', 'R05', 'R01', 'P01', 'EP01', 'F_S'],
                    epochs=1, batch_size=8, total_wall_seconds=3600, max_steps_per_condition=32,
                    patient_split='Seeded shuffle: 10% held-out validation patients; remaining patients alternate between stages',
                    image_sampling='Uniform without replacement within each patient partition; no positive enrichment',
                    label_sequence='Cumulative: seven frequency-rank-balanced labels, then all fourteen; future labels masked in stage 1',
                    new_labels=[first, second], training_pool_positive_counts=counts,
                    metric_groups=plan['metric_groups'], audit=audit,
                    training_positive_counts=[support(t['train_images']) for t in plan['tasks']],
                    validation_positive_counts=support(plan['validation_images']),
                    threshold=.5, official_test_used=False, automatic_retry=False,
                    automatic_expansion=False, source_commit=config['source_commit'])
    save(root / 'plan.private.json', plan)
    save(root / 'PROTOCOL.json', protocol)
    save(root / 'STATUS.json', dict(status='PREPARED', training_started=False))
    print(json.dumps(protocol))


def execute(config):
    root = Path(config['suite_root'])
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    if json.loads((root / 'STATUS.json').read_text())['status'] != 'PREPARED':
        raise ValueError('Suite is not a fresh prepared run; no automatic retry')
    started = time.monotonic()
    records = []
    status = dict(status='RUNNING', started=time.time(), pid=os.getpid(), completed_conditions=records)
    save(root / 'STATUS.json', status)
    try:
        for method in protocol['conditions']:
            remaining = protocol['total_wall_seconds'] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError('Suite wall budget exhausted')
            arm = dict(formal_authorized=True, data_root=config['data_root'], output=str(root / method),
                       legacy_repo=config['legacy_repo'], weight=config['weight'],
                       plan=str(root / 'plan.private.json'), method=method, device='cuda:0',
                       seed=protocol['seed'], epochs=1, batch_size=8, lr=.0003, threshold=.5,
                       max_steps=protocol['max_steps_per_condition'], max_wall_seconds=remaining)
            status.update(condition=method)
            save(root / 'STATUS.json', status)
            with (root / f'{method}.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-u', config['neutral_worker']], stdin=subprocess.PIPE,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status.update(child_pid=child.pid)
                save(root / 'STATUS.json', status)
                try:
                    child.communicate(json.dumps(arm).encode(), timeout=remaining)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            if child.returncode:
                raise RuntimeError(f'{method} exited {child.returncode}; see private log')
            receipt = json.loads((root / method / 'STATUS.json').read_text())
            if receipt['status'] != 'COMPLETE':
                raise RuntimeError(f'{method} lacks complete receipt')
            records.append(dict(method=method, elapsed_seconds=time.monotonic() - started,
                                training_steps=receipt['training_steps']))
        status.update(status='COMPLETE', completed=time.time(), elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        status.update(status='INCOMPLETE', error=str(exc), ended=time.time(), elapsed_seconds=time.monotonic() - started)
        save(root / 'STATUS.json', status)
        raise
    save(root / 'STATUS.json', status)


if __name__ == '__main__':
    configuration = json.load(sys.stdin)
    if configuration['action'] == 'prepare':
        prepare(configuration)
    elif configuration['action'] == 'execute':
        execute(configuration)
    else:
        raise ValueError('Unknown action')
