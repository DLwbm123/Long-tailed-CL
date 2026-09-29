"""Bounded recovery only: original restore, CPU ridge, fixed val; no training."""
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(os.environ['N75_ROOT'])
LEGACY = ROOT / 'source/legacy'
sys.path.insert(0, str(LEGACY / 'tools'))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from preflight_ct1 import task_lock
from run_ct1 import Run as OriginalRun
from run_ct3p import FDStudent, CountedImages
from run_ct9p import Student
from run_medical_v2 import network_hash, rng_equal, sha, write_json, WEIGHT_SHA
from ct1_statistics import ridge, joint

PUB = ROOT / 'public'
PRIVATE = ROOT / 'private'
CFG = json.loads((ROOT / 'source/INPUT.json').read_text())
STARTED = time.monotonic()
LEDGER = dict(optimizer_steps=0, D_R_engineering_steps=0, formal_training_started=False,
              ridge_solves=0, complete_control_prediction_units=0, val_image_reads=0,
              old_fit_reads=0, future_fit_reads=0, test_reserved_reads=0, oracle_reads=0,
              GPU_process_residence_seconds=0, NEXT_DECISION='STOP')


def save(name, value):
    write_json(PUB / name, value)


def budget():
    used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    # Reserve a second complete copy of new outputs, plus the six parent backups.
    projected = 2 * used + CFG['parent_backup_bytes']
    elapsed = time.monotonic() - STARTED
    LEDGER.update(GPU_process_residence_seconds=elapsed, primary_new_bytes=used,
                  projected_new_bytes_including_backup=projected,
                  free_bytes=shutil.disk_usage(ROOT).free)
    save('RESOURCE_AND_ACCESS_LEDGER.json', LEDGER)
    assert elapsed < 3600 and projected <= 3 * 1024**3
    assert LEDGER['free_bytes'] >= 1024**3


class Context:
    """Storage adapter only; inherited constructors, restore and extraction stay intact."""
    def __init__(self):
        self.cfg = CFG['legacy_config']
        self.lock = task_lock(self.cfg)
        self.code = {k: sha(LEGACY / k) for k in CFG['qualified_sha256']}
        assert self.code == CFG['qualified_sha256'], 'BLOCKED_SOURCE_BINDING'
        self.phase = 'recovery'
        self.counts = {}
        self.counter = mp.Array('q', [0, 0, 0])
        self.allowed = set()
        self.scope = None
        self.pub = ROOT / 'source'  # Exact historical CT9 PROTOCOL_LOCK bytes.
        self.parent_view = SimpleNamespace(cfg=self.cfg, lock=self.lock,
                                           code=CFG['ct1_code'], phase=self.phase, count=self.count)

    def count(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    def dataset(self, name, seed, classes, train, split='train', smoke=False):
        assert not train and split == 'val' and list(classes) == list(range(8))
        assert self.scope == (name, seed, 4)
        assert (PUB / 'CONTROL_REBUILD_LOCK.json').exists(), 'BLOCKED_UNLOCKED_VAL'
        spec = self.cfg['datasets'][name]
        ds = CountedImages(Path(spec['manifests']) / 'val.csv', spec['images'],
                           self.lock[name]['orders'][seed], classes, False, False)
        ds.rows.sort(key=lambda x: x['sample_id'])
        ds.counter, ds.split, ds.offline = self.counter, 'val', False
        self.allowed = {str((Path(spec['images']) / x['relative_path']).resolve()) for x in ds.rows}
        return ds

    extract = OriginalRun.extract


def bank_digest(bank):
    h = hashlib.sha256()
    for key in sorted(bank):
        h.update(key.encode())
        v = bank[key]
        h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray)
                 else json.dumps(v).encode())
    return h.hexdigest()


def rebuild(bank, p, method, context):
    c = p['seen']
    assert bank['space_version'] == p['task'] + 1
    assert bank['arrival'] == [1 + i // 2 for i in range(c)]
    for key, shape in [('S', (1536, 1536)), ('mu', (1536, c)), ('v', (1536, c)), ('e', (1536, c))]:
        assert bank[key].shape == shape and bank[key].dtype == np.float64
        assert np.isfinite(bank[key]).all()
    expected = [context.lock[p['dataset']]['train_counts'][c] for c in p['order'][:p['seen']]]
    np.testing.assert_array_equal(bank['n'], expected)
    before = bank_digest(bank)
    W, diag = ridge(bank)
    LEDGER['ridge_solves'] += 1
    # Independently express the unnormalised normal equation; no second ridge/self-match.
    residual = np.linalg.norm((bank['S'] @ W + bank['S'].T @ W) / 2 + .001*c*W - bank['mu']) / np.linalg.norm(bank['mu'])
    assert residual < 1e-10 and bank_digest(bank) == before
    name = f"{p['dataset']}_{p['seed']}_{method}_t{p['task']+1:02d}.npz"
    path = PRIVATE / 'readouts' / name
    assert not path.exists(), 'BLOCKED_REPEAT_SOLVE'
    np.savez_compressed(path, W=W)
    row = dict(dataset=p['dataset'], seed=p['seed'], task=p['task']+1, method=method,
               status='REBUILT_FROM_VERIFIED_PARENT', file=name, sha256=sha(path),
               bank_sha256=before, bank_unchanged=True, original_ridge=diag,
               independent_normal_equation_residual=float(residual),
               independent_historical_W_match='NOT_AVAILABLE', original_file_recovered=False)
    return row


def restore(context, entry):
    path = Path(entry['path'])
    assert sha(path) == entry['sha256'], 'BLOCKED_CHECKPOINT_FILE_SHA'
    cls = Student if entry.get('beta') == 1 else FDStudent
    l = cls(context, entry['dataset'], entry['seed'])
    p = l.restore_new(path, False)  # Unmodified original ordinary restore.
    assert (p['task']+1, p['epoch'], p['seen']) == (entry['task'], 10, 2*entry['task'])
    assert p['network_sha256'] == entry['network_sha256'] == network_hash(l._network)
    assert rng_equal(l._capture_rng_state(), p['rng'])
    assert torch.equal(l.loader_generator.get_state(), p['loader_rng'])
    assert torch.equal(l.synth.get_state(), p['synthesis_rng'])
    assert l.optimizer is None and l.scheduler is None
    row = dict(dataset=entry['dataset'], seed=entry['seed'], task=entry['task'],
               beta=p['beta'], checkpoint_sha256=entry['sha256'], network_sha256=p['network_sha256'],
               ordinary_restore='PASS', strict_state_dict=True, original_args_unchanged=True,
               original_source_weight_manifest_binding=True, RNG_loader_synthesis='PASS',
               external_historical_W_regression='NOT_AVAILABLE')
    return l, p, row


def layout(context, entry):
    dataset, seed = entry['dataset'], entry['seed']
    order = context.lock[dataset]['orders'][seed]
    spec = context.cfg['datasets'][dataset]
    with (Path(spec['manifests']) / 'val.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if int(r['original_label']) in order[:8]]
    rows.sort(key=lambda r: r['sample_id'])
    identities = [(r['sample_id'], r['identity_component'], int(r['original_label'])) for r in rows]
    assert len({r[0] for r in identities}) == len(rows) and all(r[1] for r in identities)
    assert all((Path(spec['images']) / r['relative_path']).is_file() for r in rows)
    digest = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
    return dict(n=len(rows), identity_component_label_layout_sha256=digest, order=order[:8],
                original_val_manifest_sha256=spec['manifest_sha256']['val'])


def main():
    assert not (PUB / 'PARENT_RESTORE_AUDIT.json').exists(), 'BLOCKED_REPEAT_RECOVERY'
    for name in ('readouts', 'sealed'):
        (PRIVATE / name).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    context = Context()
    # No data image is permitted until a fixed evaluation unit opens its manifest layout.
    def protect(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            assert str(p) in context.allowed, 'BLOCKED_NON_VAL_IMAGE'
        if any(x.lower() in ('test', 'reserved', 'oracle') for x in p.parts):
            raise AssertionError('BLOCKED_FORBIDDEN_ASSET')
        flags = args[2] if len(args) > 2 else 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            assert not any(p.is_relative_to(Path(x)) for x in CFG['protected_roots'])
    sys.addaudithook(protect)
    rows, readouts, controls = [], [], []
    # All sources/restores/readouts/layouts are admitted before any image forward.
    for e in CFG['parents'] + CFG['controls']:
        budget()
        l, p, audit = restore(context, e)
        rows.append(audit)
        save('PARENT_RESTORE_AUDIT.json', dict(status='IN_PROGRESS', units=rows))
        methods = [('stats', 'C2'), ('T', 'C3')] if e['task'] == 3 else [('T', e['method'])]
        for bank, method in methods:
            q = rebuild(p[bank], p, method, context)
            q['checkpoint_sha256'] = e['sha256']
            readouts.append(q)
            save('READOUT_REBUILD_AUDIT.json', dict(status='IN_PROGRESS', units=readouts))
        if e['task'] == 4:
            controls.append(dict(dataset=e['dataset'], seed=e['seed'], method=e['method'],
                                 checkpoint_sha256=e['sha256'], network_sha256=e['network_sha256'],
                                 readout_sha256=q['sha256'], readout_file=q['file'], layout=layout(context, e)))
        del l, p
        gc.collect()
        torch.cuda.empty_cache()
        print('QUALIFIED', e['dataset'], e['seed'], e['task'], e.get('method', 'parent'), flush=True)
    save('PARENT_RESTORE_AUDIT.json', dict(status='PASS', units=rows))
    save('READOUT_REBUILD_AUDIT.json', dict(status='PASS_DERIVATION_ONLY', units=readouts,
                                         independent_historical_W_match='NOT_AVAILABLE'))
    save('CONTROL_REBUILD_LOCK.json', dict(status='LOCKED_AVAILABLE_SUBSET', units=controls,
         blocked=CFG['blocked_controls'], batch_size=48, num_workers=4, shuffle=False, drop_last=False,
         dtype='float32_forward_float64_joint_and_ridge', AMP=False, TF32=False,
         route='original_pointwise_probe', source_binding=CFG['source_binding'],
         historical_protocol_sha256=sha(ROOT/'source/PROTOCOL_LOCK.json'),
         original_extraction_sha256=sha(LEGACY/'tools/run_ct1.py'),
         operator_sha256=sha(__file__), amendment_sha256=sha(PUB/'P0R_AMENDMENT_LOCK.json')))
    results = []
    for e, unit in zip(CFG['controls'], controls):
        budget()
        l, p, _ = restore(context, e)
        context.scope = (l.name, l.seed, 4)
        before = bank_digest(p['T'])
        rng = l._capture_rng_state()
        raw, _, _, y, records = context.extract(l, range(8), 'val')
        LEDGER['val_image_reads'] = int(context.counter[1])
        path = PRIVATE / 'readouts' / unit['readout_file']
        assert sha(path) == unit['readout_sha256']
        with np.load(path, allow_pickle=False) as f:
            scores = joint(raw) @ f['W']
        assert scores.shape == (unit['layout']['n'], 8) and np.isfinite(scores).all()
        assert network_hash(l._network) == e['network_sha256']
        assert bank_digest(p['T']) == before and rng_equal(l._capture_rng_state(), rng)
        order = np.array(l.order[:8])
        ids = np.array([r['sample_id'] for r in records])
        comp = np.array([r['identity_component'] for r in records])
        ident = [(str(i), str(c), int(o)) for i, c, o in zip(ids, comp, order[y])]
        assert hashlib.sha256(json.dumps(ident, separators=(',', ':')).encode()).hexdigest() == unit['layout']['identity_component_label_layout_sha256']
        outfile = PRIVATE / 'sealed' / f"{l.name}_{l.seed}_{e['method']}_t04.npz"
        assert not outfile.exists(), 'BLOCKED_REPEATED_CONTROL'
        np.savez_compressed(outfile, raw=scores, y=y, original=order[y], order=order, ids=ids, component=comp)
        LEDGER['complete_control_prediction_units'] += 1
        predicted = scores.argmax(1)
        observed, mismatches = [], []
        expected = CFG['historical_counts'][f"{l.name}_{l.seed}_{e['method']}"]
        for h in range(8):
            mask = y == h
            row = dict(head_index=h, original_label=int(order[h]), n_images=int(mask.sum()),
                       n_correct=int((predicted[mask] == h).sum()), n_components=len(set(comp[mask])))
            observed.append(row)
            for key, value in row.items():
                if value != int(expected[h][key]):
                    mismatches.append(dict(head_index=h, field=key, actual=value, expected=int(expected[h][key])))
        result = dict(dataset=l.name, seed=l.seed, method=e['method'], file=outfile.name,
                      sha256=sha(outfile), status='BLOCKED_REPRODUCTION' if mismatches else 'REBUILT_FIXED_CONTROL',
                      historical_per_sample_equality='UNKNOWN', original_file_recovered=False,
                      layout_verified=True, network_bank_RNG_unchanged=True, counts=observed,
                      historical_discrete_count_differences=mismatches)
        results.append(result)
        save('CONTROL_REPRODUCTION_AUDIT.json', dict(status=result['status'], units=results))
        print('EVALUATED', l.name, l.seed, e['method'], result['status'], flush=True)
        assert not mismatches, 'BLOCKED_REPRODUCTION_HISTORICAL_COUNTS'
        del l, p, raw, scores
        gc.collect()
        torch.cuda.empty_cache()
    save('CONTROL_REPRODUCTION_AUDIT.json', dict(status='AVAILABLE_CONTROLS_REBUILT', units=results))
    save('RECOVERY_COMPLETE.json', dict(status='BLOCKED_UPSTREAM_STATE' if CFG['blocked_controls'] else 'READY_FOR_FULL_P0',
         formal_training_started=False, NEXT_DECISION='STOP'))


if __name__ == '__main__':
    try:
        with threadpool_limits(limits=4):
            main()
    except BaseException as error:
        save('FAILURE.json', dict(status='BLOCKED_REPRODUCTION', type=type(error).__name__, reason=str(error)))
        raise
    finally:
        budget()
