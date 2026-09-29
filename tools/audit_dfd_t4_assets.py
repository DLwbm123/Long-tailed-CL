"""CPU-only asset inventory. Never grants training admission or changes parents."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from dfd_t4_core import old_directions, random_directions, math_check


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    root = Path(os.environ['N74_ROOT'])
    cfg = json.loads((root / 'source/INPUT.json').read_text())
    pub = root / 'public'
    torch.set_num_threads(4)
    # Metadata is allowed; no image, test, reserved or oracle access here.
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0]))
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png') or p.stem.lower() in ('test', 'reserved') or any(
                    x.lower() in ('test', 'reserved') for x in p.parts):
                raise RuntimeError('BLOCKED_DATA_ACCESS')
    sys.addaudithook(guard)
    def save(name, value):
        (pub / name).write_text(json.dumps(value, indent=2) + '\n')
    save('MATH_CHECK.json', math_check())
    assets = Path(cfg['assets'])
    manifest_rows, manifest_hashes = {}, {}
    for dataset in ('HK', 'ISIC'):
        manifest_rows[dataset], manifest_hashes[dataset] = {}, {}
        for split in ('train', 'val'):
            path = assets / 'manifests' / dataset / (split + '.csv')
            manifest_hashes[dataset][split] = sha(path)
            with path.open() as f:
                manifest_rows[dataset][split] = list(csv.DictReader(f))
    weight = assets / cfg['weight_relative']
    weight_sha = sha(weight)
    rows = []
    for entry in cfg['parents']:
        parent = entry['teacher_source']
        path = Path(cfg['parent_archive']) / parent['name']
        assert sha(path) == parent['sha256'], 'BLOCKED_PARENT_SHA'
        p = torch.load(path, map_location='cpu', weights_only=False)
        dataset, seed = entry['dataset'], entry['seed']
        assert (p['dataset'], p['seed'], p['stream'], p['task'], p['epoch'], p['seen'], p['beta']) == (
            dataset, seed, 'F', 2, 10, 6, 10.), 'BLOCKED_PARENT_METADATA'
        assert p['network_sha256'] == parent['network_sha256']
        assert p['weight_sha256'] == weight_sha, 'BLOCKED_WEIGHT_SHA'
        assert p['manifest_sha256'] == manifest_hashes[dataset], 'BLOCKED_MANIFEST_SHA'
        bank = p['T']
        assert bank['space_version'] == 3
        assert bank['S'].shape == (1536, 1536)
        assert all(np.isfinite(bank[k]).all() for k in ('S', 'mu', 'v', 'e'))
        train, val = manifest_rows[dataset]['train'], manifest_rows[dataset]['val']
        counts = {c: sum(int(x['original_label']) == c for x in train) for c in p['order']}
        np.testing.assert_array_equal(bank['n'], [counts[c] for c in p['order'][:6]])
        current = [x for x in train if int(x['original_label']) in p['order'][6:8]]
        seen_val = [x for x in val if int(x['original_label']) in p['order'][:8]]
        missing_images = sum(not (assets / 'images' / dataset / x['relative_path']).is_file() for x in current)
        assert missing_images == 0, 'BLOCKED_TASK4_IMAGE_PATH'
        assert all(x.get('identity_component', x.get('verified_group', x.get('component_id', ''))) for x in seen_val)
        qd, singular, threshold = old_directions(bank['mu'])
        qr, random_seed = random_directions(dataset, seed, qd.shape[1])
        qfile = root / 'private' / f'{dataset}_{seed}_Q.npz'
        np.savez(qfile, Q_D=qd, Q_R=qr, random_seed=np.array(random_seed, dtype=np.uint64))
        orth32 = max(float(np.max(np.abs(q.astype(np.float32).T @ q.astype(np.float32) - np.eye(q.shape[1])))) for q in (qd, qr))
        assert orth32 <= 1e-5
        source_mismatches = [key for key, val_hash in p['code_sha256'].items() if cfg['source_hashes'].get(key) != val_hash]
        rows.append(dict(dataset=dataset, seed=seed, parent_file=parent['name'],
                         parent_sha256=parent['sha256'], metadata_network_sha256=p['network_sha256'],
                         file_SHA_and_metadata_verified=True, ordinary_model_restore='NOT_RUN',
                         embedded_T_finite=True, T_shape=list(bank['mu'].shape), space_version=3,
                         Q_rank=qd.shape[1], singular_values=singular.tolist(), threshold=threshold,
                         Q_sha256=sha(qfile), random_seed=random_seed, fp32_orth_max_abs=orth32,
                         current_train_n=len(current), current_fit_paths_present=True,
                         current_classes=p['order'][6:8], val_seen_n=len(seen_val),
                         val_components_present=True, expected_steps_one_arm=10 * math.ceil(len(current)/48),
                         original_source_mismatches=source_mismatches,
                         shared_metadata_sha256=p['shared_sha256']))
        del p
    total = sum(x['expected_steps_one_arm'] for x in rows)
    assert total == 2750 and 2 * total == 5500, 'BLOCKED_STEPS'
    save('ASSET_AUDIT.json', dict(status='PARTIAL_NOT_ADMITTED', parents=rows,
                                weight_sha256=weight_sha, manifest_sha256=manifest_hashes,
                                CT9_recomputed_steps=total, D_R_expected_steps=2*total,
                                batch_size=48, drop_last=False,
                                missing_controls=cfg['missing_controls']))
    save('RESOURCE_AND_ACCESS_LEDGER.json', dict(status='BLOCKED_ASSET',
         elapsed_CPU_audit_wall_seconds=time.monotonic()-started,
         GPU_process_residence_seconds=0, engineering_optimizer_steps=0,
         formal_optimizer_steps=0, formal_epochs=0, new_checkpoints=0,
         image_reads=0, new_val_predictions=0, test_reserved_reads=0,
         output_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
         free_bytes=shutil.disk_usage(root).free,
         real_model_gate='NOT_RUN', full_matrix_resource_admission='NOT_RUN'))
    print(json.dumps(dict(status='BLOCKED_ASSET', verified_parent_files=len(rows),
                         ranks=[r['Q_rank'] for r in rows], expected_steps=2*total)))


if __name__ == '__main__':
    main()
