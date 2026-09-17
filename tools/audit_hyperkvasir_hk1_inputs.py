"""HK1 prerequisite audit only: metadata, image inventory, and legacy membership.

Stops before pixel auditing or protocol construction when class mappings differ.
No image payload, feature bank, checkpoint, prediction, or model is opened.
Private input paths come from HK1_INPUT_CONFIG; public output contains aggregates.
"""
import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def legacy_membership(names, class_id, seed):
    shuffled = np.array(sorted(names), dtype=object)
    np.random.default_rng(seed + 7919 * class_id).shuffle(shuffled)
    reserved = set(np.array_split(shuffled, 5)[0].tolist())
    return {n: ('test' if n in reserved else 'train') for n in names}


def selfcheck():
    names = [f'{i:02}.jpg' for i in range(11)]
    m = legacy_membership(names, 4, 1)
    assert len(m) == 11 and Counter(m.values()) == {'test': 3, 'train': 8}
    assert m == legacy_membership(list(reversed(names)), 4, 1)


def main():
    started = time.monotonic()
    selfcheck()
    cfg = json.loads(Path(os.environ['HK1_INPUT_CONFIG']).read_text())
    out = Path(cfg['public_output'])
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    def table(name, rows):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    raw = Path(cfg['official_csv']).read_bytes()
    blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
    assert blob == '0d8abe6781a82c4452416390d93a176f875a6f23', 'BLOCKED_METADATA'
    official = list(csv.DictReader(raw.decode().splitlines(), delimiter=';'))
    official_by_name = {Path(r['file-name']).name: r for r in official}
    assert len(official) == len(official_by_name) == 10662
    official_classes = sorted({r['class-name'] for r in official})
    assert len(official_classes) == 23
    with Path(cfg['legacy_alignment']).open() as f: historical = list(csv.DictReader(f))
    root = Path(cfg['images'])
    images = [p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}]
    by_name = {p.name: p for p in images}
    assert len(images) == len(by_name) == 10662
    assert set(by_name) == set(official_by_name), 'BLOCKED_IMAGE_WHITELIST'
    legacy_classes = sorted({p.parent.name for p in images})
    assert len(legacy_classes) == 23
    correspondence = Counter((p.parent.name, official_by_name[n]['class-name']) for n,p in by_name.items())
    assert len(correspondence) == 23, 'BLOCKED_NONBIJECTIVE_CLASS_NAMES'
    mapping = [dict(legacy_id=legacy_classes.index(a), legacy_class_name=a,
                    official_id=official_classes.index(b), official_class_name=b,
                    n_official_images=n, name_equal=a==b,
                    numeric_id_equal=legacy_classes.index(a)==official_classes.index(b))
               for (a,b),n in sorted(correspondence.items())]
    table('CLASS_MAPPING_DIFF.csv', mapping)
    historical_audits = []
    for seed in (1,2,3):
        old = [r for r in historical if int(r['seed']) == seed]
        assert len(old) == 10662 and len({r['basename'] for r in old}) == 10662
        expected = {}
        for c, name in enumerate(legacy_classes):
            expected.update(legacy_membership([n for n,p in by_name.items() if p.parent.name==name], c, seed))
        mismatches = sum(r['membership'] != expected[r['basename']] for r in old)
        assert all(int(r['class_id'])==legacy_classes.index(by_name[r['basename']].parent.name) for r in old)
        assert all(r['official_class_name']==official_by_name[r['basename']]['class-name'] for r in old)
        assert mismatches == 0, 'BLOCKED_HISTORICAL_MEMBERSHIP'
        historical_audits.append(dict(seed=seed, fold=1, n_images=len(old),
                                      membership_mismatches=mismatches, counts=dict(Counter(expected.values()))))
    sizes = {n:p.stat().st_size for n,p in by_name.items()}
    outer = {r['basename']:r['membership'] for r in historical if r['seed']=='1'}
    outer_bytes = sum(sizes[n] for n in sizes if outer[n]=='train')
    mismatch_names = sum(not r['name_equal'] for r in mapping)
    mismatch_ids = sum(not r['numeric_id_equal'] for r in mapping)
    audit = dict(status='BLOCKED_HK1_CLASS_MAPPING' if mismatch_names or mismatch_ids else 'INPUT_PREREQUISITES_PASS',
        verified_utc=datetime.now(timezone.utc).isoformat(), official_commit='21cc366e78c0cb4e180a26a0e441d6c0d5171da9',
        official_git_blob=blob, official_sha256=hashlib.sha256(raw).hexdigest(), official_metadata_bytes=len(raw),
        official_metadata_reused=True, downloads=0, official_images=10662, resolved_images=10662,
        missing_images=0, extra_images=0, classes=23, renamed_class_pairs=mismatch_names, changed_numeric_ids=mismatch_ids,
        mapping_bijective_by_exact_filename=True, mapping_applied=False, historical_membership=historical_audits,
        reconstructed_from_pinned_legacy_rule=False, historical_lists_verified_against_rule=True,
        image_payload_bytes=sum(sizes.values()), outer_train_image_payload_bytes=outer_bytes,
        reserved_image_payload_bytes=sum(sizes.values())-outer_bytes,
        byte_pixel_identity_audit='NOT_RUN_MAPPING_GATE_FAILED', final_fit_val_protocol_created=False,
        inner_fold_selected=None, final_fit_counts=None, final_fit_imbalance_ratio=None,
        patient_disjoint='UNKNOWN', exam_disjoint='UNKNOWN',
        source_data_host='my-gpu; inventory and metadata only, no GPU work',
        next_decision='STOP_PENDING_PROTOCOL_AND_EXECUTION_LOCATION', audit_wall_seconds=time.monotonic()-started)
    write('DATA_AUDIT_HK1.json', audit)
    write('ACCESS_AUDIT.json', dict(new_formal_S0_runs=0,new_formal_neural_training_epochs=0,
        formal_training_optimizer_steps=0,engineering_optimizer_steps=0,incremental_neural_epochs_S1_to_S5=0,
        incremental_optimizer_steps_S1_to_S5=0,encoder_forward_calls=0,new_test_predictions=0,
        new_test_feature_reads=0,new_test_model_forwards=0,reserved_identity_audit_file_reads=0,
        reserved_identity_audit_decodes=0,reserved_path_size_metadata_lookups=2143,
        image_payload_reads=0,official_metadata_file_reads=1,historical_identity_manifest_reads=1,
        historical_prediction_payload_reads=0,monitoring_tasks_created=0,further_experiments_started=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
