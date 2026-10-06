"""Verify worker parallelism preserves actual images, labels and augmentation."""
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from run_prototype_single import Images, manifests

if __name__ == '__main__':
    config = json.load(sys.stdin)
    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(config['legacy_repo']) / 'tools'))
    from run_medical_v2 import transform
    rows, _ = manifests(config)
    for training in (False, True):
        ds = Images(rows[:8], config['images'], transform(training), seed=74002)
        ds.epoch = 2
        a = list(DataLoader(ds, batch_size=4, num_workers=0))
        loader = DataLoader(ds, batch_size=4, num_workers=2, multiprocessing_context='spawn',
                            persistent_workers=not training)
        for _ in range(1 if training else 3):
            b = list(loader)
            assert len(a) == len(b)
            for left, right in zip(a, b):
                assert all(torch.equal(x, y) for x, y in zip(left, right))
        del loader
    print('PASS: canonical and augmented tensors/labels match with 0 versus 2 workers')
