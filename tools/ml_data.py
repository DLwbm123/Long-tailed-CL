"""NIH multi-hot targets and explicit patient-disjoint continual task plans."""
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

NIH_LABELS = ('Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
              'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
              'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia')


def read_manifest(path, labels=NIH_LABELS):
    with open(path, newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image', 'patient_id', 'official_split', *labels}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Manifest lacks required image, patient, split, or label fields')
        rows = {}
        for row in reader:
            name = row['image']
            if name in rows or not name or not row['patient_id']:
                raise ValueError('Duplicate image or empty image/patient ID')
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Image paths must be relative and stay inside the data root')
            if row['official_split'] not in ('train_val', 'test'):
                raise ValueError('Unknown official split')
            values, observed = [], []
            for label in labels:
                value = row[label].strip()
                known = value in ('0', '1')
                if not known and value.lower() not in ('', '-1', 'nan', '?'):
                    raise ValueError('Labels must be 0/1 or explicitly unknown')
                values.append(float(value) if known else 0.)
                observed.append(known)
            rows[name] = dict(row, target=values, observed=observed)
    if not rows:
        raise ValueError('Empty manifest')
    return rows


def validate_plan(rows, plan, labels=NIH_LABELS):
    """No inferred class split: the protocol must assign images and visible labels."""
    if not plan.get('tasks') or not plan.get('validation_images'):
        raise ValueError('Supply explicit tasks and validation images')
    assigned, train_patients, seen = set(), set(), set()
    patient_stage = {}
    for index, task in enumerate(plan['tasks']):
        names, visible = task['train_images'], task['visible_labels']
        if not names or len(names) != len(set(names)) or assigned.intersection(names):
            raise ValueError('Empty task or repeated training images across tasks')
        if not visible or len(visible) != len(set(visible)) or not set(visible) <= set(labels):
            raise ValueError('Invalid visible label set')
        for name in names:
            row = rows[name]
            if row['official_split'] != 'train_val':
                raise ValueError('Official test image cannot enter training')
            patient = row['patient_id']
            if patient in patient_stage and patient_stage[patient] != index:
                raise ValueError('Patient spans training tasks; specify a different audited protocol first')
            patient_stage[patient] = index
            train_patients.add(patient)
        assigned.update(names)
        seen.update(visible)
    val = plan['validation_images']
    if len(val) != len(set(val)) or assigned.intersection(val):
        raise ValueError('Duplicate validation images or train/validation overlap')
    for name in val:
        row = rows[name]
        if row['official_split'] != 'train_val' or row['patient_id'] in train_patients:
            raise ValueError('Validation must be held-out train_val patients, never official test')
    return dict(tasks=len(plan['tasks']), train_images=len(assigned), validation_images=len(val),
                train_patients=len(train_patients), patient_overlap=0, seen_labels=sorted(seen))


class MultiLabelImages(Dataset):
    def __init__(self, rows, image_root, names, visible_labels, transform, *,
                 labels=NIH_LABELS, training=False, allow_test=False, seed=0):
        self.rows = [rows[name] for name in names]
        self.root = Path(image_root).resolve()
        self.transform, self.seed, self.epoch = transform, int(seed), 0
        visible = set(visible_labels)
        if not visible <= set(labels):
            raise ValueError('Unknown visible label')
        if training and allow_test:
            raise ValueError('Test data can never be used for training')
        if not allow_test and any(r['official_split'] == 'test' for r in self.rows):
            raise ValueError('Test access is disabled')
        self.targets = torch.tensor([r['target'] for r in self.rows], dtype=torch.float32)
        self.observed = torch.tensor([r['observed'] for r in self.rows], dtype=torch.bool)
        self.observed &= torch.tensor([c in visible for c in labels], dtype=torch.bool)
        # Hidden values must not leak through callers that forget to use the mask.
        self.targets = torch.where(self.observed, self.targets, 0.)
        if training and (not self.rows or not self.observed.any()):
            raise ValueError('Training task has no observed labels')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = (self.root / self.rows[index]['image']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path escapes root through a symlink')
        # Augmentation does not advance the policy sampler or global Torch RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + self.epoch * 1000003 + index)
            with Image.open(path) as image:
                x = self.transform(image.convert('RGB'))
        return x, self.targets[index], self.observed[index]
