"""Unchanged data checks/transform from the existing development experiment."""
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

def read_rows(cfg, split):
    with open(Path(cfg['manifest']) / (split + '.csv')) as f:
        rows = list(csv.DictReader(f))
    with open(Path(cfg['author_labels']) / (split + '.csv')) as f:
        source = {r['id']:r for r in csv.DictReader(f)}
    for row_id, r in enumerate(rows):
        r['id'] = row_id
        name = Path(r['relative_path']).name
        r['group'] = str(int(source[name]['subject_id']))
        assert r['group'] == str(int(name.split('_')[0]))
        r['label'] = int(r['original_label'])
        assert r['split'] == split and not Path(r['relative_path']).is_absolute()
        assert '..' not in Path(r['relative_path']).parts
    assert len({r['relative_path'] for r in rows}) == len(rows)
    return rows


class UnnormalizedImages(Dataset):
    def __init__(self, rows, cfg):
        from torchvision import transforms
        self.rows, self.root = rows, Path(cfg['images'])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()])

    def __len__(self):return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.root / self.rows[i]['relative_path']) as im:
            x = self.transform(im.convert('RGB'))
        return x, i


def patient_draw(rows, allowed_labels, seed):
    import random
    rng = random.Random(seed)
    groups = {}
    for i, row in enumerate(rows):
        assert row['label'] in allowed_labels
        groups.setdefault(row['label'], {}).setdefault(row['group'], []).append(i)
    assert len(groups) >= 8
    selected, labels = [], []
    for label in rng.sample(sorted(groups), 8):
        patients = sorted(groups[label])
        for _ in range(2):
            patient = rng.choice(patients)
            selected.append(rng.choice(groups[label][patient]));labels.append(label)
    return selected, labels
