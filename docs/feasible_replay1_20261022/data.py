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
    for r in rows:
        name = Path(r['relative_path']).name
        r['group'] = str(int(source[name]['subject_id']))
        assert r['group'] == str(int(name.split('_')[0]))
        r['label'] = int(r['original_label'])
        assert r['split'] == split and not Path(r['relative_path']).is_absolute()
        assert '..' not in Path(r['relative_path']).parts
    assert len({r['relative_path'] for r in rows}) == len(rows)
    return rows


class Images(Dataset):
    def __init__(self, rows, cfg):
        from torchvision import transforms
        self.rows, self.root = rows, Path(cfg['images'])
        self.transform = transforms.Compose([
            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(), transforms.Normalize((.5,)*3,(.5,)*3)])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        with Image.open(self.root / self.rows[i]['relative_path']) as im:
            x = self.transform(im.convert('RGB'))
        return x, i
