"""Small read/decode/augmentation timing; no models or optimizer updates."""
import os, json, time, io, mmap, resource
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import Images, rows, seed_all, write

start = time.time()
cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
torch.set_num_threads(1)
data, _ = rows(cfg)
ds = Images(data['train'], 'imagenet', True, cfg['packed_cache'])
idx = np.load(Path(cfg['packed_cache'])/'pack_index.private.npz')
offsets = {str(n): (int(o), int(s)) for n, o, s in zip(idx['names'], idx['offsets'], idx['sizes'])}
records = []
qualification = []
with (Path(cfg['packed_cache'])/'images.private.pack').open('rb') as f:
    mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for j, i in enumerate(np.linspace(0, len(ds)-1, 32, dtype=int)):
        off, size = offsets[Path(ds.items[i][0]).name]
        reads = {}
        payloads = {}
        for method in (['mmap', 'pread'] if j % 2 == 0 else ['pread', 'mmap']):
            t = time.perf_counter()
            payloads[method] = mapped[off:off+size] if method == 'mmap' else os.pread(f.fileno(), size, off)
            reads[method] = time.perf_counter()-t
        assert payloads['mmap'] == payloads['pread']
        t = time.perf_counter()
        with Image.open(io.BytesIO(payloads['mmap'])) as im:
            rgb = im.convert('RGB')
        decode = time.perf_counter()-t
        seed_all(cfg['seed']+j)
        t = time.perf_counter()
        x = ds.transform(rgb)
        transform = time.perf_counter()-t
        assert x.shape == (3,224,224) and torch.isfinite(x).all()
        records.append(dict(read_seconds=reads, decode_seconds=decode, transform_seconds=transform, bytes=size))
        if j < 12:
            for train in [True, False]:
                for backbone in ['imagenet','clip']:
                    probe = Images(data['train'], backbone, train, cfg['packed_cache'])
                    tensors = []
                    times = []
                    for delayed in [False, True]:
                        seed_all(cfg['seed']+j)
                        t = time.perf_counter()
                        with Image.open(io.BytesIO(payloads['mmap'])) as im:
                            mode = im.mode
                            if delayed and mode == 'L':
                                value = im
                                for op in probe.transform.transforms[:-2]: value = op(value)
                                value = value.convert('RGB')
                                for op in probe.transform.transforms[-2:]: value = op(value)
                            else: value = probe.transform(im.convert('RGB'))
                        times.append(time.perf_counter()-t)
                        tensors.append(value)
                    assert torch.equal(*tensors), (j,train,backbone,mode)
                    qualification.append(dict(train=train,backbone=backbone,mode=mode,old_seconds=times[0],delayed_seconds=times[1],exact=True))
    mapped.close()
result = dict(state='PASS', samples=len(records), records=records,
              delayed_RGB_qualification=qualification,
              seconds_per_image={k:float(np.mean([r[k] for r in records])) for k in ['decode_seconds','transform_seconds']},
              read_seconds_per_image={k:float(np.mean([r['read_seconds'][k] for r in records])) for k in ['mmap','pread']},
              engineering_image_exposures=len(records), optimizer_updates=0,
              process_wall_seconds=time.time()-start,
              RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              paired_read_cache_order_alternated=True)
write(Path(cfg['output']).parent/'profile_input.private.json', result)
print(json.dumps({k:v for k,v in result.items() if k!='records'}), flush=True)
