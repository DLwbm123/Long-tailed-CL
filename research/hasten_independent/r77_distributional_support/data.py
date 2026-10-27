"""Single-host bounded lossless FP32 RAM cache of arrived native prefixes."""
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from common import write, guard

_store = None


def config():
    c = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    return json.loads(Path(c['root_config']).read_text()) if 'root_config' in c else c


def train_rows(cfg):
    with open(Path(cfg['manifest'])/'train.csv') as f:
        rows = list(csv.DictReader(f))
    assert all(r['split']=='train' for r in rows)
    labels = torch.tensor([cfg['order'].index(int(r['original_label'])) for r in rows])
    assert torch.bincount(labels,minlength=len(cfg['order'])).tolist()==cfg['train_counts_ordered']
    return rows,labels


def get_cache(rootcfg,dc,stage=None,preflight=False,val=False):
    assert not preflight, 'Use migration_preflight.py for the existing two TRAIN rows'
    root=Path(rootcfg['output']);rows,labels=train_rows(dc)
    if val:
        seal=json.loads((Path(dc['output'])/'ALL_HEADS_SEALED.json').read_text())
        assert seal['records']==len(dc.get('seeds',[74002,74003,74004]))*len(dc['stages'])*dc.get('branches_per_stage',1) and seal['VAL_reads']==0
        with open(Path(dc['manifest'])/'val.csv') as f: part=list(csv.DictReader(f))
        assert all(r['split']=='val' for r in part)
        ids=list(range(len(rows),len(rows)+len(part)))
        yy=[dc['order'].index(int(r['original_label'])) for r in part]
        ref=torch.load(dc['validation_cache'],map_location='cpu',weights_only=False)
        assert ref['ids']==ids;base=ref['x'];slot='val'
    else:
        ref=torch.load(Path(dc['train_cache'])/f'stage_{stage}.private.pt',map_location='cpu',weights_only=False)
        ids=[i for i,v in enumerate(labels.tolist()) if v in dc['stages'][stage]]
        assert ref['ids']==ids
        part=[rows[i] for i in ids];yy=labels[ids].tolist()
        base=F.normalize(ref['x'],dim=1) if dc['dataset']=='NIH' else ref['x'];slot=str(stage)
    assert torch.isfinite(base).all() and torch.allclose(base.norm(dim=1),torch.ones(len(base)),atol=2e-6)
    p=root/'arrival_meta'/dc['dataset']/slot;p.mkdir(parents=True,exist_ok=True)
    destination=p/'descriptor.private.pt'
    if not destination.exists():
        torch.save(dict(dataset=dc['dataset'],split='val' if val else 'train',
                        ids=ids,labels=yy,base=base,
                        relative_paths=[r['relative_path'] for r in part]),destination)
    return [p]


class Tokens(Dataset):
    def __init__(self,paths):
        refs=[torch.load(Path(p)/'descriptor.private.pt',map_location='cpu',weights_only=False) for p in paths]
        assert len({(r['dataset'],r['split']) for r in refs})==1
        self.dataset,self.split=refs[0]['dataset'],refs[0]['split']
        self.base=torch.cat([r['base'] for r in refs])
        self.labels=torch.tensor([v for r in refs for v in r['labels']])
        self.ids=[v for r in refs for v in r['ids']]
        self.relative_paths=[v for r in refs for v in r['relative_paths']]
        assert len(self.ids)==len(set(self.ids))==len(self.base)
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return i,self.base[i],self.labels[i],i


class OrderSampler:
    def __init__(self,n):self.order=list(range(n))
    def __iter__(self):return iter(self.order)
    def __len__(self):return len(self.order)


class PrefixStore:
    def __init__(self,cfg):
        self.cfg=cfg;self.root=Path(cfg['output']);self.folder=self.root/'live_prefix_cache'
        self.folder.mkdir(exist_ok=True);self.index={};self.keys=None;self.chunk=0
        self.max_abs=0.;self.encoded=0;self.hits=0;self.last_status=0.;self.max_rows=85000;self.ram_bytes=0;self.disk_index={};self.spill_bytes=0;self.spill_seq=0;self.spill_root=Path(cfg['swap_cache_root']);self.spill_root.mkdir(exist_ok=True);self.remote_spill_root=self.root/'temporary_prefix_swap';self.remote_spill_root.mkdir(exist_ok=True);self.remote_spill_bytes=0
        for p in self.folder.glob('chunk*.npy'):p.unlink()
        for p in self.spill_root.glob('prefix_*.npy'):p.unlink()
        for p in self.remote_spill_root.glob('prefix_*.npy'):p.unlink()
        self.pixel_root=self.root/'lossless_pixel_cache';self.pixel_root.mkdir(exist_ok=True)
        import threading
        self.pixel_lock=threading.Lock();self.pixel_allocated=sum(p.stat().st_blocks*512 for p in self.pixel_root.glob('*/*/*.png'))
        torch.set_num_threads(1)
        import copy
        import open_clip
        from open_clip.factory import _MODEL_CONFIGS
        assets=Path(cfg['biomed_assets']);spec=json.loads((assets/'open_clip_config.json').read_text())
        mc=copy.deepcopy(spec['model_cfg'])
        mc['text_cfg'].update(hf_model_name=str(assets/'text'),hf_tokenizer_name=str(assets),hf_model_pretrained=False)
        _MODEL_CONFIGS['fixed_biomedical']=mc
        with torch.cuda.device(1):
            model,_,self.preprocess=open_clip.create_model_and_transforms('fixed_biomedical',pretrained=None,pretrained_hf=False,precision='fp32',device='cuda:1',**{f'image_{k}':v for k,v in spec['preprocess_cfg'].items()})
            state=torch.load(cfg['biomed_checkpoint'],map_location='cpu',weights_only=False)
            state=state.get('state_dict',state);key='text.transformer.embeddings.position_ids'
            if key in state:
                assert torch.equal(state[key],model.text.transformer.embeddings.position_ids.cpu());del state[key]
            model.load_state_dict(state,strict=True)
            self.visual=model.visual.eval().requires_grad_(False)
            del model,state
        split=next(i for i,t in enumerate(self.preprocess.transforms) if type(t).__name__=='ToTensor')
        self.pixel_transforms=self.preprocess.transforms[:split];self.tensor_transforms=self.preprocess.transforms[split:]
        self.readers=concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def prepare(self,data,order):
        # All arrived frozen prefixes fit in the user-authorized 20GB RAM cache plus bounded temporary root swap cache.
        return True

    def pixel(self,data,i):
        rel=Path(data.relative_paths[i]);assert not rel.is_absolute() and '..' not in rel.parts
        with Image.open(Path(self.cfg['image_roots'][data.dataset])/rel) as im:pixel=im.convert('RGB')
        for transform in self.pixel_transforms:pixel=transform(pixel)
        assert isinstance(pixel,Image.Image) and pixel.size==(224,224)
        value=pixel
        for transform in self.tensor_transforms:value=transform(value)
        return value

    @torch.no_grad()
    def batch(self,data,indices,retain):
        guard(self.cfg);result=[None]*len(indices);missing=[]
        for j,i in enumerate(indices):
            key=(data.dataset,data.split,data.ids[i])
            if key in self.disk_index:
                path,k=self.disk_index[key];a=np.load(path,mmap_mode='r');result[j]=torch.from_numpy(np.array(a[k],copy=True));a._mmap.close();self.hits+=1;continue
            if key not in self.index:missing.append((j,i));continue
            tensor,k=self.index[key]
            result[j]=tensor[k];self.hits+=1
        for lo in range(0,len(missing),64):
            part=missing[lo:lo+64];ii=[i for j,i in part]
            pixels=torch.stack(list(self.readers.map(lambda i:self.pixel(data,i),ii)))
            with torch.cuda.device(1):
                v=self.visual;t=v.trunk;x=t.patch_embed(pixels.to('cuda:1'));x=t._pos_embed(x);x=t.patch_drop(x);x=t.norm_pre(x)
                for b in list(t.blocks)[:10]:x=b(x)
                prefix=x
                for b in list(t.blocks)[10:]:x=b(x)
                base=F.normalize(v.head(t.forward_head(t.norm(x))),dim=-1)
                error=float((base.cpu()-data.base[ii]).abs().max())
                assert error<=2e-5,('REGENERATED_ROW_PARITY',error)
                tokens=prefix.cpu();assert torch.cuda.max_memory_reserved(1)<12*2**30
                assert tokens.dtype==torch.float32 and tokens.shape==(len(ii),197,768) and torch.isfinite(tokens).all()
            self.max_abs=max(self.max_abs,error);self.encoded+=len(ii)
            if retain and len(self.index)+len(ii)<=self.max_rows:
                byte_count=tokens.numel()*tokens.element_size()
                available=int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024
                rss=int(next(v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmRSS:')).split()[1])*1024
                if self.ram_bytes+byte_count<=self.cfg['ram_cache_budget_bytes'] and rss+byte_count<self.cfg['rss_budget_bytes']-1000000000 and available>64*2**30:
                    cached=tokens;self.ram_bytes+=byte_count
                    for k,i in enumerate(ii):self.index[(data.dataset,data.split,data.ids[i])]=(cached,k)
                else:
                    disk=None;remote=False
                    free=os.statvfs(self.spill_root).f_bavail*os.statvfs(self.spill_root).f_frsize
                    if self.spill_bytes+byte_count+8192<=self.cfg['swap_cache_budget_bytes'] and free>3000000000+byte_count:disk=self.spill_root
                    elif self.remote_spill_bytes+2.1*byte_count+16384<=self.cfg['remote_swap_budget_bytes']:disk=self.remote_spill_root;remote=True
                    if disk is not None:
                        destination=disk/('prefix_'+str(self.spill_seq)+'.npy');self.spill_seq+=1
                        np.save(destination,tokens.numpy(),allow_pickle=False);used=max(destination.stat().st_size,destination.stat().st_blocks*512)
                        if remote:self.remote_spill_bytes+=used
                        else:self.spill_bytes+=used
                        for k,i in enumerate(ii):self.disk_index[(data.dataset,data.split,data.ids[i])]=(destination,k)
            for k,(j,i) in enumerate(part):result[j]=tokens[k]
        import time
        if time.time()-self.last_status>10:
            self.last_status=time.time();write(self.root/'PREFIX_CACHE_STATUS.private.json',dict(encoded_rows=self.encoded,cache_hit_rows=self.hits,current_cached_rows=len(self.index),maximum_cached_rows=self.max_rows,reference_max_abs=self.max_abs,dtype='float32',no_lossy_quantization=True,ram_data_cache_bytes=self.ram_bytes,prefix_disk_writes=True,temporary_root_swap_only=True,swap_prefix_bytes=self.spill_bytes,swap_rows=len(self.disk_index),remote_swap_prefix_bytes=self.remote_spill_bytes))
        return torch.stack(result)


def store():
    global _store
    if _store is None:_store=PrefixStore(config())
    return _store


class DataLoader:
    """One-batch RAM prefix prefetch overlaps frozen-prefix and tail compute."""
    def __init__(self,data,batch_size,sampler=None,shuffle=False,**kwargs):
        assert not shuffle
        self.data=data;self.sampler=sampler or OrderSampler(len(data));self.batch_size=batch_size
        self._iterator=None
    def __iter__(self):
        indices=list(self.sampler.order);retain=store().prepare(self.data,indices)
        batches=[indices[i:i+self.batch_size] for i in range(0,len(indices),self.batch_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=None
            for k,ii in enumerate(batches):
                if future is None:future=pool.submit(store().batch,self.data,ii,retain)
                tokens=future.result()
                future=pool.submit(store().batch,self.data,batches[k+1],retain) if k+1<len(batches) else None
                yield tokens,self.data.base[ii],self.data.labels[ii],torch.tensor(ii)
    def token_at(self,i):return store().batch(self.data,[i],False)
