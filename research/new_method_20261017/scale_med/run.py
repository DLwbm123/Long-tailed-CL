"""One fixed NIH author20 development run, with atomic per-class restart points."""
import csv
import json
import os
from pathlib import Path
import time
import traceback
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from core import group_weights, moments, covariance, scores


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


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


def metric(score, labels, seen):
    pred = torch.tensor(seen)[score.argmax(-1)]
    acc = {str(c): float((pred[labels==c] == c).double().mean()) for c in seen}
    return {'micro': float((pred == labels).double().mean()),
            'macro': sum(acc.values()) / len(acc), 'per_class':acc}


def main(cfg):
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    lock = out / 'config.private.json'
    if lock.exists(): assert json.loads(lock.read_text()) == cfg, 'Configuration changed on resume'
    else: write_json(lock,cfg)
    state_path = out / 'status.private.json'
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    if old.get('state') == 'COMPLETE': return
    start = old.get('started',time.time())
    def status(phase, **kw):
        if time.time() - start > cfg['max_wall_seconds']: raise TimeoutError('Frozen wall budget exhausted')
        write_json(state_path,dict(state='RUNNING',phase=phase,started=start,updated=time.time(),**kw))
    status('data_check')
    train, val = read_rows(cfg,'train'), read_rows(cfg,'val')
    assert not ({r['group'] for r in train} & {r['group'] for r in val})
    assert len(train)==cfg['train_count'] and len(val)==cfg['val_count']
    audit = {str(c):{'images':sum(r['label']==c for r in train),
                     'patients':len({r['group'] for r in train if r['label']==c})} for c in cfg['order']}
    write_json(out/'support.json',audit)
    import timm
    from safetensors.torch import load_file
    torch.manual_seed(cfg['seed']); torch.set_num_threads(4)
    model = timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    weights = load_file(cfg['weight'])
    weights = {k:v for k,v in weights.items() if not k.startswith('head.')}
    model.load_state_dict(weights,strict=True)
    model.eval().requires_grad_(False).cuda()
    projection = torch.linalg.qr(torch.randn(768,cfg['dim']),mode='reduced').Q.cuda()
    save(out/'projection.private.pt',projection.cpu())
    def extract(rows):
        loader = DataLoader(Images(rows,cfg),batch_size=cfg['batch_size'],num_workers=cfg['workers'],
                            shuffle=False,pin_memory=True)
        for x,idx in loader:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=model.forward_features(x.cuda(non_blocking=True))
            assert z.shape[1:]==(197,768)
            z=z.float()
            yield z[:,1:] @ projection, z[:,0] @ projection, z[:,0], idx
    stats, seen, prior = [], [], None
    stage_results=[]; offset=0
    for stage,size in enumerate(cfg['task_sizes']):
        new=cfg['order'][offset:offset+size]; offset+=size
        for c in new:
            path=out/f'class_{c:02d}.private.pt'
            if path.exists():
                stats.append(torch.load(path,weights_only=False)); continue
            rows=[r for r in train if r['label']==c]
            w=group_weights([r['group'] for r in rows],'cuda')
            d=cfg['dim']; sums=[torch.zeros(d,device='cuda',dtype=torch.float64),torch.zeros(d,d,device='cuda',dtype=torch.float64)]
            gs=[v.clone() for v in sums]
            rr=torch.zeros(769,769,device='cuda',dtype=torch.float64)
            ry=torch.zeros(769,device='cuda',dtype=torch.float64)
            processed=0
            for p,g,full,idx in extract(rows):
                ww=w[idx.cuda()]
                for target,x in [(sums,p),(gs,g[:,None])]:
                    a,b=moments(x,ww);target[0]+=a;target[1]+=b
                xx=torch.cat([full.double(),torch.ones(len(full),1,device='cuda')],1)
                rr += xx.T @ (xx*ww[:,None]);ry += (xx*ww[:,None]).sum(0)
                processed+=len(idx)
                if processed==len(idx) or processed%1024<cfg['batch_size']:
                    status('fit',stage=stage,class_id=c,processed=processed,class_images=len(rows))
            s={'class_id':c,'groups':audit[str(c)]['patients'],'images':len(rows),
               'local':(sums[0].cpu(),covariance(*sums).cpu()),
               'global':(gs[0].cpu(),covariance(*gs).cpu()),'ridge_xx':rr.cpu(),'ridge_xy':ry.cpu()}
            save(path,s);stats.append(s)
        seen+=new
        if stage==0:
            reference={}
            for key in ['local','global']:
                mu=torch.stack([s[key][0] for s in stats]).mean(0)
                sec=torch.stack([s[key][1]+s[key][0][:,None]*s[key][0][None,:] for s in stats]).mean(0)
                cov=covariance(mu,sec)
                eps=cfg['eps_relative']*float(cov.diag().mean())
                reference[key]=(mu,cov,eps)
            save(out/'reference.private.pt',reference)
        ref_gpu={k:tuple(v.cuda() if torch.is_tensor(v) else v for v in vs) for k,vs in reference.items()}
        stats_gpu=[{k:tuple(v.cuda() for v in vs) if k in ['local','global'] else vs for k,vs in s.items()} for s in stats]
        rows=[r for r in val if r['label'] in seen]
        labels=torch.tensor([r['label'] for r in rows]); outputs={};fulls=[]
        # Cache each class once so old examples use exactly the same frozen features.
        cached={}
        for c in seen:
            vp=out/f'val_{c:02d}.private.pt'
            if not vp.exists():
                vr=[r for r in val if r['label']==c]
                blocks=[(p.cpu(),g.cpu(),f.cpu()) for p,g,f,_ in extract(vr)]
                save(vp,tuple(torch.cat([block[j] for block in blocks]) for j in range(3)))
            cached[c]=torch.load(vp,weights_only=False)
        positions={c:0 for c in seen};features=[[],[],[]]
        for row in rows:
            c=row['label'];i=positions[c];positions[c]+=1
            for j in range(3): features[j].append(cached[c][j][i])
        p,g,full=[torch.stack(v) for v in features]
        for i in range(0,len(rows),cfg['batch_size']):
            with torch.inference_mode():
                results=scores(p[i:i+cfg['batch_size']].cuda(),g[i:i+cfg['batch_size']].cuda(),stats_gpu,ref_gpu,cfg)
            for k,v in results.items(): outputs.setdefault(k,[]).append(v.cpu())
        fulls=[full.double()]
        outputs={k:torch.cat(v) for k,v in outputs.items()}
        xx=torch.cat(fulls);xx=torch.cat([xx,torch.ones(len(xx),1)],1)
        rx=sum(s['ridge_xx'] for s in stats);ry=torch.stack([s['ridge_xy'] for s in stats],1)
        ridge=torch.linalg.solve(rx+cfg['ridge_alpha']*torch.eye(769),ry)
        outputs['ridge']=xx@ridge
        record={'stage':stage,'seen':seen.copy(),'metrics':{k:metric(v,labels,seen) for k,v in outputs.items()},'drift':{}}
        if prior is not None:
            old_seen,old_rows,old_scores=prior
            lookup={r['relative_path']:i for i,r in enumerate(rows)}
            ix=[lookup[r['relative_path']] for r in old_rows]
            old_labels=torch.tensor([r['label'] for r in old_rows])
            for k,v in outputs.items():
                curr=v[ix,:len(old_seen)];prev=old_scores[k]
                a=prev.argmax(-1);b=curr.argmax(-1)
                classes=torch.tensor(old_seen)
                record['drift'][k]={'max_old_score_change':float((curr-prev).abs().max()),
                    'old_pair_winner_flips':int((a!=b).sum()),
                    'harm':int(((classes[a]==old_labels)&(classes[b]!=old_labels)).sum()),
                    'rescue':int(((classes[a]!=old_labels)&(classes[b]==old_labels)).sum()),
                    'new_class_wins':int((v[ix].argmax(-1)>=len(old_seen)).sum()),'n':len(ix)}
                if k not in ['entropy','ridge']:
                    assert torch.allclose(curr,prev,atol=1e-7,rtol=1e-7), 'Old-score invariance failed: '+k
        save(out/f'stage_{stage}.private.pt',{'scores':outputs,'labels':labels,'seen':seen})
        stage_results.append(record);write_json(out/'results.json',stage_results)
        prior=(seen.copy(),rows,outputs)
        status('stage_complete',stage=stage,classes=len(seen))
    write_json(state_path,{'state':'COMPLETE','started':start,'finished':time.time(),'stages':len(stage_results)})
    print('COMPLETE',flush=True)

if __name__=='__main__':
    cfg=json.load(open(os.environ['RUN_CONFIG']))
    try: main(cfg)
    except Exception:
        p=Path(cfg['output'])/'status.private.json'
        old=json.loads(p.read_text()) if p.exists() else {}
        old.update(state='FAILED',updated=time.time(),error=traceback.format_exc())
        write_json(p,old)
        traceback.print_exc();raise
