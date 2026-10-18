"""Fixed-scale ridge + uniform local evidence; reuses completed v1 statistics."""
import json, os, time, traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core import gaussian, logpdf
from run import read_rows, Images, write_json, save, metric


def local_scores(patches, stats, ref, tau):
    raw, shrunk = [], []
    for s in stats:
        mu, cov = s['local']; beta = s['groups'] / (s['groups'] + tau)
        raw.append(logpdf(patches, gaussian(mu, cov, ref['local'][2])).mean(1))
        shrunk.append(logpdf(patches, gaussian(mu, beta*cov+(1-beta)*ref['local'][1], ref['local'][2])).mean(1))
    return torch.stack(raw,-1)/patches.shape[-1], torch.stack(shrunk,-1)/patches.shape[-1]


def centered_energy(s):
    return (s-s.mean(-1,keepdim=True)).square().mean()


def fuse(ridge, local, scales, weight):
    assert all(torch.isfinite(torch.tensor(v)) and v>0 for v in scales)
    return ridge/scales[0]+weight*local/scales[1]


def ridge_model(stats, alpha):
    return torch.linalg.solve(sum(s['ridge_xx'] for s in stats)+alpha*torch.eye(769),
                              torch.stack([s['ridge_xy'] for s in stats],1))


def main(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    lock=out/'config.private.json'
    if lock.exists(): assert json.loads(lock.read_text())==cfg
    else: write_json(lock,cfg)
    sp=out/'status.private.json';prior=json.loads(sp.read_text()) if sp.exists() else {}
    if prior.get('state')=='COMPLETE':return
    start=prior.get('started',time.time())
    def status(phase,**kw):
        if time.time()-start>cfg['max_wall_seconds']:raise TimeoutError('Original wall budget exhausted')
        write_json(sp,dict(state='RUNNING',phase=phase,started=start,updated=time.time(),**kw))
    status('binding_check');torch.set_num_threads(4)
    old=Path(cfg['source_run']);source=json.loads((old/'config.private.json').read_text())
    assert json.loads((old/'status.private.json').read_text())['state']=='COMPLETE'
    for k in ['seed','dim','tau','eps_relative','ridge_alpha','order','task_sizes','train_count','val_count']:
        assert source[k]==cfg[k],k
    old_results=json.loads((old/'results.json').read_text())
    train,val=read_rows(source,'train'),read_rows(source,'val')
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    ref=torch.load(old/'reference.private.pt',weights_only=False)
    projection=torch.load(old/'projection.private.pt',weights_only=False)
    base=cfg['order'][:cfg['task_sizes'][0]]
    stats=[torch.load(old/f'class_{c:02d}.private.pt',weights_only=False) for c in base]
    cp=out/'calibration.private.pt'
    if not cp.exists():
        import timm
        from safetensors.torch import load_file
        torch.manual_seed(cfg['seed'])
        model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
        model.load_state_dict({k:v for k,v in load_file(source['weight']).items() if not k.startswith('head.')},strict=True)
        model.eval().requires_grad_(False).cuda();projection=projection.cuda()
        ridge=ridge_model(stats,cfg['ridge_alpha']);energies=[];counts={}
        generator=torch.Generator().manual_seed(cfg['seed'])
        for c in base:
            # One deterministic image per sampled patient; equal class weight for scale fitting.
            groups={}
            for r in train:
                if r['label']==c:groups.setdefault(r['group'],[]).append(r)
            ids=sorted(groups);selection=torch.randperm(len(ids),generator=generator)[:cfg['calibration_patients_per_class']].tolist()
            rows=[sorted(groups[ids[i]],key=lambda r:r['relative_path'])[0] for i in selection]
            rs,ls=[],[]
            for x,_ in DataLoader(Images(rows,source),batch_size=cfg['batch_size'],num_workers=0):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model.forward_features(x.cuda())
                z=z.float();patch=(z[:,1:]@projection).cpu();full=z[:,0].cpu().double()
                _,local=local_scores(patch,stats,ref,cfg['tau'])
                rs.append(torch.cat([full,torch.ones(len(full),1)],1)@ridge);ls.append(local)
            energies.append([float(centered_energy(torch.cat(rs))),float(centered_energy(torch.cat(ls)))])
            counts[str(c)]=len(rows);status('calibration',class_id=c,patients=len(rows))
        scales=torch.tensor(energies,dtype=torch.float64).mean(0).sqrt().tolist()
        assert min(scales)>1e-12
        save(cp,{'scales':scales,'counts':counts})
        del model;torch.cuda.empty_cache()
    calibration=torch.load(cp,weights_only=False);scales=calibration['scales']
    write_json(out/'calibration.json',calibration)
    results=[];seen=[];stats=[];offset=0;previous=None
    for stage,size in enumerate(cfg['task_sizes']):
        new=cfg['order'][offset:offset+size];offset+=size;seen+=new
        for c in new:
            s=torch.load(old/f'class_{c:02d}.private.pt',weights_only=False);assert s['class_id']==c;stats.append(s)
        ridge=ridge_model(stats,cfg['ridge_alpha']);outputs={k:[] for k in ['ridge','uniform','local_shrink','ridge_local']};labels=[]
        for c in seen:
            patch,_,full=torch.load(old/f'val_{c:02d}.private.pt',weights_only=False)
            uniform,local=local_scores(patch,stats,ref,cfg['tau'])
            r=torch.cat([full.double(),torch.ones(len(full),1)],1)@ridge
            for k,v in dict(ridge=r,uniform=uniform,local_shrink=local,ridge_local=fuse(r,local,scales,cfg['local_lambda'])).items():
                assert torch.isfinite(v).all();outputs[k].append(v)
            labels.extend([c]*len(full))
        outputs={k:torch.cat(v) for k,v in outputs.items()};labels=torch.tensor(labels)
        metrics={k:metric(v,labels,seen) for k,v in outputs.items()}
        for k in ['ridge','uniform']: assert metrics[k]['per_class']==old_results[stage]['metrics'][k]['per_class'], 'Control reproduction failed'
        rp=outputs['ridge'].argmax(-1);fp=outputs['ridge_local'].argmax(-1);classes=torch.tensor(seen)
        record=dict(stage=stage,seen=seen.copy(),metrics=metrics,fusion_effect={'changed':int((rp!=fp).sum()),'harm':int(((classes[rp]==labels)&(classes[fp]!=labels)).sum()),'rescue':int(((classes[rp]!=labels)&(classes[fp]==labels)).sum())},drift={})
        if previous:
            n,cs,ps=previous
            for k,v in outputs.items():
                now=v[:n,:cs];before=ps[k];delta=float((now-before).abs().max())
                record['drift'][k]={'max_old_score_change':delta,'old_winner_flips':int((now.argmax(-1)!=before.argmax(-1)).sum()),'new_class_wins':int((v[:n].argmax(-1)>=cs).sum())}
                if k in ['uniform','local_shrink']:assert delta<1e-7
        save(out/f'stage_{stage}.private.pt',{'scores':outputs,'labels':labels,'seen':seen.copy()})
        previous=(len(labels),len(seen),outputs);results.append(record);write_json(out/'results.json',results);status('stage_complete',stage=stage)
    write_json(sp,dict(state='COMPLETE',started=start,finished=time.time(),stages=len(results)))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    cfg=json.load(open(os.environ['RUN_CONFIG']))
    try:main(cfg)
    except Exception:
        p=Path(cfg['output'])/'status.private.json';s=json.loads(p.read_text()) if p.exists() else {}
        s.update(state='FAILED',updated=time.time(),error=traceback.format_exc());write_json(p,s);raise
