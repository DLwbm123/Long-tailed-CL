"""V2 on sealed ISIC/HK first-transfer supports; query access follows both model seals."""
import json, os, sys, time, traceback, resource
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from core import group_weights, moments, covariance
from run import Images, save, write_json
from run_v2 import local_scores, centered_energy, fuse, ridge_model


def aggregate(score, y, old):
    assert torch.isfinite(score).all() and len(score)==len(y)
    pred=score.argmax(-1);k=score.shape[1];ce=-score.log_softmax(-1)[torch.arange(len(y)),y]
    classes=[]
    for c in range(k):
        m=y==c;assert m.any()
        classes.append(dict(class_index=c,count=int(m.sum()),correct=int((pred[m]==c).sum()),
                            recall=float((pred[m]==c).double().mean()),CE=float(ce[m].mean())))
    summary={}
    for name,ids in [('all',range(k)),('old',range(old)),('new',range(old,k))]:
        rr=[classes[i] for i in ids];mask=torch.isin(y,torch.tensor(list(ids)))
        summary[name]=dict(BA=sum(r['recall'] for r in rr)/len(rr),CE=sum(r['CE'] for r in rr)/len(rr),
                           micro=float((pred[mask]==y[mask]).double().mean()),count=int(mask.sum()))
    confusion=torch.bincount(y*k+pred,minlength=k*k).reshape(k,k).tolist()
    return dict(summary=summary,classes=classes,confusion=confusion)


def paired(a,b,y,old):
    a,b=a.argmax(-1),b.argmax(-1);result={}
    for name,mask in [('all',torch.ones_like(y,dtype=torch.bool)),('old',y<old),('new',y>=old)]:
        good,bad=a==y,b==y
        result[name]=dict(count=int(mask.sum()),changed=int(((a!=b)&mask).sum()),
                         rescue=int((~good&bad&mask).sum()),harm=int((good&~bad&mask).sum()),
                         wrong_to_wrong=int((~good&~bad&(a!=b)&mask).sum()))
    return result


def query_allowed(sealed, role):
    return role in ('S_old','S_new') or (sealed and role in ('Qfit','Qgate'))


def main(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    cp=root/'config.private.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:write_json(cp,cfg)
    sp=root/'status.private.json';before=json.loads(sp.read_text()) if sp.exists() else {}
    if before.get('state')=='COMPLETE':return
    start=before.get('started',time.time());sealed=False
    def status(phase,**kw):
        if time.time()-start>cfg['max_wall_seconds']:raise TimeoutError('Frozen original deadline reached')
        write_json(sp,dict(state='RUNNING',started=start,updated=time.time(),phase=phase,**kw))
    torch.set_num_threads(4);status('input_binding')
    prior=Path(cfg['prior_root']).resolve();a_root=Path(cfg['a_root']).resolve();b_root=Path(cfg['b_root']).resolve()
    names={d['name'] for d in cfg['datasets']};image_allow=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(prior):
            allowed={n+'.'+role+'.rows.private.json' for n in names for role in ('S_old','S_new')}
            if sealed:allowed|={n+'.'+q+suffix for n in names for q in cfg['queries'] for suffix in ('.rows.private.json','.features.private.pt')}
            assert p.parent==prior and p.name in allowed,'Historical input boundary'
        if p.is_relative_to(a_root):assert sealed and p.parent==a_root and p.name in {n+'.ridge.private.pt' for n in names}
        if p.is_relative_to(b_root):assert sealed and p.parent==b_root and p.name in {n+'.B.private.pt' for n in names}
        for spec in cfg['resources'].values():
            if p.is_relative_to(Path(spec['images']).resolve()):assert str(p) in image_allow,'Image outside frozen support/query rows'
    sys.addaudithook(audit)
    def rows_for(case,role):
        assert query_allowed(sealed,role)
        rr=json.loads((prior/(case['name']+'.'+role+'.rows.private.json')).read_text())
        assert len({r['relative_path'] for r in rr})==len(rr)
        for r in rr:
            path=Path(r['relative_path']);assert not path.is_absolute() and '..' not in path.parts and r['identity_component']
        expected=cfg['support_counts'][case['name']] if role in ('S_old','S_new') else cfg['query_counts'][case['name']][role]
        ids=range(case['old']) if role=='S_old' else range(case['old'],case['old']+case['new']) if role=='S_new' else range(case['old']+case['new'])
        assert dict(Counter(r['label'] for r in rr))=={c:expected[c] for c in ids}
        image_allow.update(str((Path(cfg['resources'][case['name']]['images'])/r['relative_path']).resolve()) for r in rr)
        return rr
    train={};support={}
    for case in cfg['datasets']:
        n=case['name'];rr=rows_for(case,'S_old')+rows_for(case,'S_new');train[n]=rr
        assert len({r['relative_path'] for r in rr})==len(rr)
        group_labels={}
        for r in rr:group_labels.setdefault(r['identity_component'],set()).add(r['label'])
        assert all(len(v)==1 for v in group_labels.values())
        support[n]={str(c):dict(images=sum(r['label']==c for r in rr),known_groups=len({r['identity_component'] for r in rr if r['label']==c})) for c in range(case['old']+case['new'])}
    write_json(root/'support.json',support)
    import timm
    from safetensors.torch import load_file
    torch.manual_seed(cfg['seed'])
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda()
    projection=torch.load(cfg['projection'],weights_only=True).cuda();assert projection.shape==(768,cfg['dim'])
    def extract(n,rr):
        for x,idx in DataLoader(Images(rr,cfg['resources'][n]),batch_size=cfg['batch_size'],num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=model.forward_features(x.cuda())
            assert z.shape[1:]==(197,768);z=z.float()
            yield z[:,1:]@projection,z[:,0],idx
    for case in cfg['datasets']:
        n=case['name'];out=root/n;out.mkdir(exist_ok=True);generator=torch.Generator().manual_seed(cfg['seed'])
        for c in range(case['old']+case['new']):
            path=out/f'class_{c:02d}.private.pt'
            rows=[r for r in train[n] if r['label']==c]
            groups={}
            for i,r in enumerate(rows):groups.setdefault(r['identity_component'],[]).append(i)
            ids=sorted(groups);sample=[]
            if c<case['old']:
                for i in torch.randperm(len(ids),generator=generator)[:cfg['calibration_patients_per_class']].tolist():
                    sample.append(min(groups[ids[i]],key=lambda j:rows[j]['relative_path']))
            if path.exists():continue
            sample=set(sample);patches=[];fulls=[]
            w=group_weights([r['identity_component'] for r in rows],'cuda');d=cfg['dim']
            mu=torch.zeros(d,dtype=torch.float64,device='cuda');second=torch.zeros(d,d,dtype=torch.float64,device='cuda')
            xx=torch.zeros(769,769,dtype=torch.float64,device='cuda');xy=torch.zeros(769,dtype=torch.float64,device='cuda');done=0
            for p,f,idx in extract(n,rows):
                weights=w[idx.cuda()];a,b=moments(p,weights);mu+=a;second+=b
                aug=torch.cat([f.double(),torch.ones(len(f),1,device='cuda')],1)
                xx+=aug.T@(aug*weights[:,None]);xy+=(aug*weights[:,None]).sum(0)
                chosen=[j for j,i in enumerate(idx.tolist()) if i in sample]
                if chosen:patches.append(p[chosen].cpu());fulls.append(f[chosen].cpu())
                done+=len(idx)
                if done==len(idx) or done%512<cfg['batch_size']:status('fit',dataset=n,class_id=c,processed=done,class_images=len(rows))
            s=dict(class_id=c,groups=len(groups),images=len(rows),local=(mu.cpu(),covariance(mu,second).cpu()),ridge_xx=xx.cpu(),ridge_xy=xy.cpu(),calibration=(torch.cat(patches),torch.cat(fulls)) if patches else None)
            save(path,s)
        stats=[torch.load(out/f'class_{c:02d}.private.pt',weights_only=False) for c in range(case['old']+case['new'])]
        base=stats[:case['old']];m=torch.stack([s['local'][0] for s in base]).mean(0)
        sec=torch.stack([s['local'][1]+s['local'][0][:,None]*s['local'][0][None,:] for s in base]).mean(0)
        cov=covariance(m,sec);ref={'local':(m,cov,cfg['eps_relative']*float(cov.diag().mean()))}
        ridge=ridge_model(base,cfg['ridge_alpha']);energies=[]
        for s in base:
            p,f=s['calibration'];_,l=local_scores(p,base,ref,cfg['tau']);r=torch.cat([f.double(),torch.ones(len(f),1)],1)@ridge
            energies.append([float(centered_energy(r)),float(centered_energy(l))])
        scales=torch.tensor(energies,dtype=torch.float64).mean(0).sqrt().tolist();assert min(scales)>1e-12
        save(out/'model.private.pt',dict(stats=stats,reference=ref,ridge=ridge_model(stats,cfg['ridge_alpha']),scales=scales))
        write_json(out/'calibration.json',dict(scales=scales,known_groups_per_class={str(s['class_id']):len(s['calibration'][0]) for s in base},patient_isolation='UNKNOWN'))
        status('model_sealed',dataset=n)
    assert all((root/c['name']/'model.private.pt').exists() for c in cfg['datasets'])
    write_json(root/'MODEL_SEAL.json',dict(datasets=sorted(names),sealed_at=time.time(),queries_used=False));sealed=True
    results=[]
    for case in cfg['datasets']:
        n=case['name'];out=root/n;obj=torch.load(out/'model.private.pt',weights_only=False);stats=obj['stats'];old=case['old'];seen=old+case['new']
        old_models={label:torch.load(base/(n+suffix),weights_only=True) for label,base,suffix in [('A',a_root,'.ridge.private.pt'),('B',b_root,'.B.private.pt')]}
        assert all(all(not v.any() for v in s['model']) and s['frame'].shape==(1536,seen) for s in old_models.values())
        support_groups={r['identity_component'] for r in train[n]};query_groups=[]
        for query in cfg['queries']:
            rr=rows_for(case,query);groups={r['identity_component'] for r in rr};assert not support_groups&groups
            assert not any(g&groups for g in query_groups);query_groups.append(groups)
            y=torch.tensor([r['label'] for r in rr]);cached=torch.load(prior/(n+'.'+query+'.features.private.pt'),weights_only=True)
            assert torch.equal(cached['y'],y) and cached['components']==[r['identity_component'] for r in rr]
            z=F.normalize(cached['x'],dim=1);outputs={label:8*z@s['frame'] for label,s in old_models.items()}
            blocks={k:[] for k in ['ridge','uniform','local_shrink','ridge_local']}
            for p,f,_ in extract(n,rr):
                u,l=local_scores(p.cpu(),stats,obj['reference'],cfg['tau']);f=f.cpu().double()
                r=torch.cat([f,torch.ones(len(f),1)],1)@obj['ridge']
                for k,v in dict(ridge=r,uniform=u,local_shrink=l,ridge_local=fuse(r,l,obj['scales'],cfg['local_lambda'])).items():blocks[k].append(v)
            outputs.update({k:torch.cat(v) for k,v in blocks.items()})
            metrics={k:aggregate(v,y,old) for k,v in outputs.items()}
            for label in ['A','B']:
                refs=[r for r in cfg['reference_metrics'] if r['dataset']==n and r['query']==query and r['model']==label]
                assert len(refs)==3
                for r in refs:
                    actual=metrics[label]['summary'][r['group']]
                    assert abs(actual['BA']-r['BA'])<1e-12 and abs(actual['CE']-r['CE'])<1e-10,'Historical reference mismatch'
            record=dict(dataset=n,query=query,old=old,seen=seen,metrics=metrics,paired={k:paired(outputs[k],outputs['ridge_local'],y,old) for k in ['ridge','A','B']})
            results.append(record);write_json(root/'results.json',results);save(out/(query+'.predictions.private.pt'),dict(scores=outputs,labels=y))
            status('query_complete',dataset=n,query=query)
    write_json(root/'costs.json',dict(wall_seconds=time.time()-start,process_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,process_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,source_feature_cost_excluded=True,encoder_updates=0,parameter_search=False,test_access=False))
    write_json(sp,dict(state='COMPLETE',started=start,finished=time.time(),conditions=len(results)));print('COMPLETE',flush=True)

if __name__=='__main__':
    cfg=json.load(open(os.environ['RUN_CONFIG']))
    try:main(cfg)
    except Exception:
        p=Path(cfg['output'])/'status.private.json';s=json.loads(p.read_text()) if p.exists() else {}
        s.update(state='FAILED',updated=time.time(),error=traceback.format_exc());write_json(p,s);raise
