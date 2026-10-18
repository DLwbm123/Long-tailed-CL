"""Frozen historical B plus existing shrunk local scores; no model fitting."""
import json, os, sys, time, traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from run import write_json
from run_transfer import aggregate, paired
from run_v2 import local_scores, centered_energy, fuse


def quantiles(x):
    return None if not len(x) else dict(zip(['min','p25','median','p75','max'],torch.quantile(x.double(),torch.tensor([0,.25,.5,.75,1.],dtype=torch.float64)).tolist()))


def diagnostics(b,l,f,y):
    bp,lp,fp=[s.argmax(1) for s in (b,l,f)]
    ix=torch.arange(len(y));bc=bp==y;lc=lp==y;fc=fp==y
    values={}
    for name,s in [('B',b),('local',l),('fusion',f)]:
        top=s.topk(2,dim=1).values
        wrong=s.clone();wrong[ix,y]=-torch.inf
        values[name+'_top2']=top[:,0]-top[:,1]
        values[name+'_true_margin']=s[ix,y]-wrong.max(1).values
        values[name+'_B_vs_local']=s[ix,bp]-s[ix,lp]
    masks={'branch_disagreement':bp!=lp,'fusion_changed':bp!=fp,
           'rescue':~bc&fc,'harm':bc&~fc}
    classes=[]
    for c in range(b.shape[1]):
        m=y==c;d=m&masks['branch_disagreement']
        classes.append(dict(class_index=c,count=int(m.sum()),B_correct=int((m&bc).sum()),
            local_correct=int((m&lc).sum()),fusion_correct=int((m&fc).sum()),
            rescue=int((m&masks['rescue']).sum()),harm=int((m&masks['harm']).sum()),
            branch_disagreement=int(d.sum()),B_only_correct=int((d&bc&~lc).sum()),
            local_only_correct=int((d&~bc&lc).sum()),both_wrong=int((d&~bc&~lc).sum())))
    summary={k:dict(count=int(m.sum()),margins={n:quantiles(v[m]) for n,v in values.items()}) for k,m in masks.items()}
    # Per-row diagnostics are private, including query row indices and labels.
    rows=[]
    for i in torch.where(masks['branch_disagreement']|masks['fusion_changed'])[0].tolist():
        rows.append(dict(row_index=i,label=int(y[i]),B_pred=int(bp[i]),local_pred=int(lp[i]),fusion_pred=int(fp[i]),
                         **{k:bool(v[i]) for k,v in masks.items()},**{k:float(v[i]) for k,v in values.items()}))
    assert sum(c['rescue']-c['harm'] for c in classes)==int(fc.sum()-bc.sum())
    return classes,summary,rows


def main(cfg):
    assert cfg['local_lambda']==.5 and cfg['parameter_scan'] is False
    torch.set_num_threads(4);start=time.time();root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.private.json',cfg)
    source=Path(cfg['source']);prior=Path(cfg['prior']);broot=Path(cfg['b_root'])
    sealed=False;names=['ISIC','HK']
    allowed={source/'config.private.json',source/'results.json',source/'status.private.json'}
    for n in names:
        allowed|={source/n/'model.private.pt',broot/(n+'.B.private.pt'),prior/(n+'.S_old.features.private.pt'),prior/(n+'.S_old.rows.private.json')}
        allowed|={source/n/f'class_{c:02d}.private.pt' for c in range(4 if n=='ISIC' else 13)}
    query_paths={source/n/(q+'.predictions.private.pt') for n in names for q in ['Qfit','Qgate']}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if any(p.is_relative_to(t) for t in [source,prior,broot]):
                assert p in allowed or (sealed and p in query_paths),'Input outside frozen allowlist'
    sys.addaudithook(audit)
    assert json.loads((source/'status.private.json').read_text())['state']=='COMPLETE'
    src=json.loads((source/'config.private.json').read_text());scales={}
    for case in src['datasets']:
        n=case['name'];old=case['old'];seen=old+case['new'];gen=torch.Generator().manual_seed(src['seed'])
        rows=json.loads((prior/(n+'.S_old.rows.private.json')).read_text())
        data=torch.load(prior/(n+'.S_old.features.private.pt'),weights_only=True)
        assert data['components']==[r['identity_component'] for r in rows]
        assert torch.equal(data['y'],torch.tensor([r['label'] for r in rows]))
        b=torch.load(broot/(n+'.B.private.pt'),weights_only=True)
        assert b['frame'].shape==(1536,seen) and all(not v.any() for v in b['model'])
        obj=torch.load(source/n/'model.private.pt',weights_only=False);energies=[];counts={}
        for c in range(old):
            indices=[i for i,r in enumerate(rows) if r['label']==c];groups={}
            for i in indices:groups.setdefault(rows[i]['identity_component'],[]).append(i)
            ids=sorted(groups);sample=[min(groups[ids[j]],key=lambda i:rows[i]['relative_path']) for j in torch.randperm(len(ids),generator=gen)[:src['calibration_patients_per_class']].tolist()]
            # The original extraction emitted sampled images in source-row order.
            sample=sorted(sample);saved=torch.load(source/n/f'class_{c:02d}.private.pt',weights_only=False)
            p=saved['calibration'][0];assert len(p)==len(sample)
            bs=8*F.normalize(data['x'][sample],dim=1)@b['frame']
            _,ls=local_scores(p,obj['stats'],obj['reference'],src['tau'])
            energies.append([float(centered_energy(bs)),float(centered_energy(ls))]);counts[str(c)]=len(sample)
        scale=torch.tensor(energies,dtype=torch.float64).mean(0).sqrt().tolist()
        assert all(v>1e-12 for v in scale);scales[n]=dict(scales=scale,counts=counts,classes=seen)
        write_json(root/'status.json',dict(state='CALIBRATING',dataset=n,elapsed=time.time()-start))
    write_json(root/'calibration.json',scales);write_json(root/'SCALE_SEAL.json',dict(sealed_at=time.time(),queries_opened=False));sealed=True
    refs=json.loads((source/'results.json').read_text());records=[]
    for ref in refs:
        n,q=ref['dataset'],ref['query'];old=ref['old']
        cached=torch.load(source/n/(q+'.predictions.private.pt'),weights_only=True)
        b,l,y=cached['scores']['B'],cached['scores']['local_shrink'],cached['labels'];scale=scales[n]['scales']
        assert torch.isfinite(b).all() and torch.isfinite(l).all()
        f=fuse(b,l,scale,cfg['local_lambda']);metrics={k:aggregate(s,y,old) for k,s in [('B',b),('local_shrink',l),('B_local',f)]}
        for k in ['B','local_shrink']:
            assert metrics[k]==ref['metrics'][k],'Frozen reference reproduction failed'
        assert torch.equal((b/scale[0]).argmax(1),b.argmax(1))
        classes,margins,rows=diagnostics(b/scale[0],l/scale[1],f,y)
        for row in rows:
            i=row['row_index']
            for name,s in [('B',b),('local',l)]:
                top=s[i].topk(2).values;row[name+'_raw_top2']=float(top[0]-top[1])
        write_json(root/(n+'_'+q+'.disagreements.private.json'),rows)
        torch.save(dict(B=b,local=l,fusion=f,labels=y),root/(n+'_'+q+'.scores.private.pt'))
        records.append(dict(dataset=n,query=q,old=old,metrics=metrics,paired=paired(b,f,y,old),classes=classes,margins=margins))
    write_json(root/'results.json',records)
    write_json(root/'status.json',dict(state='COMPLETE',conditions=len(records),wall_seconds=time.time()-start,model_fits=0,image_forwards=0,parameter_scan=False,test_access=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.json',dict(error=traceback.format_exc()));raise
