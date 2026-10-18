"""B-anchored minimum positive evidence shared by density and distance."""
import json,os,sys,time,traceback
from pathlib import Path
import torch
from run import write_json
from run_transfer import aggregate,paired
from run_v2 import fuse
from run_b_local import diagnostics


def consensus(b,l,d,scales):
    assert all(torch.isfinite(s).all() for s in [b,l,d])
    a=b.argmax(1);ix=torch.arange(len(b));bn=b/scales[0]
    u=(l-l[ix,a,None])/scales[1];v=(d-d[ix,a,None])/scales[1]
    t=bn-bn[ix,a,None]+.5*torch.minimum(u,v).clamp_min(0)
    # Exact zero ties retain B; this nextafter changes no meaningful CE precision.
    stay=t.max(1).values<=0
    t[ix[stay],a[stay]]=torch.nextafter(torch.tensor(0.,dtype=t.dtype),torch.tensor(float('inf'),dtype=t.dtype))
    return t


def retention(b,refs,new,y,old):
    bc=b.argmax(1)==y;nc=new.argmax(1)==y;result={}
    for group,m in [('all',torch.ones_like(y,dtype=torch.bool)),('old',y<old),('new',y>=old)]:
        entry={}
        for name,s in refs.items():
            rc=s.argmax(1)==y;rescued=m&~bc&rc
            entry[name]=dict(reference_rescues=int(rescued.sum()),retained=int((rescued&nc).sum()),lost=int((rescued&~nc).sum()),additional_rescues=int((m&~bc&~rc&nc).sum()))
        common=m&bc
        for s in refs.values():common&=s.argmax(1)!=y
        harm=m&bc&~nc
        assert not (harm&~common).any(),'Harm subset theorem violated'
        entry['harm_bound']=dict(upper_bound=int(common.sum()),observed=int(harm.sum()),subset_verified=True)
        result[group]=entry
    return result


def main(cfg):
    torch.set_num_threads(4);start=time.time();root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.private.json',cfg);a=Path(cfg['density']);d=Path(cfg['distance'])
    allowed={p/n for p in [a,d] for n in ['status.json','calibration.json','results.json']}
    allowed|={p/(n+'_'+q+'.scores.private.pt') for p in [a,d] for n in ['ISIC','HK'] for q in ['Qfit','Qgate']}
    allowed.add(a/'HK_Qgate.disagreements.private.json')
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(a) or p.is_relative_to(d):assert p in allowed,'Input allowlist'
    sys.addaudithook(audit)
    assert cfg['local_lambda']==.5 and cfg['parameter_scan'] is False
    for p in [a,d]:assert json.loads((p/'status.json').read_text())['state']=='COMPLETE'
    scales=json.loads((a/'calibration.json').read_text());assert scales==json.loads((d/'calibration.json').read_text())
    write_json(root/'calibration.json',scales)
    refs=json.loads((d/'results.json').read_text());records=[]
    for ref in refs:
        n,q,old=ref['dataset'],ref['query'],ref['old'];s=torch.load(a/(n+'_'+q+'.scores.private.pt'),weights_only=True);z=torch.load(d/(n+'_'+q+'.scores.private.pt'),weights_only=True)
        b,l,dl,y=s['B'],s['local'],z['local'],s['labels'];sc=scales[n]['scales']
        assert torch.equal(b,z['B']) and torch.equal(y,z['labels'])
        assert torch.equal(fuse(b,l,sc,.5),s['fusion']) and torch.equal(fuse(b,dl,sc,.5),z['fusion'])
        t=consensus(b,l,dl,sc);outputs={'B':b,'density_fusion':s['fusion'],'distance_fusion':z['fusion'],'consensus':t}
        metrics={k:aggregate(v,y,old) for k,v in outputs.items()}
        for k,other in [('B','B'),('density_fusion','previous_fusion'),('distance_fusion','B_distance')]:assert metrics[k]==ref['metrics'][other]
        prior={k:outputs[k] for k in ['density_fusion','distance_fusion']}
        rr=retention(b,prior,t,y,old)
        # Stronger label-free check: a change requires both existing fusions to leave B.
        changed=t.argmax(1)!=b.argmax(1)
        assert all(not (changed&(v.argmax(1)==b.argmax(1))).any() for v in prior.values())
        classes,margins,private=diagnostics(b/sc[0],dl/sc[1],t,y)
        record=dict(dataset=n,query=q,old=old,metrics=metrics,retention=rr,paired={k:paired(v,t,y,old) for k,v in outputs.items() if k!='consensus'},classes=classes,margins=margins)
        if n=='HK' and q=='Qgate':
            targets=[r for r in json.loads((a/'HK_Qgate.disagreements.private.json').read_text()) if r['harm']];assert len(targets)==2
            record['target_audit']=[dict(class_index=r['label'],correct=bool(t[r['row_index']].argmax()==y[r['row_index']])) for r in targets]
            assert all(r['correct'] for r in record['target_audit'])
        write_json(root/(n+'_'+q+'.disagreements.private.json'),private)
        torch.save(dict(B=b,scores=t,labels=y),root/(n+'_'+q+'.scores.private.pt'));records.append(record)
    write_json(root/'results.json',records)
    write_json(root/'status.json',dict(state='COMPLETE',conditions=4,wall_seconds=time.time()-start,model_fits=0,image_forwards=0,scales_refitted=False,parameter_scan=False,test_access=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.json',dict(error=traceback.format_exc()));raise
