"""Remove Gaussian volume offsets uniformly; retain frozen B, scales and lambda."""
import json,os,sys,time,traceback
from pathlib import Path
import torch
from run import write_json
from run_transfer import aggregate,paired
from run_v2 import fuse
from run_b_local import diagnostics


def offsets(obj,tau):
    values=[]
    for s in obj['stats']:
        beta=s['groups']/(s['groups']+tau);d=len(s['local'][0])
        cov=beta*s['local'][1]+(1-beta)*obj['reference']['local'][1]
        cov=cov+obj['reference']['local'][2]*torch.eye(d,dtype=cov.dtype)
        # Match the Cholesky log-determinant used by the original density scorer.
        values.append(torch.linalg.cholesky(cov).diag().log().sum()/d)
    return torch.stack(values)


def main(cfg):
    torch.set_num_threads(4);start=time.time()
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    write_json(root/'config.private.json',cfg)
    prior=Path(cfg['prior']);source=Path(cfg['source']);sealed=False
    allowed={prior/'status.json',prior/'calibration.json',prior/'results.json',source/'config.private.json'}
    allowed|={source/n/'model.private.pt' for n in ['ISIC','HK']}
    queries={prior/(n+'_'+q+'.scores.private.pt') for n in ['ISIC','HK'] for q in ['Qfit','Qgate']}
    queries.add(prior/'HK_Qgate.disagreements.private.json')
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(prior) or p.is_relative_to(source):
                assert p in allowed or (sealed and p in queries),'Frozen input allowlist'
    sys.addaudithook(audit)
    assert json.loads((prior/'status.json').read_text())['state']=='COMPLETE'
    sc=json.loads((prior/'calibration.json').read_text());src=json.loads((source/'config.private.json').read_text())
    assert cfg['local_lambda']==.5 and cfg['parameter_scan'] is False
    shift={n:offsets(torch.load(source/n/'model.private.pt',weights_only=False),src['tau']) for n in ['ISIC','HK']}
    write_json(root/'calibration.json',sc)
    write_json(root/'offsets.json',{n:v.tolist() for n,v in shift.items()})
    write_json(root/'SEAL.json',dict(sealed_at=time.time(),queries_opened=False,scales_refitted=False));sealed=True
    refs=json.loads((prior/'results.json').read_text());results=[]
    for ref in refs:
        n,q=ref['dataset'],ref['query'];old=ref['old'];scales=sc[n]['scales']
        s=torch.load(prior/(n+'_'+q+'.scores.private.pt'),weights_only=True)
        b,l,y,previous=s['B'],s['local'],s['labels'],s['fusion']
        assert torch.equal(fuse(b,l,scales,.5),previous)
        distance=l+shift[n][None,:];f=fuse(b,distance,scales,.5)
        assert torch.isfinite(f).all()
        metrics={k:aggregate(v,y,old) for k,v in [('B',b),('previous_fusion',previous),('distance_local',distance),('B_distance',f)]}
        assert metrics['B']==ref['metrics']['B'] and metrics['previous_fusion']==ref['metrics']['B_local']
        classes,margins,rows=diagnostics(b/scales[0],distance/scales[1],f,y)
        prevclasses,prevmargins,_=diagnostics(previous,distance/scales[1],f,y)
        record=dict(dataset=n,query=q,old=old,metrics=metrics,paired_B=paired(b,f,y,old),paired_previous=paired(previous,f,y,old),classes=classes,classes_previous=prevclasses,margins=margins)
        # The targeted audit is descriptive only and never enters the decision rule.
        if n=='HK' and q=='Qgate':
            targets=[r for r in json.loads((prior/'HK_Qgate.disagreements.private.json').read_text()) if r['harm']]
            assert len(targets)==2
            record['target_audit']=[dict(class_index=r['label'],B_correct=bool(b[r['row_index']].argmax()==y[r['row_index']]),previous_correct=bool(previous[r['row_index']].argmax()==y[r['row_index']]),current_correct=bool(f[r['row_index']].argmax()==y[r['row_index']])) for r in targets]
        write_json(root/(n+'_'+q+'.disagreements.private.json'),rows)
        torch.save(dict(B=b,local=distance,fusion=f,labels=y),root/(n+'_'+q+'.scores.private.pt'))
        results.append(record)
    write_json(root/'results.json',results)
    write_json(root/'status.json',dict(state='COMPLETE',conditions=4,wall_seconds=time.time()-start,model_fits=0,image_forwards=0,scales_refitted=False,parameter_scan=False,test_access=False))
    print('COMPLETE',flush=True)


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        write_json(Path(cfg['output'])/'failure.json',dict(error=traceback.format_exc()));raise
