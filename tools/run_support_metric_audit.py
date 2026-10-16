"""Read-only support CE versus balanced accuracy at three frozen states."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,training_batch
from run_real_query_node import load,metrics,save


def classify(delta_ce,delta_ba):
    if delta_ce>=-1e-12:return 'SUPPORT_CE_NOT_IMPROVED'
    return 'SUPPORT_CE_BA_MISMATCH' if delta_ba<-1e-12 else 'SUPPORT_CE_IMPROVED_BA_NONWORSE'


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT)
    assert classify(-.1,-.1)=='SUPPORT_CE_BA_MISMATCH'
    assert classify(-.1,0.)==classify(-.1,.1)=='SUPPORT_CE_IMPROVED_BA_NONWORSE'
    assert classify(0.,-.1)==classify(.1,.1)=='SUPPORT_CE_NOT_IMPROVED'
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];prior=Path(cfg['prior_root'])
    paths={key:Path(path)/(name+'.HH.private.pt') for key,path in cfg['model_roots'].items()}
    allowed={prior/(name+'.teacher.private.pt'),prior/(name+'.S_new.features.private.pt'),*paths.values()}
    protected=[prior]+[Path(p) for p in cfg['model_roots'].values()]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if any(p.is_relative_to(r) for r in protected):assert p in allowed,'Support metric boundary'
    sys.addaudithook(audit)
    for suffix in ('S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    state=load(prior/(name+'.teacher.private.pt'));bank=load(root/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    for x,p in bank.values():assert bool(torch.allclose(p,torch.full_like(p,1/len(p)),atol=1e-12,rtol=0))
    current={c:new['x'][new['y']==c] for c in range(old,seen)};x,y,p=training_batch(current,bank)
    assert all(abs(float(p[y==c].sum())-1/seen)<1e-12 for c in range(seen))
    rows=[];summary=[]
    for label in lock['states']:
        save(root/(job['id']+'.progress.json'),dict(phase='support_metrics_'+label,dataset=name))
        obj=dict(model=state['teacher'],frame=state['frame']) if label=='teacher' else load(paths[label])
        _,rr,ba=metrics(obj,dict(x=x,y=y),old,seen)
        rows.extend(dict(dataset=name,model=label,**v) for v in rr)
        for group in ('all','old','new'):
            selected=[v for v in rr if group=='all' or v['group']==group]
            summary.append(dict(dataset=name,model=label,group=group,BA=ba[group],CE=sum(v['CE'] for v in selected)/len(selected),count=sum(v['count'] for v in selected)))
    comparisons=[]
    for reference in ('teacher','original_HH'):
        for group in ('all','old','new'):
            a=next(v for v in summary if v['model']=='projected' and v['group']==group);b=next(v for v in summary if v['model']==reference and v['group']==group)
            dc=a['CE']-b['CE'];db=a['BA']-b['BA'];comparisons.append(dict(reference=reference,group=group,delta_CE=dc,delta_BA=db,category=classify(dc,db)))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,class_rows=rows,summaries=summary,comparisons=comparisons,
        primary_old_vs_HH=next(v for v in comparisons if v['reference']=='original_HH' and v['group']=='old'),
        support_samples=len(y),sample_forward_evaluations=len(y)*len(lock['states']),gradient_evaluations=0,adapter_updates=0,model_updates=0,
        old_S_access=False,query_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,
        scope='Development support metrics on H and new S only; historical query comparison belongs to report, no new query inference'))


if __name__=='__main__':main()
