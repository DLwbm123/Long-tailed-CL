"""Fixed query-supervision control: fit class offsets with the base model frozen."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT, scores
from query_node_weights import select
from run_real_query_node import load, metrics, save


def fit_bias(logits,y,steps,learning_rate):
    seen=logits.shape[1];counts=torch.bincount(y,minlength=seen);assert bool((counts>0).all())
    w=1/(seen*counts[y].to(DT));assert abs(float(w.sum())-1)<1e-12
    b=torch.zeros(seen,dtype=DT,requires_grad=True)
    def loss(v):return -(w*(logits+v).log_softmax(1)[torch.arange(len(y)),y]).sum()+.5*v.square().mean()
    trace=[float(loss(b).detach())]
    for _ in range(steps):
        g=torch.autograd.grad(loss(b),b)[0]
        b=(b-g*learning_rate).detach();b=(b-b.mean()).requires_grad_()
        value=float(loss(b).detach());assert value<=trace[-1]+1e-12 and bool(torch.isfinite(b).all());trace.append(value)
    assert abs(float(b.detach().mean()))<1e-12
    return b.detach(),trace


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());prior=Path(cfg['prior_root']);diagnostic=Path(cfg['diagnostic_root'])
    lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(2);torch.set_default_dtype(DT)
    allowed={name+'.'+s for s in ('baseline.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed,'Only frozen baseline and query features allowed'
        if p.is_relative_to(diagnostic):assert p.name==name+'_half.json','Only prior aggregate diagnostic receipt allowed'
    sys.addaudithook(audit)
    for suffix in ('E.features.private.pt','S_old.features.private.pt','S_new.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Data role guard failed')
    start=time.monotonic();save(root/(job['id']+'.progress.json'),dict(phase='fit_query_class_bias',dataset=name))
    obj=load(prior/(name+'.baseline.private.pt'));fit=load(prior/(name+'.Qfit.features.private.pt'))
    with torch.no_grad():logits=scores(fit['x'],obj['model'],obj['frame'])[:,:seen]
    b,trace=fit_bias(logits,fit['y'],lock['bias_steps'],lock['learning_rate'])
    prior_result=json.loads((diagnostic/(name+'_half.json')).read_text())
    expected=next(r['CE'] for r in prior_result['summaries'] if r['query']=='Qfit' and r['model']=='baseline' and r['group']=='all')
    zero_error=abs(trace[0]-expected);assert zero_error<1e-10
    obj['bias']=b.detach();torch.save(obj,root/(name+'.bias.private.pt'));sealed_at=time.time()
    rows=list(prior_result['class_rows']);summary=list(prior_result['summaries'])
    for query,data in [('Qfit',fit),('Qgate',load(prior/(name+'.Qgate.features.private.pt')))]:
        _,rr,ba=metrics(obj,data,old,seen);rows.extend(dict(dataset=name,query=query,model='bias',**r) for r in rr)
        for group in ('all','old','new'):
            subset=[r for r in rr if group=='all' or r['group']==group]
            summary.append(dict(dataset=name,query=query,model='bias',group=group,BA=ba[group],CE=sum(r['CE'] for r in subset)/len(subset)))
    get=lambda mode:{r['group']:r['BA'] for r in summary if r['query']=='Qgate' and r['model']==mode}
    result=dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
        bias_steps=lock['bias_steps'],adapter_updates=0,encoder_updates=0,new_labels=0,E_access=False,S_access=False,
        additional_inference_parameters=seen,bias_storage_bytes=seen*b.element_size(),
        bias=b.detach().tolist(),bias_sum=float(b.detach().sum()),objective_trace=trace,objective_nonincreasing=True,
        zero_bias_CE_reproduction_error=zero_error,
        class_rows=rows,summaries=summary,gate=select(get('baseline'),get('bias')),
        comparison='All four models are development-only; prior baseline/full/half metrics reused without inference',
        operation_seconds=time.monotonic()-start)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
