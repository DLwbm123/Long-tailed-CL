"""One support-only old/new logit offset, solved without query fitting."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,scores,training_batch
from query_node_weights import select
from run_real_query_node import load,metrics,save


def solve_offset(ss,y,p,old,steps):
    assert abs(float(p.sum())-1)<1e-12 and bool((p>0).all()) and bool(torch.isfinite(ss).all())
    t=ss[:,old:].logsumexp(1)-ss[:,:old].logsumexp(1);pi=p[y>=old].sum();assert 0<float(pi)<1
    center=torch.logit(pi);lo=center-t.max();hi=center-t.min()
    residual=lambda b:(p*torch.sigmoid(t+b)).sum()-pi
    assert float(residual(lo))<=1e-12 and float(residual(hi))>=-1e-12
    bracket=[float(lo),float(hi)]
    for _ in range(steps):
        mid=(lo+hi)/2
        if float(residual(mid))>0:hi=mid
        else:lo=mid
    b=(lo+hi)/2;assert abs(float(residual(b)))<1e-12 and bool(torch.isfinite(b))
    return float(b),dict(iterations=steps,initial_bracket=bracket,target_new_group_mass=float(pi),root_residual=float(residual(b)))


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE']);job=json.loads(os.environ['Q356_JOB'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT)
    b,_=solve_offset(torch.tensor([[0.,3.],[0.,3.]],dtype=DT),torch.tensor([0,1]),torch.tensor([.5,.5],dtype=DT),1,64)
    assert abs(b+3)<1e-12
    name=job['dataset'];case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);reference=Path(cfg['reference_root']);sealed=False
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Support calibration role boundary'
        if p.is_relative_to(reference):assert p.name in {name+'.HH.private.pt',name+'.herding.private.pt'} or (sealed and p.name==name+'_factorial.json'),'HH references after sealing'
    sys.addaudithook(audit)
    for suffix in ('E.features.private.pt','S_old.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Pre-seal access guard failed')
    start=time.monotonic();save(root/(job['id']+'.progress.json'),dict(phase='support_only_scalar_calibration',dataset=name))
    obj=load(reference/(name+'.HH.private.pt'));memory=load(reference/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    current={c:new['x'][new['y']==c] for c in range(old,seen)};x,y,p=training_batch(current,memory)
    assert all(abs(float(p[y==c].sum())-1/seen)<1e-12 for c in range(seen))
    with torch.no_grad():
        ss=torch.cat([scores(x[i:i+256],obj['model'],obj['frame'])[:,:seen] for i in range(0,len(y),256)])
        b,fit=solve_offset(ss,y,p,old,lock['bisection_steps']);bias=torch.zeros(seen,dtype=DT);bias[old:]=b
        before=float(-(p*ss.log_softmax(1)[torch.arange(len(y)),y]).sum());after=float(-(p*(ss+bias).log_softmax(1)[torch.arange(len(y)),y]).sum())
        assert after<=before+1e-12
        fit.update(offset=b,support_count=len(y),old_support_count=int((y<old).sum()),support_CE_before=before,support_CE_after=after)
    obj['bias']=bias;torch.save(obj,root/(name+'.group_bias.private.pt'));sealed=True;sealed_at=time.time()
    ref=json.loads((reference/(name+'_factorial.json')).read_text());rows=[r for r in ref['class_rows'] if r['model'] in ('MM','HH')];summary=[]
    for query in ('Qfit','Qgate'):
        _,rr,_=metrics(obj,load(prior/(name+'.'+query+'.features.private.pt')),old,seen)
        rows.extend(dict(dataset=name,query=query,model='group_bias',**r) for r in rr)
        for mode in ('MM','HH','group_bias'):
            for group in ('all','old','new'):
                rs=[r for r in rows if r['query']==query and r['model']==mode and (group=='all' or r['group']==group)]
                summary.append(dict(dataset=name,query=query,model=mode,group=group,BA=sum(r['recall'] for r in rs)/len(rs),CE=sum(r['CE'] for r in rs)/len(rs)))
    gates={}
    for query in ('Qfit','Qgate'):
        get=lambda mode:{r['group']:r['BA'] for r in summary if r['query']==query and r['model']==mode}
        gates[query]=dict(vs_MM=select(get('MM'),get('group_bias')),vs_HH=select(get('HH'),get('group_bias')))
    result=dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
        fit=fit,primary='group_bias_vs_MM',primary_both_queries_pass=all(g['vs_MM']['selected']=='candidate' for g in gates.values()),
        gates=gates,class_rows=rows,summaries=summary,extra_scalar_parameters=1,adapter_updates=0,gradient_evaluations=0,
        scalar_root_iterations=lock['bisection_steps'],engineering_unit_root_iterations=64,unit_passed=True,
        query_used_for_fitting=False,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
