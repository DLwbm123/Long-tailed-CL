"""One fixed half-direction control on already exposed development queries."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT, clone_model
from query_node_weights import rollout, select
from run_real_query_node import load, metrics, save


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());prior=Path(cfg['prior_root'])
    lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    case=next(c for c in lock['datasets'] if c['name']==name)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(2);torch.set_default_dtype(DT)
    allowed={name+'.'+s for s in ('teacher.private.pt','S_new.features.private.pt','Qfit.features.private.pt',
        'Qgate.features.private.pt','baseline.private.pt','candidate.private.pt','direction.private.pt')}
    allowed.update(('choose.json',name+'_candidate.json'))
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed,'Only sealed mechanism inputs allowed; E and old S denied'
    sys.addaudithook(audit)
    try:(prior/(name+'.E.features.private.pt')).open('rb')
    except AssertionError:pass
    else:raise AssertionError('E access guard failed')
    begin=time.monotonic();save(root/(job['id']+'.progress.json'),dict(phase='fixed_half_direction',dataset=name))
    state=load(prior/(name+'.teacher.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    d=load(prior/(name+'.direction.private.pt'))['logit_change'];half=.5*d
    assert bool(d.any()) and bool(torch.isfinite(half).all())
    assert torch.equal(half*2,d) and float(half.abs().max())<=lock['half_logit_cap']+1e-12
    old=case['old'];seen=old+case['new'];current={c:new['x'][new['y']==c] for c in range(old,seen)}
    for c,(_,p) in state['memory'].items():
        w=(p.log()+half[c]).softmax(0);assert abs(float(w.sum())-1)<1e-12 and bool((w>0).all())
    model=clone_model(rollout(state['teacher'],current,state['memory'],state['frame'],half))
    obj=dict(model=model,frame=state['frame'])
    torch.save(obj,root/(name+'.half.private.pt'))
    sealed_at=time.time();rows=[];summary=[]
    fit=load(prior/(name+'.Qfit.features.private.pt'))
    for mode,item in [('baseline',load(prior/(name+'.baseline.private.pt'))),
                      ('full',load(prior/(name+'.candidate.private.pt'))),('half',obj)]:
        _,rr,ba=metrics(item,fit,old,seen)
        rows.extend(dict(dataset=name,query='Qfit',model=mode,**r) for r in rr)
        for group in ('all','old','new'):
            subset=[r for r in rr if group=='all' or r['group']==group]
            summary.append(dict(dataset=name,query='Qfit',model=mode,group=group,BA=ba[group],CE=sum(r['CE'] for r in subset)/len(subset)))
    gate=load(prior/(name+'.Qgate.features.private.pt'));_,rr,ba=metrics(obj,gate,old,seen)
    rows.extend(dict(dataset=name,query='Qgate',model='half',**r) for r in rr)
    saved=json.loads((prior/'choose.json').read_text())
    rr_saved=[r for r in saved['class_rows'] if r['dataset']==name]
    rows.extend(dict(query='Qgate',**dict(r,model='full' if r['model']=='candidate' else 'baseline')) for r in rr_saved)
    for mode in ('baseline','full','half'):
        for group in ('all','old','new'):
            subset=[r for r in rows if r['query']=='Qgate' and r['model']==mode and (group=='all' or r['group']==group)]
            summary.append(dict(dataset=name,query='Qgate',model=mode,group=group,
                BA=sum(r['recall'] for r in subset)/len(subset),CE=sum(r['CE'] for r in subset)/len(subset)))
    prior_info=json.loads((prior/(name+'_candidate.json')).read_text())
    diagnostic=[]
    for mode,scale in [('full',1.),('half',.5)]:
        changes={}
        for group in ('old','new'):
            get=lambda m:next(r['CE'] for r in summary if r['query']=='Qfit' and r['model']==m and r['group']==group)
            assert abs(get('baseline')-prior_info['baseline_Qfit_'+group+'_CE'])<1e-10
            predicted=scale*prior_info['direction'][group+'_directional_derivative'];actual=get(mode)-get('baseline')
            changes[group]=dict(predicted_CE_change=predicted,actual_CE_change=actual,nonlinear_residual=actual-predicted)
        diagnostic.append(dict(model=mode,groups=changes,realized_joint_descent=all(v['actual_CE_change']<=1e-10 for v in changes.values())))
    base={r['group']:r['BA'] for r in summary if r['query']=='Qgate' and r['model']=='baseline'}
    receipt=dict(status='COMPLETED',dataset=name,source_commit=os.environ['Q356_SHA'],sealed_at=sealed_at,
        half_adapter_updates=48,baseline_retrained=False,teacher_retrained=False,old_S_access=False,E_access=False,
        new_labels=0,new_feature_extraction=False,query_exposure='Previously exposed development Qfit and Qgate',
        class_rows=rows,summaries=summary,mechanism=diagnostic,half_gate_descriptive=select(base,ba),
        finite_step_checks=True,operation_seconds=time.monotonic()-begin)
    save(root/(job['id']+'.json'),receipt)


if __name__=='__main__':main()
