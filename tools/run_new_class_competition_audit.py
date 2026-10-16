"""Inference-only new-class support/query audit of two already sealed endpoints."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT,scores
from run_real_query_node import load,save


def summarize(ss,y,old,seen):
    assert ss.shape==(len(y),seen) and bool(((y>=old)&(y<seen)).all())
    pred=ss.argmax(1);restricted=ss[:,old:].argmax(1)+old
    category=torch.where(pred==y,0,torch.where(pred<old,1,2))
    true=ss[torch.arange(len(y)),y];other=ss[:,old:].clone()
    other[torch.arange(len(y)),y-old]=-torch.inf
    margins={'true_minus_best_old':true-ss[:,:old].max(1).values,
             'true_minus_other_new':true-other.max(1).values}
    ce=-ss.log_softmax(1)[torch.arange(len(y)),y];rows=[]
    for c in range(old,seen):
        mask=y==c;n=int(mask.sum());assert n>0
        correct=int(((pred==y)&mask).sum());forced=int(((restricted==y)&mask).sum())
        row=dict(class_index=c,count=n,correct=correct,recall=correct/n,CE=float(ce[mask].mean()),
            errors_to_old=int(((category==1)&mask).sum()),errors_to_new=int(((category==2)&mask).sum()),
            oracle_new_only_correct=forced,oracle_new_only_recall=forced/n,oracle_rescued=forced-correct)
        assert row['correct']+row['errors_to_old']+row['errors_to_new']==n and forced>=correct
        for key,value in margins.items():
            z=value[mask];qs=torch.quantile(z,torch.tensor([.1,.5,.9],dtype=DT))
            row.update({key+'_mean':float(z.mean()),key+'_q10':float(qs[0]),key+'_median':float(qs[1]),key+'_q90':float(qs[2])})
        rows.append(row)
    return rows,category


def unit():
    rows,cat=summarize(torch.tensor([[5.,0.,4.,1.],[0.,0.,1.,2.],[0.,0.,1.,3.]],dtype=DT),torch.tensor([2,2,3]),2,4)
    assert cat.tolist()==[1,2,0] and rows[0]['correct']==0 and rows[0]['oracle_new_only_correct']==1
    assert rows[1]['correct']==1 and rows[0]['true_minus_best_old_mean']==0.


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE']);job=json.loads(os.environ['Q356_JOB'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    name=job['dataset'];case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new']
    prior=Path(cfg['prior_root']);reference=Path(cfg['reference_root'])
    allowed={name+'.'+s for s in ('baseline.private.pt','S_new.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed,'Competition audit role boundary'
        if p.is_relative_to(reference):assert p.name in {name+'.HH.private.pt',name+'_factorial.json'},'Frozen HH endpoint only'
    sys.addaudithook(audit)
    for suffix in ('E.features.private.pt','S_old.features.private.pt'):
        try:(prior/(name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Forbidden access guard failed')
    start=time.monotonic();models={'MM':load(prior/(name+'.baseline.private.pt')),'HH':load(reference/(name+'.HH.private.pt'))}
    frozen=json.loads((reference/(name+'_factorial.json')).read_text());rows=[];summary=[];transitions=[];evaluations=0
    with torch.no_grad():
        for role in ('S_new','Qfit','Qgate'):
            save(root/(job['id']+'.progress.json'),dict(phase='audit_'+role,dataset=name))
            data=load(prior/(name+'.'+role+'.features.private.pt'));mask=data['y']>=old;x=data['x'][mask];y=data['y'][mask]
            assert len(y)==lock['expected_new_rows'][name][role]
            categories={}
            for mode,obj in models.items():
                ss=torch.cat([scores(x[i:i+256],obj['model'],obj['frame'])[:,:seen] for i in range(0,len(y),256)])
                rr,categories[mode]=summarize(ss,y,old,seen);evaluations+=len(y)
                if role!='S_new':
                    for row in rr:
                        ref=next(v for v in frozen['class_rows'] if v['query']==role and v['model']==mode and v['class_index']==row['class_index'])
                        assert (row['count'],row['correct'])==(ref['count'],ref['correct']),'Frozen prediction aggregate mismatch'
                rows.extend(dict(dataset=name,role=role,model=mode,**v) for v in rr)
                summary.append(dict(dataset=name,role=role,model=mode,count=len(y),new_BA=sum(v['recall'] for v in rr)/len(rr),
                                    oracle_new_only_BA=sum(v['oracle_new_only_recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
            labels=['correct','wrong_old','wrong_new']
            for c in range(old,seen):
                for i,a in enumerate(labels):
                    for j,b in enumerate(labels):
                        transitions.append(dict(dataset=name,role=role,class_index=c,MM=a,HH=b,count=int(((y==c)&(categories['MM']==i)&(categories['HH']==j)).sum())))
    delta={role:next(v['new_BA'] for v in summary if v['role']==role and v['model']=='HH')-next(v['new_BA'] for v in summary if v['role']==role and v['model']=='MM') for role in ('S_new','Qfit','Qgate')}
    result=dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,class_rows=rows,summaries=summary,transitions=transitions,
        HH_minus_MM_new_BA=delta,training_support_drop_over_1pp=delta['S_new']<-.01,query_drop_over_1pp={q:delta[q]<-.01 for q in ('Qfit','Qgate')},
        query_count_correct_parity=True,unit_check_passed=True,adapter_updates=0,gradient_evaluations=0,model_sample_evaluations=evaluations,
        E_access=False,old_S_access=False,query_used_for_fitting=False,new_labels=0,GPU_used=False,
        oracle_scope='Label-conditioned new-only diagnostic, not deployable classifier or replacement metric',operation_seconds=time.monotonic()-start)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
