"""Fixed 2x2 CE/retention support diagnostic, with presealed real representatives."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import DT
from paper_medical_models import herding_indices
from query_node_weights import select
from run_real_query_node import load,metrics,save
from run_real_support_ce_reference import train_reference


def build_banks(root,prior,lock):
    selection=lock.get('selection_mode','herding')
    assert selection in ('herding','random')
    def indices(x,c):
        if selection=='herding':return herding_indices(x,lock['nodes_per_class'])
        return torch.randperm(len(x),generator=torch.Generator().manual_seed(lock['seed']+c))[:min(lock['nodes_per_class'],len(x))].sort().values.tolist()
    # Ties are resolved by source order; scarce classes keep all available samples.
    example=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],dtype=DT)
    assert herding_indices(example,1)==[0]
    assert sorted(herding_indices(example,16))==[0,1,2]
    assert sorted(indices(example,0))==[0,1,2]
    probe=torch.zeros(23,2,dtype=DT);idx=indices(probe,0)
    assert idx==indices(probe,0) and len(set(idx))==16 and all(0<=i<23 for i in idx)
    banks=[]
    for case in lock['datasets']:
        name=case['name'];data=load(prior/(name+'.S_old.features.private.pt'));bank={};counts=[]
        for c in range(case['old']):
            x=data['x'][data['y']==c];idx=indices(x,c)
            assert len(idx)==min(16,len(x)) and len(set(idx))==len(idx)>0
            bank[c]=(x[idx].clone(),torch.full((len(idx),),1/len(idx),dtype=DT))
            counts.append(dict(class_index=c,available=len(x),selected=len(idx)))
        torch.save(bank,root/(name+'.'+selection+'.private.pt'))
        banks.append(dict(dataset=name,counts=counts,node_count=sum(len(z) for z,p in bank.values()),
                          feature_bytes=sum(z.numel()*z.element_size() for z,p in bank.values()),
                          probability_bytes=sum(p.numel()*p.element_size() for z,p in bank.values())))
    return dict(status='COMPLETED',selection=selection,banks=banks,banks_sealed_at=time.time(),query_access=False,
                E_access=False,source_commit=os.environ['Q356_SHA'],engineering_adapter_updates=0)


def run_cells(root,prior,diagnostic,lock,job,seal):
    name=job['dataset'];case=next(c for c in lock['datasets'] if c['name']==name)
    old=case['old'];seen=old+case['new'];start=time.monotonic()
    bank_receipt=json.loads((root/'banks.json').read_text());assert bank_receipt['status']=='COMPLETED'
    state=load(prior/(name+'.teacher.private.pt'));real=load(root/(name+'.'+lock.get('selection_mode','herding')+'.private.pt'))
    moment=state['memory'];new=load(prior/(name+'.S_new.features.private.pt'))
    current={c:new['x'][new['y']==c] for c in range(old,seen)}
    training={}
    for cell in lock['cells']:
        save(root/(job['id']+'.progress.json'),dict(phase='fit_'+cell,dataset=name))
        ce=moment if cell[0]=='M' else real;retention=moment if cell[1]=='M' else real
        model,info=train_reference(state['teacher'],current,ce,retention,state['frame'],lock['adapter_steps'])
        torch.save(dict(model=model,frame=state['frame']),root/(name+'.'+cell+'.private.pt'))
        training[cell]=info
    sealed_at=time.time();seal()
    ref=json.loads((diagnostic/(name+'_half.json')).read_text())
    rows=[dict(r,model='MM') for r in ref['class_rows'] if r['model']=='baseline'];summary=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'))
        for cell in lock['cells']:
            _,rr,_=metrics(load(root/(name+'.'+cell+'.private.pt')),data,old,seen)
            rows.extend(dict(dataset=name,query=query,model=cell,**r) for r in rr)
        for cell in ['MM']+lock['cells']:
            for group in ('all','old','new'):
                rs=[r for r in rows if r['query']==query and r['model']==cell and (group=='all' or r['group']==group)]
                summary.append(dict(dataset=name,query=query,model=cell,group=group,
                                    BA=sum(r['recall'] for r in rs)/len(rs),CE=sum(r['CE'] for r in rs)/len(rs)))
    gates={};interaction=[]
    for query in ('Qfit','Qgate'):
        get=lambda cell:{r['group']:r['BA'] for r in summary if r['query']==query and r['model']==cell}
        gates[query]={cell:select(get('MM'),get(cell)) for cell in lock['cells']}
        for group in ('all','old','new'):
            rs={r['model']:r for r in summary if r['query']==query and r['group']==group}
            if all(cell in rs for cell in ('HH','HM','MH')):
                interaction.append(dict(query=query,group=group,**{metric:rs['HH'][metric]-rs['HM'][metric]-rs['MH'][metric]+rs['MM'][metric] for metric in ('BA','CE')}))
    primary=lock.get('primary_cell','HH')
    return dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,
        training=training,adapter_updates=len(lock['cells'])*lock['adapter_steps'],primary=primary,
        primary_both_queries_pass=all(gates[q][primary]['selected']=='candidate' for q in gates),
        gates=gates,class_rows=rows,summaries=summary,interaction_HH_minus_HM_minus_MH_plus_MM=interaction,
        bank=next(b for b in bank_receipt['banks'] if b['dataset']==name),
        original_moment_node_count=sum(len(z) for z,p in moment.values()),
        mixed_cells_use_both_banks=any('M' in cell for cell in lock['cells']),old_S_access_in_training=False,query_used_for_fitting=False,
        E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start)


def main():
    root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE']);job=json.loads(os.environ['Q356_JOB'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    assert lock['cells']==(['RR'] if lock.get('selection_mode')=='random' else ['HM','MH','HH'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';torch.set_num_threads(2);torch.set_default_dtype(DT)
    prior=Path(cfg['prior_root']);diagnostic=Path(cfg['diagnostic_root']);sealed=False
    name=job.get('dataset');bank_job=job['role']=='banks'
    allowed={c['name']+'.S_old.features.private.pt' for c in lock['datasets']} if bank_job else {name+'.'+s for s in ('teacher.private.pt','S_new.features.private.pt')}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name in allowed or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Herding role boundary'
        if p.is_relative_to(diagnostic):assert sealed and p.name==name+'_half.json','References after all models sealed'
    sys.addaudithook(audit)
    def seal():
        nonlocal sealed
        sealed=True
    test_name=lock['datasets'][0]['name'] if bank_job else name
    blocked=['E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt']+([] if bank_job else ['S_old.features.private.pt'])
    for suffix in blocked:
        try:(prior/(test_name+'.'+suffix)).open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    save(root/(job['id']+'.progress.json'),dict(phase='construct_banks' if bank_job else 'initializing'))
    result=build_banks(root,prior,lock) if bank_job else run_cells(root,prior,diagnostic,lock,job,seal)
    save(root/(job['id']+'.json'),result)


if __name__=='__main__':main()
