"""Complete the information-by-readout factorial with two bounded CE fits."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA, DT
from query_node_weights import select
from run_affine_pair_diagnostic import paired
from run_linear_ce_head import fit, unit
from run_offline_information_ab import validate
from run_real_query_node import load, save

MODELS=('R_H','R_F','C_H','C_F')
COMPARISONS=(('R_H','R_F'),('C_H','C_F'),('R_H','C_H'),('R_F','C_F'))


def interaction(values):
    return values['C_F']-values['C_H']-values['R_F']+values['R_H']


def check_head(obj,seen):
    assert obj['frame'].shape==(1536,seen) and obj['frame'].dtype==DT and bool(torch.isfinite(obj['frame']).all()) and 'bias' not in obj
    assert len(obj['model'])==2 and all(v.shape==(1536,8) and v.dtype==DT and bool(torch.isfinite(v).all()) and not bool(v.any()) for v in obj['model'])


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE']);sha=os.environ['Q356_SHA']
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    job=json.loads(os.environ['Q356_JOB']);role=job['role'];name=job.get('dataset')
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==.001 and lock['logit_scale']==BETA==8
    assert list(MODELS)==lock['models'] and [list(v) for v in COMPARISONS]==lock['comparisons']
    torch.set_num_threads(2);torch.set_default_dtype(DT)
    bases={key:Path(cfg[key+'_root']) for key in ('prior','ridge','offline','ce')}
    allowed={key:{'DELIVERY.private.json'} for key in bases}
    if name:
        allowed['ridge'].add(name+'.ridge.private.pt')
        if role=='fit':allowed['prior'].update(name+'.'+s+'.features.private.pt' for s in ('S_old','S_new'))
        if role=='evaluate':
            # Both fits must be sealed before either query can be opened.
            for n in ('ISIC','HK'):
                rec=json.loads((root/(n+'_fit.json')).read_text())
                assert rec['status']=='COMPLETED' and rec['source_commit']==sha and (root/(n+'.ce.private.pt')).is_file()
            allowed['prior'].update(name+'.'+q+'.features.private.pt' for q in ('Qfit','Qgate'))
            allowed['offline'].update((name+'.B.private.pt',name+'_evaluate.json'))
            allowed['ce'].update((name+'.ce.private.pt',name+'_ce.json'))
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        for key,base in bases.items():
            if p.is_relative_to(base):assert p.parent==base and p.name in allowed[key],'Frozen role input boundary'
        if p.is_relative_to(bases['prior'].parent) and not p.is_relative_to(root) and not any(p.is_relative_to(b) for b in bases.values()):
            raise AssertionError('Unfrozen experiment input')
    sys.addaudithook(audit)
    for n in ('ISIC','HK'):
        for suffix in ('S_old.features.private.pt','S_new.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt','teacher.private.pt'):
            p=bases['prior']/(n+'.'+suffix)
            if p.name in allowed['prior']:continue
            try:p.open('rb')
            except AssertionError:pass
            else:raise AssertionError('Access guard failed')
    for key,base in bases.items():
        rec=json.loads((base/'DELIVERY.private.json').read_text())
        assert rec['design_source_commit']==lock['inputs'][key]['source'] and rec['public_commit']==lock['inputs'][key]['delivery']
    out=dict(status='COMPLETED',source_commit=sha,role=role,dataset=name,role_guard_passed=True,
             E_access=False,GPU_used=False,encoder_updates=0,adapter_updates=0,new_labels=0,
             production_linear_solves=0,formal_head_fits=0,old_S_access=role=='fit',query_used_for_fitting=False)
    if role=='unit':
        info=unit();assert info['converged']
        assert abs(interaction(dict(R_H=.2,R_F=.3,C_H=.4,C_F=.6))-.1)<1e-12
        p=paired(torch.tensor([0,0,0,1,1]),torch.tensor([0,1,1,0,1]),torch.tensor([0,1,2,1,2]))
        assert (p['rescued'],p['harmed'],p['wrong_to_other_wrong'])==(1,1,1)
        out.update(unit_fit=info,engineering_head_fits=1,engineering_gradient_evaluations=info['closure_gradient_evaluations']+3,interaction_and_pairing_check=True)
    else:
        case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new']
        rh=load(bases['ridge']/(name+'.ridge.private.pt'));check_head(rh,seen);model=rh['model']
        if role=='fit':
            zo,yo=validate(load(bases['prior']/(name+'.S_old.features.private.pt')),lock['old_counts'][name],0,model)
            zn,yn=validate(load(bases['prior']/(name+'.S_new.features.private.pt')),lock['new_counts'][name],old,model)
            z=torch.cat((torch.from_numpy(zo),torch.from_numpy(zn)));y=torch.cat((torch.from_numpy(yo),torch.from_numpy(yn)))
            counts=lock['old_counts'][name]+lock['new_counts'][name]
            mass=1/(seen*torch.tensor(counts,dtype=DT)[y])
            assert abs(float(mass.sum())-1)<1e-12
            for c in range(seen):assert abs(float(mass[y==c].sum())-1/seen)<1e-12
            w,info=fit(z,y,mass,rh['frame'],penalty=.001)
            path=root/(name+'.ce.private.pt');assert not path.exists();torch.save(dict(model=model,frame=w),path)
            out.update(sealed_at=time.time(),fit=info,formal_head_fits=1,formal_gradient_evaluations=info['closure_gradient_evaluations']+1,
                       head_bytes=w.numel()*w.element_size(),normalized_feature_samples=len(y),support_counts=counts,initial_head='R_H',
                       old_cache_file_bytes=(bases['prior']/(name+'.S_old.features.private.pt')).stat().st_size,
                       head_stats=[dict(class_index=c,head_norm=float(w[:,c].norm()),change_from_R_H=float((w[:,c]-rh['frame'][:,c]).norm())) for c in range(seen)])
        elif role=='evaluate':
            objects=dict(R_H=rh,R_F=load(bases['offline']/(name+'.B.private.pt')),C_H=load(bases['ce']/(name+'.ce.private.pt')),C_F=load(root/(name+'.ce.private.pt')))
            for obj in objects.values():
                check_head(obj,seen);assert all(torch.equal(a,b) for a,b in zip(obj['model'],model))
            refs={key:json.loads((bases[key]/(name+suffix)).read_text()) for key,suffix in [('offline','_evaluate.json'),('ce','_ce.json')]}
            for key,rec in refs.items():assert rec['source_commit']==lock['inputs'][key]['source']
            classes=[];summaries=[];pairs=[];group_pairs=[];confusion=[];gates=[]
            for query in ('Qfit','Qgate'):
                z,y=validate(load(bases['prior']/(name+'.'+query+'.features.private.pt')),lock['query_counts'][name][query],0,model)
                z=torch.from_numpy(z);y=torch.from_numpy(y);predictions={}
                for key,obj in objects.items():
                    score=BETA*z@obj['frame'];assert bool(torch.isfinite(score).all())
                    pred=score.argmax(1);predictions[key]=pred;ce=-score.log_softmax(1)[torch.arange(len(y)),y]
                    cm=torch.bincount(y*seen+pred,minlength=seen*seen).reshape(seen,seen)
                    for c in range(seen):
                        mask=y==c;pr=pred[mask]
                        row=dict(dataset=name,query=query,model=key,class_index=c,group='old' if c<old else 'new',count=int(mask.sum()),correct=int((pr==c).sum()),recall=float((pr==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum()))
                        classes.append(row);assert int(cm[c].sum())==row['count'] and int(cm[c,c])==row['correct']
                        if key!='C_F':
                            refkey,refmodel=('ce','linear_ce') if key=='C_H' else ('offline','A' if key=='R_H' else 'B')
                            ref=next(v for v in refs[refkey]['class_rows'] if v['query']==query and v['model']==refmodel and v['class_index']==c)
                            assert all(row[k]==ref[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-ref['CE'])<1e-10
                        for j in range(seen):confusion.append(dict(dataset=name,query=query,model=key,true_class=c,predicted_class=j,count=int(cm[c,j])))
                    for group in ('all','old','new'):
                        rr=[v for v in classes if v['query']==query and v['model']==key and (group=='all' or v['group']==group)]
                        summaries.append(dict(dataset=name,query=query,model=key,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
                get=lambda key:{r['group']:r['BA'] for r in summaries if r['query']==query and r['model']==key}
                for a,b in COMPARISONS:
                    comparison=a+'__to__'+b
                    for c in range(seen):
                        mask=y==c;p=paired(predictions[a][mask],predictions[b][mask],y[mask])
                        pairs.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,group='old' if c<old else 'new',**p))
                    for group in ('all','old','new'):
                        mask=torch.ones_like(y,dtype=torch.bool) if group=='all' else y<old if group=='old' else y>=old
                        group_pairs.append(dict(dataset=name,query=query,comparison=comparison,group=group,**paired(predictions[a][mask],predictions[b][mask],y[mask])))
                    gate=select(get(a),get(b));gates.append(dict(dataset=name,query=query,comparison=comparison,pass_gate=gate['selected']=='candidate',**gate))
            out.update(class_rows=classes,summaries=summaries,paired_rows=pairs,paired_group_rows=group_pairs,confusion_rows=confusion,gates=gates,
                       published_three_cells_matched=True,normalized_feature_samples=sum(sum(v) for v in lock['query_counts'][name].values()),
                       model_logit_sample_evaluations=4*sum(sum(v) for v in lock['query_counts'][name].values()))
        else:raise ValueError('Unknown role')
    out['operation_seconds']=time.monotonic()-start;save(root/(job['id']+'.json'),out)


if __name__=='__main__':main()
