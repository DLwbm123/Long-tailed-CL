"""Score three sealed ISIC heads on support; reuse query aggregates only."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA, DT
from run_affine_pair_diagnostic import paired
from run_full_support_linear_ce import check_head
from run_head_competition_diagnostic import competition, stats, unit
from run_offline_information_ab import validate
from run_real_query_node import load, save

MODELS=('C_H','C_F','R_F')
COMPARISONS=(('C_H','C_F'),('R_F','C_F'))


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    job=json.loads(os.environ['Q356_JOB']);assert job['dataset']=='ISIC' and job['role']=='audit'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA==8
    assert list(MODELS)==lock['models'] and [list(v) for v in COMPARISONS]==lock['comparisons']
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    bases={k:Path(cfg[k+'_root']) for k in ('prior','ce','full','offline')}
    allowed=dict(prior={'ISIC.S_old.features.private.pt','ISIC.S_new.features.private.pt'},
                 ce={'ISIC.ce.private.pt'},full={'ISIC.ce.private.pt','ISIC_evaluate.json'},offline={'ISIC.B.private.pt'})
    for names in allowed.values():names.add('DELIVERY.private.json')
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        for key,base in bases.items():
            if p.is_relative_to(base):assert p.parent==base and p.name in allowed[key],'Frozen input boundary'
        if p.is_relative_to(bases['prior'].parent) and not p.is_relative_to(root) and not any(p.is_relative_to(b) for b in bases.values()):
            raise AssertionError('Unfrozen experiment input')
    sys.addaudithook(audit)
    for n in ('ISIC','HK'):
        for suffix in ('Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt','teacher.private.pt'):
            try:(bases['prior']/(n+'.'+suffix)).open('rb')
            except AssertionError:pass
            else:raise AssertionError('Query/E guard failed')
    for key,base in bases.items():
        rec=json.loads((base/'DELIVERY.private.json').read_text())
        assert rec['design_source_commit']==lock['inputs'][key]['source'] and rec['public_commit']==lock['inputs'][key]['delivery']
    heads={key:load(bases[base]/file) for key,base,file in [('C_H','ce','ISIC.ce.private.pt'),('C_F','full','ISIC.ce.private.pt'),('R_F','offline','ISIC.B.private.pt')]}
    for obj in heads.values():
        check_head(obj,6);assert all(torch.equal(a,b) for a,b in zip(obj['model'],heads['C_F']['model']))
    zo,yo=validate(load(bases['prior']/'ISIC.S_old.features.private.pt'),lock['old_counts'],0,heads['C_F']['model'])
    zn,yn=validate(load(bases['prior']/'ISIC.S_new.features.private.pt'),lock['new_counts'],4,heads['C_F']['model'])
    z=torch.cat((torch.from_numpy(zo),torch.from_numpy(zn)));y=torch.cat((torch.from_numpy(yo),torch.from_numpy(yn)))
    classes=[];summary=[];pairs=[];group_pairs=[];confusion=[];margins=[];predictions={}
    with torch.no_grad():
        for key,obj in heads.items():
            s=BETA*z@obj['frame'];assert bool(torch.isfinite(s).all());pr=s.argmax(1);predictions[key]=pr
            ce=-s.log_softmax(1)[torch.arange(len(y)),y];rival,margin,rank,ties=competition(s,y)
            cm=torch.bincount(y*6+pr,minlength=36).reshape(6,6)
            for c in range(6):
                mask=y==c;pc=pr[mask];count=int(mask.sum())
                row=dict(dataset='ISIC',query='S_full',model=key,class_index=c,group='old' if c<4 else 'new',count=count,correct=int((pc==c).sum()),recall=float((pc==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((pc!=c)&(pc<4)).sum()),errors_to_new=int(((pc!=c)&(pc>=4)).sum()))
                assert row['correct']+row['errors_to_old']+row['errors_to_new']==count and int(cm[c,c])==row['correct'];classes.append(row)
                margins.append(dict(dataset='ISIC',query='S_full',model=key,class_index=c,count=count,true_score_tie_samples=int((ties[mask]>0).sum()),margin=stats(margin[mask]),rank=stats(rank[mask])))
                for j in range(6):confusion.append(dict(dataset='ISIC',query='S_full',model=key,true_class=c,predicted_class=j,count=int(cm[c,j])))
            for group in ('all','old','new'):
                rr=[v for v in classes if v['model']==key and (group=='all' or v['group']==group)]
                summary.append(dict(dataset='ISIC',query='S_full',model=key,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
        for a,b in COMPARISONS:
            for c in range(6):
                mask=y==c;pairs.append(dict(dataset='ISIC',query='S_full',comparison=a+'__to__'+b,class_index=c,group='old' if c<4 else 'new',**paired(predictions[a][mask],predictions[b][mask],y[mask])))
            for group in ('all','old','new'):
                mask=torch.ones_like(y,dtype=torch.bool) if group=='all' else y<4 if group=='old' else y>=4
                group_pairs.append(dict(dataset='ISIC',query='S_full',comparison=a+'__to__'+b,group=group,**paired(predictions[a][mask],predictions[b][mask],y[mask])))
    # Existing query records are data, not instructions; no query features are opened.
    ref=json.loads((bases['full']/'ISIC_evaluate.json').read_text())
    assert ref['status']=='COMPLETED' and ref['source_commit']==lock['inputs']['full']['source'] and ref['published_three_cells_matched']
    for q in ('Qfit','Qgate'):
        for key in MODELS:
            rr=sorted((v for v in ref['class_rows'] if v['query']==q and v['model']==key),key=lambda v:v['class_index'])
            assert [v['class_index'] for v in rr]==list(range(6)) and [v['count'] for v in rr]==lock['query_counts'][q]
    comparisons={a+'__to__'+b for a,b in COMPARISONS}
    classes.extend(v for v in ref['class_rows'] if v['model'] in MODELS)
    summary.extend(v for v in ref['summaries'] if v['model'] in MODELS)
    confusion.extend(v for v in ref['confusion_rows'] if v['model'] in MODELS)
    pairs.extend(v for v in ref['paired_rows'] if v['comparison'] in comparisons)
    group_pairs.extend(v for v in ref['paired_group_rows'] if v['comparison'] in comparisons)
    assert len(classes)==54 and len(summary)==27 and len(pairs)==36 and len(group_pairs)==18 and len(confusion)==324
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset='ISIC',class_rows=classes,summaries=summary,
         paired_rows=pairs,paired_group_rows=group_pairs,confusion_rows=confusion,support_margin_rows=margins,
         source_query_commit=lock['inputs']['full']['source'],source_query_delivery=lock['inputs']['full']['delivery'],
         role_guard_passed=True,engineering_checks_passed=True,query_aggregates_reused=True,raw_query_access=False,E_access=False,
         head_fits=0,linear_solves=0,gradient_evaluations=0,encoder_updates=0,adapter_updates=0,new_head_bytes=0,
         support_normalized_samples=len(y),model_logit_sample_evaluations=3*len(y),query_feature_evaluations=0,GPU_used=False,
         operation_seconds=time.monotonic()-start,scope='Shared support-domain audit, not a common training set for all heads: C_H used only H plus new S. No new fit, query scoring, selection, independent validation or representation-capacity claim.'))


if __name__=='__main__':main()
