"""New offline information intervention; historical S is builder-only input."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import prototype_analytic as analytic
from check_reserved_competition_memory import BETA, DT, features
from query_node_weights import select
from run_affine_pair_diagnostic import paired
from run_real_query_node import load, save


def unit():
    x=np.array([[1.,0.,0.],[0.,1.,0.],[.6,.8,0.],[0.,0.,1.],[.8,0.,.6]])
    y=np.array([0,0,1,2,2]);old=analytic.append(analytic.empty(3),x[:3],y[:3],[0,1])
    sequential=analytic.append(old,x[3:],y[3:],[2]);direct=analytic.append(analytic.empty(3),x,y,[0,1,2])
    weights=np.array([1/6,1/6,1/3,1/6,1/6]);rhs=x.T@(weights[:,None]*np.eye(3)[y])
    assert np.allclose(sequential['S']/3,x.T@(weights[:,None]*x),atol=1e-14,rtol=0)
    assert np.allclose(sequential['mu'].T/3,rhs,atol=1e-14,rtol=0)
    a,_=analytic.ridge(sequential);b,_=analytic.ridge(direct)
    assert np.allclose(a,b,atol=1e-12,rtol=0)
    duplicated=analytic.append(analytic.empty(3),x[[0,1,0,1,2,3,4]],y[[0,1,0,1,2,3,4]],[0,1,2])
    c,_=analytic.ridge(duplicated);assert np.allclose(a,c,atol=1e-12,rtol=0)
    assert np.linalg.norm((x.T@(weights[:,None]*x)+.001*np.eye(3))@a-rhs)<1e-12
    assert not np.allclose(old['S']/2+(sequential['S']-old['S']),sequential['S']/3)
    return dict(status='PASS',engineering_small_linear_solves=3,streaming_equals_direct=True,
                explicit_class_balanced_normal_equation=True,class_duplication_invariant=True)


def validate(data,counts,start,model):
    x,y=data['x'],data['y'];assert x.shape==(sum(counts),1536) and x.dtype==DT and y.dtype==torch.int64
    assert y.shape==(len(x),) and bool(torch.isfinite(x).all())
    assert bool(((y>=start)&(y<start+len(counts))).all())
    assert [int((y==c).sum()) for c in range(start,start+len(counts))]==counts
    with torch.no_grad():z=torch.cat([features(x[i:i+256],model) for i in range(0,len(x),256)])
    return z.numpy(),y.numpy()


def bank_load(path,old):
    obj=load(path);bank={k:obj[k].numpy() for k in ('S','mu','n')}
    assert bank['S'].shape==(1536,1536) and bank['mu'].shape==(old,1536) and bank['n'].shape==(old,)
    assert bank['S'].dtype==np.float64 and bank['mu'].dtype==np.float64 and bank['n'].dtype==np.int64
    assert all(np.isfinite(v).all() for v in bank.values()) and (bank['n']>0).all()
    return bank,obj['provenance']


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    job=json.loads(os.environ['Q356_JOB']);role=job['role'];name=job.get('dataset');sha=os.environ['Q356_SHA']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==.001 and lock['logit_scale']==BETA
    torch.set_num_threads(2);torch.set_default_dtype(DT)
    prior=Path(cfg['prior_root']);ridge=Path(cfg['ridge_root'])
    prior_names={'DELIVERY.private.json'};ridge_names={'DELIVERY.private.json'}
    if name:
        ridge_names.add(name+'.ridge.private.pt')
        if role=='build':prior_names.add(name+'.S_old.features.private.pt')
        if role=='fit':prior_names.add(name+'.S_new.features.private.pt');ridge_names.add(name+'.herding.private.pt')
        if role=='evaluate':
            prior_names.update(name+'.'+q+'.features.private.pt' for q in ('Qfit','Qgate'))
            ridge_names.add(name+'_ridge.json')
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(prior):assert p.parent==prior and p.name in prior_names,'Historical input role boundary'
        if p.is_relative_to(ridge):assert p.parent==ridge and p.name in ridge_names,'Reference role boundary'
        if p.is_relative_to(prior.parent) and not p.is_relative_to(root) and not p.is_relative_to(prior) and not p.is_relative_to(ridge):
            raise AssertionError('Unfrozen experiment input')
    sys.addaudithook(audit)
    blocked=[prior/'ISIC.E.features.private.pt',prior/'HK.E.features.private.pt']
    if role!='build':blocked += [prior/(n+'.S_old.features.private.pt') for n in ('ISIC','HK')]
    if role!='evaluate':blocked += [prior/(n+'.'+q+'.features.private.pt') for n in ('ISIC','HK') for q in ('Qfit','Qgate')]
    for p in blocked:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Input guard failed')
    for key,base in [('prior',prior),('ridge',ridge)]:
        receipt=json.loads((base/'DELIVERY.private.json').read_text())
        assert receipt['design_source_commit']==lock['inputs'][key]['source'] and receipt['public_commit']==lock['inputs'][key]['delivery']
    out=dict(status='COMPLETED',source_commit=sha,role=role,dataset=name,role_guard_passed=True,
             E_access=False,GPU_used=False,encoder_updates=0,adapter_updates=0,gradient_evaluations=0,
             old_S_access=role=='build',new_labels=0,production_linear_solves=0)
    if role=='unit':out['unit']=unit()
    else:
        case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new']
        a=load(ridge/(name+'.ridge.private.pt'));model=a['model']
        assert len(model)==2 and all(v.shape==(1536,8) and v.dtype==DT and not bool(v.any()) for v in model)
        assert a['frame'].shape==(1536,seen) and a['frame'].dtype==DT and bool(torch.isfinite(a['frame']).all()) and 'bias' not in a
        if role=='build':
            data=load(prior/(name+'.S_old.features.private.pt'))
            z,y=validate(data,lock['old_counts'][name],0,model)
            bank=analytic.append(analytic.empty(1536),z,y,list(range(old)))
            provenance=dict(source_commit=sha,cache_source_commit=lock['inputs']['prior']['source'],dataset=name,
                            order=case['order'][:old],seed=lock['seed'],counts=bank['n'].tolist(),coordinate=lock['coordinate'],
                            weighting='sum of class-mean second moments; per-class arithmetic means',
                            newly_constructed_offline=True,preexisting_protocol_memory=False,sealed_at=time.time())
            path=root/(name+'.moments.private.pt');assert not path.exists()
            torch.save(dict(**{k:torch.from_numpy(v) for k,v in bank.items()},provenance=provenance),path)
            loaded,receipt=bank_load(path,old);assert receipt==provenance and loaded['n'].tolist()==lock['old_counts'][name]
            out.update(provenance=provenance,statistic_bytes=sum(v.nbytes for v in bank.values()),
                       statistic_file_bytes=path.stat().st_size,old_cache_file_bytes=(prior/(name+'.S_old.features.private.pt')).stat().st_size,
                       normalized_feature_samples=len(z),class_counts=bank['n'].tolist(),statistic_constructions=1)
        elif role=='fit':
            old_bank,provenance=bank_load(root/(name+'.moments.private.pt'),old)
            assert provenance['source_commit']==sha and provenance['order']==case['order'][:old]
            assert old_bank['n'].tolist()==lock['old_counts'][name]
            new=load(prior/(name+'.S_new.features.private.pt'));z,y=validate(new,lock['new_counts'][name],old,model)
            full=analytic.append(old_bank,z,y,list(range(old,seen)))
            h=load(ridge/(name+'.herding.private.pt'));hx=[];hy=[]
            assert sorted(h)==list(range(old))
            for c,(x,p) in h.items():
                assert len(x)==lock['H_counts'][name][c] and p.dtype==DT and p.shape==(len(x),)
                assert torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0)
                hx.append(x);hy.append(torch.full((len(x),),c,dtype=torch.int64))
            zh,yh=validate(dict(x=torch.cat(hx),y=torch.cat(hy)),lock['H_counts'][name],0,model)
            hb=analytic.append(analytic.empty(1536),zh,yh,list(range(old)))
            base=analytic.append(hb,z,y,list(range(old,seen)))
            rhs=base['mu'].T/seen
            residual=float(np.linalg.norm((base['S']/seen+.001*np.eye(1536))@a['frame'].numpy()-rhs)/max(np.linalg.norm(rhs),1e-15))
            assert residual<=1e-7,'A head does not match retained support'
            w,solve_residual=analytic.ridge(full,.001)
            path=root/(name+'.B.private.pt');assert not path.exists()
            torch.save(dict(model=model,frame=torch.from_numpy(w)),path)
            out.update(sealed_at=time.time(),production_linear_solves=1,solve_relative_residual=solve_residual,
                       A_normal_relative_residual=residual,full_support_counts=full['n'].tolist(),
                       normalized_feature_samples=len(z)+len(zh),old_moment_relative_change=float(np.linalg.norm(old_bank['S']-hb['S'])/np.linalg.norm(old_bank['S'])),
                       old_mean_relative_change=float(np.linalg.norm(old_bank['mu']-hb['mu'])/np.linalg.norm(old_bank['mu'])),
                       head_bytes=w.nbytes,head_stats=[dict(class_index=c,head_norm=float(np.linalg.norm(w[:,c])),change_from_A=float(np.linalg.norm(w[:,c]-a['frame'].numpy()[:,c]))) for c in range(seen)])
        elif role=='evaluate':
            for n in ('ISIC','HK'):
                receipt=json.loads((root/(n+'_fit.json')).read_text())
                assert receipt['status']=='COMPLETED' and receipt['source_commit']==sha and (root/(n+'.B.private.pt')).is_file()
            b=load(root/(name+'.B.private.pt'));assert b['frame'].shape==a['frame'].shape and all(torch.equal(v,w) for v,w in zip(b['model'],model))
            reference=json.loads((ridge/(name+'_ridge.json')).read_text());assert reference['source_commit']==lock['inputs']['ridge']['source']
            rows=[];summaries=[];pairs=[];group_pairs=[];gates={};queries=[]
            for query in ('Qfit','Qgate'):
                data=load(prior/(name+'.'+query+'.features.private.pt'))
                z,y=validate(data,lock['query_counts'][name][query],0,model);z=torch.from_numpy(z);y=torch.from_numpy(y)
                predictions={};queries.append(dict(query=query,count=len(y)))
                for label,obj in [('A',a),('B',b)]:
                    score=BETA*z@obj['frame'];assert bool(torch.isfinite(score).all())
                    pred=score.argmax(1);predictions[label]=pred;ce=-score.log_softmax(1)[torch.arange(len(y)),y]
                    for c in range(seen):
                        mask=y==c;pr=pred[mask]
                        row=dict(dataset=name,query=query,model=label,class_index=c,group='old' if c<old else 'new',count=int(mask.sum()),
                                 correct=int((pr==c).sum()),recall=float((pr==c).to(DT).mean()),CE=float(ce[mask].mean()),
                                 errors_to_old=int(((pr!=c)&(pr<old)).sum()),errors_to_new=int(((pr!=c)&(pr>=old)).sum()))
                        if label=='A':
                            ref=next(v for v in reference['class_rows'] if v['query']==query and v['model']=='raw_ridge' and v['class_index']==c)
                            assert all(row[k]==ref[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-ref['CE'])<1e-10
                        rows.append(row)
                    for group in ('all','old','new'):
                        rr=[v for v in rows if v['query']==query and v['model']==label and (group=='all' or v['group']==group)]
                        summaries.append(dict(dataset=name,query=query,model=label,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
                for c in range(seen):
                    mask=y==c;p=paired(predictions['A'][mask],predictions['B'][mask],y[mask])
                    pairs.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',**p))
                for group in ('all','old','new'):
                    mask=torch.ones_like(y,dtype=torch.bool) if group=='all' else y<old if group=='old' else y>=old
                    group_pairs.append(dict(dataset=name,query=query,group=group,**paired(predictions['A'][mask],predictions['B'][mask],y[mask])))
                get=lambda label:{v['group']:v['BA'] for v in summaries if v['query']==query and v['model']==label}
                gates[query]=select(get('A'),get('B'))
            out.update(class_rows=rows,summaries=summaries,paired_rows=pairs,group_paired_rows=group_pairs,gates=gates,
                       primary_both_queries_pass=all(v['selected']=='candidate' for v in gates.values()),
                       reference_aggregates_matched=True,query_counts=queries,normalized_feature_samples=sum(v['count'] for v in queries),
                       model_logit_sample_evaluations=2*sum(v['count'] for v in queries),query_used_for_fitting=False,
                       gate_used_for_model_selection=False)
        else:raise ValueError('Unknown role')
    out['operation_seconds']=time.monotonic()-start;save(root/(job['id']+'.json'),out)


if __name__=='__main__':main()
