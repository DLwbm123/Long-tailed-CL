"""Matched centroid/ridge stress test on two fixed thinnings of legal memory."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import prototype_analytic as analytic
from check_reserved_competition_memory import BETA, DT, features
from run_affine_pair_diagnostic import paired
from run_head_competition_diagnostic import competition, stats
from run_raw_ridge_head import REGULARIZATION, unit as ridge_unit
from run_real_query_node import load, save


def halves(n):
    assert n > 0
    a = np.arange(0, n, 2); b = np.arange(1, n, 2) if n > 1 else a.copy()
    assert len(a) and len(b)
    if n > 1: assert not np.intersect1d(a, b).size and np.array_equal(np.sort(np.r_[a, b]), np.arange(n))
    return a, b


def unit():
    for n, expected in [(1, ([0], [0])), (2, ([0], [1])), (5, ([0, 2, 4], [1, 3])), (16, (list(range(0,16,2)), list(range(1,16,2))))]:
        assert tuple(v.tolist() for v in halves(n)) == expected
    ridge_unit()


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['regularization']==REGULARIZATION and lock['logit_scale']==BETA
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(v for v in lock['datasets'] if v['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    prior=Path(cfg['prior_root']);cr=Path(cfg['centroid_root']);rr=Path(cfg['ridge_root']);ar=Path(cfg['affine_root'])
    allowed={cr:{'DELIVERY.private.json',name+'.centroid.private.pt',name+'.herding.private.pt'},rr:{'DELIVERY.private.json',name+'.ridge.private.pt'}}
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert p.name==name+'.S_new.features.private.pt' or (sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'}),'Fit/query boundary'
        for base,names in allowed.items():
            if p.is_relative_to(base):assert p.name in names,'Sealed source boundary'
        if p.is_relative_to(ar):assert p.name=='DELIVERY.private.json' or (sealed and p.name==name+'_affine.json'),'Reference before seal'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+s) for s in ('S_old.features.private.pt','teacher.private.pt','E.features.private.pt','Qfit.features.private.pt','Qgate.features.private.pt')]+[ar/(name+'_affine.json')]:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    for k,base in (('centroid',cr),('ridge',rr),('affine',ar)):
        rec=json.loads((base/'DELIVERY.private.json').read_text());assert rec['design_source_commit']==lock['inputs'][k]['source'] and rec['public_commit']==lock['inputs'][k]['delivery']
    centroid=load(cr/(name+'.centroid.private.pt'));ridge=load(rr/(name+'.ridge.private.pt'));bank=load(cr/(name+'.herding.private.pt'));new=load(prior/(name+'.S_new.features.private.pt'))
    for obj in (centroid,ridge):
        assert obj['frame'].shape==(1536,seen) and obj['frame'].dtype==DT and bool(torch.isfinite(obj['frame']).all()) and 'bias' not in obj
        assert len(obj['model'])==2 and all(v.ndim==2 and v.shape[0]==1536 and v.dtype==DT and bool(torch.isfinite(v).all()) and not bool(v.any()) for v in obj['model'])
    assert all(torch.equal(a,b) for a,b in zip(centroid['model'],ridge['model']))
    assert new['x'].dtype==DT and new['x'].ndim==2 and new['x'].shape[1]==1536 and bool(torch.isfinite(new['x']).all()) and new['y'].dtype==torch.int64 and bool(((new['y']>=old)&(new['y']<seen)).all())
    zz=[];counts=[];partitions={p:[] for p in ('A','B')}
    with torch.no_grad():
        for c in range(seen):
            if c<old:
                x,p=bank[c];assert p.dtype==DT and p.shape==(len(x),) and len(x)>0
                assert torch.allclose(p,torch.full_like(p,1/len(x)),atol=1e-12,rtol=0)
            else:x=new['x'][new['y']==c]
            n=len(x);assert n==lock['support_counts'][name][c] and x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all())
            z=torch.cat([features(x[i:i+256],centroid['model']) for i in range(0,n,256)]).numpy();zz.append(z)
            ia,ib=halves(n) if c<old else (np.arange(n),np.arange(n))
            partitions['A'].append(z[ia]);partitions['B'].append(z[ib]);counts.append(dict(dataset=name,class_index=c,group='old' if c<old else 'new',full=n,A=len(ia),B=len(ib),old_singleton_unperturbed=c<old and n==1))
    def moments(parts):
        return analytic.append(analytic.empty(1536),np.concatenate(parts),np.concatenate([np.full(len(z),c,dtype=np.int64) for c,z in enumerate(parts)]),list(range(seen)))
    full=moments(zz);mu=full['mu'];norm=np.linalg.norm(mu,axis=1);assert np.all(norm>1e-12)
    mean_error=float(np.max(np.abs((mu/norm[:,None]).T-centroid['frame'].numpy())))
    G=full['S']/seen;B=mu.T/seen;normal_error=float(np.linalg.norm((G+REGULARIZATION*np.eye(1536))@ridge['frame'].numpy()-B)/max(np.linalg.norm(B),1e-15))
    assert mean_error<1e-12 and normal_error<=1e-7,'Full support/reference mismatch'
    heads={'centroid_full':centroid['frame'],'ridge_full':ridge['frame']};fit_rows=[];head_rows=[]
    for part,parts in partitions.items():
        m=moments(parts);w,res=analytic.ridge(m,REGULARIZATION);norm=np.linalg.norm(m['mu'],axis=1);assert np.all(norm>1e-12)
        heads['centroid_'+part]=torch.from_numpy((m['mu']/norm[:,None]).T);heads['ridge_'+part]=torch.from_numpy(w)
        fit_rows.append(dict(dataset=name,partition=part,support_count=int(m['n'].sum()),solve_relative_residual=res,second_moment_relative_change=float(np.linalg.norm(m['S']-full['S'])/max(np.linalg.norm(full['S']),1e-15)),class_mean_relative_change=float(np.linalg.norm(m['mu']-mu)/max(np.linalg.norm(mu),1e-15))))
        for method in ('centroid','ridge'):
            h=heads[method+'_'+part];assert bool(torch.isfinite(h).all()) and h.shape==(1536,seen)
            for c in range(seen):head_rows.append(dict(dataset=name,partition=part,method=method,class_index=c,head_norm=float(h[:,c].norm()),head_change_norm=float((h[:,c]-heads[method+'_full'][:,c]).norm()),class_mean_change_norm=float(np.linalg.norm(m['mu'][c]-mu[c]))))
    torch.save(dict(model=centroid['model'],heads={k:v for k,v in heads.items() if not k.endswith('_full')}),root/(name+'.halves.private.pt'))
    sealed=True;sealed_at=time.time();save(root/(job['id']+'.progress.json'),dict(phase='four_diagnostic_heads_sealed',fits=fit_rows))
    ref=json.loads((ar/(name+'_affine.json')).read_text());classes=[];pairs=[];instability=[];margin_rows=[];competitors=[];queries=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));x=data['x'];y=data['y'];n=len(y)
        assert x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():z=torch.cat([features(x[i:i+256],centroid['model']) for i in range(0,n,256)]);scores={k:BETA*z@h for k,h in heads.items()}
        pred={k:s.argmax(1) for k,s in scores.items()};queries.append(dict(query=query,count=n))
        for model,s in scores.items():
            pr=pred[model];rv,margin,rank,ties=competition(s,y);ce=-s.log_softmax(1)[torch.arange(n),y]
            for c in range(seen):
                mask=y==c;count=int(mask.sum());assert count>0;p=pr[mask]
                row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=count,correct=int((p==c).sum()),recall=float((p==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((p!=c)&(p<old)).sum()),errors_to_new=int(((p!=c)&(p>=old)).sum()));classes.append(row)
                if model.endswith('_full'):
                    label='raw_centroid' if model.startswith('centroid') else 'raw_ridge';r=next(v for v in ref['class_rows'] if v['query']==query and v['model']==label and v['class_index']==c)
                    assert all(row[t]==r[t] for t in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-r['CE'])<1e-10
                margin_rows.append(dict(dataset=name,query=query,model=model,class_index=c,count=count,margin=stats(margin[mask]),rank=stats(rank[mask]),top2=int((rank[mask]<=2).sum()),top3=int((rank[mask]<=3).sum()),true_score_tie_samples=int((ties[mask]>0).sum())))
                for j in sorted(set(rv[mask].tolist())):
                    v=mask&(rv==j);competitors.append(dict(dataset=name,query=query,model=model,class_index=c,competitor=j,count=int(v.sum()),wrong_count=int((v&(pr!=y)).sum())))
        for method in ('centroid','ridge'):
            a,b,f=(pred[method+'_'+p] for p in ('A','B','full'))
            for c in range(seen):
                m=y==c;count=int(m.sum());instability.append(dict(dataset=name,query=query,method=method,class_index=c,group='old' if c<old else 'new',count=count,any_disagreement=int(((a!=f)|(b!=f))[m].sum()),halves_disagree=int((a!=b)[m].sum()),both_halves_harm_full_correct=int(((f==y)&(a!=y)&(b!=y))[m].sum()),both_halves_rescue_full_wrong=int(((f!=y)&(a==y)&(b==y))[m].sum())))
                for aa,bb in (('full','A'),('full','B'),('A','B')):pairs.append(dict(dataset=name,query=query,method=method,comparison=aa+'__to__'+bb,class_index=c,group='old' if c<old else 'new',**paired(pred[method+'_'+aa][m],pred[method+'_'+bb][m],y[m])))
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,partition_counts=counts,fit_rows=fit_rows,head_rows=head_rows,class_rows=classes,paired_rows=pairs,instability_rows=instability,margin_rows=margin_rows,competitor_rows=competitors,query_counts=queries,full_centroid_max_abs_reconstruction_error=mean_error,full_ridge_normal_relative_residual=normal_error,published_aggregates_matched=True,engineering_checks_passed=True,support_feature_sample_forwards=sum(v['full'] for v in counts),shared_query_feature_sample_forwards=sum(v['count'] for v in queries),model_logit_sample_evaluations=6*sum(v['count'] for v in queries),head_fits=4,centroid_head_fits=2,ridge_head_fits=2,production_linear_solves=2,engineering_small_linear_solves=2,full_reference_head_fits=0,full_support_moment_reconstructions=1,diagnostic_head_elements=4*1536*seen,diagnostic_head_bytes=4*1536*seen*8,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,old_S_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Two deterministic old-memory thinnings with unchanged new support. Diagnostic heads only, no model selection or tuning. Changes support count/composition; not an uncertainty interval, compression-loss proof or same-budget method. Singleton old class unchanged.'))


if __name__=='__main__':main()
