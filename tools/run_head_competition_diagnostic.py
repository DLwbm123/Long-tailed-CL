"""Fixed three-head competition diagnostics; no fitting or model selection."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA, DT, features
from run_affine_pair_diagnostic import paired, unit as paired_unit
from run_real_query_node import load, save


def competition(s, y):
    assert s.ndim == 2 and s.shape[1] > 1 and bool(torch.isfinite(s).all())
    i = torch.arange(len(y)); true = s[i, y]
    other = s.clone(); other[i, y] = -torch.inf
    rival = other.argmax(1); margin = true - other[i, rival]
    tied = s == true[:, None]; columns = torch.arange(s.shape[1])
    rank = 1 + (s > true[:, None]).sum(1) + (tied & (columns < y[:, None])).sum(1)
    assert torch.equal(rank == 1, s.argmax(1) == y)
    return rival, margin, rank, tied.sum(1) - 1


def stats(v):
    assert len(v) and bool(torch.isfinite(v).all())
    return dict(min=float(v.min()), median=float(torch.quantile(v.to(DT), .5)), mean=float(v.to(DT).mean()), max=float(v.max()))


def unit():
    paired_unit()
    s = torch.tensor([[3., 2., 1.], [3., 3., 1.], [1., 2., 4.]], dtype=DT)
    y = torch.tensor([0, 1, 0]); rival, margin, rank, ties = competition(s, y)
    assert rival.tolist() == [1, 0, 2] and margin.tolist() == [1., 0., -3.]
    assert rank.tolist() == [1, 2, 3] and ties.tolist() == [0, 1, 0]
    shifted = s + torch.tensor([2., -3., 8.], dtype=DT)[:, None]
    assert all(torch.equal(a, b) for a, b in zip(competition(s, y), competition(shifted, y)))
    ss = stats(torch.tensor([1., 3.], dtype=DT)); assert ss == dict(min=1., median=2., mean=2., max=3.)


def main():
    start = time.monotonic(); root = Path(os.environ['Q356_ROOT']); source = Path(os.environ['Q356_SOURCE'])
    cfg = json.loads((root/'CONFIG.private.json').read_text()); lock = json.loads((source/os.environ['Q356_PROTOCOL']).read_text())
    job = json.loads(os.environ['Q356_JOB']); name = job['dataset']; case = next(v for v in lock['datasets'] if v['name'] == name)
    old = case['old']; seen = old + case['new']; prior = Path(cfg['prior_root']); sealed = False
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and lock['logit_scale'] == BETA == 8
    torch.set_num_threads(2); torch.set_default_dtype(DT); unit()
    bases = {k: Path(cfg[k+'_root']) for k in ('centroid', 'ridge', 'affine')}
    files = {k: name + ('.centroid.private.pt' if k == 'centroid' else '.ridge.private.pt') for k in bases}
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior): assert sealed and p.name in {name+'.Qfit.features.private.pt', name+'.Qgate.features.private.pt'}, 'Query/support boundary'
        for k, base in bases.items():
            if p.is_relative_to(base): assert p.name in {files[k], 'DELIVERY.private.json'} or (k == 'affine' and sealed and p.name == name+'_affine.json'), 'Fixed artifact boundary'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+s) for s in ('S_old.features.private.pt', 'S_new.features.private.pt', 'teacher.private.pt', 'E.features.private.pt', 'Qfit.features.private.pt', 'Qgate.features.private.pt')] + [bases['affine']/(name+'_affine.json')]:
        try: p.open('rb')
        except AssertionError: pass
        else: raise AssertionError('Access guard failed')
    objects = {}
    for k, base in bases.items():
        receipt = json.loads((base/'DELIVERY.private.json').read_text())
        assert receipt['design_source_commit'] == lock['inputs'][k]['source'] and receipt['public_commit'] == lock['inputs'][k]['delivery']
        obj = load(base/files[k]); objects[k] = obj
        assert obj['frame'].shape == (1536, seen) and obj['frame'].dtype == DT and bool(torch.isfinite(obj['frame']).all())
        assert len(obj['model']) == 2 and all(v.ndim == 2 and v.shape[0] == 1536 and v.dtype == DT and bool(torch.isfinite(v).all()) and not bool(v.any()) for v in obj['model'])
        if k == 'affine': assert obj['bias'].shape == (seen,) and obj['bias'].dtype == DT and bool(torch.isfinite(obj['bias']).all())
        else: assert 'bias' not in obj
    assert all(all(torch.equal(a,b) for a,b in zip(o['model'],objects['ridge']['model'])) for o in objects.values())
    assert torch.allclose(objects['centroid']['frame'].norm(dim=0), torch.ones(seen,dtype=DT), atol=1e-12, rtol=0)
    sealed = True; sealed_at = time.time(); save(root/(job['id']+'.progress.json'), dict(phase='fixed_heads_verified'))
    ref = json.loads((bases['affine']/(name+'_affine.json')).read_text())
    labels = dict(centroid='raw_centroid', ridge='raw_ridge', affine='raw_affine_ridge')
    classes=[]; margins=[]; competitors=[]; pairs=[]; switches=[]; effects=[]; queries=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));x=data['x'];y=data['y'];n=len(y);i=torch.arange(n)
        assert x.shape==(n,1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():
            z=torch.cat([features(x[j:j+256],objects['ridge']['model']) for j in range(0,n,256)])
            scores={k:BETA*z@o['frame']+o.get('bias',0) for k,o in objects.items()}
            predictions={k:s.argmax(1) for k,s in scores.items()}; comps={k:competition(s,y) for k,s in scores.items()}
        queries.append(dict(query=query,count=n))
        for k,s in scores.items():
            pred=predictions[k];rival,margin,rank,ties=comps[k];ce=-s.log_softmax(1)[i,y]
            for c in range(seen):
                m=y==c;count=int(m.sum());assert count>0;p=pred[m]
                row=dict(dataset=name,query=query,model=labels[k],class_index=c,group='old' if c<old else 'new',count=count,correct=int((p==c).sum()),recall=float((p==c).to(DT).mean()),CE=float(ce[m].mean()),errors_to_old=int(((p!=c)&(p<old)).sum()),errors_to_new=int(((p!=c)&(p>=old)).sum()));classes.append(row)
                previous=next(v for v in ref['class_rows'] if v['query']==query and v['model']==labels[k] and v['class_index']==c)
                assert all(row[t]==previous[t] for t in ('count','correct','errors_to_old','errors_to_new')) and abs(row['CE']-previous['CE'])<1e-10,'Published aggregate mismatch'
                margins.append(dict(dataset=name,query=query,model=labels[k],class_index=c,group=row['group'],count=count,margin=stats(margin[m]),rank=stats(rank[m]),top1=int((rank[m]<=1).sum()),top2=int((rank[m]<=2).sum()),top3=int((rank[m]<=3).sum()),true_score_tie_samples=int((ties[m]>0).sum()),zero_margin_samples=int((margin[m]==0).sum())))
                for j in sorted(set(rival[m].tolist())):
                    mask=m&(rival==j);competitors.append(dict(dataset=name,query=query,model=labels[k],class_index=c,competitor=j,competitor_group='old' if j<old else 'new',count=int(mask.sum()),wrong_count=int((mask&(pred!=y)).sum())))
        for a,b in (('centroid','ridge'),('centroid','affine')):
            pa=predictions[a];pb=predictions[b];ca=pa==y;cb=pb==y;changed=pa!=pb;comparison=labels[a]+'__to__'+labels[b]
            rival0=comps[a][0];fixed_delta=(scores[b][i,y]-scores[b][i,rival0])-(scores[a][i,y]-scores[a][i,rival0])
            masks=dict(rescued=~ca&cb,harmed=ca&~cb,both_correct=ca&cb,both_wrong=~ca&~cb)
            for c in range(seen):
                m=y==c;pairs.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,group='old' if c<old else 'new',**paired(pa[m],pb[m],y[m])))
                for outcome,mask in masks.items():
                    v=m&mask
                    if not bool(v.any()):continue
                    effects.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,outcome=outcome,count=int(v.sum()),from_margin=stats(comps[a][1][v]),to_margin=stats(comps[b][1][v]),fixed_baseline_competitor_margin_delta=stats(fixed_delta[v]),competitor_changed=int((comps[a][0][v]!=comps[b][0][v]).sum())))
            for c,aa,bb in sorted(set(zip(y[changed].tolist(),pa[changed].tolist(),pb[changed].tolist()))):
                switches.append(dict(dataset=name,query=query,comparison=comparison,class_index=c,from_prediction=aa,to_prediction=bb,from_group='old' if aa<old else 'new',to_group='old' if bb<old else 'new',outcome='harmed' if aa==c else 'rescued' if bb==c else 'wrong_to_other_wrong',count=int(((y==c)&(pa==aa)&(pb==bb)).sum())))
            assert sum(v['count'] for v in switches if v['query']==query and v['comparison']==comparison)==int(changed.sum())
    n=sum(v['count'] for v in queries)
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,class_rows=classes,margin_rows=margins,competitor_rows=competitors,paired_rows=pairs,switch_rows=switches,effect_rows=effects,query_counts=queries,published_aggregates_matched=True,engineering_checks_passed=True,shared_feature_sample_forwards=n,model_logit_sample_evaluations=3*n,decision_sample_evaluations=3*n,head_fits=0,head_constructions=0,production_linear_solves=0,engineering_linear_solves=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,support_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Development diagnostic of fixed head competition. All classes reported; highlighted classes were chosen after prior results. Margin scales differ across heads; no causal, reliability or independent-generalization claim. No new candidate, fit or selection.'))


if __name__=='__main__':main()
