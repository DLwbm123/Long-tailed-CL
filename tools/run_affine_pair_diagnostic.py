"""Paired audit of sealed affine and no-intercept ridge heads; no new fit."""
import json
import os
from pathlib import Path
import sys
import time
import torch
from check_reserved_competition_memory import BETA,DT,features
from run_real_query_node import load,save


def reconstruct(w0,w1,b):
    assert bool(torch.isfinite(b).all()) and float(b.norm())>1e-12
    delta=w1-w0;g=-(delta@b)/b.square().sum()
    residual=float((delta+g[:,None]*b).norm()/delta.norm().clamp_min(1e-15))
    assert bool(torch.isfinite(g).all()) and residual<=1e-7
    return g,residual


def paired(a,b,y):
    ca=a==y;cb=b==y;changed=a!=b
    r=dict(count=len(y),rescued=int((~ca&cb).sum()),harmed=int((ca&~cb).sum()),both_correct=int((ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),changed=int(changed.sum()),wrong_to_other_wrong=int((~ca&~cb&changed).sum()))
    assert r['rescued']+r['harmed']+r['both_correct']+r['both_wrong']==r['count']
    assert r['changed']==r['rescued']+r['harmed']+r['wrong_to_other_wrong']
    assert r['rescued']-r['harmed']==int(cb.sum())-int(ca.sum())
    return r


def unit():
    w0=torch.eye(2,dtype=DT);g=torch.tensor([.25,-.5],dtype=DT);b=torch.tensor([.4,-.2],dtype=DT);w1=w0-g[:,None]*b
    gg,_=reconstruct(w0,w1,b);assert torch.allclose(gg,g,atol=1e-12,rtol=0)
    z=torch.tensor([[4.,0.],[0.,2.]],dtype=DT);delta=z@w1+b-z@w0
    assert torch.allclose(delta,(1-z@g)[:,None]*b,atol=1e-12,rtol=0) and torch.allclose(delta[0],torch.zeros(2,dtype=DT),atol=1e-12,rtol=0)
    try:reconstruct(w0,w1,torch.zeros(2,dtype=DT))
    except AssertionError:pass
    else:raise AssertionError('Zero bias accepted')
    r=paired(torch.tensor([0,0,0,1,1]),torch.tensor([0,1,1,0,1]),torch.tensor([0,1,2,1,2]))
    assert r==dict(count=5,rescued=1,harmed=1,both_correct=1,both_wrong=2,changed=3,wrong_to_other_wrong=1)


def main():
    start=time.monotonic();root=Path(os.environ['Q356_ROOT']);source=Path(os.environ['Q356_SOURCE'])
    cfg=json.loads((root/'CONFIG.private.json').read_text());lock=json.loads((source/os.environ['Q356_PROTOCOL']).read_text());job=json.loads(os.environ['Q356_JOB']);name=job['dataset']
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and lock['logit_scale']==BETA==8
    torch.set_num_threads(2);torch.set_default_dtype(DT);unit()
    case=next(c for c in lock['datasets'] if c['name']==name);old=case['old'];seen=old+case['new'];sealed=False
    prior=Path(cfg['prior_root']);bases={k:Path(cfg[k+'_root']) for k in ('ridge','affine','intercept','centered')}
    files=dict(ridge=name+'.ridge.private.pt',affine=name+'.ridge.private.pt',intercept=name+'.intercept.private.pt',centered=name+'.centered.private.pt');refname=name+'_affine.json'
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(prior):assert sealed and p.name in {name+'.Qfit.features.private.pt',name+'.Qgate.features.private.pt'},'Query/support boundary'
        for key,base in bases.items():
            if p.is_relative_to(base):assert p.name in {files[key],'DELIVERY.private.json'} or (key=='affine' and sealed and p.name==refname),'Sealed artifact boundary'
    sys.addaudithook(audit)
    for p in [prior/(name+'.'+s) for s in ('S_old.features.private.pt','S_new.features.private.pt','teacher.private.pt','Qfit.features.private.pt','Qgate.features.private.pt','E.features.private.pt')]+[bases['affine']/refname]:
        try:p.open('rb')
        except AssertionError:pass
        else:raise AssertionError('Access guard failed')
    objects={}
    for key,base in bases.items():
        receipt=json.loads((base/'DELIVERY.private.json').read_text());assert receipt['design_source_commit']==lock['inputs'][key]['source'] and receipt['public_commit']==lock['inputs'][key]['delivery']
        objects[key]=load(base/files[key]);obj=objects[key]
        assert obj['frame'].shape==(1536,seen) and obj['frame'].dtype==DT and bool(torch.isfinite(obj['frame']).all())
        assert len(obj['model'])==2 and all(v.ndim==2 and v.shape[0]==1536 and v.dtype==DT and bool(torch.isfinite(v).all()) and not bool(v.any()) for v in obj['model'])
    raw=objects['ridge'];affine=objects['affine'];conditional=objects['intercept'];center=objects['centered']['center']
    assert 'bias' not in raw and torch.equal(conditional['frame'],raw['frame'])
    assert all(all(torch.equal(a,b) for a,b in zip(obj['model'],raw['model'])) for obj in objects.values())
    assert center.shape==(1536,) and center.dtype==DT and bool(torch.isfinite(center).all())
    for obj in (affine,conditional):assert obj['bias'].shape==(seen,) and obj['bias'].dtype==DT and bool(torch.isfinite(obj['bias']).all())
    b=affine['bias']/BETA;bc=conditional['bias']/BETA;g,rank_res=reconstruct(raw['frame'],affine['frame'],b)
    denominator=float(1-center@g);assert denominator>1e-12
    bias_res=float((denominator*b-bc).norm()/bc.norm().clamp_min(1e-15));assert bias_res<=1e-7
    geometry=dict(rank_one_relative_residual=rank_res,bias_identity_relative_residual=bias_res,one_minus_center_dot_g=denominator,amplification=1/denominator,g_norm=float(g.norm()),affine_bias_norm=float(b.norm()),conditional_bias_norm=float(bc.norm()))
    save(root/(job['id']+'.progress.json'),dict(phase='sealed_algebra_audit',geometry=geometry));sealed=True;sealed_at=time.time()
    ref=json.loads((bases['affine']/refname).read_text());classes=[];summaries=[];pairs=[];groups=[];switches=[];coefficients=[];counts=[];score_errors=[]
    for query in ('Qfit','Qgate'):
        data=load(prior/(name+'.'+query+'.features.private.pt'));y=data['y'];x=data['x']
        assert x.shape==(len(y),1536) and x.dtype==DT and bool(torch.isfinite(x).all()) and y.dtype==torch.int64 and bool(((y>=0)&(y<seen)).all())
        with torch.no_grad():
            z=torch.cat([features(x[i:i+256],raw['model']) for i in range(0,len(y),256)]);a=BETA*z@raw['frame'];bb=BETA*z@affine['frame']+affine['bias'];alpha=1-z@g
            error=float((bb-a-BETA*alpha[:,None]*b).abs().max());assert error<=1e-9
            assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(bb).all()) and bool(torch.isfinite(alpha).all())
            pred0=a.argmax(1);pred1=bb.argmax(1);changed=pred0!=pred1;perturb=(bb-a).abs().max(1).values
        counts.append(dict(query=query,count=len(y)));score_errors.append(dict(query=query,score_identity_max_abs_error=error))
        for model,ss,pr in (('raw_ridge',a,pred0),('raw_affine_ridge',bb,pred1)):
            ce=-ss.log_softmax(1)[torch.arange(len(y)),y]
            for c in range(seen):
                mask=y==c;n=int(mask.sum());assert n>0;p=pr[mask]
                row=dict(dataset=name,query=query,model=model,class_index=c,group='old' if c<old else 'new',count=n,correct=int((p==c).sum()),recall=float((p==c).to(DT).mean()),CE=float(ce[mask].mean()),errors_to_old=int(((p!=c)&(p<old)).sum()),errors_to_new=int(((p!=c)&(p>=old)).sum()));classes.append(row)
                rr=next(v for v in ref['class_rows'] if v['model']==model and v['query']==query and v['class_index']==c)
                assert all(rr[k]==row[k] for k in ('count','correct','errors_to_old','errors_to_new')) and abs(rr['CE']-row['CE'])<1e-10,'Published aggregate mismatch'
            for group in ('all','old','new'):
                rr=[v for v in classes if v['query']==query and v['model']==model and (group=='all' or v['group']==group)]
                summaries.append(dict(dataset=name,query=query,model=model,group=group,BA=sum(v['recall'] for v in rr)/len(rr),CE=sum(v['CE'] for v in rr)/len(rr)))
        for c in range(seen):
            m=y==c;v=alpha[m];p=perturb[m]
            pairs.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',**paired(pred0[m],pred1[m],y[m])))
            coefficients.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',count=len(v),alpha_min=float(v.min()),alpha_median=float(torch.quantile(v,.5)),alpha_mean=float(v.mean()),alpha_max=float(v.max()),alpha_negative_count=int((v<0).sum()),mean_abs_alpha=float(v.abs().mean()),max_abs_score_change_mean=float(p.mean()),max_abs_score_change_max=float(p.max())))
        for group in ('all','old','new'):
            m=torch.ones_like(y,dtype=torch.bool) if group=='all' else y<old if group=='old' else y>=old
            groups.append(dict(dataset=name,query=query,group=group,**paired(pred0[m],pred1[m],y[m])))
        for c,aa,bb in sorted(set(zip(y[changed].tolist(),pred0[changed].tolist(),pred1[changed].tolist()))):
            switches.append(dict(dataset=name,query=query,class_index=c,group='old' if c<old else 'new',from_prediction=aa,to_prediction=bb,from_group='old' if aa<old else 'new',to_group='old' if bb<old else 'new',outcome='harmed' if aa==c else 'rescued' if bb==c else 'wrong_to_other_wrong',count=int(((y==c)&(pred0==aa)&(pred1==bb)).sum())))
        assert sum(v['count'] for v in switches if v['query']==query)==int(changed.sum())
    n=sum(v['count'] for v in counts)
    save(root/(job['id']+'.json'),dict(status='COMPLETED',source_commit=os.environ['Q356_SHA'],dataset=name,sealed_at=sealed_at,geometry=geometry,score_identity_errors=score_errors,class_rows=classes,summaries=summaries,paired_rows=pairs,paired_group_rows=groups,switch_rows=switches,coefficient_rows=coefficients,query_counts=counts,published_aggregates_matched=True,engineering_checks_passed=True,shared_feature_sample_forwards=n,model_logit_sample_evaluations=2*n,decision_sample_evaluations=2*n,diagnostic_score_difference_sample_evaluations=n,head_fits=0,head_constructions=0,bias_fits=0,center_fits=0,production_linear_solves=0,engineering_linear_solves=0,gradient_evaluations=0,adapter_updates=0,teacher_updates=0,encoder_updates=0,memory_updates=0,support_access=False,E_access=False,new_labels=0,GPU_used=False,operation_seconds=time.monotonic()-start,scope='Fixed paired predictions and algebraic reconstruction from sealed heads. Reconstructed g, not an independently solved inverse direction. No new candidate, tuning, deployment selection, population causal or independent validation claim.'))


if __name__=='__main__':main()
