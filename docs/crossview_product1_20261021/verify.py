"""Independent explicit affine maps, Cholesky solves and saved-query metrics."""
import json,os,time,traceback
from pathlib import Path
import torch
from run import module,write


def explicit_map(mc,mp,arm):
    # Separate implementation: construct raw-to-centered design matrix explicitly.
    d=len(mc);a=torch.zeros(4*d+1,3*d+1,dtype=torch.float64);i=torch.arange(d)
    a[i,i]=1;a[d+i,d+i]=1;a[-1,-1]=1
    perm=i if arm=='cross_aligned' else (i+d//2)%d
    product=2*d if arm=='cross_aligned' else 3*d
    a[product+i,2*d+i]=1
    a[i,2*d+i]=-mp[perm];a[d+perm,2*d+i]=-mc
    a[-1,2*d+i]=mc*mp[perm]
    return a


def main(cfg):
    start=time.time();torch.set_num_threads(4)
    root=Path(cfg['output']);base=Path(cfg['remote_base'])
    data=module(base/'case_hierarchical1_20261018/data.py','cross_audit_data')
    diag=module(base/'joint_origin1_20261019/diagnose.py','cross_audit_diagnostics')
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300 and not ({r['group'] for r in train}&{r['group'] for r in val})
    costs=json.loads((root/'costs.json').read_text())
    assert costs['science_training_encodes']==68058 and costs['qualification_training_encodes']==32
    assert costs['total_training_encodes']==68090 and costs['validation_encodes']==300 and costs['optimizer_updates']==0
    assert json.loads((root/'QUALIFICATION.json').read_text())['state']=='PASS'
    seal=json.loads((root/'FIT_SEAL.json').read_text());bs=json.loads((root/'BASE_SEAL.json').read_text())
    opened=json.loads((root/'QUERY_OPEN.private.json').read_text())
    assert seal['heads']==12 and seal['query_opened'] is False and bs['frozen_at']<seal['frozen_at']<opened['opened_at']
    assert bs['future_class_images']==bs['validation_images']==opened['validation_encodes']==0
    stats=[torch.load(root/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
    for c,s in zip(cfg['order'],stats):
        rows=[r for r in train if r['label']==c]
        assert s['gram'].shape==(3073,3073) and s['images']==len(rows) and s['groups']==len({r['group'] for r in rows})
        assert torch.equal(s['target'],s['gram'][:,-1]) and abs(float(s['gram'][-1,-1])-1)<1e-9
    assert bs['training_images']==sum(s['images'] for s in stats[:10])==60129
    assert bs['base_classes']==cfg['order'][:10]
    bg=sum(s['gram'] for s in stats[:10]);mc=bg[:768,-1]/10;mp=bg[768:1536,-1]/10
    assert torch.allclose(mc,torch.tensor(bs['mu_c'],dtype=torch.float64),atol=1e-12,rtol=0)
    assert torch.allclose(mp,torch.tensor(bs['mu_p'],dtype=torch.float64),atol=1e-12,rtol=0)
    maps={};factor_error=0.
    for arm in cfg['arms']:
        a=explicit_map(mc,mp,arm);sub=a.T@bg@a
        ec=float(sub[:768,:768].trace());ep=float(sub[768:1536,768:1536].trace());eh=float(sub[1536:2304,1536:2304].trace())
        assert min(ec,ep,eh)>0
        f=torch.ones(2305,dtype=torch.float64);f[:768]=2**-.5;f[768:1536]=(ec/ep)**.5*2**-.5;f[1536:2304]=(ec/eh)**.5
        delta=float((f-torch.tensor(seal['factors'][arm],dtype=torch.float64)).abs().max());factor_error=max(delta,factor_error)
        assert torch.allclose(f,torch.tensor(seal['factors'][arm],dtype=torch.float64),atol=1e-8,rtol=1e-8)
        maps[arm]=(a*f[None,:],f)
    query=torch.load(root/'query.private.pt',weights_only=True)
    assert set(query)==set(cfg['order']) and all(v.shape==(15,3073) and torch.isfinite(v).all() for v in query.values())
    assert max(json.loads((root/'VALIDATION_PARITY.json').read_text()).values())<=2e-5
    assert all(max(e.values())<=2e-5 for e in seal['training_parity'].values()) and len(seal['training_parity'])==20
    records=[];diagnostics=[];previous=None;score_error=0.;df_error=0.
    for stage in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        n=sum(cfg['task_sizes'][:stage+1]);seen=cfg['order'][:n];rg=sum(s['gram'] for s in stats[:n])
        rt=torch.stack([s['target'] for s in stats[:n]],1)
        cls_ids=torch.tensor([*range(768),3072]);cg=rg[cls_ids][:,cls_ids]
        target_df=769-float(torch.cholesky_inverse(torch.linalg.cholesky(cg+torch.eye(769,dtype=torch.float64))).trace())
        labels=torch.tensor([r['label'] for r in val if r['label'] in seen]);scores={}
        for directory in cfg['reference_runs']:
            old=torch.load(Path(directory)/f'stage_{stage}.private.pt',weights_only=True)
            assert old['seen']==seen and torch.equal(labels,old['labels'])
            for arm,value in old['scores'].items():
                if arm in scores:assert torch.allclose(scores[arm],value,atol=1e-10,rtol=0)
                scores[arm]=value
        pos={c:0 for c in seen};xs=[]
        for c in labels.tolist():xs.append(query[c][pos[c]]);pos[c]+=1
        x=torch.stack(xs)
        for arm,(a,fac) in maps.items():
            g=a.T@rg@a;t=a.T@rt
            head=torch.load(root/f'{arm}_heads_{stage}.private.pt',weights_only=True)
            assert torch.allclose(head['factors'],fac,atol=1e-8,rtol=1e-8)
            alpha=head['alpha'];ch=torch.linalg.cholesky((g+g.T)/2+alpha*torch.eye(2305,dtype=torch.float64))
            actual_df=2305-alpha*float(torch.cholesky_inverse(ch).trace());delta=abs(actual_df-target_df)
            df_error=max(df_error,delta);assert delta<1e-6
            w=torch.cholesky_solve(t,ch)
            assert torch.allclose(w,head['head'],atol=1e-8,rtol=1e-7)
            value=(x@a)@w
            err=float((value-(x@a)@head['head']).abs().max());score_error=max(score_error,err);assert err<1e-8
            scores[arm]=value
        predictions={a:torch.tensor(seen)[s.argmax(1)] for a,s in scores.items()};metrics={}
        for arm,p in predictions.items():
            counts=[int((labels==c).sum()) for c in seen];assert counts==[15]*n
            correct=[int(((labels==c)&(p==c)).sum()) for c in seen]
            metrics[arm]=dict(macro=sum(correct)/(15*n),counts=counts,correct=correct,recall=[c/15 for c in correct])
        contrasts={}
        for arm in cfg['arms']:
            contrasts[arm]={}
            for ref in ['df_matched_joint','scale_joint','cross_shifted']:
                if arm==ref:continue
                p,q=predictions[arm],predictions[ref]
                contrasts[arm][ref]=dict(rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
        saved=dict(scores=scores,labels=labels,seen=seen);torch.save(saved,root/f'stage_{stage}.private.pt')
        diagnostics.append(diag.stage_diagnostics(saved,previous));previous=saved
        records.append(dict(stage=stage,seen=seen,metrics=metrics,contrasts=contrasts))
    support=json.loads((Path(cfg['prior_run'])/'support.json').read_text());groups={}
    for criterion in ['images','patients']:
        ordered=sorted(cfg['order'],key=lambda c:(support[str(c)][criterion],c));bins=dict(tail7=ordered[:7],middle6=ordered[7:13],head7=ordered[13:],initial10=cfg['order'][:10],later10=cfg['order'][10:])
        groups[criterion]=dict(bins=bins,correct={a:{k:sum(m['correct'][cfg['order'].index(c)] for c in cs) for k,cs in bins.items()} for a,m in records[-1]['metrics'].items()})
    mean={a:sum(r['metrics'][a]['macro'] for r in records)/6 for a in records[-1]['metrics']}
    a='cross_aligned';control='cross_shifted';final=records[-1]['metrics'];fg=groups['images']['correct'][a]
    gates=dict(final_correct=sum(final[a]['correct'])>=86,mean_gain=mean[a]-mean['df_matched_joint']>=.01,tail7=fg['tail7']>=22,later10=fg['later10']>=38,base_correct=sum(records[0]['metrics'][a]['correct'])>=66,aligned_beats_shifted_final=sum(final[a]['correct'])>sum(final[control]['correct']),aligned_not_worse_shifted_mean=mean[a]>=mean[control])
    resources=json.loads((root.parent/'guardian_resources.private.json').read_text());assert resources['peak_output_bytes']<cfg['output_budget_bytes'] and costs['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
    for name,value in [('RESULTS.json',records),('DIAGNOSTICS.json',diagnostics),('FREQUENCY.json',groups),('DECISION.json',dict(gates=gates,proceed=all(gates.values()),mean_stage_BA=mean)),('VERIFICATION.json',dict(state='PASS',heads_checked=12,explicit_affine_reconstruction=True,base_means_recomputed=True,head_score_max_error=score_error,df_max_error=df_error,factor_max_error=factor_error,patient_overlap=0,new_image_encodes=0,wall_seconds=time.time()-start))]:write(root/name,value)
    write(root/'status.private.json',dict(state='COMPLETE',updated=time.time(),proceed=all(gates.values()),delivery_pending=True))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        write(Path(cfg['output'])/'VERIFY_FAILURE.private.json',dict(updated=time.time(),error=traceback.format_exc()))
        raise
