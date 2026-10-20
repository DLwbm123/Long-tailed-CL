"""Independent base selection, Cholesky heads and saved-query metric audit."""
import json, os, time
from pathlib import Path
import numpy as np
import torch
from run import module, write


def main(cfg):
    start=time.time();torch.set_num_threads(4)
    root=Path(cfg['output']);base=Path(cfg['remote_base'])
    data=module(base/'case_hierarchical1_20261018/data.py','audit_data')
    diag=module(base/'joint_origin1_20261019/diagnose.py','audit_diagnostics')
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    assert json.loads((root/'QUALIFICATION.json').read_text())['state']=='PASS'
    costs=json.loads((root/'costs.json').read_text())
    assert costs['total_training_encodes']==68122 and costs['validation_encodes']==300 and costs['optimizer_updates']==0
    seal=json.loads((root/'FIT_SEAL.json').read_text());sel=json.loads((root/'BASE_SELECTION.json').read_text())
    assert seal['query_opened'] is False and seal['heads']==12
    assert sel['future_class_images']==0 and sel['validation_images']==0 and sel['frozen_at']<seal['frozen_at']
    assert sel['base_classes']==cfg['order'][:10]
    stats=[torch.load(root/f'class_{c:02d}.private.pt',weights_only=True) for c in cfg['order']]
    for c,s in zip(cfg['order'],stats):
        rows=[r for r in train if r['label']==c]
        assert s['images']==len(rows) and s['groups']==len({r['group'] for r in rows})
        assert abs(float(s['gram'][-1,-1])-1)<1e-9 and torch.equal(s['target'],s['gram'][:,-1])
    assert sel['training_images']==sum(s['images'] for s in stats[:10])
    # NumPy's explicit class means/variance and lexicographic tie rule are separate from core.
    means=np.stack([s['gram'][1536:-1,-1].numpy() for s in stats[:10]])
    seconds=np.stack([s['gram'].diagonal()[1536:-1].numpy() for s in stats[:10]])
    within_each=seconds-means**2
    assert np.all(within_each>=-1e-10*np.maximum(np.abs(seconds),np.finfo(float).tiny))
    within=np.maximum(within_each,0).mean(0);between=((means-means.mean(0))**2).mean(0)
    score=np.zeros(3072);positive=within>0;score[positive]=between[positive]/within[positive]
    score[(~positive)&(between>0)]=np.inf
    selected=(np.lexsort((np.arange(3072),-score))[:768]+1536).tolist()
    assert selected==sel['raw_hidden_indices']
    ids_by_arm={'native_pre':torch.tensor([*range(768),*selected,4608]),
                'native_post':torch.tensor([*range(1536),4608])}
    cls_ids=torch.tensor([*range(768),4608]);query=torch.load(root/'query.private.pt',weights_only=True)
    assert set(query)==set(cfg['order']) and all(x.shape==(15,4609) for x in query.values())
    records=[];diagnostics=[];previous=None;solve_error=0.;df_error=0.
    for stage in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        n=sum(cfg['task_sizes'][:stage+1]);seen=cfg['order'][:n]
        labels=torch.tensor([r['label'] for r in val if r['label'] in seen])
        scores={}
        for directory in cfg['reference_runs']:
            reference=torch.load(Path(directory)/f'stage_{stage}.private.pt',weights_only=True)
            assert reference['seen']==seen and torch.equal(reference['labels'],labels)
            for arm,value in reference['scores'].items():
                if arm in scores:assert torch.allclose(scores[arm],value,atol=1e-10,rtol=0)
                scores[arm]=value
        pos={c:0 for c in seen};xs=[]
        for c in labels.tolist():xs.append(query[c][pos[c]]);pos[c]+=1
        raw=torch.stack(xs)
        cls_g=sum((s['gram'][cls_ids][:,cls_ids] for s in stats[:n]))
        cls_inv=torch.cholesky_inverse(torch.linalg.cholesky(cls_g+torch.eye(769,dtype=torch.float64)))
        target_df=769-float(cls_inv.trace())
        for arm,ids in ids_by_arm.items():
            sub=[s['gram'][ids][:,ids] for s in stats[:n]]
            base_sub=[s['gram'][ids][:,ids] for s in stats[:10]]
            ec=sum(float(g[:768,:768].trace()) for g in base_sub)/10
            ep=sum(float(g[768:1536,768:1536].trace()) for g in base_sub)/10
            scale=(ec/ep)**.5;assert abs(scale-seal['scales'][arm])<1e-9
            factors=torch.ones(1537,dtype=torch.float64);factors[:768]=2**-.5;factors[768:1536]=scale*2**-.5
            g=sum(sub)*factors[:,None]*factors[None,:]
            t=torch.stack([s[:,-1] for s in sub],1)*factors[:,None]
            head=torch.load(root/f'{arm}_heads_{stage}.private.pt',weights_only=True)
            assert torch.equal(head['indices'],ids)
            factor=torch.linalg.cholesky(g+head['alpha']*torch.eye(1537,dtype=torch.float64))
            inverse=torch.cholesky_inverse(factor)
            actual_df=1537-head['alpha']*float(inverse.trace())
            df_error=max(df_error,abs(actual_df-target_df));assert abs(actual_df-target_df)<1e-6
            reconstructed=torch.cholesky_solve(t,factor)
            assert torch.allclose(reconstructed,head['head'],atol=1e-8,rtol=1e-7)
            xx=raw[:,ids]*factors;value=xx@reconstructed
            error=float((value-xx@head['head']).abs().max());solve_error=max(solve_error,error);assert error<1e-8
            scores[arm]=value
        predictions={a:torch.tensor(seen)[s.argmax(1)] for a,s in scores.items()}
        metrics={}
        for arm,p in predictions.items():
            counts=[int((labels==c).sum()) for c in seen]
            correct=[int(((labels==c)&(p==c)).sum()) for c in seen]
            assert counts==[15]*n
            recalls=[x/y for x,y in zip(correct,counts)]
            metrics[arm]=dict(macro=sum(recalls)/n,counts=counts,correct=correct,recall=recalls)
        contrasts={}
        for arm in cfg['arms']:
            contrasts[arm]={}
            for ref in ['df_matched_joint','scale_joint','native_post']:
                if arm==ref:continue
                p,q=predictions[arm],predictions[ref]
                contrasts[arm][ref]=dict(rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
        saved=dict(scores=scores,labels=labels,seen=seen)
        diagnostics.append(diag.stage_diagnostics(saved,previous));previous=saved
        records.append(dict(stage=stage,seen=seen,metrics=metrics,contrasts=contrasts))
        torch.save(saved,root/f'stage_{stage}.private.pt')
    support=json.loads((Path(cfg['prior_run'])/'support.json').read_text());groups={}
    for criterion in ['images','patients']:
        ordered=sorted(cfg['order'],key=lambda c:(support[str(c)][criterion],c))
        bins=dict(tail7=ordered[:7],middle6=ordered[7:13],head7=ordered[13:],initial10=cfg['order'][:10],later10=cfg['order'][10:])
        groups[criterion]=dict(bins=bins,correct={a:{k:sum(m['correct'][cfg['order'].index(c)] for c in members) for k,members in bins.items()} for a,m in records[-1]['metrics'].items()})
    averages={a:sum(r['metrics'][a]['macro'] for r in records)/6 for a in records[-1]['metrics']}
    final=records[-1]['metrics'];pre=groups['images']['correct']['native_pre']
    gates=dict(final_correct=sum(final['native_pre']['correct'])>=86,mean_gain=averages['native_pre']-averages['df_matched_joint']>=.01,
               tail7=pre['tail7']>=22,later10=pre['later10']>=38,base_correct=sum(records[0]['metrics']['native_pre']['correct'])>=66,
               pre_beats_post_final=sum(final['native_pre']['correct'])>sum(final['native_post']['correct']),pre_not_worse_post_mean=averages['native_pre']>=averages['native_post'])
    resources=json.loads((root.parent/'guardian_resources.private.json').read_text())
    assert resources['peak_output_bytes']<cfg['output_budget_bytes']
    assert costs['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
    for name,value in [('results.json',records),('DIAGNOSTICS.json',diagnostics),('FREQUENCY.json',groups),('DECISION.json',dict(gates=gates,proceed=all(gates.values()),mean_stage_BA=averages,practical_not_significance=True))]:write(root/name,value)
    write(root/'VERIFICATION.json',dict(state='PASS',heads_checked=12,stages=6,new_image_encodes=0,base_selection_independently_recomputed=True,head_score_max_error=solve_error,df_max_error=df_error,wall_seconds=time.time()-start,patient_overlap=0))
    write(root/'status.private.json',dict(state='COMPLETE',updated=time.time(),proceed=all(gates.values()),delivery_pending=True))


if __name__=='__main__':main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
