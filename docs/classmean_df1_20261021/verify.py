"""Independent Cholesky/mean-risk/df audit; scoring only after fit seal."""
import json, os, time
from pathlib import Path
import torch
from run import statistics, module, write


def main(cfg):
    start=time.time();torch.set_num_threads(4)
    root=Path(cfg['output']);seal=json.loads((root/'FIT_SEAL.json').read_text())
    assert seal['heads']==12 and seal['query_opened'] is False
    assert json.loads((root/'QUALIFICATION.json').read_text())['state']=='PASS'
    cls,joint,targets,coverage,scale=statistics(cfg)
    assert coverage==seal['coverage'] and abs(scale-seal['patch_scale'])<1e-12
    data=module(Path(cfg['remote_base'])/'case_hierarchical1_20261018/data.py','metadata_audit')
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    for item in coverage:
        rows=[r for r in train if r['label']==item['class_id']]
        assert len(rows)==item['images'] and len({r['group'] for r in rows})==item['groups']
    query=torch.load(Path(cfg['joint_run'])/'query768.private.pt',weights_only=True)
    records=[];errors=[];base_errors=[]
    for stage in range(6):
        assert time.time()<cfg['hard_deadline_epoch']-60
        n=sum(cfg['task_sizes'][:stage+1]);seen=cfg['order'][:n]
        cg=sum(cls[:n]);ct=torch.stack([g[:,-1] for g in cls[:n]],1)
        ic=torch.eye(769,dtype=torch.float64);alpha_c=n/10
        # Solve the normalized risk equation directly, rather than reusing worker solve.
        cc=torch.linalg.cholesky(cg/n+.1*ic)
        wc=torch.cholesky_solve(ct/n,cc)
        target_df=769-.1*float(torch.cholesky_inverse(cc).trace())
        factors=torch.ones(1537,dtype=torch.float64);factors[:768]=2**-.5;factors[768:1536]=scale*2**-.5
        gj=sum(joint[:n])*factors[:,None]*factors[None,:]
        tj=torch.stack(targets[:n],1)*factors[:,None]
        saved=torch.load(root/f'heads_{stage}.private.pt',weights_only=True)
        alpha_j=saved['joint_classmean_df']['alpha'];ij=torch.eye(1537,dtype=torch.float64)
        cj=torch.linalg.cholesky(gj+alpha_j*ij)
        wj=torch.cholesky_solve(tj,cj)
        actual_df=1537-alpha_j*float(torch.cholesky_inverse(cj).trace())
        assert abs(actual_df-target_df)<1e-6
        assert saved['cls_classmean']['alpha']==alpha_c
        assert torch.equal(factors,saved['joint_classmean_df']['factors'])
        for arm,w in [('cls_classmean',wc),('joint_classmean_df',wj)]:
            assert torch.allclose(w,saved[arm]['head'],atol=1e-8,rtol=1e-7)
        labels=torch.tensor([r['label'] for r in val if r['label'] in seen]);scores={}
        for directory in cfg['reference_runs']:
            old=torch.load(Path(directory)/f'stage_{stage}.private.pt',weights_only=True)
            assert old['seen']==seen and torch.equal(old['labels'],labels)
            for arm,value in old['scores'].items():
                if arm in scores:assert torch.allclose(scores[arm],value,atol=1e-10,rtol=0)
                scores[arm]=value
        pos={c:0 for c in seen};xs=[]
        for c in labels.tolist():
            i=pos[c];pos[c]+=1;xs.append(torch.cat([query[c][0][i],query[c][1][i],torch.ones(1,dtype=torch.float64)]))
        x=torch.stack(xs);xc=torch.cat([x[:,:768],x[:,-1:]],1);xj=x*factors
        scores['cls_classmean']=xc@wc;scores['joint_classmean_df']=xj@wj
        error=max(float((scores['cls_classmean']-xc@saved['cls_classmean']['head']).abs().max()),float((scores['joint_classmean_df']-xj@saved['joint_classmean_df']['head']).abs().max()))
        assert error<1e-8;errors.append(error)
        if stage==0:
            for a,b in [('cls_classmean','cls768_ridge'),('joint_classmean_df','df_matched_joint')]:
                error=float((scores[a]-scores[b]).abs().max());base_errors.append(error);assert error<1e-8
        pred={a:torch.tensor(seen)[v.argmax(1)] for a,v in scores.items()};metrics={}
        for arm,p in pred.items():
            counts=[int((labels==c).sum()) for c in seen];assert counts==[15]*n
            correct=[int(((labels==c)&(p==c)).sum()) for c in seen]
            metrics[arm]=dict(counts=counts,correct=correct,macro=sum(correct)/(15*n))
        contrasts={}
        for arm in cfg['arms']:
            contrasts[arm]={}
            for ref in ['df_matched_joint','cls768_ridge','cls_classmean']:
                if arm==ref:continue
                p,q=pred[arm],pred[ref];contrasts[arm][ref]=dict(rescue=int(((p==labels)&(q!=labels)).sum()),harm=int(((p!=labels)&(q==labels)).sum()),changed=int((p!=q).sum()))
        records.append(dict(stage=stage,seen=seen,metrics=metrics,contrasts=contrasts,target_df=target_df,actual_joint_df=actual_df))
        torch.save(dict(scores=scores,labels=labels,seen=seen),root/f'stage_{stage}.private.pt')
    support=json.loads((Path(cfg['prior_run'])/'support.json').read_text())
    ordered=sorted(cfg['order'],key=lambda c:(support[str(c)]['images'],c))
    bins=dict(tail7=ordered[:7],middle6=ordered[7:13],head7=ordered[13:],initial10=cfg['order'][:10],later10=cfg['order'][10:])
    groups={a:{k:sum(m['correct'][cfg['order'].index(c)] for c in cs) for k,cs in bins.items()} for a,m in records[-1]['metrics'].items()}
    mean={a:sum(r['metrics'][a]['macro'] for r in records)/6 for a in records[-1]['metrics']}
    a='joint_classmean_df';z='cls_classmean';final=records[-1]['metrics']
    gates=dict(final_correct=sum(final[a]['correct'])>=86,mean_gain=mean[a]-mean['df_matched_joint']>=.01,tail7=groups[a]['tail7']>=22,later10=groups[a]['later10']>=38,base_correct=sum(records[0]['metrics'][a]['correct'])>=66,joint_beats_cls_final=sum(final[a]['correct'])>sum(final[z]['correct']),joint_not_worse_cls_mean=mean[a]>=mean[z])
    for name,value in [('RESULTS.json',records),('FREQUENCY.json',dict(bins=bins,correct=groups)),('DECISION.json',dict(gates=gates,proceed=all(gates.values()),mean_stage_BA=mean)),('VERIFICATION.json',dict(state='PASS',heads_checked=12,base_score_max_error=max(base_errors),head_score_max_error=max(errors),df_max_error=max(abs(r['target_df']-r['actual_joint_df']) for r in records),patient_overlap=0,new_image_encodes=0,wall_seconds=time.time()-start))]:write(root/name,value)
    write(root/'status.private.json',dict(state='COMPLETE',updated=time.time(),proceed=all(gates.values()),delivery_pending=True))


if __name__=='__main__':
    try:main(json.loads(Path(os.environ['RUN_CONFIG']).read_text()))
    except BaseException:
        import traceback
        root=Path(json.loads(Path(os.environ['RUN_CONFIG']).read_text())['output'])
        write(root/'VERIFY_FAILURE.private.json',dict(updated=time.time(),error=traceback.format_exc()))
        raise
