"""Independent float64 density evaluation, explicit window reductions and metrics."""
import json,math,os,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)

def manual_ratio(u,condition,state):
    x=u;det=u.new_zeros(u.shape[:-1]);d=u.shape[-1]//2
    for j in range(2):
        first=x[...,:d];second=x[...,d:]
        h=F.linear(torch.cat((first,condition),-1),state[f'layers.{j}.0.weight'],state[f'layers.{j}.0.bias'])
        h=h*torch.sigmoid(h)
        out=F.linear(h,state[f'layers.{j}.2.weight'],state[f'layers.{j}.2.bias']);logscale=torch.tanh(out[...,:d])/2
        new=(second-out[...,d:])/torch.exp(logscale);x=torch.cat((new,first),-1);det=det-logscale.sum(-1)
    return (u.square().sum(-1)-x.square().sum(-1))/2+det

def manual_score(r,arm):
    if arm=='G':return r.sum(-1)
    if arm=='E_shuffle':
        perm=torch.randperm(196,generator=torch.Generator().manual_seed(74386)).to(r.device);r=r.index_select(1,perm)
    r=r.reshape(-1,14,14);terms=[]
    for k in (1,2,4,7,14):
        starts=list(range(0,15-k,max(1,k//2)));lp=-math.log(5*len(starts)**2)
        for y in starts:
            for x in starts:terms.append(r[:,y:y+k,x:x+k].sum((1,2))+lp)
    return torch.logsumexp(torch.stack(terms,1),1)

def main(cfg):
    start=time.time();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(cfg['output']);dev='cuda'
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('verify_data',Path(cfg['remote_base'])/'case_hierarchical1_20261018/data.py');data=module_from_spec(spec);spec.loader.exec_module(data)
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val');assert len(train)==68058 and len(val)==300
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    seal=json.loads((root/'FIT_SEAL.json').read_text());bs=json.loads((root/'REFERENCE_SEAL.json').read_text());opened=json.loads((root/'QUERY_OPEN.private.json').read_text())
    assert bs['at']<seal['at']<opened['at'] and bs['future_images']==bs['val_images']==0 and seal['models']==60
    assert [v['class_id'] for v in seal['classes']]==cfg['order'] and all(v['val_images']==v['future_class_encodes']==0 for v in seal['classes'])
    q={k:v.to(dev).double() for k,v in torch.load(root/'reference.private.pt',weights_only=True).items()}
    stats=torch.load(root/'reference_moments.private.pt',weights_only=True);g=stats['g'].to(dev);xz=stats['xz'].to(dev);zz=stats['zz'].to(dev)
    reg=torch.eye(33,device=dev,dtype=torch.float64)*.001;reg[-1,-1]=0
    beta=torch.linalg.solve(g+reg,xz.permute(1,0,2).reshape(33,-1)).reshape(33,196,768).permute(1,0,2)
    sigma=(zz-2*(beta*xz).sum(1)+torch.einsum('jid,ik,jkd->jd',beta,g,beta)).clamp_min(.0001).sqrt()
    assert (beta-q['beta']).abs().max()<1e-9 and (sigma-q['sigma']).abs().max()<1e-9
    query=torch.load(root/'query.private.pt',weights_only=True);saved=torch.load(root/'scores.private.pt',weights_only=True)
    labels=query['labels'];assert labels.tolist()==[c for c in cfg['order'] for _ in range(15)] and torch.equal(labels,saved['labels'])
    yy,xx=torch.meshgrid(torch.linspace(-1,1,14,dtype=torch.float64),torch.linspace(-1,1,14,dtype=torch.float64),indexing='ij');pos=torch.stack((yy.flatten(),xx.flatten()),-1).to(dev).double()
    # Worker uses FP32 position inputs; reconstruct their rounded coordinates exactly.
    scores={a:torch.empty(300,20,dtype=torch.float64) for a in cfg['arms']};err=0.;mismatches=0
    with torch.no_grad():
        for ci,c in enumerate(cfg['order']):
            for arm in cfg['arms']:
                assert time.time()<cfg['hard_deadline_epoch']-60
                state={k:v.to(dev).double() for k,v in torch.load(root/f'{arm}_{c:02d}.private.pt',weights_only=True).items()}
                for ids in torch.arange(300).split(16):
                    cls=query['cls'][ids].to(dev).double();ctx=cls@q['projection'];aug=torch.cat((ctx,torch.ones(len(ids),1,device=dev,dtype=torch.float64)),1)
                    u=((query['patch'][ids].to(dev).double()-torch.einsum('bi,jid->bjd',aug,q['beta']))/q['sigma'][None])
                    cond=torch.cat((ctx[:,None].expand(-1,196,-1),pos[None].expand(len(ids),-1,-1)),-1)
                    score=manual_score(manual_ratio(u,cond,state),arm);scores[arm][ids,ci]=score.cpu()
                err=max(err,float((scores[arm][:,ci]-saved['scores'][arm][:,ci]).abs().max()))
    assert err<=cfg['score_absolute_tolerance'],f'Score mismatch {err}'
    for a in cfg['arms']:
        mismatches+=int((scores[a].argmax(1)!=saved['scores'][a].argmax(1)).sum())
    assert mismatches==0,'Independent prediction mismatch; do not relax parity'
    records=[];finalgroups={};paired=[];tail=[15,16,7,14,18,12,5]
    for task in range(6):
        n=sum(cfg['task_sizes'][:task+1]);seen=cfg['order'][:n];mask=torch.isin(labels,torch.tensor(seen));y=labels[mask];metrics={};pred={}
        for a in cfg['arms']:
            pred[a]=torch.tensor(seen)[scores[a][mask,:n].argmax(1)]
            correct={str(c):int(((y==c)&(pred[a]==c)).sum()) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=sum(correct.values())/(15*n)*100)
        records.append(dict(task=task,seen=seen,metrics=metrics))
        paired.append(dict(task=task,comparisons={a:dict(rescue=int(((pred['E']==y)&(pred[a]!=y)).sum()),harm=int(((pred['E']!=y)&(pred[a]==y)).sum())) for a in ['G','E_shuffle']}))
    for a in cfg['arms']:
        c=records[-1]['metrics'][a]['correct'];finalgroups[a]={name:sum(c[str(k)] for k in cs) for name,cs in dict(tail7=tail,initial10=cfg['order'][:10],later10=cfg['order'][10:]).items()}
    means={a:sum(r['metrics'][a]['BA_percent'] for r in records)/6 for a in cfg['arms']};f=records[-1]['metrics'];fg=finalgroups['E']
    gates=dict(final=f['E']['total_correct']>=91,mean=means['E']>=36.005291005291,tail=fg['tail7']>=25,later=fg['later10']>=41,initial=fg['initial10']>=44,
      beats_G_final=f['E']['total_correct']>=f['G']['total_correct']+4,beats_G_mean=means['E']>=means['G']+1,not_hurt_G_tail=fg['tail7']>=finalgroups['G']['tail7'],
      spatial_final=f['E']['total_correct']>f['E_shuffle']['total_correct'],spatial_mean=means['E']>=means['E_shuffle'])
    cost=json.loads((root/'COSTS.json').read_text());expected=5*sum(math.ceil(sum(r['label']==c for r in train)/32) for c in cfg['order'])
    assert cost['qualification_encodes']==64 and cost['qualification_updates']==3 and cost['reference_encodes']==60129 and cost['science_training_encodes']==68058 and cost['validation_encodes']==300
    assert cost['optimizer_updates']==3*expected and all(v==expected for v in cost['arm_updates'].values())
    assert all(v==340290 for v in cost['feature_image_exposures'].values())
    resource=json.loads((root.parent/'guardian_resources.private.json').read_text());assert resource['peak_output_bytes']<cfg['output_budget_bytes'] and resource['RSS_sampled_peak_bytes']<cfg['RSS_budget_bytes']
    for name,obj in [('RESULTS.json',dict(stages=records,mean_BA_percent=means,frequency=finalgroups,paired=paired)),('DECISION.json',dict(gates=gates,proceed=all(gates.values()),primary_comparator='G',no_parameter_rescue=True)),('VERIFICATION.json',dict(state='PASS',score_max_absolute_error=err,prediction_mismatches=0,model_count=60,reference_reconstructed=True,patient_overlap=0,new_image_encodes=0,wall_seconds=time.time()-start))]:write(root/name,obj)
    write(root/'status.private.json',dict(state='COMPLETE_VERIFIED',updated=time.time(),method_gate_pass=all(gates.values()),delivery_pending=True))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        write(Path(cfg['output'])/'VERIFY_FAILURE.private.json',dict(at=time.time(),error=traceback.format_exc()));raise
