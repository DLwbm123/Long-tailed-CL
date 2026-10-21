"""One finite, guarded three-arm empirical coverage experiment."""
import json
import os
import signal
import time
import traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import ARMS, make_cover, tensors, objective, fit, math_check
from data import read_rows, Images
from verify import geometry_audit, numpy_objective


def write(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,
        qualification_updates=0,optimizer_updates={a:0 for a in ARMS},
        objective_evaluations={a:0 for a in ARMS},objective_feature_rows={a:0 for a in ARMS},
        current_training_risk_rows={a:0 for a in ARMS},cover_rows=0,independent_cover_rows=0,
        encoder_seconds={})
    def status(phase,**kwargs):
        cost.update(wall_from_first=time.time()-cfg['original_start_epoch'],
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(root/'COSTS.private.json',cost)
        write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**kwargs))
        assert time.time()<cfg['hard_deadline_epoch']-60
        assert cost['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
    def interrupt(sig,frame):
        write(root/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupt)
    status('mathematical_qualification');maths=math_check(cfg)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    write(root/'COVERAGE.json',dict(train_counts=dict(Counter(str(r['label']) for r in train)),
        patient_overlap=0,base_images=sum(r['label'] in cfg['order'][:10] for r in train),
        total_train=68058,total_val=300,retained_feature_exemplars=True,old_image_replay=False))
    import timm
    from safetensors.torch import load_file
    status('encoder_loading')
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    def extract(rows,field):
        out=torch.empty(len(rows),1536,dtype=torch.float32);tick=time.time()
        for images,idx in DataLoader(Images(rows,cfg),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):raw=encoder.forward_features(images.cuda())
            assert raw.shape[1:]==(197,768) and torch.isfinite(raw).all()
            raw=raw.float();out[idx]=torch.cat([raw[:,0],raw[:,1:].mean(1)],1).cpu()
            torch.cuda.synchronize();cost[field]+=len(idx)
            cost['encoder_seconds'][field]=cost['encoder_seconds'].get(field,0)+time.time()-tick
            status(field,encoded_in_current_set=int(idx[-1])+1,current_count=len(rows));tick=time.time()
        return torch.nn.functional.normalize(out.double(),dim=1)
    qrows=[]
    for c in cfg['order'][:10]:
        groups=set()
        for r in train:
            if r['label']==c and r['group'] not in groups:
                qrows.append(r);groups.add(r['group'])
                if len(groups)==2:break
        assert len(groups)==2
    q=extract(qrows,'qualification_encodes');qcover={}
    for c in cfg['order'][:10]:
        ids=[i for i,r in enumerate(qrows) if r['label']==c];groups=[qrows[i]['group'] for i in ids]
        cover,chosen,assign,mass=make_cover(q[ids],groups,1)
        geometry_audit(q[ids],groups,cover,chosen,assign);qcover[c]=cover
    receipts={}
    for arm in ARMS:
        x,y,r,m=tensors(qcover,cfg['order'][:10],arm)
        t=torch.zeros(10,1537,dtype=torch.float64,requires_grad=True)
        loss=objective(t,x,y,r,m,cfg);loss.backward()
        value,grad=numpy_objective(t.detach().numpy(),x.numpy(),y.numpy(),r.numpy(),m.numpy(),cfg,True)
        error=float(np.max(abs(grad-t.grad.numpy())))
        assert abs(value-float(loss.detach()))<1e-9 and error<1e-9
        with torch.no_grad():t-=.01*t.grad
        cost['qualification_updates']+=1
        receipts[arm]=dict(objective_error=abs(value-float(loss.detach())),gradient_error=error,engineering_updates=1)
    assert cost['qualification_encodes']==20
    write(root/'QUALIFICATION.json',dict(state='PASS',math=maths,arms=receipts,real_encodes=20,
        engineering_updates=3,validation_encodes=0))
    del q,qcover,qrows,t,loss,x,y,r,m,grad;torch.cuda.empty_cache()
    covers={};audits={};seals=[];all_fits=[];bound_checks=[]
    for task in range(6):
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])]
        new=seen[sum(cfg['task_sizes'][:task]):]
        rows=[r for r in train if r['label'] in new]
        z=extract(rows,'science_train_encodes')
        current={}
        for c in new:
            ids=[i for i,r in enumerate(rows) if r['label']==c];groups=[rows[i]['group'] for i in ids]
            points=z[ids]
            cover,chosen,assignment,mass=make_cover(points,groups,cfg['centers_per_class'])
            audits[str(c)]=geometry_audit(points,groups,cover,chosen,assignment)
            audits[str(c)]['at']=time.time();audits[str(c)]['task']=task
            current[c]=(ids,mass)
            covers[c]=cover;cost['cover_rows']+=len(ids);cost['independent_cover_rows']+=len(ids)
            status('cover_complete',task=task,class_id=c,centers=len(chosen))
        write(root/'GEOMETRY_AUDITS.json',audits);torch.save(covers,root/'covers.private.pt')
        point=torch.stack([v['weights']@v['centers'] for v in covers.values()]).mean(0)
        slack=min(float((v['radii']-torch.linalg.vector_norm(v['centers']-point,dim=1)).min()) for v in covers.values())
        if slack>=0:
            write(root/'STOP_THEOREM.private.json',dict(task=task,seen=seen,common_point=point.tolist(),minimum_slack=slack,
                lower_bound=float(np.log(len(seen))),smoothing_gap_at_zero_upper=max(float(v['radii'].max()) for v in covers.values())*cfg['norm_smoothing']))
            write(root/'DECISION.json',dict(proceed=False,reason='THEOREM_NONINFORMATIVE_STOP',no_parameter_rescue=True))
            write(root/'COSTS.json',cost);status('theorem_stop',task=task)
            return
        models={};fits={}
        for arm in ARMS:
            previous_calls=cost['objective_evaluations'][arm]
            previous_updates=cost['optimizer_updates'][arm]
            previous_rows=cost['objective_feature_rows'][arm]
            count_rows=sum(len(v['weights']) for v in covers.values())
            def progress(calls,value,gradient,updates):
                cost['objective_evaluations'][arm]=previous_calls+calls
                cost['optimizer_updates'][arm]=previous_updates+updates
                cost['objective_feature_rows'][arm]=previous_rows+calls*count_rows
                if calls==1 or calls%10==0:
                    status('head_fit',task=task,arm=arm,evaluations=calls,loss=value,gradient_inf=gradient)
            try:
                theta,receipt=fit(covers,seen,arm,cfg,progress)
            except BaseException:
                write(root/'COSTS.private.json',cost)
                raise
            models[arm]=theta;fits[arm]=receipt
            all_fits.append(dict(task=task,arm=arm,**receipt));write(root/'FITS.json',all_fits)
            torch.save(dict(seen=seen,covers=covers,models=models,fits=fits),root/f'partial_stage_{task}.private.pt')
            assert receipt['converged'], 'Frozen solver did not reach independent-acceptance gradient threshold'
            for c in new:
                ids,mass=current[c];label=seen.index(c)
                full=0.
                for off in range(0,len(ids),1024):
                    xx=z[ids[off:off+1024]]
                    ce=torch.nn.functional.cross_entropy(xx@theta[:,:-1].T+theta[:,-1],torch.full((len(xx),),label),reduction='none')
                    full+=float(mass[off:off+len(xx)]@ce)
                    cost['current_training_risk_rows'][arm]+=len(xx)
                single={c:covers[c]}
                xx=covers[c]['centers'];yy=torch.full((len(xx),),label)
                radius=covers[c]['radii'].clone()
                if arm=='POINT':radius.zero_()
                if arm=='CLASS':radius.fill_(float(radius.max()))
                upper=float(objective(theta,xx,yy,radius,covers[c]['weights'],cfg)-cfg['ridge']/2*theta.square().sum())
                gap=full-upper
                bound_checks.append(dict(task=task,arm=arm,class_id=c,current_full_CE=full,cover_loss=upper,full_minus_cover=gap))
                if arm!='POINT':assert gap<=cfg['gates']['coverage_bound_violation_tolerance']
        write(root/'TRAINING_BOUND_CHECKS.json',bound_checks)
        torch.save(dict(seen=seen,covers=covers,models=models,fits=fits),root/f'stage_{task}.private.pt')
        seals.append(dict(task=task,at=time.time(),seen=seen,validation_encodes=0,training_encodes=cost['science_train_encodes'],
            center_count=sum(len(v['weights']) for v in covers.values()),parameters_per_arm=len(seen)*1537,
            common_witness_minimum_slack=slack))
        write(root/'STAGE_SEALS.json',seals)
        del z,points,current,rows,models;torch.cuda.empty_cache()
    assert cost['science_train_encodes']==68058
    write(root/'FIT_SEAL.json',dict(at=time.time(),stages=6,query_opened=False,models=18))
    write(root/'QUERY_OPEN.private.json',dict(at=time.time(),validation_encodes=0))
    query=extract(val,'validation_encodes');labels=torch.tensor([r['label'] for r in val])
    torch.save(dict(features=query,labels=labels),root/'query.private.pt')
    stages=[]
    for task in range(6):
        pack=torch.load(root/f'stage_{task}.private.pt',weights_only=False)
        seen=pack['seen'];idx=torch.where(torch.isin(labels,torch.tensor(seen)))[0];metrics={}
        for arm in ARMS:
            theta=pack['models'][arm];score=query[idx]@theta[:,:-1].T+theta[:,-1]
            pred=torch.tensor(seen)[score.argmax(1)]
            torch.save(score,root/f'scores_{task}_{arm}.private.pt')
            correct={str(c):int(((pred==labels[idx])&(labels[idx]==c)).sum()) for c in seen}
            metrics[arm]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/len(idx))
        stages.append(dict(task=task,seen=seen,metrics=metrics))
    mean={a:float(np.mean([s['metrics'][a]['BA_percent'] for s in stages])) for a in ARMS}
    freq={}
    for a in ARMS:
        cor=stages[-1]['metrics'][a]['correct']
        freq[a]={k:sum(cor[str(c)] for c in ids) for k,ids in dict(tail4=[7,14,15,16],tail7=[c for c,n in Counter(r['label'] for r in train).items() if n<250],initial10=cfg['order'][:10],later10=cfg['order'][10:]).items()}
    result=dict(stages=stages,mean_BA_percent=mean,frequency=freq)
    write(root/'RESULTS.json',result)
    final=lambda a:stages[-1]['metrics'][a]['total_correct']
    gates=dict(final=final('CELL')>=91,mean=mean['CELL']>=cfg['gates']['mean_min_percent'],tail7=freq['CELL']['tail7']>=25,
        later10=freq['CELL']['later10']>=41,initial10=freq['CELL']['initial10']>=44,
        vs_POINT_final=final('CELL')>=final('POINT')+4,vs_POINT_mean=mean['CELL']>=mean['POINT']+1,
        vs_POINT_tail4=freq['CELL']['tail4']>=freq['POINT']['tail4']+2,vs_POINT_later=freq['CELL']['later10']>=freq['POINT']['later10'],
        vs_CLASS_final=final('CELL')>final('CLASS'),vs_CLASS_mean=mean['CELL']>=mean['CLASS'],all_fits_converged=all(f['converged'] for f in all_fits),
        bound_valid=all(d['full_minus_cover']<=1e-9 for d in bound_checks if d['arm']!='POINT'))
    write(root/'DECISION.json',dict(gates=gates,proceed=all(gates.values()),no_parameter_rescue=True))
    status('scientific_complete');write(root/'COSTS.json',cost)
    write(root/'status.private.json',dict(state='COMPLETE_PENDING_VERIFY',updated=time.time(),cost=cost))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException as error:
        root=Path(cfg['output'])
        if root.exists():write(root/'FAILURE.private.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
