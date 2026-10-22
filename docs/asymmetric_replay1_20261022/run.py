"""Arriving-data acquisition and admitted all-class-memory consolidation, paired arms."""
import copy
import json
import os
import signal
import time
import traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS, Model, objective, memory_indices
from data import read_rows, Images
from verify import numpy_forward, numpy_loss, summarize


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def math_check(cfg):
    start=time.time()
    torch.manual_seed(cfg['seed'])
    m=Model(8,3,torch.randn(4,8,dtype=torch.float64)).double()
    with torch.no_grad():m.up.weight.normal_(0,.1)
    x=torch.randn(16,8,dtype=torch.float64)
    y=torch.tensor([2,3]*4+[0,1,2,3]*2)
    receipt={}
    for a in ARMS:
        m.zero_grad(); loss=objective(m,x,y,8,2,a,cfg)[0];loss.backward()
        s={k:v.numpy().astype(np.float64) for k,v in state(m).items()}
        error=abs(float(loss.detach())-numpy_loss(x.numpy(),y.numpy(),8,2,s,a,cfg))
        fd_errors=[]
        for name,ix in [('head',(0,0)),('up.weight',(2,1)),('down.bias',(0,))]:
            plus={k:v.copy() for k,v in s.items()};minus={k:v.copy() for k,v in s.items()}
            plus[name][ix]+=1e-5;minus[name][ix]-=1e-5
            fd=(numpy_loss(x.numpy(),y.numpy(),8,2,plus,a,cfg)-numpy_loss(x.numpy(),y.numpy(),8,2,minus,a,cfg))/2e-5
            fd_errors.append(abs(fd-float(dict(m.named_parameters())[name].grad[ix])))
        assert error<1e-9 and max(fd_errors)<2e-6
        receipt[a]=dict(objective_error=error,finite_difference_error=max(fd_errors),torch_forward_backward_rows=16,numpy_objective_rows=7*16)
    # Exact routing: acquisition cannot update old head rows directly, consolidation can.
    m.zero_grad();_,z=m(x[:8],cfg['logit_scale']);F.cross_entropy(z[:,2:],y[:8]-2).backward()
    assert torch.count_nonzero(m.head.grad[:2])==0 and torch.count_nonzero(m.head.grad[2:])>0
    m.zero_grad();_,z=m(x[8:],cfg['logit_scale']);F.cross_entropy(z,y[8:]).backward()
    assert torch.count_nonzero(m.head.grad[:2])>0 and torch.count_nonzero(m.head.grad[2:])>0
    receipt['routing']=dict(old_head_acquisition_gradient_exact_zero=True,replay_reaches_all_head_rows=True,extra_torch_forward_backward_rows=16)
    receipt['seconds']=time.time()-start
    return receipt


def pools(rows):
    out={}
    for i,r in enumerate(rows):out.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
    return out


def draw(pool, classes, count, rng):
    result=[]
    for _ in range(count):
        c=classes[int(rng.integers(len(classes)))];gs=list(pool[c]);g=gs[int(rng.integers(len(gs)))];ix=pool[c][g]
        result.append(ix[int(rng.integers(len(ix)))])
    return result


def minibatch(features, rows, memory, memory_rows, seen, new, rng, view_rng, cfg, channel_pools=None):
    cp,rp=channel_pools if channel_pools is not None else (pools(rows),pools(memory_rows))
    ci=draw(cp,new,cfg['current_draws'],rng)
    ri=draw(rp,seen,cfg['replay_draws'],rng)
    # Channel grouping survives views; all arms consume these exact same tensors.
    cx=features[ci].repeat(cfg['views_per_draw'],1)
    rx=memory[ri].repeat(cfg['views_per_draw'],1)
    x=torch.cat([cx,rx])
    x=x*((torch.rand(x.shape,generator=view_rng)>=cfg['input_dropout'])/(1-cfg['input_dropout']))
    y=torch.tensor([seen.index(rows[i]['label']) for i in ci]*cfg['views_per_draw']+
                   [seen.index(memory_rows[i]['label']) for i in ri]*cfg['views_per_draw'])
    ids=[rows[i]['id'] for i in ci]*cfg['views_per_draw']+[memory_rows[i]['id'] for i in ri]*cfg['views_per_draw']
    return x,y,len(cx),ci,ri,ids


def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,qualification_updates=0,
              updates={a:0 for a in ARMS},sampled_image_rows={a:0 for a in ARMS},training_view_rows={a:0 for a in ARMS},
              incoming_view_rows={a:0 for a in ARMS},replay_view_rows={a:0 for a in ARMS},
              qualification_model_rows={a:0 for a in ARMS},qualification_FP64_rows={a:0 for a in ARMS},qualification_numpy_rows={a:0 for a in ARMS},
              validation_model_rows={a:0 for a in ARMS},training_probe_model_rows={a:0 for a in ARMS},
              encoder_parameter_updates=0,encoder_seconds={},head_initialization_feature_rows=0)
    models={};opts={};curves=[];seals=[];admissions=[];sampling=[]
    def status(phase,**kw):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch']
        cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(root/'COSTS.private.json',cost)
        write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def interrupted(sig,frame):
        save(root/'interrupted.private.pt',dict(models={a:state(m) for a,m in models.items()},cost=cost))
        write(root/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupted)
    status('mathematical_qualification');math=math_check(cfg);write(root/'MATH_QUALIFICATION.json',math)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(train):r['id']=i
    assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    write(root/'COVERAGE.json',dict(total_train=len(train),total_val=len(val),patient_overlap=0,train_counts=dict(Counter(str(r['label']) for r in train)),old_image_replay=False,old_cache_reuse=False))
    import timm
    from safetensors.torch import load_file
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    def extract(rows,field):
        out=torch.empty(len(rows),cfg['input_dimension']);start=time.time()
        prior=cost['encoder_seconds'].get(field,0.)
        for images,ix in DataLoader(Images(rows,cfg),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):raw=encoder.forward_features(images.cuda())
            assert raw.shape[1:]==(197,768) and torch.isfinite(raw).all()
            raw=raw.float();out[ix]=F.normalize(torch.cat([raw[:,0],raw[:,1:].mean(1)],1),dim=1).cpu()
            torch.cuda.synchronize();cost[field]+=len(ix)
            cost['encoder_seconds'][field]=prior+time.time()-start
            status(field,encoded_in_current_set=int(ix[-1])+1,current_count=len(rows))
        # This timer includes data delivery/decoding/wait and each status write.
        cost.setdefault('encoder_stage_seconds',[]).append(dict(field=field,rows=len(rows),seconds=time.time()-start))
        return out
    def means(features,rows,classes):
        pp=pools(rows)
        return torch.stack([torch.stack([features[ix].double().mean(0) for ix in pp[c].values()]).mean(0).float() for c in classes])
    qrows=[]
    for c in cfg['order'][:10]:
        gs=set()
        for r in train:
            if r['label']==c and r['group'] not in gs:qrows.append(r);gs.add(r['group'])
            if len(gs)==2:break
        assert len(gs)==2
    q=extract(qrows,'qualification_encodes');qstart=time.time();qreceipt={}
    torch.manual_seed(cfg['seed']);base=Model(cfg['input_dimension'],cfg['adapter_rank'],means(q[:16],qrows[:16],cfg['order'][:8]))
    qm={a:copy.deepcopy(base).cuda() for a in ARMS}
    rng=np.random.default_rng(cfg['seed']+10);vg=torch.Generator().manual_seed(cfg['seed']+11)
    for phase in range(2):
        seen=cfg['order'][:8] if phase==0 else cfg['order'][:10];new=seen if phase==0 else seen[8:]
        xf=q[:16] if phase==0 else q[16:];rr=qrows[:16] if phase==0 else qrows[16:]
        if phase:
            for a in ARMS:qm[a].expand(means(xf,rr,new).cuda())
        mem=q[:16] if phase==0 else q;mr=qrows[:16] if phase==0 else qrows
        x,y,n,ci,ri,ids=minibatch(xf,rr,mem,mr,seen,new,rng,vg,cfg);x=x.cuda();y=y.cuda()
        for a in ARMS:
            opt=torch.optim.AdamW(qm[a].parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);opt.zero_grad()
            loss=objective(qm[a],x,y,n,len(seen)-len(new),a,cfg)[0];loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in qm[a].parameters())
            gn=torch.nn.utils.clip_grad_norm_(qm[a].parameters(),cfg['gradient_clip']);assert torch.isfinite(gn);opt.step()
            cost['qualification_updates']+=1;cost['qualification_model_rows'][a]+=len(x)
        if phase==0:
            assert all(torch.equal(state(qm['ER'])[k],state(qm['ASYMMETRIC'])[k]) for k in state(qm['ER']))
    for a in ARMS:
        save(root/f'qualification_{a}.private.pt',state(qm[a]));saved=torch.load(root/f'qualification_{a}.private.pt',weights_only=True)
        m=Model(cfg['input_dimension'],cfg['adapter_rank'],saved['head']).double();m.load_state_dict({k:v.double() for k,v in saved.items()})
        with torch.no_grad():score=m(x.cpu().double(),cfg['logit_scale'])[1]
        ns={k:v.numpy().astype(np.float64) for k,v in saved.items()};ref=numpy_forward(x.cpu().numpy(),ns,cfg['logit_scale'])[1]
        err=float(abs(score.numpy()-ref).max());assert err<=cfg['score_absolute_tolerance']
        obj=numpy_loss(x.cpu().numpy(),y.cpu().numpy(),n,8,ns,a,cfg)
        with torch.no_grad():tl=float(objective(m,x.cpu().double(),y.cpu(),n,8,a,cfg)[0])
        assert abs(obj-tl)<=cfg['score_absolute_tolerance']
        cost['qualification_FP64_rows'][a]+=2*len(x);cost['qualification_numpy_rows'][a]+=2*len(x)
        delta=float((saved['up.weight']-base.up.weight.detach().cpu()).abs().max())
        assert delta>0
        qreceipt[a]=dict(actual_residual_weight_change=delta,score_error=err,objective_error=abs(obj-tl),engineering_updates=2,model_forward_backward_rows=256,FP64_rows=256,numpy_rows=256)
    write(root/'QUALIFICATION.json',dict(state='PASS',math=math,real_GPU=qreceipt,legal_base_encodes=20,discarded_optimization=4,validation_encodes=0,actual_arrival_branch=True,save_load=True,post_encoding_seconds=time.time()-qstart))
    del qm,base,q,x,y,loss,opt,m,score,ref,ns,saved
    torch.cuda.empty_cache();status('qualified_before_bulk')
    memory=torch.empty(0,cfg['input_dimension']);memory_rows=[]
    for task in range(6):
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];old=sum(cfg['task_sizes'][:task]);new=seen[old:]
        rows=[r for r in train if r['label'] in new];features=extract(rows,'science_train_encodes');mm=means(features,rows,new)
        cost['head_initialization_feature_rows']+=len(rows)
        if task==0:
            torch.manual_seed(cfg['seed']);template=Model(cfg['input_dimension'],cfg['adapter_rank'],mm)
            models={a:copy.deepcopy(template).cuda() for a in ARMS};del template
        else:
            for m in models.values():m.expand(mm.cuda())
        # Admit each current class before its first update: global replay has current positives.
        chosen=memory_indices(rows,cfg['memory_per_class'],cfg['seed'])
        memory=torch.cat([memory,features[chosen]]);memory_rows.extend(rows[i] for i in chosen)
        assert set(r['label'] for r in memory_rows)==set(seen)
        admitted=time.time();admissions.append(dict(task=task,at=admitted,validation_encodes=0,seen=seen,rows=len(memory_rows),current_rows=len(chosen)))
        write(root/'MEMORY_ADMISSIONS.json',admissions)
        opts={a:torch.optim.AdamW(models[a].parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for a in ARMS}
        rng=np.random.default_rng(cfg['seed']+1000+task);vg=torch.Generator().manual_seed(cfg['seed']+2000+task)
        channel_pools=(pools(rows),pools(memory_rows))
        steps=cfg['base_updates'] if task==0 else cfg['later_updates'];start=time.time()
        audit=dict(task=task,current_counts={str(c):0 for c in seen},replay_counts={str(c):0 for c in seen},paired_same_draws=True,paired_same_views=True,first_update_at=None)
        for step in range(1,steps+1):
            x,y,n,ci,ri,ids=minibatch(features,rows,memory,memory_rows,seen,new,rng,vg,cfg,channel_pools);x=x.cuda();y=y.cuda()
            for i in ci:audit['current_counts'][str(rows[i]['label'])]+=1
            for i in ri:audit['replay_counts'][str(memory_rows[i]['label'])]+=1
            if step==1:audit['first_update_at']=time.time();assert admitted<audit['first_update_at']
            details={}
            for a in ARMS:
                opt=opts[a];opt.zero_grad();loss,acq,cons=objective(models[a],x,y,n,old,a,cfg);assert torch.isfinite(loss)
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(models[a].parameters(),cfg['gradient_clip']);assert torch.isfinite(gn);opt.step()
                cost['updates'][a]+=1;cost['sampled_image_rows'][a]+=len(ci)+len(ri);cost['training_view_rows'][a]+=len(x)
                cost['incoming_view_rows'][a]+=n;cost['replay_view_rows'][a]+=len(x)-n
                details[a]=dict(loss=float(loss.detach()),acquisition_CE=float(acq.detach()),weighted_consolidation=float(cons.detach()),gradient_norm=float(gn))
            if step==1 or step%100==0 or step==steps:
                torch.cuda.synchronize();curves.append(dict(task=task,step=step,at=time.time(),elapsed_stage=time.time()-start,arms=details))
                write(root/'TRAINING_CURVES.json',curves);status('actual_stage_training',task=task,step=step,steps=steps)
        probe=dict(x=x.cpu().double(),y=y.cpu(),incoming_rows=n,old_count=old,ids=ids,losses={})
        for a in ARMS:
            pm=copy.deepcopy(models[a]).cpu().double()
            with torch.no_grad():probe['losses'][a]=float(objective(pm,probe['x'],probe['y'],n,old,a,cfg)[0])
            cost['training_probe_model_rows'][a]+=len(x);del pm
        save(root/f'stage_{task}.private.pt',dict(seen=seen,models={a:state(m) for a,m in models.items()},memory_features=memory,memory_rows=memory_rows,training_probe=probe,new_head_initialization_means=mm))
        seals.append(dict(task=task,at=time.time(),seen=seen,validation_encodes=0,science_train_encodes=cost['science_train_encodes'],updates=dict(cost['updates']),memory_rows=len(memory_rows),parameters_per_arm=sum(p.numel() for p in models['ER'].parameters()),training_loop_seconds=time.time()-start))
        sampling.append(audit);write(root/'SAMPLING_AUDIT.json',sampling);write(root/'STAGE_SEALS.json',seals)
        del features,rows,x,y,probe,mm,loss,acq,cons;torch.cuda.empty_cache()
    assert cost['science_train_encodes']==68058
    write(root/'FIT_SEAL.json',dict(at=time.time(),models=12,validation_encodes=0));write(root/'QUERY_OPEN.private.json',dict(at=time.time(),validation_encodes=0))
    query=extract(val,'validation_encodes');labels=torch.tensor([r['label'] for r in val]);save(root/'query.private.pt',dict(features=query,labels=labels));stages=[]
    for task in range(6):
        pack=torch.load(root/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);seen=pack['seen'];mask=torch.isin(labels,torch.tensor(seen));metrics={}
        for a in ARMS:
            m=Model(cfg['input_dimension'],cfg['adapter_rank'],pack['models'][a]['head']).double();m.load_state_dict({k:v.double() for k,v in pack['models'][a].items()})
            with torch.inference_mode():score=m(query[mask].double(),cfg['logit_scale'])[1]
            cost['validation_model_rows'][a]+=int(mask.sum());save(root/f'scores_{task}_{a}.private.pt',score)
            pred=torch.tensor(seen)[score.argmax(1)];correct={str(c):int(((pred==c)&(labels[mask]==c)).sum()) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
        stages.append(dict(task=task,seen=seen,metrics=metrics));status('sealed_model_scoring',task=task)
    results,decision=summarize(stages,cfg,Counter(r['label'] for r in train));write(root/'RESULTS.json',results);write(root/'DECISION.json',decision)
    cost['memory_rows_final']=len(memory_rows);status('scientific_complete');write(root/'COSTS.json',cost);write(root/'status.private.json',dict(state='COMPLETE_PENDING_VERIFY',updated=time.time()))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException as error:
        root=Path(cfg['output'])
        if root.exists():write(root/'FAILURE.private.json',dict(at=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise
