"""Frozen native patch extraction, patient-token replay, all-head sealing and scoring."""
import copy,json,os,signal,time,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS,Model,global_features,affine,weights,memory_indices,update,cpu_scores
from data import read_rows,Images
from foundation import write,save,prefix,build_backbone,body_state,np_tokens
from checks import math_check,qualify,optimizer,state,counter


def main(cfg):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=Path(cfg['output']);p.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,interface_encodes=0,encoder_compute_copy_seconds={},arms={a:counter() for a in ARMS[1:]},native_solves=0,full_NumPy_qualification_rows=0,full_FP64_qualification_rows=0,qualification_discarded_updates=0,initialization_global_rows=0,validation_model_rows={a:0 for a in ARMS})
    def status(phase,**extra):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.json',cost);write(p/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def stop(sig,frame):write(p/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    status('math_qualification');tick=time.time();math=math_check();math['wall_seconds']=time.time()-tick;write(p/'MATH_TORCH_NUMPY.json',math)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert len(train)==68058 and len(val)==300
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    for i,r in enumerate(train):r['id']=i
    write(p/'COVERAGE.json',dict(train=68058,val=300,train_counts=dict(Counter(str(r['label']) for r in train)),patient_overlap=0))
    b=build_backbone(cfg,'cuda');original={k:v.detach().cpu().clone() for k,v in b.state_dict().items()}
    def extract(rows,field,proof=False):
        out=torch.empty((len(rows),197,768),dtype=torch.float16);prefixes=torch.empty_like(out) if field=='validation_encodes' else None
        wanted=set(memory_indices(rows,2,cfg['seed'])) if proof else set();saved={}
        for images,ix in DataLoader(Images(rows,cfg),batch_size=cfg['encoder_batch'],num_workers=0,shuffle=False):
            tick=time.time()
            with torch.no_grad():
                with torch.autocast('cuda',dtype=torch.float16):pre=prefix(b,images.cuda()).half()
                raw=b.norm(b.blocks(pre.float()))
            assert torch.isfinite(raw).all() and raw.shape==(len(ix),197,768)
            out[ix]=raw.half().cpu()
            if prefixes is not None:prefixes[ix]=pre.cpu()
            for j,z in enumerate(ix.tolist()):
                if z in wanted:saved[z]=dict(prefix=pre[j].cpu().clone(),raw=raw[j].cpu().clone(),id=rows[z].get('id',-1),label=rows[z]['label'])
            cost[field]+=len(ix);torch.cuda.synchronize();cost['encoder_compute_copy_seconds'][field]=cost['encoder_compute_copy_seconds'].get(field,0.)+time.time()-tick
            status(field,encoded=int(ix[-1])+1,total_current=len(rows))
        return out,prefixes,saved
    def descriptors(tokens):
        z=torch.empty((len(tokens),1536),dtype=torch.float64);part=torch.empty((len(tokens),768),dtype=torch.float32)
        for i in range(0,len(tokens),64):
            x=tokens[i:i+64].float();z[i:i+64]=global_features(x).double();part[i:i+64]=F.normalize(x[:,1:],dim=-1).mean(1)
        return z,part
    def means(rows,z,part,classes):
        zz=[];pp=[]
        for c in classes:
            ids=[i for i,r in enumerate(rows) if r['label']==c];w,_=weights([rows[i] for i in ids]);ww=torch.from_numpy(w)
            zz.append((ww[:,None]*z[ids]).sum(0));pp.append((ww[:,None]*part[ids].double()).sum(0))
        return torch.stack(zz).float(),torch.stack(pp).float()
    qrows=[]
    for c in cfg['order'][:10]:
        selected=memory_indices([r for r in train if r['label']==c],2,cfg['seed']);rr=[r for r in train if r['label']==c];qrows.extend(rr[i] for i in selected)
    q,_,qproof=extract(qrows,'qualification_encodes',proof=True);qz,qp=descriptors(q);gm,pm=means(qrows,qz,qp,cfg['order'][:10])
    qual=qualify(q,gm,pm,cfg,p);cost['qualification_discarded_updates']=qual['discarded_updates']
    # Re-read two images through the canonical quantized-prefix/full-body interface.
    ii=Images(qrows[:2],cfg);images=torch.stack([ii[i][0] for i in range(2)]).cuda()
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.float16):pre=prefix(b,images).half()
        raw=b.norm(b.blocks(pre.float())).cpu().double();ref=np_tokens(pre.cpu().numpy(),body_state(b));cost['interface_encodes']=2
    error=float(abs(raw.numpy()-ref).max());assert error<cfg['FP32_probe_tolerance'];cost['full_NumPy_qualification_rows']=2
    cpu=build_backbone(cfg,'cpu').double()
    with torch.no_grad():cpu_raw=cpu.norm(cpu.blocks(pre.cpu().double())).numpy()
    fp64_error=float(abs(cpu_raw-ref).max());assert fp64_error<=cfg['score_absolute_tolerance'];cost['full_FP64_qualification_rows']=2
    assert all(torch.equal(v,b.state_dict()[k].cpu()) for k,v in original.items())
    save(p/'qualification_inputs.private.pt',dict(tokens=q,rows=qrows,proof={i:qproof[i] for i in sorted(qproof)[:2]}));del cpu,pre,raw,ref,images,ii,q,qz,qp,gm,pm,qproof,qrows
    qual.update(full_GPU_FP32_NumPy_error=error,full_FP64_NumPy_error=fp64_error,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),qualification_new_images=20,additional_interface_images=2)
    forecast=cfg['feasibility_factor']*(cfg['base_updates']+5*cfg['later_updates'])*qual['paired_step_seconds']+cfg['feasibility_reserve_seconds'];remaining=cfg['hard_deadline_epoch']-time.time()
    qual['forecast_seconds']=forecast;qual['remaining_seconds']=remaining;write(p/'QUALIFICATION.json',qual)
    assert forecast<remaining-60,'Frozen original-clock budget gate failed'
    assert torch.cuda.max_memory_reserved()<cfg['gpu_reserved_budget_bytes']
    models={};memory=torch.empty((0,197,768),dtype=torch.float16);memory_rows=[];sumgram=torch.zeros(1537,1537,dtype=torch.float64);classmeans=[];seals=[];curves=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];rows=[r for r in train if r['label'] in new]
        tokens,_,proof=extract(rows,'science_train_encodes',proof=True);z,part=descriptors(tokens);cost['initialization_global_rows']+=len(rows);gm,pm=means(rows,z,part,new)
        arrived_gram=torch.zeros_like(sumgram);arrival_counts=[]
        for c in new:
            ids=[i for i,r in enumerate(rows) if r['label']==c];rr=[rows[i] for i in ids];w,n=weights(rr);phi=affine(z[ids]);ww=torch.from_numpy(w);gram=phi.T@(ww[:,None]*phi);mu=(ww[:,None]*phi).sum(0)
            # Current arriving matrix is independently checked without retaining full historical images.
            check=phi.numpy().T@(w[:,None]*phi.numpy());err=float(abs(check-gram.numpy()).max());assert err<1e-9
            assert float(abs(phi.numpy().T@w-mu.numpy()).max())<1e-9
            sumgram+=gram;arrived_gram+=gram;classmeans.append(mu);arrival_counts.append(dict(label=c,images=len(rr),patients=n))
            chosen=memory_indices(rr,cfg['memory_per_class'],cfg['seed']);save(p/f'memory_{c}.private.pt',dict(tokens=tokens[[ids[i] for i in chosen]],ids=[rr[i]['id'] for i in chosen],label=c))
            items=[v for v in proof.values() if v['label']==c];save(p/f'proof_{c}.private.pt',dict(label=c,images=len(rr),patients=n,proof=items,independent_current_gram_error=err))
            del phi,gram,check
        mat=sumgram/len(seen)+cfg['ridge']*torch.eye(1537,dtype=torch.float64);rhs=torch.stack(classmeans,1)/len(seen);head=torch.linalg.solve(mat,rhs);assert float(abs(mat@head-rhs).max())<1e-9;cost['native_solves']+=1
        for arm in ARMS[1:]:
            if task==0:models[arm]=Model(gm,pm,arm).cuda()
            else:models[arm].expand(gm.cuda(),pm.cuda())
        opts={a:optimizer(models[a],cfg) for a in ARMS[1:]};source=memory_rows+rows;pools={}
        for i,r in enumerate(source):pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
        rng=np.random.default_rng(cfg['seed']+1000+task);steps=cfg['base_updates'] if task==0 else cfg['later_updates'];draws=[];last={}
        for step in range(1,steps+1):
            ix=[]
            for c in rng.choice(seen,8,replace=False):
                groups=list(pools[c])
                for _ in range(8):
                    values=pools[c][groups[int(rng.integers(len(groups)))]];ix.append(values[int(rng.integers(len(values)))])
            x=torch.stack([memory[i] if i<len(memory_rows) else tokens[i-len(memory_rows)] for i in ix]);y=torch.tensor([seen.index(source[i]['label']) for i in ix]);draws.append([source[i]['id'] for i in ix]);last={}
            if step==steps:prestates={a:state(models[a]) for a in ARMS[1:]}
            for a in ARMS[1:]:last[a]=update(models[a],opts[a],x,y,cfg,cost['arms'][a])
            if step==1 or step%100==0 or step==steps:
                curves.append(dict(task=task,step=step,arms=last,at=time.time()));write(p/'TRAINING_CURVES.json',curves);status('training',task=task,step=step,total=steps)
        selected=memory_indices(rows,cfg['memory_per_class'],cfg['seed']);memory=torch.cat([memory,tokens[selected]]);memory_rows.extend(rows[i] for i in selected)
        assert len(memory_rows)==len({(r['label'],r['group']) for r in memory_rows})
        save(p/f'stage_{task}.private.pt',dict(seen=seen,models={a:state(models[a]) for a in ARMS[1:]},native_head=head,native_mat=mat,native_rhs=rhs,arrived_gram=arrived_gram,arrival_counts=arrival_counts,memory_ids=[r['id'] for r in memory_rows],draws=torch.tensor(draws,dtype=torch.int32),probe=dict(tokens=x,labels=y,models_before_update=prestates,losses={a:last[a]['CE'] for a in ARMS[1:]})))
        seals.append(dict(task=task,seen=seen,at=time.time(),models=4,train_encodes=cost['science_train_encodes'],memory_images=len(memory_rows),updates={a:cost['arms'][a]['updates'] for a in ARMS[1:]},parameters={a:sum(v.numel() for v in models[a].parameters()) for a in ARMS[1:]}));write(p/'STAGE_SEALS.json',seals)
        del tokens,z,part,proof,rows,arrived_gram,opts,source,pools,x,y,draws,prestates;torch.cuda.empty_cache()
    assert cost['science_train_encodes']==68058 and all(torch.equal(v,b.state_dict()[k].cpu()) for k,v in original.items())
    write(p/'FIT_SEAL.json',dict(models=24,at=time.time(),backbone_frozen_exact=True));write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
    query,prefixes,_=extract(val,'validation_encodes');labels=torch.tensor([r['label'] for r in val]);save(p/'query.private.pt',dict(tokens=query,prefix=prefixes,labels=labels));del prefixes
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);seen=pack['seen'];mask=torch.isin(labels,torch.tensor(seen));qt=query[mask]
        scores={'NATIVE_RIDGE':affine(global_features(qt.double()))@pack['native_head']}
        scores.update({a:cpu_scores(qt,pack['models'][a],a,cfg['microbatch']) for a in ARMS[1:]})
        save(p/f'scores_{task}.private.pt',scores)
        for a in ARMS:cost['validation_model_rows'][a]+=len(qt)
        status('deployment_scored',task=task)
    write(p/'COSTS.json',cost);status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        p=Path(cfg['output'])
        if p.exists():write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()))
        raise
