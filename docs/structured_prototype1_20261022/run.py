"""Base-only evidence learning, immutable patient modes and paired all-seen training."""
import copy,json,os,time,traceback,signal
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS,EvidenceMap,Classifier,primitive_objective,objective,fit_classes,base_prior,stack_stats,memory_indices
from data import read_rows,Images
from verify import arrays,numpy_map,numpy_score,numpy_loss,independent_fit

def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');torch.save(x,q);q.replace(p)
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def pools(rows):
    d={}
    for i,r in enumerate(rows):d.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
    return d
def draw(pool,classes,rng):
    ids=[]
    for c in rng.choice(classes,8,replace=False if len(classes)>=8 else True):
        gs=list(pool[c])
        for _ in range(8):
            group=gs[int(rng.integers(len(gs)))];ix=pool[c][group];ids.append(ix[int(rng.integers(len(ix)))])
    return ids

def check_stats(features,rows,new,prior,cfg):
    records,cost=fit_classes(features,rows,new,prior,cfg)
    independent,proof=independent_fit(features.numpy(),rows,new,prior,cfg)
    err=max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for a,b in zip(records,independent) for k in ['mean','variance','calibrated','weights','global_mean','counts'])
    assert err<cfg['score_absolute_tolerance']
    return records,proof,cost,dict(state='PASS',max_error=err,patient_rows=cost['patient_rows'],at=time.time(),validation_encodes=0)

def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,encoder_parameter_updates=0,primitive_updates=0,primitive_forward_backward_rows=0,descriptor_rows=0,head_updates={a:0 for a in ARMS},head_training_rows={a:0 for a in ARMS},probe_rows={a:0 for a in ARMS},validation_rows={a:0 for a in ARMS},encoder_seconds={},qualification_primitive_updates=0,qualification_head_updates={a:0 for a in ARMS},qualification_head_rows={a:0 for a in ARMS},statistics_patient_rows=0,kmeans_distance_pairs=0,online_independent_statistics_patient_rows=0)
    models={};seals=[];curves=[];stage_task=-1
    def status(phase,**kw):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(root/'COSTS.private.json',cost);write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def interrupted(sig,frame):
        write(root/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupted)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(train):r['id']=i
    assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    write(root/'COVERAGE.json',dict(train=len(train),val=len(val),patient_overlap=0,old_image_replay=False,old_cache_reuse=False))
    status('loading_encoder')
    import timm
    from safetensors.torch import load_file
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True);encoder.eval().requires_grad_(False).cuda()
    def extract(rows,field):
        out=torch.empty(len(rows),197,768,dtype=torch.float16);start=time.time();prior=cost['encoder_seconds'].get(field,0.)
        for images,ix in DataLoader(Images(rows,cfg),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):tokens=encoder.forward_features(images.cuda())
            assert tokens.shape[1:]==(197,768) and torch.isfinite(tokens).all();out[ix]=tokens.half().cpu();cost[field]+=len(ix)
            torch.cuda.synchronize();cost['encoder_seconds'][field]=prior+time.time()-start;status(field,current_rows=int(ix[-1])+1,total_rows=len(rows))
        return out
    def describe(tokens,mapping):
        out=torch.empty(len(tokens),cfg['descriptor_dimension']);mapping.eval()
        for i in range(0,len(tokens),32):
            with torch.no_grad():out[i:i+32]=mapping(tokens[i:i+32].float().cuda()).cpu()
            cost['descriptor_rows']+=len(out[i:i+32])
        return out
    def head_step(model,x,y,records,arm,opt):
        stats=stack_stats(records,'cuda');opt.zero_grad();loss=objective(model,x,y,stats,arm,cfg)[0];assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);assert torch.isfinite(gn);opt.step();return float(loss.detach())
    # The only legal GPU qualification data are two distinct patients from each base class.
    qr=[]
    for c in cfg['order'][:10]:
        groups=set()
        for r in train:
            if r['label']==c and r['group'] not in groups:qr.append(r);groups.add(r['group'])
            if len(groups)==2:break
        assert len(groups)==2
    qt=extract(qr,'qualification_encodes');qstart=time.time();mapping=EvidenceMap(cfg).cuda();initial=state(mapping)
    qhead=torch.nn.Parameter(torch.randn(10,cfg['descriptor_dimension'],device='cuda'));opt=torch.optim.AdamW(list(mapping.parameters())+[qhead],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    # Match the formal 64 logical/microbatch8 primitive optimizer path.
    opt.zero_grad()
    for i in range(8):
        ids=torch.arange(i*8,i*8+8)%20;x=qt[ids].float().cuda();y=ids.cuda()//2
        loss=primitive_objective(mapping,qhead,x,y,cfg)[0]/8;assert torch.isfinite(loss);loss.backward()
    torch.nn.utils.clip_grad_norm_(list(mapping.parameters())+[qhead],cfg['gradient_clip']);opt.step();cost['qualification_primitive_updates']=1
    delta=max(float((state(mapping)[k]-initial[k]).abs().max()) for k in initial);assert delta>0
    mapping.eval().requires_grad_(False);qf=describe(qt,mapping);prior=base_prior(qf,qr,cfg['order'][:8],cfg)
    math_errors=[];fd_errors=[];head_receipt={};qm={}
    for phase in range(2):
        seen=cfg['order'][:8] if phase==0 else cfg['order'][:10];new=seen if phase==0 else seen[8:];ix=list(range(16)) if phase==0 else list(range(16,20))
        records,proof,sc,check=check_stats(qf[ix],[qr[i] for i in ix],new,prior,cfg)
        if phase==0:
            allrecords=records;template=Classifier(stack_stats(records)['global_mean']);qm={a:copy.deepcopy(template).cuda() for a in ARMS}
        else:
            allrecords+=records
            for model in qm.values():model.expand(stack_stats(records,'cuda')['global_mean'])
        ids=np.arange(64)%len(seen)*2;x=qf[ids].cuda();y=torch.tensor(ids//2,device='cuda')
        for a in ARMS:
            opt=torch.optim.AdamW(qm[a].parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);head_step(qm[a],x,y,allrecords,a,opt)
            cost['qualification_head_updates'][a]+=1;cost['qualification_head_rows'][a]+=len(x)
    # Saved model/map reloading and independent full NumPy attention + density/objective.
    save(root/'qualification.private.pt',dict(mapping=state(mapping),models={a:state(m) for a,m in qm.items()},records=allrecords))
    pack=torch.load(root/'qualification.private.pt',map_location='cpu',weights_only=False)
    dm=EvidenceMap(cfg).double();dm.load_state_dict({k:v.double() for k,v in pack['mapping'].items()})
    with torch.no_grad():df=dm(qt.double())
    ref=numpy_map(qt.numpy(),arrays(pack['mapping']),cfg);map_error=float(abs(df.numpy()-ref).max());assert map_error<=cfg['score_absolute_tolerance']
    for a in ARMS:
        model=Classifier(pack['models'][a]['head']).double();model.load_state_dict({k:v.double() for k,v in pack['models'][a].items()});xx=df.repeat(4,1)[:64];yy=torch.arange(64)%20//2;stats=stack_stats(allrecords,dtype=torch.float64)
        with torch.no_grad():ss=model(xx,stats,a,cfg).numpy();tl=float(objective(model,xx,yy,stats,a,cfg)[0])
        ns=arrays(pack['models'][a]);nr=numpy_score(xx.numpy(),ns,allrecords,a,cfg);err=float(abs(ss-nr).max());oe=abs(tl-numpy_loss(xx.numpy(),yy.numpy(),ns,allrecords,a,cfg));assert max(err,oe)<=cfg['score_absolute_tolerance']
        # Check real gradients against independent NumPy differences, including mixture routing.
        model.zero_grad();objective(model,xx[:8],yy[:8],stats,a,cfg)[0].backward()
        local_fd=[]
        for name,index in [('head',(0,0)),('mixing',(0,))]:
            plus={k:v.copy() for k,v in ns.items()};minus={k:v.copy() for k,v in ns.items()};plus[name][index]+=1e-5;minus[name][index]-=1e-5
            fd=(numpy_loss(xx.numpy()[:8],yy.numpy()[:8],plus,allrecords,a,cfg)-numpy_loss(xx.numpy()[:8],yy.numpy()[:8],minus,allrecords,a,cfg))/2e-5
            local_fd.append(abs(fd-float(dict(model.named_parameters())[name].grad[index])))
        assert max(local_fd)<2e-5
        head_receipt[a]=dict(score_error=err,objective_error=oe,finite_difference_error=max(local_fd),engineering_updates=2,GPU_forward_backward_rows=128,FP64_rows=128,numpy_score_and_objective_rows=128,finite_difference_numpy_objective_rows=32)
    # Empty modes/singletons retain finite floor and no fabricated positive mass.
    assert all(np.isfinite(r['calibrated']).all() and abs(r['weights'].sum()-1)<1e-12 for r in allrecords)
    write(root/'QUALIFICATION.json',dict(state='PASS',real_base_encodes=20,full_GPU_encoder=True,primitive_GPU64_micro8_update=True,primitive_actual_parameter_delta=delta,primitive_forward_backward_rows=64,head_arrival_memory_modes=True,save_load=True,map_FP64_numpy_error=map_error,arms=head_receipt,discarded_optimizer_updates=7,validation_encodes=0,post_encoding_seconds=time.time()-qstart))
    del mapping,qhead,qt,qf,qm,pack,dm,df,xx,yy,x,y,model,opt,loss;torch.cuda.empty_cache();status('qualified_before_bulk')
    torch.manual_seed(cfg['seed']);mapping=EvidenceMap(cfg).cuda();memory=torch.empty(0,cfg['descriptor_dimension']);memory_rows=[];allrecords=[];allproof=[];prior=None
    for task in range(6):
        stage_task=task;seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];old=sum(cfg['task_sizes'][:task]);new=seen[old:];rows=[r for r in train if r['label'] in new]
        tokens=extract(rows,'science_train_encodes')
        if task==0:
            head=torch.nn.Parameter(torch.randn(10,cfg['descriptor_dimension'],device='cuda'));params=list(mapping.parameters())+[head];opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);pool=pools(rows);rng=np.random.default_rng(cfg['seed'])
            for step in range(cfg['primitive_updates']):
                ids=draw(pool,new,rng);opt.zero_grad();lv=0.
                for j in range(0,64,8):
                    ii=ids[j:j+8];x=tokens[ii].float().cuda();y=torch.tensor([seen.index(rows[k]['label']) for k in ii],device='cuda')
                    loss=primitive_objective(mapping,head,x,y,cfg)[0]/8;assert torch.isfinite(loss);loss.backward();lv+=float(loss.detach());cost['primitive_forward_backward_rows']+=len(ii)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);opt.step();cost['primitive_updates']+=1
                if (step+1)%50==0:curves.append(dict(phase='primitive',step=step+1,loss=lv));write(root/'TRAINING_CURVES.json',curves);status('base_evidence_training',step=step+1)
            mapping.eval().requires_grad_(False);cost['mapping_frozen_after_base']=True;save(root/'evidence_map.private.pt',state(mapping));write(root/'MAP_SEAL.json',dict(at=time.time(),science_train_encodes=cost['science_train_encodes'],validation_encodes=0,base_only=True))
            del head,params,opt,loss,x,y
        features=describe(tokens,mapping);del tokens;torch.cuda.empty_cache()
        if task==0:prior=base_prior(features,rows,new,cfg)
        records,proof,sc,check=check_stats(features,rows,new,prior,cfg);allrecords+=records;allproof+=proof;cost['statistics_patient_rows']+=sc['patient_rows'];cost['online_independent_statistics_patient_rows']+=sc['patient_rows'];cost['kmeans_distance_pairs']+=sc['kmeans_distance_pairs']
        mm=stack_stats(records,'cuda')['global_mean']
        if task==0:
            template=Classifier(mm);models={a:copy.deepcopy(template).cuda() for a in ARMS};del template
        else:
            for model in models.values():model.expand(mm)
        # Old finite memory plus current full descriptors: no duplicate current-memory exposure.
        source=torch.cat([memory,features]);source_rows=memory_rows+rows;pool=pools(source_rows);rng=np.random.default_rng(cfg['seed']+task*19+100)
        opts={a:torch.optim.AdamW(models[a].parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for a in ARMS};steps=cfg['base_updates'] if task==0 else cfg['later_updates'];loop=time.time();class_counts=Counter()
        for step in range(steps):
            ids=draw(pool,seen,rng);x=source[ids].cuda();y=torch.tensor([seen.index(source_rows[i]['label']) for i in ids],device='cuda');class_counts.update(str(source_rows[i]['label']) for i in ids)
            losses={}
            for a in ARMS:
                losses[a]=head_step(models[a],x,y,allrecords,a,opts[a]);cost['head_updates'][a]+=1;cost['head_training_rows'][a]+=64
            if (step+1)%100==0:curves.append(dict(phase='head',task=task,step=step+1,loss=losses));write(root/'TRAINING_CURVES.json',curves);status('actual_stage_training',task=task,step=step+1)
        # Probe actual final batch at sealed FP32 parameters using FP64 deployment, before validation.
        probe=dict(x=x.detach().cpu(),y=y.detach().cpu(),ids=[source_rows[i]['id'] for i in ids],losses={})
        for a in ARMS:
            model=Classifier(models[a].head.detach().cpu()).double();model.load_state_dict({k:v.double() for k,v in state(models[a]).items()})
            with torch.no_grad():probe['losses'][a]=float(objective(model,probe['x'].double(),probe['y'],stack_stats(allrecords,dtype=torch.float64),a,cfg)[0])
            cost['probe_rows'][a]+=64
        chosen=memory_indices(rows,cfg['memory_per_class'],cfg['seed']);memory=torch.cat([memory,features[chosen]]);memory_rows+= [rows[i] for i in chosen]
        save(root/f'stage_{task}.private.pt',dict(seen=seen,mapping=state(mapping),models={a:state(m) for a,m in models.items()},memory=memory,memory_rows=memory_rows,records=allrecords,proofs=allproof,prior=prior,probe=probe,online_independent_stats=check,sampling=dict(class_counts=dict(class_counts),paired_draws=True),cost=copy.deepcopy(cost)))
        seals.append(dict(task=task,at=time.time(),seen=seen,validation_encodes=0,memory_rows=len(memory_rows),updates=copy.deepcopy(cost['head_updates']),science_train_encodes=cost['science_train_encodes'],loop_seconds=time.time()-loop));write(root/'STAGE_SEALS.json',seals)
        del features,source,source_rows,opts,x,y,probe,model;status('stage_sealed',task=task)
    cost['memory_rows_final']=len(memory_rows);write(root/'FIT_SEAL.json',dict(at=time.time(),models=18,validation_encodes=0,science_train_encodes=cost['science_train_encodes']))
    write(root/'QUERY_OPEN.private.json',dict(at=time.time()));qt=extract(val,'validation_encodes');dm=EvidenceMap(cfg).double();dm.load_state_dict({k:v.double() for k,v in state(mapping).items()})
    qf=[]
    for i in range(0,len(qt),8):
        with torch.no_grad():qf.append(dm(qt[i:i+8].double()))
    qf=torch.cat(qf);save(root/'query.private.pt',dict(tokens=qt,features=qf,labels=torch.tensor([r['label'] for r in val])))
    for task in range(6):
        pack=torch.load(root/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);mask=torch.tensor([r['label'] in pack['seen'] for r in val]);stats=stack_stats(pack['records'],dtype=torch.float64)
        for a in ARMS:
            model=Classifier(pack['models'][a]['head']).double();model.load_state_dict({k:v.double() for k,v in pack['models'][a].items()})
            with torch.no_grad():score=model(qf[mask],stats,a,cfg)
            save(root/f'scores_{task}_{a}.private.pt',score);cost['validation_rows'][a]+=len(score)
    write(root/'COSTS.json',cost);status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception as exc:
        root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True);write(root/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),trace=traceback.format_exc(),at=time.time()));raise
