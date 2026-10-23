"""One original-clock fit with mandatory real numerical/resource admission before bulk."""
import json,os,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import ARMS,ArrivalFrame
from data import read_rows,Images
from foundation import write,save,prefix,build_backbone
from training import selected,weights,affine,optimizer,update,counter
from inference import native_features,body_state,scores,restore,frozen_parameter_delta
from checks import make_models,qualify


def main(cfg,cost):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction((7*1024**3)/torch.cuda.get_device_properties(0).total_memory)
    p=Path(cfg['output']);p.mkdir(exist_ok=False);events=[]
    def status(phase,**extra):
        cost['elapsed_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.json',cost);write(p/'status.private.json',dict(phase=phase,updated=time.time(),**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def record(value):events.append(value);write(p/'NUMERIC_GATES.json',events)
    def stop(sig,frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,stop)
    status('checking_existing_small_qualification');small=json.loads(Path(cfg['small_math_receipt']).read_text());assert small['state']=='PASS' and small['NIH_images']==small['GPU_calls']==small['optimizer_updates']==0
    # The completed small qualification is not run again.
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    for i,r in enumerate(train):r['id']=i
    write(p/'COVERAGE.json',dict(train=len(train),val=len(val),patient_overlap=0))
    b=build_backbone(cfg,'cuda');cpu=build_backbone(cfg,'cpu');original={k:v.detach().cpu().clone() for k,v in b.state_dict().items()}
    image_proof={}
    def extract(rows,field,proof=False):
        out=torch.empty((len(rows),197,768),dtype=torch.float16);embeddings=torch.empty_like(out) if field=='validation_encodes' else None
        wanted=set(selected(rows,2,cfg['seed'])) if proof else set();saved={}
        for images,ix in DataLoader(Images(rows,cfg),batch_size=cfg['encoder_batch'],num_workers=0,shuffle=False):
            tick=time.time()
            with torch.no_grad():raw_embedding=prefix(b,images.cuda().double());emb=raw_embedding.half();lower=b.blocks[:8](emb.double())
            finite=bool(torch.isfinite(lower).all());write(p/'LOWER_FINITE_GATE.json',dict(check='finite_lower8',phase=field,through=int(ix[-1])+1,finite=finite));assert finite
            out[ix]=lower.half().cpu()
            if embeddings is not None:embeddings[ix]=emb.cpu()
            if field=='qualification_encodes' and not image_proof:
                image_proof.update(images=images[:2].clone(),raw_embedding=raw_embedding[:2].cpu())
            for j,i in enumerate(ix.tolist()):
                if i in wanted:saved[i]=dict(embedding=emb[j].cpu().clone(),lower=lower[j].cpu().clone(),id=rows[i]['id'],label=rows[i]['label'])
            cost[field]+=len(ix);torch.cuda.synchronize();cost['encoding_seconds'][field]=cost['encoding_seconds'].get(field,0.)+time.time()-tick
            status(field,encoded=int(ix[-1])+1,total_current=len(rows))
        return out,embeddings,saved
    def means(rows,z,classes):
        result=[]
        for c in classes:
            ii=[i for i,r in enumerate(rows) if r['label']==c];w,_=weights([rows[i] for i in ii]);result.append((torch.from_numpy(w)[:,None]*z[ii]).sum(0))
        return torch.stack(result)
    qrows=[]
    for c in cfg['order'][:10]:
        rr=[r for r in train if r['label']==c];qrows.extend(rr[i] for i in selected(rr,2,cfg['seed']))
    qt,_,qp=extract(qrows,'qualification_encodes',True);qz=native_features(b,qt,cfg['microbatch']);cost['native_initialization_rows']+=len(qt)
    qm=means(qrows,qz,cfg['order'][:10]);status('real_GPU_qualification')
    proof=[qp[i] for i in range(len(qrows))];save(p/'qualification_inputs.private.pt',dict(tokens=qt,proof=proof,image_proof=image_proof,means=qm,native_features=qz))
    real=qualify(b,cpu,qt,qm,cfg['order'][:10],proof,image_proof,qz,cfg,p,cost,status)
    # Empirical NFS block accounting on actual saved >=64KiB files, before any bulk.
    factors=[];used=0
    for loc in [p.parent,Path(cfg['neutral_entry'])]:
        for f in loc.rglob('*'):
            if f.is_file():st=f.stat();used+=max(st.st_size,st.st_blocks*512)
            else:continue
            if st.st_size>=65536:factors.append(max(st.st_size,st.st_blocks*512)/st.st_size)
    factor=max([1.]+factors)*1.1
    forecast_bytes=used+factor*(900000000+192000000)
    forecast=cfg['feasibility_factor']*(cfg['base_updates']+5*cfg['later_updates'])*real['paired_step_seconds']+cfg['feasibility_reserve_seconds'];remaining=cfg['hard_deadline_epoch']-time.time()
    admission=dict(state='PENDING',paired_stage_step_seconds=real['paired_stage_step_seconds'],paired_step_seconds=real['paired_step_seconds'],time_forecast_seconds=forecast,original_clock_remaining_seconds=remaining,physical_accounting_factor_with_margin=factor,physical_forecast_bytes=forecast_bytes,actual_existing_new_root_entry_bytes=used,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),time_reserve_scope='All unmeasured bulk encoding, initialization, moment fitting, 24 sealed scoring and independent full acceptance; not ETA.')
    write(p/'QUALIFICATION.json',admission)
    assert forecast<remaining-60 and forecast_bytes<cfg['storage_stop_threshold_bytes'] and torch.cuda.max_memory_reserved()<cfg['gpu_reserved_budget_bytes']
    admission['state']='PASS';write(p/'QUALIFICATION.json',admission)
    del qt,qz,qm,qrows,qp,proof,image_proof;torch.cuda.empty_cache()
    models=make_models(b,cfg);frame=ArrivalFrame(record);memory=torch.empty((0,197,768),dtype=torch.float16);memory_rows=[];gram=torch.zeros(1537,1537,dtype=torch.float64);mus=[];seals=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];rows=[r for r in train if r['label'] in new]
        tokens,_,proof=extract(rows,'science_train_encodes',True);z=native_features(b,tokens,cfg['microbatch']);cost['native_initialization_rows']+=len(tokens);mm=means(rows,z,new)
        for i,v in proof.items():v['native_descriptor']=z[i].clone()
        old_frame=None if frame.rows is None else frame.rows.clone();frame.append(mm,new);fr=frame.receipt();fr['old_delta']=0. if old_frame is None else float(abs(frame.rows[:len(old_frame)]-old_frame).max());record(dict(check='scientific_frame',task=task,**fr));assert fr['Gram_error']<1e-10 and fr['old_delta']==0
        arrived=torch.zeros_like(gram)
        for c in new:
            ii=[i for i,r in enumerate(rows) if r['label']==c];rr=[rows[i] for i in ii];w,n=weights(rr);phi=affine(z[ii]);gg=phi.T@(torch.from_numpy(w)[:,None]*phi);mu=(torch.from_numpy(w)[:,None]*phi).sum(0)
            err=float(abs(phi.numpy().T@(w[:,None]*phi.numpy())-gg.numpy()).max());record(dict(check='current_full_native_moment',label=c,error=err));assert err<1e-9
            gram+=gg;arrived+=gg;mus.append(mu);pick=selected(rr,cfg['memory_per_class'],cfg['seed']);save(p/f'memory_{c}.private.pt',dict(tokens=tokens[[ii[i] for i in pick]],ids=[rr[i]['id'] for i in pick]))
            save(p/f'proof_{c}.private.pt',dict(label=c,images=len(rr),patients=n,gram_error=err,proof=[v for v in proof.values() if v['label']==c]))
        mat=gram/len(seen)+cfg['ridge']*torch.eye(1537,dtype=torch.float64);rhs=torch.stack(mus,1)/len(seen);head=torch.linalg.solve(mat,rhs);error=float(abs(mat@head-rhs).max());record(dict(check='native_normal_system',task=task,error=error));assert error<1e-9;cost['native_solves']+=1
        for m in models.values():m.arrive(mm,frame.rows,new)
        opts={a:optimizer(m,cfg) for a,m in models.items()};source=memory_rows+rows;pools={}
        for i,r in enumerate(source):pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
        rng=np.random.default_rng(cfg['seed']+1000+task);steps=cfg['base_updates'] if task==0 else cfg['later_updates'];draws=[]
        for step in range(1,steps+1):
            ix=[]
            for c in rng.choice(seen,8,replace=False):
                groups=list(pools[c])
                for _ in range(8):
                    values=pools[c][groups[int(rng.integers(len(groups)))]];ix.append(values[int(rng.integers(len(values)))])
            x=torch.stack([memory[i] if i<len(memory_rows) else tokens[i-len(memory_rows)] for i in ix]);y=torch.tensor([seen.index(source[i]['label']) for i in ix]);draws.append([source[i]['id'] for i in ix])
            if step==steps:before={a:m.adapter_state() for a,m in models.items()}
            last={a:update(m,opts[a],x,y,cfg,cost['arms'][a],record) for a,m in models.items()}
            if step==1 or step%100==0 or step==steps:status('training',task=task,step=step,total=steps,details=last)
        adapters={a:m.adapter_state() for a,m in models.items()}
        for a,m in models.items():
            delta=frozen_parameter_delta(m,b);record(dict(check='frozen_original_parameters',task=task,arm=a,max_delta=delta));assert delta==0
        pick=selected(rows,cfg['memory_per_class'],cfg['seed']);memory=torch.cat([memory,tokens[pick]]);memory_rows.extend(rows[i] for i in pick)
        save(p/f'stage_{task}.private.pt',dict(seen=seen,adapters=adapters,frame=frame.rows.clone(),means=mm,native_head=head,native_mat=mat,native_rhs=rhs,arrived_gram=arrived,memory_ids=[r['id'] for r in memory_rows],draws=torch.tensor(draws,dtype=torch.int32),probe=dict(tokens=x,labels=y,before=before,losses={a:last[a]['objective'] for a in ARMS[1:]})))
        seals.append(dict(task=task,at=time.time(),models=4,updates={a:cost['arms'][a]['updates'] for a in ARMS[1:]},memory_images=len(memory)));write(p/'STAGE_SEALS.json',seals)
        del tokens,z,rows,source,pools,opts,x,y,before,adapters,draws,proof;torch.cuda.empty_cache()
    delta=max(float(abs(v-b.state_dict()[k].cpu()).max()) for k,v in original.items());record(dict(check='entire_original_backbone_frozen',max_delta=delta));assert delta==0 and cost['science_train_encodes']==68058
    write(p/'FIT_SEAL.json',dict(models=24,at=time.time(),backbone_frozen_exact=True));del models;torch.cuda.empty_cache();write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
    query,emb,_=extract(val,'validation_encodes');labels=torch.tensor([r['label'] for r in val]);save(p/'query.private.pt',dict(tokens=query,embedding=emb,labels=labels));del emb
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);seen=pack['seen'];qt=query[torch.isin(labels,torch.tensor(seen))];values={'NATIVE_RIDGE':affine(native_features(b,qt,cfg['microbatch']))@pack['native_head']}
        for a in ARMS[1:]:
            m=restore(b,a,pack['adapters'][a],seen);values[a]=scores(m,qt,cfg['microbatch']);cost['validation_model_rows'][a]+=len(qt);del m;torch.cuda.empty_cache()
        cost['validation_model_rows']['NATIVE_RIDGE']+=len(qt);save(p/f'scores_{task}.private.pt',values);status('scored',task=task)
    status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());cost=dict(qualification_encodes=0,qualification_full_chain_rows=0,qualification_zero_B_rows=0,qualification_native_probe_rows=0,qualification_CPU_score_rows=0,qualification_NumPy_score_rows=0,qualification_GPU_score_rows=0,science_train_encodes=0,validation_encodes=0,qualification_arms={a:counter() for a in ARMS[1:]},arms={a:counter() for a in ARMS[1:]},native_initialization_rows=0,native_solves=0,encoding_seconds={},validation_model_rows={a:0 for a in ARMS})
    try:main(cfg,cost)
    except BaseException:
        p=Path(cfg['output']);p.mkdir(exist_ok=True);write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc(),cost=cost,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None));raise
    finally:
        p=Path(cfg['output'])
        if p.exists():write(p/'COSTS.json',cost)
