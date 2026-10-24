"""Base-only representation fit, immutable subsequent maps and 24 sealed systems."""
import json,os,time,signal,traceback,copy,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import NativeUpper,affine_descriptor
from data import read_rows,UnnormalizedImages,patient_draw
from geometry import augment,view_receipt
from foundation import write,save,prefix,build_backbone
from training import ARMS,Learner,optimizer,update,counter,canonical_outputs
from inference import body_state
from moments import add_current,solve


def main(cfg,cost):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=Path(cfg['output']);p.mkdir(exist_ok=False);events=[]
    def record(value):
        def safe(x):
            if isinstance(x,float) and not math.isfinite(x):return str(x)
            if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [safe(v) for v in x]
            return x
        value=safe(value)
        events.append(value)
        with (p/'NUMERIC_GATES.private.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    def status(phase,**extra):
        cost['elapsed_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.json',cost);write(p/'NUMERIC_GATES.json',events);write(p/'status.private.json',dict(phase=phase,updated=time.time(),**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(SystemExit(143)))
    small=json.loads(Path(cfg['small_math_receipt']).read_text());assert small['state']=='PASS_REFERENCE_ONLY'
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    for i,r in enumerate(train):r['id']=i
    write(p/'COVERAGE.json',dict(train=len(train),val=len(val),patient_overlap=0))
    backbone=build_backbone(cfg,'cuda');native=NativeUpper(backbone).requires_grad_(False)
    base=[r for r in train if r['label'] in cfg['order'][:10]]
    def encode(images,field,capture=False):
        out=[];embeddings=[];lowers=[]
        with torch.no_grad():
            for i in range(0,len(images),cfg['microbatch']):
                x=images[i:i+cfg['microbatch']].cuda().double();raw=prefix(backbone,x);emb=raw.half();lower=backbone.blocks[:8](emb.double())
                record(dict(check='finite_lower',field=field,through=i+len(x),PASS=bool(torch.isfinite(lower).all())))
                assert bool(torch.isfinite(lower).all());out.append(lower.half().cpu());
                if capture:embeddings.append(emb.cpu());lowers.append(lower.cpu())
                cost[field]=cost.get(field,0)+len(x)
        return (torch.cat(out),torch.cat(embeddings),torch.cat(lowers)) if capture else torch.cat(out)
    def paired(rows,seed,field):
        ids,labels=patient_draw(rows,cfg['order'][:10],seed)
        dataset=UnnormalizedImages(rows,cfg);images=torch.stack([dataset[i][0] for i in ids]);receipt=view_receipt(seed+1,len(ids));views=augment(images,receipt)
        tokens=encode(torch.cat(views),field)
        return tokens,torch.tensor([cfg['order'][:10].index(c) for c in labels]),receipt,[rows[i]['id'] for i in ids]
    def describe(rows,uppers,field,keep_query=False):
        tables={a:torch.empty(len(rows),1537,dtype=torch.float64) for a in uppers};proof={};query_emb=[];query_lower=[]
        wanted=set()
        for c in sorted({r['label'] for r in rows}):
            candidates=sorted((i for i,r in enumerate(rows) if r['label']==c),key=lambda i:(rows[i]['group'],rows[i]['relative_path']))
            patients=set()
            for i in candidates:
                if rows[i]['group'] not in patients:wanted.add(i);patients.add(rows[i]['group'])
                if len(patients)==2:break
        with torch.no_grad():
            for images,ix in DataLoader(UnnormalizedImages(rows,cfg),batch_size=cfg['microbatch'],num_workers=0,shuffle=False):
                raw=prefix(backbone,((images-.5)/.5).cuda().double());emb=raw.half();lower=backbone.blocks[:8](emb.double());stored=lower.half()
                finite=bool(torch.isfinite(lower).all());record(dict(check='finite_canonical_lower',field=field,through=int(ix[-1])+1,PASS=finite));assert finite
                cost[field]=cost.get(field,0)+len(ix)
                for a,upper in uppers.items():
                    result=upper(stored.to(next(upper.parameters())));tables[a][ix]=affine_descriptor(result).cpu();cost['descriptor_rows'][a]+=len(ix)
                if keep_query:query_emb.append(emb.cpu());query_lower.append(stored.cpu())
                for j,i in enumerate(ix.tolist()):
                    if i in wanted:proof[rows[i]['id']]=dict(embedding=emb[j].cpu().clone(),lower=stored[j].cpu().clone(),lower_FP64=lower[j].cpu().clone(),label=rows[i]['label'],features={a:tables[a][i].clone() for a in uppers})
                if int(ix[-1])%800<cfg['microbatch']:status(field,through=int(ix[-1])+1,total=len(rows))
        query=None if not keep_query else dict(embedding=torch.cat(query_emb),lower=torch.cat(query_lower),labels=torch.tensor([r['label'] for r in rows]))
        return tables,proof,query
    from checks import qualify
    status('real_qualification');qualification=qualify(backbone,native,base,cfg,p,cost,record,status,encode,paired)
    factors=[];used=0
    for root in [p.parent,Path(cfg['neutral_entry'])]:
        for f in root.rglob('*'):
            if f.is_file():
                st=f.stat();used+=max(st.st_size,st.st_blocks*512)
                if st.st_size>=65536:factors.append(max(st.st_size,st.st_blocks*512)/st.st_size)
    factor=1.1*max([1.]+factors)
    # 3 final dense upper states, 24 normal systems plus arrival proofs and query/input/projection proofs.
    upper_parameters=sum(v.numel() for block in backbone.blocks[8:] for v in block.parameters())+sum(v.numel() for v in backbone.norm.parameters())
    logical_future=3*upper_parameters*8+24*(1537**2*8+(1537*1538//2)*8+1537*20*16)+300*197*768*4+40*(197*768*12+1537*8*4)+64*197*768*2+24000000+64000000
    forecast=cfg['feasibility_factor']*cfg['base_updates']*qualification['paired_step_seconds']+cfg['feasibility_reserve_seconds'];remaining=cfg['hard_deadline_epoch']-time.time()
    admission=dict(state='PENDING',paired_step_seconds=qualification['paired_step_seconds'],time_forecast_seconds=forecast,remaining_seconds=remaining,physical_factor_with_margin=factor,physical_forecast_bytes=used+factor*logical_future,existing_new_bytes=used,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),ETA=False)
    write(p/'QUALIFICATION.json',admission)
    assert forecast<remaining-60 and admission['physical_forecast_bytes']<cfg['storage_stop_threshold_bytes'] and torch.cuda.max_memory_reserved()<cfg['gpu_reserved_budget_bytes']
    admission['state']='PASS';write(p/'QUALIFICATION.json',admission);status('science_initial_native_base')
    initial,proof,_=describe(base,{'NATIVE_RIDGE':native},'native_base_initial_encodes')
    grams={a:torch.zeros(1537,1537,dtype=torch.float64) for a in ARMS};means={a:[] for a in ARMS}
    add_current(base,initial['NATIVE_RIDGE'],cfg['order'][:10],grams['NATIVE_RIDGE'],means['NATIVE_RIDGE'],record)
    base_means=torch.stack([x[:1536]*np.sqrt(2) for x in means['NATIVE_RIDGE']]);del initial
    models={}
    for a in ARMS[1:]:
        torch.manual_seed(cfg['seed']);models[a]=Learner(NativeUpper(backbone),a,base_means).cuda().double()
    opts={a:optimizer(m,cfg) for a,m in models.items()};draws=[];last_probe=None
    for step in range(1,cfg['base_updates']+1):
        tokens,labels,receipt,ids=paired(base,cfg['seed']+100000+step,'base_augmented_encodes');draws.append(ids)
        detail={a:update(m,opts[a],tokens,labels,receipt,cfg,cost['arms'][a],lambda v:record(dict(arm=a,step=step,**v))) for a,m in models.items()}
        if step==cfg['base_updates']:
            last_probe=dict(tokens=tokens,labels=labels,receipt=receipt,ids=ids,post_update_objectives={})
            for a,m in models.items():
                out=canonical_outputs(m,tokens,cfg['microbatch']);last_probe['post_update_objectives'][a]=float(m.objective(out,labels,receipt));cost['endpoint_probe_rows'][a]+=len(tokens)
                save(p/f'probe_head_{a}.private.pt',dict(projectors=m.projectors.state_dict(),head=m.head))
        if step==1 or step%50==0 or step==cfg['base_updates']:status('base_training',step=step,total=cfg['base_updates'],detail=detail)
    save(p/'training_endpoint.private.pt',last_probe);del opts,tokens,last_probe
    uppers={'NATIVE_RIDGE':native}
    for a,m in models.items():
        m.upper.requires_grad_(False);save(p/f'map_{a}.private.pt',m.upper.state_dict());uppers[a]=m.upper
    original=build_backbone(cfg,'cpu');delta=max(float((v.detach().cpu()-original.state_dict()[k]).abs().max()) for k,v in backbone.state_dict().items());record(dict(check='original_backbone_fixed',max_delta=delta,PASS=delta==0));assert delta==0;del original
    write(p/'MAP_SEAL.json',dict(at=time.time(),base_updates=cfg['base_updates'],labels=cfg['order'][:10],maps=4,all_later_optimization_forbidden=True));del models
    stage_seals=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];rows=base if task==0 else [r for r in train if r['label'] in new]
        current_uppers={a:u for a,u in uppers.items() if task>0 or a!='NATIVE_RIDGE'}
        table,current_proof,_=describe(rows,current_uppers,'adapted_base_encodes' if task==0 else 'later_canonical_encodes')
        for identity,value in current_proof.items():
            if identity in proof:value['features']={**proof[identity]['features'],**value['features']}
            proof[identity]=value
        save(p/'training_proof.private.pt',proof)
        heads={};systems={}
        for a in ARMS:
            before=torch.zeros_like(grams[a]) if task==0 and a=='NATIVE_RIDGE' else grams[a].clone()
            if task>0 or a!='NATIVE_RIDGE':add_current(rows,table[a],new,grams[a],means[a],lambda v:record(dict(arm=a,task=task,**v)))
            h,mat,rhs=solve(grams[a],means[a],cfg['ridge']);residual=float(abs(mat@h-rhs).max());record(dict(check='full_normal_system',arm=a,task=task,residual=residual,PASS=residual<=1e-9));assert residual<=1e-9
            heads[a]=h;systems[a]=dict(matrix=mat,rhs=rhs,arrived_upper_triangle=(grams[a]-before)[torch.triu_indices(1537,1537)[0],torch.triu_indices(1537,1537)[1]]);cost['ridge_solves'][a]+=1
            if a!='NATIVE_RIDGE':
                saved=torch.load(p/f'map_{a}.private.pt',map_location='cpu',weights_only=False)
                delta=max(float((v.detach().cpu()-saved[k]).abs().max()) for k,v in uppers[a].state_dict().items());record(dict(check='map_frozen_after_base',arm=a,task=task,max_delta=delta,PASS=delta==0));assert delta==0
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=heads,systems=systems));stage_seals.append(dict(task=task,at=time.time(),models=4));write(p/'STAGE_SEALS.json',stage_seals);del table,systems,heads
        status('sealed',task=task)
    write(p/'FIT_SEAL.json',dict(at=time.time(),models=24));write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
    table,_,query=describe(val,uppers,'validation_encodes',True);save(p/'query.private.pt',query);save(p/'query_features.private.pt',table)
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);ix=torch.isin(query['labels'],torch.tensor(pack['seen']));scores={a:table[a][ix]@pack['heads'][a] for a in ARMS}
        save(p/f'scores_{task}.private.pt',scores)
        for a in ARMS:cost['score_rows'][a]+=int(ix.sum())
        status('scored',task=task)
    status('COMPLETE_PENDING_VERIFY')


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());cost=dict(arms={a:counter() for a in ARMS[1:]},qualification_arms={a:counter() for a in ARMS[1:]},descriptor_rows={a:0 for a in ARMS},endpoint_probe_rows={a:0 for a in ARMS[1:]},ridge_solves={a:0 for a in ARMS},score_rows={a:0 for a in ARMS})
    try:main(cfg,cost)
    except BaseException:
        p=Path(cfg['output']);p.mkdir(exist_ok=True);write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc(),cost=cost));raise
    finally:
        p=Path(cfg['output'])
        if p.exists():write(p/'COSTS.json',cost)
