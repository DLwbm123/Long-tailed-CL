"""Frozen visual-word base fit, immutable full maps, and24sealed arrival systems."""
import json,os,time,signal,traceback,math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import ARMS,Model,descriptor,shuffle_receipt,corrupt
from word_data import patient_means,discover,word_draw,render
from data import read_rows,UnnormalizedImages,patient_draw
from foundation import build_backbone,save,write
from training import optimizer,update,counter
from moments import add_current,solve

def main(cfg,cost):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=Path(cfg['output']);p.mkdir(exist_ok=False);events=[]
    def record(value):
        def safe(v):
            if isinstance(v,float) and not math.isfinite(v):return str(v)
            if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
            if isinstance(v,list):return [safe(x) for x in v]
            return v
        value=safe(value);events.append(value)
        with (p/'NUMERIC_GATES.private.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
    def status(phase,**extra):
        cost['elapsed_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved();write(p/'COSTS.json',cost);write(p/'NUMERIC_GATES.json',events);write(p/'status.private.json',dict(phase=phase,updated=time.time(),**extra));assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(SystemExit(143)))
    small=json.loads(Path(cfg['small_math_receipt']).read_text());assert small['state']=='PASS'
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val};assert [r['id'] for r in val]==list(range(300))
    write(p/'COVERAGE.json',dict(train=len(train),val=len(val),patient_overlap=0,val_id_schema_checked_without_pixels=True))
    base=[r for r in train if r['label'] in cfg['order'][:10]];backbone=build_backbone(cfg,'cpu')
    from checks import qualify
    status('real_qualification');qualification=qualify(backbone,base,cfg,p,cost,record,status)
    # Calibrate a representative physical file; retain it as budget evidence.
    save(p/'physical_calibration.private.pt',torch.linspace(-1,1,1024**2,dtype=torch.float64).reshape(1024,1024));f=p/'physical_calibration.private.pt';s=f.stat();factor=1.1*max(1.,s.st_blocks*512/s.st_size);used=0
    for root in [p.parent,Path(cfg['neutral_entry'])]:
        for f in root.rglob('*'):
            if f.is_file():s=f.stat();used+=max(s.st_size,s.st_blocks*512)
    modelbytes=sum(x.numel()*x.element_size() for x in backbone.state_dict().values());patient_count=len({r['group'] for r in base});logical=patient_count*1536*8+3*modelbytes+24*(1537**2*8+1537*1538//2*8+1537*20*16)+(300+40)*3*224*224*4+16*3*224*224*4+3*10*1536*8+100000000
    window_used=0
    for root in cfg['window_accounting_roots']:
        root=Path(root)
        if root.exists():
            for f in root.rglob('*'):
                if f.is_file():s=f.stat();window_used+=max(s.st_size,s.st_blocks*512)
    remaining=cfg['hard_deadline_epoch']-time.time();forecast=cfg['feasibility_factor']*cfg['base_updates']*qualification['paired_step_seconds']+cfg['feasibility_reserve_seconds'];admission=dict(state='PENDING',paired_step_seconds=qualification['paired_step_seconds'],time_forecast_seconds=forecast,remaining_seconds=remaining,physical_factor_with_margin=factor,existing_bytes=used,physical_forecast_bytes=used+factor*logical,window_existing_bytes=window_used,window_forecast_bytes=window_used+factor*logical,ETA=False);write(p/'QUALIFICATION.json',admission);assert forecast<remaining-60 and admission['physical_forecast_bytes']<cfg['storage_stop_threshold_bytes'] and admission['window_forecast_bytes']<cfg['window_storage_stop_bytes'];admission['state']='PASS';write(p/'QUALIFICATION.json',admission)
    def describe(rows,maps,field,keep_query=False):
        tables={a:torch.empty(len(rows),1537,dtype=torch.float64) for a in maps};proof={};wanted=set();query=[]
        for c in sorted({r['label'] for r in rows}):
            patients=set()
            for i in sorted([i for i,r in enumerate(rows) if r['label']==c],key=lambda i:(rows[i]['group'],rows[i]['relative_path'])):
                if rows[i]['group'] not in patients:wanted.add(i);patients.add(rows[i]['group'])
                if len(patients)==2:break
        with torch.no_grad():
            for raw,ix in DataLoader(UnnormalizedImages(rows,cfg),batch_size=cfg['microbatch'],shuffle=False,num_workers=0):
                x=raw.cuda().double()
                for a,b in maps.items():
                    z=b.forward_features((x-.5)/.5);finite=bool(torch.isfinite(z).all());record(dict(check='finite_canonical',arm=a,field=field,through=int(ix[-1])+1,PASS=finite));assert finite;tables[a][ix]=descriptor(z).cpu();cost['descriptor_rows'][a]+=len(ix)
                cost[field]=cost.get(field,0)+len(ix)
                if keep_query:query.append(raw.clone())
                for j,i in enumerate(ix.tolist()):
                    if i in wanted:proof[rows[i]['id']]=dict(raw=raw[j].clone(),label=rows[i]['label'],features={a:tables[a][i].clone() for a in maps})
                if (int(ix[-1])+1)%800==0 or int(ix[-1])+1==len(rows):status(field,through=int(ix[-1])+1,total=len(rows))
        return tables,proof,(dict(raw=torch.cat(query),labels=torch.tensor([r['label'] for r in rows])) if keep_query else None)
    backbone.cuda();status('science_native_base');initial,proof,_=describe(base,{'NATIVE_RIDGE':backbone},'native_base_encodes');backbone.cpu();groups,patient_vectors=patient_means(base,initial['NATIVE_RIDGE'],record);words=discover(patient_vectors,cfg['seed']+1,record);save(p/'discovery.private.pt',dict(groups=groups,means=patient_vectors,words=words));cost['discovery_patients']=len(groups);del patient_vectors
    grams={a:torch.zeros(1537,1537,dtype=torch.float64) for a in ARMS};means={a:[] for a in ARMS};add_current(base,initial['NATIVE_RIDGE'],cfg['order'][:10],grams['NATIVE_RIDGE'],means['NATIVE_RIDGE'],record);head=torch.stack([v[:1536]*np.sqrt(2) for v in means['NATIVE_RIDGE']]);del initial
    models={}
    for a in ARMS[1:]:torch.manual_seed(cfg['seed']);models[a]=Model(backbone,head).eval()
    ds=UnnormalizedImages(base,cfg);opts={a:optimizer(m,cfg) for a,m in models.items()}
    for step in range(1,cfg['base_updates']+1):
        seed=cfg['seed']+100000+step;receipt=word_draw(words,groups,seed);target=render(ds,receipt);shuffle=shuffle_receipt(seed+1);inputs=corrupt(target,shuffle);wy=torch.tensor([r['word'] for r in receipt]);ids,classes=patient_draw(base,cfg['order'][:10],seed+2);sup=torch.stack([ds[i][0] for i in ids]);sy=torch.tensor([cfg['order'][:10].index(c) for c in classes]);cost['word_source_pixel_rows']=cost.get('word_source_pixel_rows',0)+16;cost['supervised_pixel_rows']=cost.get('supervised_pixel_rows',0)+16
        detail={}
        for a,m in models.items():
            x,t,y=(sup,sup,sy) if a==ARMS[1] else (inputs,target,wy);detail[a]=update(m,opts[a],x,t,y,a,cfg,cost['arms'][a],lambda v:record(dict(arm=a,step=step,**v)))
        if step==cfg['base_updates']:
            cost['endpoint_source_pixel_rows']=16
            endpoint=dict(word_receipt=receipt,shuffle=shuffle,word_raw=torch.stack([ds[r['index']][0] for r in receipt]),word_input=inputs,word_target=target,word_labels=wy,supervised_raw=sup,supervised_ids=[base[i]['id'] for i in ids],supervised_labels=sy,objectives={})
            for a,m in models.items():
                m.cuda();x,t,y=(sup,sup,sy) if a==ARMS[1] else (inputs,target,wy)
                with torch.no_grad():endpoint['objectives'][a]=sum(float(m.objective(x[i:i+1].cuda().double(),t[i:i+1].cuda().double(),y[i:i+1].cuda(),a))/16 for i in range(16))
                cost['endpoint_rows'][a]+=16;m.cpu();save(p/f'endpoint_head_{a}.private.pt',dict(decoder=m.decoder.state_dict(),word_head=m.word_head.state_dict(),disease_head=m.disease_head))
            save(p/'endpoint.private.pt',endpoint)
        if step==1 or step%50==0:status('base_training',step=step,total=cfg['base_updates'],detail=detail)
    del opts
    maps={'NATIVE_RIDGE':backbone.cuda()}
    for a,m in models.items():m.backbone.requires_grad_(False);save(p/f'map_{a}.private.pt',m.backbone.state_dict());maps[a]=m.backbone.cuda()
    save(p/'training_proof.private.pt',proof);write(p/'MAP_SEAL.json',dict(at=time.time(),maps=4,all_later_optimization_forbidden=True,base_updates=cfg['base_updates'],labels=cfg['order'][:10]));del models
    seals=[]
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];rows=base if task==0 else [r for r in train if r['label'] in new];active={a:b for a,b in maps.items() if task>0 or a!='NATIVE_RIDGE'};table,pf,_=describe(rows,active,'adapted_base_encodes' if task==0 else 'later_encodes')
        for identity,value in pf.items():
            if identity in proof:value['features']={**proof[identity]['features'],**value['features']}
            proof[identity]=value
        save(p/'training_proof.private.pt',proof);heads={};systems={}
        for a in ARMS:
            before=torch.zeros_like(grams[a]) if task==0 and a=='NATIVE_RIDGE' else grams[a].clone()
            if task>0 or a!='NATIVE_RIDGE':add_current(rows,table[a],new,grams[a],means[a],record)
            h,mat,rhs=solve(grams[a],means[a],cfg['ridge']);residual=float(abs(mat@h-rhs).max());record(dict(check='normal_system',arm=a,task=task,residual=residual,PASS=residual<=1e-9));assert residual<=1e-9
            tri=torch.triu_indices(1537,1537);heads[a]=h;systems[a]=dict(matrix=mat,rhs=rhs,arrived_upper_triangle=(grams[a]-before)[tri[0],tri[1]]);cost['ridge_solves'][a]+=1
            if a!='NATIVE_RIDGE':
                saved=torch.load(p/f'map_{a}.private.pt',map_location='cpu',weights_only=False);delta=max(float(abs(v.detach().cpu()-saved[k]).max()) for k,v in maps[a].state_dict().items());record(dict(check='frozen_full_map',arm=a,task=task,error=delta,PASS=delta==0));assert delta==0;del saved
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=heads,systems=systems));seals.append(dict(task=task,at=time.time(),models=4));write(p/'STAGE_SEALS.json',seals);status('sealed',task=task);del table,heads,systems
    write(p/'FIT_SEAL.json',dict(at=time.time(),models=24));write(p/'QUERY_OPEN.private.json',dict(at=time.time()));table,_,query=describe(val,maps,'validation_encodes',True);save(p/'query.private.pt',query);save(p/'query_features.private.pt',table)
    for task in range(6):
        pack=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);ix=torch.isin(query['labels'],torch.tensor(pack['seen']));scores={a:table[a][ix]@pack['heads'][a] for a in ARMS};save(p/f'scores_{task}.private.pt',scores)
        for a in ARMS:cost['score_rows'][a]+=int(ix.sum())
        status('scored',task=task)
    status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());cost=dict(arms={a:counter() for a in ARMS[1:]},qualification_arms={a:counter() for a in ARMS[1:]},descriptor_rows={a:0 for a in ARMS},ridge_solves={a:0 for a in ARMS},score_rows={a:0 for a in ARMS},endpoint_rows={a:0 for a in ARMS[1:]})
    try:main(cfg,cost)
    except BaseException:
        p=Path(cfg['output']);p.mkdir(exist_ok=True);write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc(),cost=cost));raise
    finally:
        p=Path(cfg['output'])
        if p.exists():write(p/'COSTS.json',cost)
