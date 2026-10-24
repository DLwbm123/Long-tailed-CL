"""Base-only SSF trajectory and crossed fixed nonlinear readouts,24sealed systems."""
import json,math,os,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from core import ARMS,SSF,affine,pack
from data import read_rows,UnnormalizedImages,patient_draw
from foundation import build_backbone,write,save
from moments import add_current,solve

def main(cfg,cost):
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8';torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    p=Path(cfg['output']);p.mkdir(exist_ok=False);events=[]
    def record(v):
        def safe(x):
            if isinstance(x,float) and not math.isfinite(x):return str(x)
            if isinstance(x,dict):return {k:safe(y) for k,y in x.items()}
            if isinstance(x,list):return [safe(y) for y in x]
            return x
        v=safe(v);events.append(v)
        with (p/'NUMERIC_GATES.private.jsonl').open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
    def status(phase,**v):
        cost.update(elapsed_from_first=time.time()-cfg['original_start_epoch'],GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved());write(p/'COSTS.json',cost);write(p/'NUMERIC_GATES.json',events);write(p/'status.private.json',dict(phase=phase,updated=time.time(),**v));assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(SystemExit(143)))
    assert json.loads(Path(cfg['small_math_receipt']).read_text())['state']=='PASS';train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val} and [r['id'] for r in val]==list(range(300));base=[r for r in train if r['label'] in cfg['order'][:10]];assert len(base)==60129
    from checks import select_rows
    qualified_rows=select_rows(base,cfg,record)
    write(p/'COVERAGE.json',dict(train=68058,val=300,patient_overlap=0,val_id_schema_checked_without_pixels=True,qualification_patients=len({r['group'] for r in qualified_rows})));b=build_backbone(cfg,'cuda');assert not any(x.requires_grad for x in b.parameters());original={k:v.detach().cpu().clone() for k,v in b.state_dict().items()}
    m=SSF(b,cfg['seed']).cuda();W=torch.randn(cfg['projection_input'],cfg['projection_output'],generator=torch.Generator().manual_seed(cfg['seed']),dtype=torch.float64).cuda()
    whitelist={n for n,w in m.named_parameters() if w.requires_grad};record(dict(check='trainable_whitelist',names=sorted(whitelist),PASS=all(n=='head' or n.startswith('mods.') for n in whitelist) and len(whitelist)==149));assert all(n=='head' or n.startswith('mods.') for n in whitelist) and len(whitelist)==149
    from checks import qualify
    status('real_qualification');q=qualify(b,m,W,base,qualified_rows,cfg,p,cost,record,status)
    delta=max(float(abs(v.detach().cpu()-original[k]).max()) for k,v in b.state_dict().items());record(dict(check='qualification_original_unchanged',error=delta,PASS=delta==0));assert delta==0
    save(p/'physical_interfaces.private.pt',dict(ssf=m.small_state(),projection=W,raw=torch.zeros(1,3,224,224,dtype=torch.float32)))
    calibration=[]
    for name in ['qualification_system.private.pt','physical_interfaces.private.pt']:
        st=(p/name).stat();calibration.append(dict(file=name,logical_bytes=st.st_size,blocks_bytes=st.st_blocks*512,ratio=max(1.,st.st_blocks*512/st.st_size)))
    factor=1.1*max(v['ratio'] for v in calibration);write(p/'PHYSICAL_CALIBRATION.json',dict(representatives=calibration,factor_with_margin=factor))
    def physical(roots):
        used=0
        for root in roots:
            for f in Path(root).rglob('*'):
                try:
                    if f.is_file():s=f.stat();used+=max(s.st_size,s.st_blocks*512)
                except FileNotFoundError:pass
        return used
    dims=cfg['dimensions'];logical=12*sum(d*(d+1)//2*8 for d in dims)+12*sum(d*20*8 for d in dims)+340*3*224*224*4+340*sum(dims)*8+16*3*224*224*4+cfg['projection_input']*cfg['projection_output']*8+50000000;used=physical([p.parent,cfg['neutral_entry']]);window=physical(cfg['window_accounting_roots']);forecast=1.5*(1000*q['production_step_seconds']+68358*q['production_seconds_per_image']+68058*q['paired_Gram_seconds_per_row']+6*q['paired_system_seconds']+680*q['NumPy_seconds_per_image'])+3600;remaining=cfg['hard_deadline_epoch']-time.time();admission=dict(state='PENDING',time_forecast_seconds=forecast,remaining_seconds=remaining,physical_factor_with_margin=factor,new_root_forecast_bytes=used+factor*logical,window_forecast_bytes=window+factor*logical,existing_root_bytes=used,existing_window_bytes=window,ETA=False);write(p/'QUALIFICATION.json',admission);assert forecast<remaining-60 and admission['new_root_forecast_bytes']<cfg['storage_stop_threshold_bytes'] and admission['window_forecast_bytes']<cfg['window_storage_stop_bytes'];admission['state']='PASS';write(p/'QUALIFICATION.json',admission)
    from training import optimizer,update
    dataset=UnnormalizedImages(base,cfg);opt=optimizer(m,cfg);training_count=dict(updates=0,forward_rows=0,backward_rows=0)
    for step in range(cfg['supervised_updates']):
        ix,labels=patient_draw(base,cfg['order'][:10],cfg['seed']+step);raw=torch.stack([dataset[i][0] for i in ix]);y=torch.tensor([cfg['order'].index(c) for c in labels]);receipt=update(m,opt,raw,y,cfg,training_count,record,step=step);cost['optimizer_steps']=training_count['updates'];cost['formal_optimizer']=dict(training_count)
        if step==cfg['supervised_updates']-1:
            with torch.no_grad():
                endpoint=torch.cat([m.forward_features(raw[i:i+1].cuda().double()).cpu() for i in range(len(raw))]);obj=float(np.mean([float(m.objective(raw[i:i+1].cuda().double(),y[i:i+1].cuda())) for i in range(len(raw))]))
            save(p/'endpoint.private.pt',dict(raw=raw,labels=y,source_ids=[base[i]['id'] for i in ix],features=endpoint,head=m.head,objective=obj,step=step));cost['endpoint_full_SSF_forward_rows']=32
        if (step+1)%25==0 or step==0:status('science_SSF_training',step=step+1,total=cfg['supervised_updates'],objective=receipt['objective'])
    save(p/'ssf.private.pt',m.small_state());save(p/'projection.private.pt',W);loaded=torch.load(p/'ssf.private.pt',map_location='cpu',weights_only=False);check_error=max(float(abs(v-loaded[k]).max()) for k,v in m.small_state().items());record(dict(check='formal_SSF_disk_persistence',error=check_error,PASS=check_error==0));assert check_error==0
    del opt;m.requires_grad_(False);sealed=m.small_state();frozen_W=W.detach().cpu().clone()
    def describe(rows,field,query=False):
        tables={a:torch.empty(len(rows),d,dtype=torch.float64) for a,d in zip(ARMS,dims)};proof={};wanted=set();raws=[]
        for c in sorted({r['label'] for r in rows}):
            patients=set()
            for i in sorted([i for i,r in enumerate(rows) if r['label']==c],key=lambda i:(rows[i]['group'],rows[i]['relative_path'])):
                if rows[i]['group'] not in patients:wanted.add(i);patients.add(rows[i]['group'])
                if len(patients)==2:break
        with torch.no_grad():
            for raw,ix in DataLoader(UnnormalizedImages(rows,cfg),batch_size=cfg['microbatch'],shuffle=False,num_workers=0):
                zn=b.forward_features((raw.cuda().double()-.5)/.5);za=m.forward_features(raw.cuda().double());finite=bool(torch.isfinite(zn).all() and torch.isfinite(za).all());record(dict(check='finite_original_native_tokens',through=int(ix[-1])+1,field=field,PASS=finite));assert finite
                for a in ARMS:tables[a][ix]=affine(za if a.startswith('SSF') else zn,a,W).cpu();cost['descriptor_rows'][a]+=len(ix)
                cost[field]=cost.get(field,0)+len(ix);cost['SSF_'+field]=cost.get('SSF_'+field,0)+len(ix)
                if query:raws.append(raw.clone())
                for j,i in enumerate(ix.tolist()):
                    if i in wanted:proof[rows[i]['id']]=dict(raw=raw[j].clone(),label=rows[i]['label'],features={a:tables[a][i].clone() for a in ARMS})
                if (int(ix[-1])+1)%800==0 or int(ix[-1])+1==len(rows):status(field,through=int(ix[-1])+1,total=len(rows))
        return tables,proof,(dict(raw=torch.cat(raws),labels=torch.tensor([r['label'] for r in rows])) if query else None)
    write(p/'MAP_SEAL.json',dict(at=time.time(),maps=4,trainable_backbone_parameters=0,optimizer_steps=1000,SSF_frozen=True,RP_frozen=True,seed=cfg['seed'],dimensions=dims));grams={a:torch.zeros(d,d,dtype=torch.float64) for a,d in zip(ARMS,dims)};means={a:[] for a in ARMS};proof={};seals=[];status('science_started_native_base')
    for task in range(6):
        seen=cfg['order'][:10+2*task];new=seen[:10] if task==0 else seen[-2:];rows=base if task==0 else [r for r in train if r['label'] in new];tables,pf,_=describe(rows,'training_native_encodes');proof.update(pf);save(p/'training_proof.private.pt',proof);heads={};systems={}
        for a in ARMS:
            before=grams[a].clone();add_current(rows,tables[a],new,grams[a],means[a],lambda v:record(dict(arm=a,**v)));h,mat,rhs=solve(grams[a],means[a],cfg['ridge']);residual=float(abs(mat@h-rhs).max());record(dict(check='normal_system',arm=a,task=task,residual=residual,PASS=residual<=1e-9));assert residual<=1e-9;heads[a]=h;systems[a]=dict(matrix_upper=pack(mat),arrival_upper=pack(grams[a]-before),rhs=rhs,d=len(mat));cost['ridge_solves'][a]+=1
        delta=max(float(abs(v.detach().cpu()-original[k]).max()) for k,v in b.state_dict().items());record(dict(check='original_backbone_fixed',task=task,error=delta,PASS=delta==0));assert delta==0
        frozen_error=max(float(abs(v-sealed[k]).max()) for k,v in m.small_state().items());rp_error=float(abs(W.cpu()-frozen_W).max());record(dict(check='SSF_and_RP_fixed_after_base',task=task,error=frozen_error,RP_error=rp_error,PASS=frozen_error==0 and rp_error==0));assert frozen_error==rp_error==0
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=heads,systems=systems));seals.append(dict(task=task,at=time.time(),models=4));write(p/'STAGE_SEALS.json',seals);status('sealed',task=task);del tables,heads,systems
    write(p/'FIT_SEAL.json',dict(at=time.time(),models=24));write(p/'QUERY_OPEN.private.json',dict(at=time.time()));table,_,query=describe(val,'validation_encodes',True);save(p/'query.private.pt',query);save(p/'query_features.private.pt',table)
    for task in range(6):
        saved=torch.load(p/f'stage_{task}.private.pt',map_location='cpu',weights_only=False);take=torch.isin(query['labels'],torch.tensor(saved['seen']));scores={a:table[a][take]@saved['heads'][a] for a in ARMS};save(p/f'scores_{task}.private.pt',scores)
        for a in ARMS:cost['score_rows'][a]+=int(take.sum())
        status('scored',task=task)
    status('COMPLETE_PENDING_VERIFY')

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());cost=dict(optimizer_steps=0,descriptor_rows={a:0 for a in ARMS},ridge_solves={a:0 for a in ARMS},score_rows={a:0 for a in ARMS})
    try:main(cfg,cost)
    except BaseException:
        p=Path(cfg['output']);p.mkdir(exist_ok=True);write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc(),cost=cost));raise
    finally:
        p=Path(cfg['output'])
        if p.exists():write(p/'COSTS.json',cost)
