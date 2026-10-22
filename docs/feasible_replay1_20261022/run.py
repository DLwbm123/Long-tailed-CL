"""Actual arrival stages; mandatory real-image CUDA full-chain gate before bulk encoding."""
import os
assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
import copy,json,signal,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS,Model,memory_indices,state,risks,step,flat,project
from data import Images,read_rows
from verify import numpy_forward,numpy_risks,nstate,ce,summarize

def write(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(v,tmp);tmp.replace(p)
def armcost():return dict(opportunities=0,accepted_updates=0,batch_forward_rows=0,batch_backward_rows=0,constraint_forward_rows=0,constraint_backward_rows=0,feasibility_forward_rows=0,candidate_checks=0,training_view_rows=0,sampled_image_rows=0)

def mathematical_check(cfg):
    torch.manual_seed(cfg['seed']);m=Model(8,3,torch.randn(3,8,dtype=torch.float64)).double()
    with torch.no_grad():m.up.weight.normal_(0,.1)
    x=torch.randn(12,8,dtype=torch.float64);y=torch.arange(3).repeat(4)
    loss=F.cross_entropy(m(x,cfg['logit_scale'])[1],y);loss.backward();ns=nstate(state(m))
    objective_error=abs(float(loss.detach())-ce(numpy_forward(x.numpy(),ns,cfg['logit_scale'])[1],y.numpy()).mean())
    errors=[]
    for name,pos in [('head',(0,0)),('up.weight',(2,1)),('down.bias',(0,)),('bias',(0,))]:
        plus={k:v.copy() for k,v in ns.items()};minus={k:v.copy() for k,v in ns.items()};plus[name][pos]+=1e-5;minus[name][pos]-=1e-5
        fd=(ce(numpy_forward(x.numpy(),plus,cfg['logit_scale'])[1],y.numpy()).mean()-ce(numpy_forward(x.numpy(),minus,cfg['logit_scale'])[1],y.numpy()).mean())/2e-5
        errors.append(abs(fd-float(dict(m.named_parameters())[name].grad[pos])))
    assert objective_error<1e-9 and max(errors)<2e-6
    g=torch.tensor([-2.,-1.,3.],dtype=torch.float64);G=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    d,receipt=project(g,G,cfg);assert torch.allclose(d,torch.tensor([0.,0.,3.],dtype=torch.float64),atol=1e-9)
    # First-order tangent feasibility alone does not imply finite-step nonlinear feasibility.
    # r(theta)=theta^2 at theta=0 has zero gradient; d=1 satisfies tangent but r(-eta)>r(0).
    assert (0.*1.)>=0 and (-.1)**2>0
    return dict(objective_error=objective_error,finite_difference_error=max(errors),independent_orthant_projection_error=float((d-torch.tensor([0.,0.,3.])).abs().max()),nonlinear_tangent_counterexample=True,projection=receipt)

def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,encoder_parameter_updates=0,encoder_seconds={},qualification_arms={a:armcost() for a in ARMS},arms={a:armcost() for a in ARMS},validation_model_rows={a:0 for a in ARMS},head_initialization_rows={a:0 for a in ARMS},ledger_forward_rows={a:0 for a in ARMS},probe_FP64_rows={a:0 for a in ARMS},qualification_FP64_rows={a:0 for a in ARMS},qualification_numpy_rows={a:0 for a in ARMS},qualification_arrival_ledger_rows={a:0 for a in ARMS})
    models={};curves=[];seals=[];audits=[]
    def status(phase,**kw):
        cost.update(wall_from_first=time.time()-cfg['original_start_epoch'],GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(root/'COSTS.private.json',cost);write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def interrupt(sig,frame):
        save(root/'interrupted.private.pt',dict(models={a:state(m) for a,m in models.items()},cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupt)
    status('mathematical_qualification');math=mathematical_check(cfg);write(root/'MATH_QUALIFICATION.json',math)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    for i,r in enumerate(train):r['id']=i
    assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    write(root/'COVERAGE.json',dict(train_counts=dict(Counter(str(r['label']) for r in train)),patient_overlap=0,total_train=len(train),total_val=len(val),old_image_replay=False,cache_reconstruction=False))
    import timm
    from safetensors.torch import load_file
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    def extract(rows,field):
        out=torch.empty(len(rows),cfg['input_dimension']);tick=time.time()
        for images,ix in DataLoader(Images(rows,cfg),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):raw=encoder.forward_features(images.cuda())
            assert raw.shape[1:]==(197,768) and torch.isfinite(raw).all();raw=raw.float()
            out[ix]=F.normalize(torch.cat([raw[:,0],raw[:,1:].mean(1)],1),dim=1).cpu();torch.cuda.synchronize()
            cost[field]+=len(ix);cost['encoder_seconds'][field]=cost['encoder_seconds'].get(field,0)+time.time()-tick
            status(field,encoded_in_current_set=int(ix[-1])+1,current_count=len(rows));tick=time.time()
        return out
    # Two patients for each of ten actually available base classes; later classes are not read.
    qr=[]
    for c in cfg['order'][:10]:
        groups=set()
        for r in train:
            if r['label']==c and r['group'] not in groups:
                qr.append(r);groups.add(r['group'])
                if len(groups)==2:break
        assert len(groups)==2
    q=extract(qr,'qualification_encodes').cuda();qy=torch.arange(10,device='cuda').repeat_interleave(2)
    torch.manual_seed(cfg['seed']);qm=torch.stack([q[i*2:i*2+2].mean(0) for i in range(10)])
    template=Model(cfg['input_dimension'],cfg['adapter_rank'],qm[:8]).cuda();qual={};qualification_started=time.time()
    for a in ARMS:
        m=copy.deepcopy(template);qc=cost['qualification_arms'][a]
        ix=torch.arange(64,device='cuda')%16;x=q[ix].repeat(2,1);yy=qy[ix].repeat(2)
        # Actual dropout, FP32 classifier forward/backward/clip/plain-SGD on CUDA.
        torch.manual_seed(cfg['seed']+10);x=x*(torch.rand_like(x)>=cfg['input_dropout'])/(1-cfg['input_dropout'])
        first=step(m,x,yy,q[:0],qy[:0],q.new_empty(0),a,cfg,qc)
        with torch.no_grad():before=risks(m,q[:16],qy[:16],cfg['logit_scale'])
        cost['qualification_arrival_ledger_rows'][a]+=16*3
        shift=m.expand(qm[8:],q[:16],cfg['logit_scale'],cfg['expansion_CE_allowance']);ceiling=before+cfg['stage_CE_allowance']
        ix=torch.arange(64,device='cuda')%20;x=q[ix].repeat(2,1);yy=qy[ix].repeat(2)
        torch.manual_seed(cfg['seed']+11);x=x*(torch.rand_like(x)>=cfg['input_dropout'])/(1-cfg['input_dropout'])
        second=step(m,x,yy,q[:16],qy[:16],ceiling,a,cfg,qc)
        save(root/f'qualification_{a}.private.pt',state(m))
        loaded=Model(cfg['input_dimension'],cfg['adapter_rank'],qm.cpu()).double()
        loaded.load_state_dict({k:v.double() for k,v in torch.load(root/f'qualification_{a}.private.pt',map_location='cpu',weights_only=True).items()})
        with torch.no_grad():score=loaded(x.cpu().double(),cfg['logit_scale'])[1].numpy()
        ns=nstate(state(loaded));reference=numpy_forward(x.cpu().numpy(),ns,cfg['logit_scale'])[1]
        error=float(abs(score-reference).max());assert error<=cfg['score_absolute_tolerance']
        cost['qualification_FP64_rows'][a]+=len(x);cost['qualification_numpy_rows'][a]+=len(x)+16
        r=numpy_risks(q[:16].cpu().numpy(),qy[:16].cpu().numpy(),ns,cfg['logit_scale'])
        if a=='FEASIBLE':assert np.all(r<=ceiling.cpu().numpy()+cfg['risk_verification_tolerance'])
        qual[a]=dict(first=first,arrival=second,arrival_bias=shift,score_error=error,CUDA_FP32=True,saved_loaded=True)
    write(root/'QUALIFICATION.json',dict(state='PASS',real_training_images=20,validation_encodes=0,base_internal_simulated_arrival_only=True,actual_GPU_encoder=True,actual_GPU_FP32_batch_views=128,all_arms_backward_clip_optimizer_memory_expansion_serialization_FP64_NumPy=True,math=math,arms=qual,seconds=time.time()-qualification_started,scope='Discarded engineering states; not formal accuracy or ETA.'))
    del m,template,loaded,q,qm,qr,x,yy;torch.cuda.empty_cache()
    assert json.loads((root/'QUALIFICATION.json').read_text())['state']=='PASS'
    memory=torch.empty(0,cfg['input_dimension']);memory_rows=[]
    for task in range(6):
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];new=seen[sum(cfg['task_sizes'][:task]):]
        rows=[r for r in train if r['label'] in new];features=extract(rows,'science_train_encodes')
        means=[]
        for c in new:
            groups={}
            for i,r in enumerate(rows):
                if r['label']==c:groups.setdefault(r['group'],[]).append(i)
            means.append(torch.stack([features[ix].double().mean(0) for ix in groups.values()]).mean(0).float())
        means=torch.stack(means)
        mx=memory.cuda();my=torch.tensor([seen.index(r['label']) for r in memory_rows],device='cuda',dtype=torch.long)
        ceilings={};arrival_models={};arrival_receipts={}
        if task==0:
            torch.manual_seed(cfg['seed']);template=Model(cfg['input_dimension'],cfg['adapter_rank'],means)
            models={a:copy.deepcopy(template).cuda() for a in ARMS};del template
        for a,m in models.items():
            if task:
                with torch.no_grad():before=risks(m,mx,my,cfg['logit_scale'])
                cost['ledger_forward_rows'][a]+=len(mx);ceilings[a]=before+cfg['stage_CE_allowance']
                shift=m.expand(means.cuda(),mx,cfg['logit_scale'],cfg['expansion_CE_allowance'])
                cost['head_initialization_rows'][a]+=2*len(mx)
                with torch.no_grad():admitted=risks(m,mx,my,cfg['logit_scale'])
                cost['ledger_forward_rows'][a]+=len(mx)
                assert bool((admitted<=before+cfg['expansion_CE_allowance']+cfg['risk_training_tolerance']).all())
                arrival_models[a]=state(m);arrival_receipts[a]=dict(before=before.cpu().tolist(),admitted=admitted.cpu().tolist(),bias=shift,ceiling=ceilings[a].cpu().tolist())
            else:ceilings[a]=torch.empty(0,device='cuda')
        source=torch.cat([memory,features]);source_rows=memory_rows+rows;pools={}
        for i,r in enumerate(source_rows):pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
        assert set(pools)==set(seen)
        rng=np.random.default_rng(cfg['seed']+1000+task);view_rng=torch.Generator().manual_seed(cfg['seed']+2000+task)
        steps=cfg['base_updates'] if task==0 else cfg['later_updates'];start=time.time();transition={};stage_max_violation=-1e30
        for it in range(1,steps+1):
            chosen=rng.choice(seen,size=cfg['classes_per_batch'],replace=False).tolist();ix=[]
            for c in chosen:
                gs=list(pools[c])
                for _ in range(cfg['draws_per_class']):
                    g=gs[int(rng.integers(len(gs)))];v=pools[c][g];ix.append(v[int(rng.integers(len(v)))])
            x=source[ix].repeat(2,1);x*=((torch.rand(x.shape,generator=view_rng)>=cfg['input_dropout'])/(1-cfg['input_dropout']));x=x.cuda()
            yy=torch.tensor([seen.index(source_rows[i]['label']) for i in ix]*2,device='cuda');details={}
            for a in ARMS:
                if a=='FEASIBLE' and task and it==steps:transition['before']=state(models[a]);transition['parameter_order']=[n for n,_ in models[a].named_parameters()]
                d=step(models[a],x,yy,mx,my,ceilings[a],a,cfg,cost['arms'][a],capture=(a=='FEASIBLE' and task>0 and it==steps));witness=d.pop('_witness',None);details[a]=d
                cost['arms'][a]['training_view_rows']+=len(x);cost['arms'][a]['sampled_image_rows']+=len(ix)
                if a=='FEASIBLE' and task:
                    stage_max_violation=max(stage_max_violation,max(np.array(d['risks_after'])-ceilings[a].cpu().numpy()))
                    if it==steps:
                        transition.update(after=state(models[a]),factor=d['step_factor'])
                        transition.update(witness)
            if it==1 or it%100==0 or it==steps:
                torch.cuda.synchronize();curves.append(dict(task=task,step=it,arms=details,elapsed_stage=time.time()-start,at=time.time()))
                write(root/'TRAINING_CURVES.json',curves);status('actual_stage_training',task=task,step=it,steps=steps)
        probe=dict(x=x.cpu().double(),y=yy.cpu(),CE={});final_risks={}
        for a,m in models.items():
            pm=copy.deepcopy(m).cpu().double()
            with torch.no_grad():probe['CE'][a]=float(F.cross_entropy(pm(probe['x'],cfg['logit_scale'])[1],probe['y']))
            cost['probe_FP64_rows'][a]+=len(x)
            with torch.no_grad():final_risks[a]=risks(m,mx,my,cfg['logit_scale']).cpu().tolist()
            cost['ledger_forward_rows'][a]+=len(mx)
            del pm
        selected=memory_indices(rows,cfg['memory_per_class'],cfg['seed']);memory=torch.cat([memory,features[selected]]);memory_rows.extend(rows[i] for i in selected)
        assert len(memory_rows)<=len(seen)*cfg['memory_per_class'] and len(memory_rows)==len({(r['label'],r['group']) for r in memory_rows})
        save(root/f'stage_{task}.private.pt',dict(seen=seen,models={a:state(m) for a,m in models.items()},memory_features=memory,memory_rows=memory_rows,arrival_models=arrival_models,ceilings={a:v.cpu().tolist() for a,v in ceilings.items()},final_old_risks=final_risks,probe=probe,transition_probe=transition,new_head_initialization_means=means))
        seals.append(dict(task=task,at=time.time(),seen=seen,validation_encodes=0,science_train_encodes=cost['science_train_encodes'],opportunities={a:cost['arms'][a]['opportunities'] for a in ARMS},accepted_updates={a:cost['arms'][a]['accepted_updates'] for a in ARMS},memory_images=len(memory_rows),parameters_per_arm=sum(p.numel() for p in models['ER'].parameters()),training_residence_seconds=time.time()-start))
        audits.append(dict(task=task,arrival=arrival_receipts,final_old_risks=final_risks,FEASIBLE_max_online_violation=stage_max_violation if task else None))
        write(root/'STAGE_SEALS.json',seals);write(root/'RISK_LEDGERS.json',audits)
        del source,source_rows,features,rows,pools,mx,my,x,yy;torch.cuda.empty_cache()
    assert cost['science_train_encodes']==68058
    write(root/'FIT_SEAL.json',dict(at=time.time(),models=18,stages=6,validation_encodes=0));write(root/'QUERY_OPEN.private.json',dict(at=time.time(),validation_encodes=0))
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
    result,decision=summarize(stages,cfg,Counter(r['label'] for r in train));write(root/'RESULTS.json',result);write(root/'DECISION.json',decision)
    status('scientific_complete');write(root/'COSTS.json',cost);write(root/'status.private.json',dict(state='COMPLETE_PENDING_VERIFY',updated=time.time(),cost=cost))
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException as e:
        root=Path(cfg['output'])
        if root.exists():write(root/'FAILURE.private.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()))
        raise
