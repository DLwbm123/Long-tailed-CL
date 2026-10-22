"""Full real-GPU qualification precedes base-only training and all-arrival statistics."""
import os,json,time,signal,traceback,copy
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS,NativeEvidence,base_objective,np_evidence,affine,patient_weights,moments,solve
from data import read_rows,Images

def write(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def compact(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone().contiguous()
    if isinstance(v,dict):return {k:compact(x) for k,x in v.items()}
    if isinstance(v,list):return [compact(x) for x in v]
    return v

def save(p,v):
    v=compact(v)
    def check(x):
        if isinstance(x,torch.Tensor):assert x.untyped_storage().nbytes()==x.numel()*x.element_size()
        elif isinstance(x,dict):
            for y in x.values():check(y)
        elif isinstance(x,list):
            for y in x:check(y)
    check(v);t=p.with_suffix(p.suffix+'.tmp');torch.save(v,t);t.replace(p)
def prefix(b,images):return b.norm_pre(b.patch_drop(b._pos_embed(b.patch_embed(images))))
def build_backbone(cfg,device):
    import timm
    from safetensors.torch import load_file
    b=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    b.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    b.eval().requires_grad_(False)
    for block in b.blocks:
        block.attn.fused_attn=False
        assert isinstance(block.attn.q_norm,torch.nn.Identity) and isinstance(block.attn.k_norm,torch.nn.Identity)
        assert isinstance(block.ls1,torch.nn.Identity) and isinstance(block.ls2,torch.nn.Identity)
        assert isinstance(block.mlp.norm,torch.nn.Identity) and block.mlp.act.approximate=='none'
        assert block.norm1.eps==block.norm2.eps==b.norm.eps==1e-6
    for m in b.modules():
        if isinstance(m,torch.nn.Dropout):assert m.p==0
        if hasattr(m,'drop_prob'):assert m.drop_prob==0
    return b.to(device)
def body_state(b):return {k:v.detach().cpu().numpy() for k,v in b.state_dict().items() if k.startswith(('blocks.','norm.'))}
def evidence(model,tokens,micro,counter=None,field=None):
    p=next(model.backbone.parameters());out=[]
    with torch.no_grad():
        for i in range(0,len(tokens),micro):out.append(model(tokens[i:i+micro].to(device=p.device,dtype=p.dtype)).cpu())
    if counter is not None:counter[field]+=len(tokens)
    return torch.cat(out)
def sampling_groups(rows,classes):
    groups={c:{} for c in classes}
    for i,r in enumerate(rows):groups[r['label']].setdefault(r['group'],[]).append(i)
    return {c:list(groups[c].values()) for c in classes}
def draw(rng,groups,classes):
    cs=rng.choice(classes,8,replace=False);ix=[]
    for c in cs:
        pp=groups[int(c)]
        for j in rng.choice(len(pp),8,replace=len(pp)<8):ix.append(int(rng.choice(pp[int(j)])))
    return ix

def step(model,head,opt,x,teacher,y,weight,cfg,cost):
    opt.zero_grad(set_to_none=True);total=0.;ceval=0.;rcval=0.
    for start in range(0,len(x),cfg['microbatch']):
        sl=slice(start,start+cfg['microbatch']);z=model(x[sl].cuda().float());loss,ce,rc=base_objective(z,teacher[sl].cuda().float(),head,y[sl].cuda(),weight)
        scale=len(z)/len(x);(loss*scale).backward();total+=float(loss.detach())*scale;ceval+=float(ce.detach())*scale;rcval+=float(rc.detach())*scale
        cost['body_forward_rows']+=len(z);cost['body_backward_rows']+=len(z)
    norm=torch.nn.utils.clip_grad_norm_([model.prompts,head],cfg['gradient_clip']);assert torch.isfinite(norm)
    opt.step();cost['updates']+=1;cost['logical_draws']+=len(x)
    return dict(objective=total,CE=ceval,response_consistency=rcval,gradient_norm=float(norm))
def math_check():
    from types import SimpleNamespace
    from timm.models.vision_transformer import Block
    from functools import partial
    norm=partial(torch.nn.LayerNorm,eps=1e-6);b=torch.nn.Module();b.blocks=torch.nn.Sequential(*[Block(12,3,mlp_ratio=2,qkv_bias=True,norm_layer=norm) for _ in range(2)]);b.norm=norm(12)
    torch.manual_seed(74002);m=NativeEvidence(b,torch.randn(2,2,12)*.02).double();x=torch.randn(2,5,12,dtype=torch.float64);h=torch.randn(3,24,dtype=torch.float64,requires_grad=True);t=evidence(NativeEvidence(b),x,2);y=torch.tensor([0,2]);loss,_,_=base_objective(m(x),t,h,y,.05);loss.backward();s=body_state(b)
    def objective(prompt):
        z=np_evidence(x.numpy(),s,prompt,heads=3);hh=h.detach().numpy().copy();hh/=np.linalg.norm(hh,axis=1,keepdims=True);v=25*z@hh.T;v-=v.max(1,keepdims=True)
        return np.mean(np.log(np.exp(v).sum(1))-v[np.arange(2),y.numpy()])+.05*np.mean(1-(z*t.numpy()).sum(1))
    p=m.prompts.detach().numpy().copy();err=abs(objective(p)-float(loss.detach()));plus=p.copy();minus=p.copy();plus[0,0,0]+=1e-5;minus[0,0,0]-=1e-5;fd=abs((objective(plus)-objective(minus))/2e-5-float(m.prompts.grad[0,0,0]));assert err<1e-9 and fd<2e-6
    return dict(state='PASS',objective_error=err,finite_difference_error=fd,CPU_forward_backward_rows=2,numpy_objectives=3,numpy_rows=6,optimization=0)

def main(cfg):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=Path(cfg['output']);p.mkdir(exist_ok=False)
    cost=dict(qualification_prefix_rows=0,science_prefix_rows=0,validation_prefix_rows=0,prefix_seconds={},formal_solves=0,qualification_solves=0,teacher_body_rows=0,
              arms={a:dict(updates=0,logical_draws=0,body_forward_rows=0,body_backward_rows=0,mapping_rows=0,deploy_query_rows=0) for a in ARMS},qualification_arms={a:dict(updates=0,logical_draws=0,body_forward_rows=0,body_backward_rows=0) for a in ARMS})
    def status(phase,**extra):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.json',cost);write(p/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def term(sig,frame):
        write(p/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,term)
    status('mathematical_qualification');tick=time.time();math=math_check();math['seconds']=time.time()-tick;write(p/'MATH_QUALIFICATION.json',math)
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val');assert len(train)==68058 and len(val)==300 and not {r['group'] for r in train}&{r['group'] for r in val}
    write(p/'COVERAGE.json',dict(train=68058,val=300,train_counts=dict(Counter(str(r['label']) for r in train)),patient_overlap=0,permission='All arriving patient-balanced second moments in a base-frozen map; no raw-image replay.'))
    b=build_backbone(cfg,'cuda');backbone_before={k:v.detach().cpu().clone() for k,v in b.state_dict().items()};baseclasses=cfg['order'][:10]
    def extract(rows,field):
        out=torch.empty((len(rows),197,768),dtype=torch.float16)
        for images,idx in DataLoader(Images(rows,cfg),batch_size=cfg['encoder_batch'],num_workers=0,shuffle=False):
            tick=time.time()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):z=prefix(b,images.cuda())
            assert torch.isfinite(z).all();out[idx]=z.cpu().half();torch.cuda.synchronize();cost[field]+=len(idx);cost['prefix_seconds'][field]=cost['prefix_seconds'].get(field,0)+time.time()-tick
            status(field,encoded_in_current_set=int(idx[-1])+1,current_size=len(rows))
        return out
    prompts=torch.randn(8,6,768)*cfg['prompt_std'];save(p/'initial_prompts.private.pt',prompts)
    qrows=[]
    for c in baseclasses:
        seen=set()
        for r in train:
            if r['label']==c and r['group'] not in seen:qrows.append(r);seen.add(r['group'])
            if len(seen)==2:break
        assert len(seen)==2
    q=extract(qrows,'qualification_prefix_rows');qstart=time.time();teacher=evidence(NativeEvidence(b),q,cfg['microbatch']);qual={};qm={};qheads={};qerr=0.;qstorage=[]
    for a in ARMS:
        m=NativeEvidence(b,None if a==ARMS[0] else prompts.cuda());head=torch.nn.Parameter(F.normalize(torch.stack([teacher[2*i:2*i+2].mean(0) for i in range(8)]),dim=1).cuda());before=None if m.prompts is None else m.prompts.detach().clone()
        if m.prompts is not None:
            opt=torch.optim.AdamW([m.prompts,head],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            ix=torch.arange(64)%16;step(m,head,opt,q[ix],teacher[ix],(ix//2),.05 if a==ARMS[-1] else 0.,cfg,cost['qualification_arms'][a])
            head=torch.nn.Parameter(torch.cat([head.detach(),F.normalize(torch.stack([teacher[16:18].mean(0),teacher[18:20].mean(0)]),dim=1).cuda()]));opt=torch.optim.AdamW([m.prompts,head],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);ix=torch.arange(64)%20;step(m,head,opt,q[ix],teacher[ix],ix//2,.05 if a==ARMS[-1] else 0.,cfg,cost['qualification_arms'][a]);delta=float((m.prompts-before).abs().max());assert delta>0
        else:delta=0.
        m.requires_grad_(False);z=evidence(m,q,cfg['microbatch']);phi=affine(z);records=[]
        for i,c in enumerate(baseclasses):
            gram,mean=moments(phi[2*i:2*i+2],np.ones(2)/2);records.append({a:dict(gram=gram,mean=mean)})
        heads=[]
        for n in [8,10]:h,res=solve(records[:n],a,cfg['ridge']);heads.append(h);cost['qualification_solves']+=1
        obj=dict(prompts=None if m.prompts is None else m.prompts.detach(),records=records,heads=heads,bounded_tokens=q[:2]);target=p/f'qualification_{a}.private.pt';save(target,obj);loaded=torch.load(target,map_location='cpu',weights_only=False)
        assert target.stat().st_size<400_000_000
        for rr in loaded['records']:
            for v in rr[a].values():assert v.untyped_storage().nbytes()==v.numel()*v.element_size()
        cpu=NativeEvidence(b.cpu().double(),loaded['prompts']).double();ref=evidence(cpu,q[:2],2).numpy();ind=np_evidence(q[:2].numpy(),body_state(b),None if loaded['prompts'] is None else loaded['prompts'].numpy());error=float(abs(ref-ind).max());qerr=max(qerr,error);assert error<cfg['score_absolute_tolerance']
        # Verify saved systems and actual logits after full class-state load.
        for n,h in zip([8,10],loaded['heads']):
            mat=sum(r[a]['gram'].numpy() for r in loaded['records'][:n])/n+cfg['ridge']*np.eye(1537);rhs=np.stack([r[a]['mean'].numpy()/n for r in loaded['records'][:n]],1);nh=np.linalg.solve(mat,rhs);qerr=max(qerr,float(abs(nh-h.numpy()).max()));assert abs(nh-h.numpy()).max()<2e-6
        qual[a]=dict(actual_prompt_maxdelta=delta,numpy_feature_error=error,GPU_forward_backward_rows=cost['qualification_arms'][a]['body_backward_rows'],saved_classes=10,saved_systems=2,FP64_rows=2,NumPy_rows=2)
        del cpu,m,head,records,loaded,obj,heads
        if a!=ARMS[0]:del opt
        b.cuda().float()
    assert all(not v.requires_grad for v in b.parameters())
    unchanged=max(float((v.detach().cpu()-backbone_before[k]).abs().max()) for k,v in b.state_dict().items());assert unchanged==0
    write(p/'QUALIFICATION.json',dict(state='PASS',math=math,arms=qual,max_error=qerr,after_encoding_seconds=time.time()-qstart,real_prefix_rows=20,formal_batch=64,microbatch=8,qualification_solves=6,discarded_optimizer_updates=4,all_class_storage_loaded=True,raw_table_saved=False,pretrained_parameter_maxdelta=unchanged,teacher_body_rows=20,postupdate_mapping_rows=60))
    del q,teacher
    status('qualified_base_prefix_encoding')
    base=[r for r in train if r['label'] in baseclasses];tokens=extract(base,'science_prefix_rows');models={a:NativeEvidence(b,None if a==ARMS[0] else prompts.cuda()) for a in ARMS}
    teacher_tick=time.time();teacher=evidence(models[ARMS[0]],tokens,cfg['microbatch']);cost['teacher_body_rows']+=len(tokens);cost['teacher_body_seconds']=time.time()-teacher_tick;means=[]
    for c in baseclasses:
        ix=[i for i,r in enumerate(base) if r['label']==c];weights,_=patient_weights([base[i] for i in ix]);means.append((teacher[ix].double()*torch.from_numpy(weights)[:,None]).sum(0).float())
    initial=F.normalize(torch.stack(means),dim=1);heads={a:torch.nn.Parameter(initial.clone().cuda()) for a in ARMS[1:]};opts={a:torch.optim.AdamW([models[a].prompts,heads[a]],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for a in ARMS[1:]}
    groups=sampling_groups(base,baseclasses);rng=np.random.default_rng(cfg['seed']);draws=[];curves=[];labels=torch.tensor([baseclasses.index(r['label']) for r in base])
    train_tick=time.time()
    for iteration in range(cfg['base_updates']):
        ix=draw(rng,groups,baseclasses);draws.append(ix)
        for a in ARMS[1:]:
            row=step(models[a],heads[a],opts[a],tokens[ix],teacher[ix],labels[ix],cfg['consistency_weight'] if a==ARMS[-1] else 0.,cfg,cost['arms'][a])
            if iteration%100==0:curves.append(dict(iteration=iteration,arm=a,**row))
        if iteration%20==0:status('base_prompt_training',iteration=iteration)
    cost['base_training_resident_seconds']=time.time()-train_tick
    save(p/'paired_draws.private.pt',torch.tensor(draws,dtype=torch.int32));write(p/'TRAINING_CURVES.json',curves)
    for a in ARMS:
        models[a].requires_grad_(False);save(p/f'prompts_{a}.private.pt',None if models[a].prompts is None else models[a].prompts.detach())
    del opts,heads,teacher,draws
    unchanged=max(float((v.detach().cpu()-backbone_before[k]).abs().max()) for k,v in b.state_dict().items());assert unchanged==0
    del backbone_before
    write(p/'MAP_SEAL.json',dict(pretrained_parameter_maxdelta=unchanged,at=time.time(),base_only=True,backbone_frozen=True,prompt_updates={a:cost['arms'][a]['updates'] for a in ARMS},prompt_parameters=36864,dimension=1536))
    records=[];seals=[];cost['stage_sections']=[]
    for task,size in enumerate(cfg['task_sizes']):
        old=sum(cfg['task_sizes'][:task]);cs=cfg['order'][old:old+size];rows=base if task==0 else [r for r in train if r['label'] in cs]
        if task:tokens=extract(rows,'science_prefix_rows')
        stage_tick=time.time()
        new={c:dict(label=c,images=sum(r['label']==c for r in rows),patients=len(set(r['group'] for r in rows if r['label']==c))) for c in cs}
        for a in ARMS:
            z=evidence(models[a],tokens,cfg['microbatch'],cost['arms'][a],'mapping_rows');phi=affine(z);inderr=0.
            for c in cs:
                ix=[i for i,r in enumerate(rows) if r['label']==c];rr=[rows[i] for i in ix];w,pc=patient_weights(rr);gram,mean=moments(phi[ix],w)
                # Separate full current-arrival NumPy patient aggregation before release.
                from verify import np_stats
                ng,nm,npats=np_stats(phi[ix].numpy(),rr);err=max(float(abs(ng-gram.numpy()).max()),float(abs(nm-mean.numpy()).max()));assert err<1e-9 and npats==pc
                new[c][a]=dict(gram=gram,mean=mean,online_independent_error=err,proof_tokens=tokens[ix[:2]],proof_features=z[ix[:2]])
            del z,phi
        for c in cs:save(p/f'class_{c}.private.pt',new[c]);records.append(new[c])
        seen=cfg['order'][:old+size];hh={};res={}
        for a in ARMS:hh[a],res[a]=solve(records,a,cfg['ridge']);cost['formal_solves']+=1
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=hh,residual=res));seals.append(dict(task=task,seen=seen,at=time.time(),models=3));write(p/'STAGE_SEALS.json',seals);status('stage_sealed',task=task)
        cost['stage_sections'].append(dict(task=task,body_mapping_stats_solve_save_seconds=time.time()-stage_tick))
        del tokens,rows,new
    assert cost['science_prefix_rows']==68058 and cost['formal_solves']==18
    write(p/'FIT_SEAL.json',dict(at=time.time(),models=18,all_models_before_validation=True));write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
    q=extract(val,'validation_prefix_rows');save(p/'query_tokens.private.pt',q)
    # Deployment uses FP64 full response once per arm, reused for every seen prefix.
    deploy_tick=time.time();b.cpu().double();query={}
    for a in ARMS:
        models[a].cpu().double();z=evidence(models[a],q,cfg['microbatch']);query[a]=z;cost['arms'][a]['deploy_query_rows']+=300
        for task in range(6):
            pack=torch.load(p/f'stage_{task}.private.pt',weights_only=False);mask=torch.tensor([r['label'] in pack['seen'] for r in val]);save(p/f'scores_{task}_{a}.private.pt',affine(z[mask])@pack['heads'][a])
        status('deployment_scored',arm=a)
    cost['deployment_resident_seconds']=time.time()-deploy_tick
    save(p/'query_features.private.pt',query);write(p/'COSTS.json',cost);status('COMPLETE_PENDING_VERIFY')
if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except Exception:
        p=Path(cfg['output']);p.mkdir(exist_ok=True,parents=True);write(p/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));raise
