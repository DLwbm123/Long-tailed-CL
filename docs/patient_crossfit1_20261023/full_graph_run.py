"""Original scientific runner; composed qualification and exact retained full graph."""
import os,json,time,signal,traceback,copy
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS,NativeEvidence,np_evidence,affine,patient_weights,moments,solve,pools,episode,cached_update
from checks import math_checks,real_gradient_check,np_objective
from execution_core import graph_update as cached_update
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
def main(cfg):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=Path(cfg['output']);p.mkdir(exist_ok=False)
    def arm_cost():return dict(updates=0,logical_rows=0,cached_body_forward_rows=0,recomputed_body_forward_rows=0,body_backward_rows=0,mapping_rows=0,deploy_query_rows=0,inner_dual_solves=0)
    cost=dict(qualification_prefix_rows=0,science_prefix_rows=0,validation_prefix_rows=0,prefix_seconds={},formal_solves=0,qualification_solves=0,teacher_body_rows=0,arms={a:arm_cost() for a in ARMS},qualification_arms={a:arm_cost() for a in ARMS})
    def status(phase,**extra):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch'];cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(p/'COSTS.json',cost);write(p/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60 and cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def term(sig,frame):
        write(p/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost));raise SystemExit(143)
    signal.signal(signal.SIGTERM,term)
    status('mathematical_qualification');tick=time.time();math=math_checks(cfg);math['seconds']=time.time()-tick;write(p/'MATH_QUALIFICATION.json',math)
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
    old=json.loads(Path(cfg['prior_class_qualification']).read_text());eqroot=Path(cfg['execution_qualification_root'])
    eq=json.loads((eqroot/'EXECUTION_COMPLETION.json').read_text());feas=json.loads((eqroot/'EXECUTION_FEASIBILITY.json').read_text());gpu=json.loads((eqroot/'GPU_RESOURCES.json').read_text())
    assert old['state']=='PASS' and eq['state']=='PASS' and feas['state']=='PASS' and gpu['peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    assert old['all_class_storage_loaded'] and old['pretrained_parameter_maxdelta']==0
    assert feas['original_start_epoch']==cfg['original_start_epoch'] and feas['hard_deadline_epoch']==cfg['hard_deadline_epoch']
    for name in ['CPU_EQUIVALENCE','GPU_EQUIVALENCE']:
        checks=json.loads((eqroot/(name+'.json')).read_text());assert set(checks)==set(ARMS[1:])
        for check in checks.values():assert check['gradient_error']<3e-4 and check['actual_AdamW_error']<2e-5 and check['objective_error']<2e-6
    write(p/'QUALIFICATION.json',dict(state='PASS',evidence_composed=True,full_class_chain_prior=old,changed_execution_completion=eq,changed_execution_GPU=gpu,new_GPU_qualification_rows_in_this_worker=0))
    forecast=feas['forecast_seconds'];remaining=cfg['hard_deadline_epoch']-time.time();write(p/'FEASIBILITY.json',dict(**feas,rechecked_remaining_seconds=remaining,bulk_admission_at=time.time()));assert forecast<remaining-60,'Original budget no longer admits bulk'
    cost['execution']='retained_full_graph_original_attention_no_checkpoint';cost['prior_qualification_is_composed']=True
    status('qualified_base_prefix_encoding')
    base=[r for r in train if r['label'] in baseclasses];tokens=extract(base,'science_prefix_rows');models={a:NativeEvidence(b,None if a==ARMS[0] else prompts.cuda()) for a in ARMS}
    teacher_tick=time.time();teacher=evidence(models[ARMS[0]],tokens,cfg['microbatch']);cost['teacher_body_rows']+=len(tokens);cost['teacher_body_seconds']=time.time()-teacher_tick;means=[]
    for c in baseclasses:
        ix=[i for i,r in enumerate(base) if r['label']==c];weights,_=patient_weights([base[i] for i in ix]);means.append((teacher[ix].double()*torch.from_numpy(weights)[:,None]).sum(0).float())
    initial=F.normalize(torch.stack(means),dim=1);head=torch.nn.Parameter(initial.clone().cuda());opts={a:torch.optim.AdamW([models[a].prompts]+([head] if a=='CE_RC' else []),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for a in ARMS[1:]}
    pp=pools(base,baseclasses,cfg['seed']);rng=np.random.default_rng(cfg['seed']);draws=[];curves=[];train_tick=time.time()
    for iteration in range(cfg['base_updates']):
        selected,ids,yy,ww=episode(rng,pp,baseclasses);ix=ids['S']+ids['Q'];labels=torch.tensor(yy['S']+yy['Q']);weights=torch.tensor(ww['S']+ww['Q']);draws.append(ix+[-1]*(64-len(ix)))
        assert not {base[i]['group'] for i in ids['S']}&{base[i]['group'] for i in ids['Q']}
        for a in ARMS[1:]:
            row=cached_update(models[a],opts[a],tokens[ix],teacher[ix],labels,weights,a,cfg,cost['arms'][a],head if a=='CE_RC' else None)
            if iteration%100==0:curves.append(dict(iteration=iteration,arm=a,logical_rows=len(ix),**row))
        if iteration%20==0:status('base_crossfit_training',iteration=iteration)
    cost['base_training_resident_seconds']=time.time()-train_tick;save(p/'paired_draws.private.pt',torch.tensor(draws,dtype=torch.int32));write(p/'TRAINING_CURVES.json',curves)
    for a in ARMS:
        models[a].requires_grad_(False);save(p/f'prompts_{a}.private.pt',None if models[a].prompts is None else models[a].prompts.detach())
    del opts,head,teacher,draws
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
        save(p/f'stage_{task}.private.pt',dict(seen=seen,heads=hh,residual=res));seals.append(dict(task=task,seen=seen,at=time.time(),models=4));write(p/'STAGE_SEALS.json',seals);status('stage_sealed',task=task)
        cost['stage_sections'].append(dict(task=task,body_mapping_stats_solve_save_seconds=time.time()-stage_tick))
        del tokens,rows,new
    assert cost['science_prefix_rows']==68058 and cost['formal_solves']==24
    write(p/'FIT_SEAL.json',dict(at=time.time(),models=24,all_models_before_validation=True));write(p/'QUERY_OPEN.private.json',dict(at=time.time()))
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
