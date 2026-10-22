"""Actual-stage native latent replay; qualification gates every bulk encode."""
import copy
import json
import os
import signal
import time
import traceback
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from core import ARMS, Upper, Model, memory_indices, parameters_for_adam, update, empty_arm_cost, np_upper
from data import read_rows, Images


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp); tmp.replace(path)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def forward_rows(model, tokens, microbatch):
    output=[]
    with torch.inference_mode():
        for begin in range(0,len(tokens),microbatch):
            output.append(model(tokens[begin:begin+microbatch].to(next(model.parameters()).device,
                         dtype=next(model.parameters()).dtype)).cpu())
    return torch.cat(output)


def prefix(encoder, images):
    x=encoder.patch_embed(images)
    x=encoder._pos_embed(x)
    x=encoder.patch_drop(x)
    x=encoder.norm_pre(x)
    return encoder.blocks[:10](x)


def make_means(upper, tokens, rows, classes, cfg, arm_cost):
    # A compact 1536-dimensional temporary table, never a second token table.
    z=forward_rows(upper,tokens,cfg['microbatch']).double()
    arm_cost['initialization_rows']+=len(tokens)
    means=[]
    for c in classes:
        patients={}
        for i,r in enumerate(rows):
            if r['label']==c: patients.setdefault(r['group'],[]).append(i)
        means.append(torch.stack([z[ix].mean(0) for ix in patients.values()]).mean(0))
    return torch.stack(means).float()


def numpy_scores(tokens, weights, eps):
    upper={k.removeprefix('upper.'):v.numpy() for k,v in weights.items() if k.startswith('upper.')}
    z=np_upper(tokens,upper,eps)
    h=weights['head'].numpy().astype(np.float64)
    h/=np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-12)
    return 25.*z@h.T


def mathematical_check():
    from types import SimpleNamespace
    from timm.models.vision_transformer import Block
    from functools import partial
    torch.manual_seed(74002)
    norm=partial(torch.nn.LayerNorm,eps=1e-6)
    b=SimpleNamespace(blocks=torch.nn.Sequential(*([torch.nn.Identity() for _ in range(10)]+[
         Block(24,3,mlp_ratio=2,qkv_bias=True,norm_layer=norm) for _ in range(2)])),norm=norm(24))
    m=Model(Upper(b),torch.randn(3,48),True).double()
    x=torch.randn(2,3,24,dtype=torch.float64);y=torch.tensor([0,2])
    loss=F.cross_entropy(m(x),y);loss.backward();weights=state(m)
    def reference(w):
        z=np_upper(x.numpy(),{k.removeprefix('upper.'):v.numpy() for k,v in w.items() if k.startswith('upper.')},1e-6,heads=3)
        h=w['head'].numpy();h=h/np.linalg.norm(h,axis=1,keepdims=True)
        v=25.*z@h.T;v-=v.max(1,keepdims=True)
        return float((np.log(np.exp(v).sum(1))-v[np.arange(2),y.numpy()]).mean())
    objective_error=abs(reference(weights)-float(loss.detach()));errors=[]
    for name,pos in [('head',(0,1)),('upper.blocks.0.attn.qkv.weight',(1,2)),('upper.blocks.1.mlp.fc1.weight',(0,1))]:
        plus={k:v.clone() for k,v in weights.items()};minus={k:v.clone() for k,v in weights.items()}
        plus[name][pos]+=1e-5;minus[name][pos]-=1e-5
        fd=(reference(plus)-reference(minus))/2e-5
        errors.append(abs(fd-float(dict(m.named_parameters())[name].grad[pos])))
    assert objective_error<1e-9 and max(errors)<2e-6
    return dict(state='PASS',objective_error=objective_error,finite_difference_error=max(errors),
                synthetic_torch_rows=2,synthetic_backward_rows=2,numpy_objectives=7,numpy_rows=14,
                optimization_updates=0,NIH_images=0)


def main(cfg):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['output']); root.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,science_train_encodes=0,validation_encodes=0,
              encoder_seconds={},arms={a:empty_arm_cost() for a in ARMS},
              qualification_arms={a:empty_arm_cost() for a in ARMS},prefix_parameter_updates=0)
    models={};opts={};seals=[];curves=[]
    def status(phase,**extra):
        cost['wall_from_first']=time.time()-cfg['original_start_epoch']
        cost['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0
        write(root/'COSTS.private.json',cost)
        write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**extra))
        assert time.time()<cfg['hard_deadline_epoch']-60
        assert cost['GPU_peak_reserved_bytes']<cfg['gpu_reserved_budget_bytes']
    def interrupt(sig,frame):
        write(root/'INTERRUPTED.private.json',dict(at=time.time(),cost=cost))
        raise SystemExit(143)
    signal.signal(signal.SIGTERM,interrupt)
    status('mathematical_qualification')
    math_tick=time.time();math_receipt=mathematical_check();math_receipt['internal_seconds']=time.time()-math_tick
    write(root/'MATH_QUALIFICATION.json',math_receipt)
    status('loading_metadata')
    train,val=read_rows(cfg,'train'),read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300
    assert not {r['group'] for r in train}&{r['group'] for r in val}
    for i,r in enumerate(train):r['id']=i
    write(root/'COVERAGE.json',dict(train_counts=dict(Counter(str(r['label']) for r in train)),
          train=68058,val=300,patient_overlap=0,old_raw_image_replay=False,
          token_memory_permission='197x768 FP16 per retained patient; new permission, not old pooled features'))
    import timm
    from safetensors.torch import load_file
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    for block in encoder.blocks:block.attn.fused_attn=False
    for module in encoder.modules():
        if isinstance(module,torch.nn.Dropout):assert module.p==0
        if hasattr(module,'drop_prob'):assert module.drop_prob==0
    upper=Upper(encoder).cpu().float()
    assert upper.norm.eps==cfg['layernorm_epsilon']
    save(root/'initial_upper.private.pt',state(upper))
    # Original shared read-only weights reconstruct this seal independently.
    lower_before={k:v for k,v in state(encoder).items() if not k.startswith(('blocks.10.','blocks.11.','norm.'))}
    def extract(rows,field):
        out=torch.empty((len(rows),197,768),dtype=torch.float16)
        for images,idx in DataLoader(Images(rows,cfg),batch_size=cfg['encoder_batch'],num_workers=0,shuffle=False):
            tick=time.time()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):raw=prefix(encoder,images.cuda())
            assert raw.shape==(len(idx),197,768) and torch.isfinite(raw).all()
            out[idx]=raw.half().cpu();torch.cuda.synchronize()
            cost[field]+=len(idx);cost['encoder_seconds'][field]=cost['encoder_seconds'].get(field,0.)+time.time()-tick
            status(field,current_count=len(rows),encoded_in_current_set=int(idx[-1])+1)
        return out
    qrows=[]
    for c in cfg['order'][:10]:
        patients=set()
        for r in train:
            if r['label']==c and r['group'] not in patients:
                qrows.append(r);patients.add(r['group'])
                if len(patients)==2:break
        assert len(patients)==2
    q=extract(qrows,'qualification_encodes')
    qstart=time.time();qual={}
    # Prefix interface against the full original network, using no new images.
    images=torch.stack([Images(qrows,cfg)[i][0] for i in range(2)]).cuda()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        raw=prefix(encoder,images)
        full=encoder.forward_features(images)
        joined=encoder.norm(encoder.blocks[10:](raw))
    interface_error=float((full.float()-joined.float()).abs().max())
    assert interface_error<1e-4
    cost['qualification_interface_full_encoder_rows']=2
    cost['qualification_interface_prefix_rows']=2
    cost['qualification_interface_upper_rows']=2
    del images,raw,full,joined
    for a in ARMS:
        ac=cost['qualification_arms'][a]
        u=copy.deepcopy(upper).cuda()
        means=make_means(u,q[:16],qrows[:16],cfg['order'][:8],cfg,ac)
        m=Model(u,means.cuda(),a=='PLASTIC_UPPER').cuda()
        opt=parameters_for_adam(m,cfg)
        ix=torch.arange(64)%16
        update(m,opt,q[ix],torch.arange(8).repeat_interleave(2)[ix],cfg,ac)
        newmeans=make_means(m.upper,q[16:],qrows[16:],cfg['order'][8:10],cfg,ac)
        m.expand(newmeans.cuda());opt=parameters_for_adam(m,cfg)
        ix=torch.arange(64)%20
        update(m,opt,q[ix],torch.arange(10).repeat_interleave(2)[ix],cfg,ac)
        seal=state(m);save(root/f'qualification_{a}.private.pt',seal)
        loaded=Model(upper,seal['head'],a=='PLASTIC_UPPER').double()
        loaded.load_state_dict(torch.load(root/f'qualification_{a}.private.pt',weights_only=False))
        fp64=forward_rows(loaded,q[:2],1).numpy()
        ref=numpy_scores(q[:2].numpy(),seal,cfg['layernorm_epsilon'])
        error=float(abs(fp64-ref).max());assert error<=cfg['score_absolute_tolerance']
        ac['FP64_probe_rows']+=2;ac['numpy_probe_rows']+=2
        upper_delta=max(float((v-seal['upper.'+k]).abs().max()) for k,v in state(upper).items())
        assert upper_delta>0 if a=='PLASTIC_UPPER' else upper_delta==0
        qual[a]=dict(state='PASS',score_error=error,updates=2,actual_upper_parameter_max_delta=upper_delta,
                     total_parameters=sum(p.numel() for p in m.parameters()),
                     trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
        del u,m,opt,loaded,seal,fp64,ref
    assert all(torch.equal(v,encoder.state_dict()[k].cpu()) for k,v in lower_before.items())
    write(root/'QUALIFICATION.json',dict(state='PASS',arms=qual,interface_error=interface_error,
          qualification_encodes=20,engineering_updates=4,validation_encodes=0,
          residence_after_encoding_seconds=time.time()-qstart,scope='Real GPU full chain before bulk; not performance/ETA'))
    del q,qrows;torch.cuda.empty_cache()
    memory=torch.empty((0,197,768),dtype=torch.float16);memory_rows=[]
    for task in range(6):
        seen=cfg['order'][:sum(cfg['task_sizes'][:task+1])];new=seen[sum(cfg['task_sizes'][:task]):]
        rows=[r for r in train if r['label'] in new]
        tokens=extract(rows,'science_train_encodes')
        for a in ARMS:
            u=copy.deepcopy(upper).cuda() if task==0 else models[a].upper
            means=make_means(u,tokens,rows,new,cfg,cost['arms'][a])
            if task==0:
                models[a]=Model(u,means.cuda(),a=='PLASTIC_UPPER').cuda()
                del u
            else:models[a].expand(means.cuda())
            # Explicit per-stage optimizer reset, identical rule, no hidden resume.
            opts[a]=parameters_for_adam(models[a],cfg)
        source_rows=memory_rows+rows;pools={}
        for i,r in enumerate(source_rows):pools.setdefault(r['label'],{}).setdefault(r['group'],[]).append(i)
        assert set(pools)==set(seen)
        def gather(indices):
            return torch.stack([memory[i] if i<len(memory_rows) else tokens[i-len(memory_rows)] for i in indices])
        rng=np.random.default_rng(cfg['seed']+1000+task)
        steps=cfg['base_updates'] if task==0 else cfg['later_updates'];tick=time.time()
        for step in range(1,steps+1):
            indices=[]
            for c in rng.choice(seen,8,replace=False):
                groups=list(pools[c])
                for _ in range(8):
                    values=pools[c][groups[int(rng.integers(len(groups)))]]
                    indices.append(values[int(rng.integers(len(values)))])
            x=gather(indices);y=torch.tensor([seen.index(source_rows[i]['label']) for i in indices])
            details={a:update(models[a],opts[a],x,y,cfg,cost['arms'][a]) for a in ARMS}
            if step==1 or step%100==0 or step==steps:
                torch.cuda.synchronize();curves.append(dict(task=task,step=step,at=time.time(),arms=details,stage_residence=time.time()-tick))
                write(root/'TRAINING_CURVES.json',curves);status('training',task=task,step=step,steps=steps)
        probe=dict(tokens=x[:2].clone(),labels=y[:2].clone(),scores={})
        for a in ARMS:
            m=copy.deepcopy(models[a]).cpu().double()
            probe['scores'][a]=forward_rows(m,probe['tokens'],1)
            cost['arms'][a]['FP64_probe_rows']+=2;del m
        selected=memory_indices(rows,cfg['memory_per_class'],cfg['seed'])
        memory=torch.cat([memory,tokens[selected]])
        memory_rows.extend(rows[i] for i in selected)
        assert len(memory_rows)<=len(seen)*cfg['memory_per_class']
        assert len(memory_rows)==len({(r['label'],r['group']) for r in memory_rows})
        save(root/f'stage_{task}.private.pt',dict(seen=seen,models={a:state(models[a]) for a in ARMS},
             memory_tokens=memory,memory_rows=memory_rows,training_probe=probe))
        seals.append(dict(task=task,at=time.time(),seen=seen,validation_encodes=0,
                     science_train_encodes=cost['science_train_encodes'],updates={a:cost['arms'][a]['updates'] for a in ARMS},memory_images=len(memory_rows),
                     parameters={a:sum(p.numel() for p in models[a].parameters()) for a in ARMS},
                     trainable_parameters={a:sum(p.numel() for p in models[a].parameters() if p.requires_grad) for a in ARMS}))
        write(root/'STAGE_SEALS.json',seals)
        del tokens,rows,pools,source_rows,x,y,probe;torch.cuda.empty_cache()
    assert cost['science_train_encodes']==68058
    assert all(torch.equal(v,encoder.state_dict()[k].cpu()) for k,v in lower_before.items())
    save(root/'prefix_final.private.pt',{k:encoder.state_dict()[k].detach().cpu().clone() for k in lower_before})
    write(root/'FIT_SEAL.json',dict(at=time.time(),models=12,validation_encodes=0,prefix_parameters_unchanged=True))
    write(root/'QUERY_OPEN.private.json',dict(at=time.time(),validation_encodes=0))
    query=extract(val,'validation_encodes');labels=torch.tensor([r['label'] for r in val])
    save(root/'query.private.pt',dict(tokens=query,labels=labels))
    stages=[]
    for task in range(6):
        pack=torch.load(root/f'stage_{task}.private.pt',map_location='cpu',weights_only=False)
        seen=pack['seen'];mask=torch.isin(labels,torch.tensor(seen));metrics={}
        for a in ARMS:
            w=pack['models'][a];m=Model(upper,w['head'],a=='PLASTIC_UPPER').double();m.load_state_dict(w)
            score=forward_rows(m,query[mask],cfg['microbatch']);cost['arms'][a]['validation_rows']+=int(mask.sum())
            save(root/f'scores_{task}_{a}.private.pt',score)
            pred=torch.tensor(seen)[score.argmax(1)]
            correct={str(c):int(((pred==c)&(labels[mask]==c)).sum()) for c in seen}
            metrics[a]=dict(correct=correct,total_correct=sum(correct.values()),BA_percent=100*sum(correct.values())/int(mask.sum()))
            del m
        stages.append(dict(task=task,seen=seen,metrics=metrics));status('sealed_scoring',task=task)
    from verify import summarize
    result,decision=summarize(stages,cfg,Counter(r['label'] for r in train))
    write(root/'RESULTS.json',result);write(root/'DECISION.json',decision)
    status('science_complete');write(root/'COSTS.json',cost)
    write(root/'status.private.json',dict(state='COMPLETE_PENDING_VERIFY',at=time.time(),cost=cost))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException as e:
        root=Path(cfg['output'])
        if root.exists():write(root/'FAILURE.private.json',dict(at=time.time(),error=repr(e),traceback=traceback.format_exc()))
        raise
