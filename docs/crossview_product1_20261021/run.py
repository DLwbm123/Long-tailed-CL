"""One scientific encoding pass, two fixed cross-view products, no optimizer."""
import importlib.util
import json, os, time, traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core import raw_dictionary, base_means, center_gram, center_targets, factors, self_check


def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def write(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    base=Path(cfg['remote_base']);prior=Path(cfg['prior_run'])
    data=module(base/'case_hierarchical1_20261018/data.py','product_data')
    common=module(base/'matched_heads1_20261018/core.py','product_weights')
    df=module(base/'joint_controls1_20261019/core.py','product_df')
    nt=nv=nq=0
    def status(phase,**kw):
        write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),science_training_images=nt,validation_images=nv,qualification_encodes=nq,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60
        assert nt<=68058 and nv<=300 and nt+nq<=cfg['new_training_encode_cap']
        if torch.cuda.is_initialized():assert torch.cuda.max_memory_reserved()<=cfg['gpu_reserved_budget_bytes']
    status('math_qualification');assert self_check()
    toy=torch.tensor([[3.,1.],[1.,2.]],dtype=torch.float64);ev=df.spectrum(toy)
    assert abs(df.degrees(ev,1.)-float(torch.linalg.solve(toy+torch.eye(2,dtype=torch.float64),toy).trace()))<1e-12
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300
    assert not ({r['group'] for r in train}&{r['group'] for r in val})
    old_cfg=json.loads((prior/'config.private.json').read_text())
    for k in ['seed','order','task_sizes','manifest','images','weight','author_labels']:assert cfg[k]==old_cfg[k]
    import timm
    from safetensors.torch import load_file
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    def extract(rows):
        for images,idx in DataLoader(data.Images(rows,cfg),batch_size=64,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=encoder.forward_features(images.cuda())
            assert z.shape[1:]==(197,768) and torch.isfinite(z).all()
            yield z[:,0].double(),z[:,1:].float().mean(1).double(),idx
    qstart=time.time();qrows=[r for r in train if r['label']==cfg['order'][0]][:32];qualification={}
    for c,p,idx in extract(qrows):
        nq+=len(idx);status('real_product_qualification')
        raw=raw_dictionary(c,p);assert raw.shape==(32,3073) and torch.isfinite(raw).all()
        mc,mp=c.mean(0),p.mean(0)
        for arm in cfg['arms']:
            perm=torch.arange(768,device=c.device)
            if arm=='cross_shifted':perm=(perm+384)%768
            direct=torch.cat([c,p,(c-mc)*(p[:,perm]-mp[perm]),torch.ones(32,1,dtype=c.dtype,device=c.device)],1)
            feature_error=float((center_targets(raw.T,mc,mp,arm).T-direct).abs().max())
            gram_error=float((center_gram(raw.T@raw,mc,mp,arm)-direct.T@direct).abs().max())
            assert feature_error<1e-8 and gram_error<cfg['feature_atol']
            qualification[arm]=dict(feature_error=feature_error,gram_error=gram_error)
    torch.cuda.synchronize();qseconds=time.time()-qstart
    estimate=qseconds/nq*68058*1.5+1800
    assert estimate<cfg['max_wall_seconds']-300,'Frozen budget infeasible in qualification screen'
    write(root/'QUALIFICATION.json',dict(state='PASS',math=True,real_training_images=nq,training_encodes=nq,validation_encodes=0,optimizer_updates=0,errors=qualification,wall_seconds=qseconds,conditional_projected_seconds=estimate,forecast_not_upper_bound=True))
    grams=[];parity={}
    for ci,label in enumerate(cfg['order']):
        if ci>=10:assert (root/'BASE_SEAL.json').exists()
        rows=[r for r in train if r['label']==label]
        a=torch.load(prior/f'class_{label:02d}.private.pt',weights_only=True)
        b=torch.load(Path(cfg['matched_run'])/f'class_{label:02d}.private.pt',weights_only=True)
        assert len({r['group'] for r in rows})==a['groups']
        weights=common.weights([r['group'] for r in rows],'cuda')
        g=torch.zeros(3073,3073,dtype=torch.float64,device='cuda')
        for c,p,idx in extract(rows):
            nt+=len(idx);raw=raw_dictionary(c,p);mass=weights[idx.cuda()]
            g+=raw.T@(raw*mass[:,None]);status('training_moments',class_id=label,class_images=int(idx[-1])+1)
        g=g.cpu();ids_c=torch.tensor([*range(768),3072]);ids_p=torch.tensor([*range(768,1536),3072])
        errors=dict(CLS=float((g[ids_c][:,ids_c]-a['ridge_xx']).abs().max()),patch=float((g[ids_p][:,ids_p]-b['patch_mean_gram']).abs().max()))
        assert max(errors.values())<=cfg['feature_atol'] and abs(float(g[-1,-1])-1)<1e-9
        parity[str(label)]=errors
        torch.save(dict(gram=g,target=g[:,-1].clone(),images=len(rows),groups=a['groups']),root/f'class_{label:02d}.private.pt')
        grams.append(g)
        print(json.dumps(dict(event='class_complete',class_id=label,training_images=nt,parity=errors)),flush=True)
        if ci==9:
            mc,mp=base_means(grams)
            write(root/'BASE_SEAL.json',dict(frozen_at=time.time(),base_classes=cfg['order'][:10],training_images=nt,future_class_images=0,validation_images=nv,mu_c=mc.tolist(),mu_p=mp.tolist()))
    assert nt==68058 and nq==32
    base_g=sum(grams[:10]);mc,mp=base_means(grams[:10]);audits={};scales={}
    for arm in cfg['arms']:
        fac=factors(center_gram(base_g,mc,mp,arm));scales[arm]=fac.tolist();audits[arm]=[]
        for stage in range(6):
            n=sum(cfg['task_sizes'][:stage+1]);rg=sum(grams[:n])
            g=center_gram(rg,mc,mp,arm)*fac[:,None]*fac[None,:]
            target=center_targets(torch.stack([s[:,-1] for s in grams[:n]],1),mc,mp,arm)*fac[:,None]
            cls_ids=torch.tensor([*range(768),3072]);cg=rg[cls_ids][:,cls_ids]
            target_df=df.degrees(df.spectrum(cg),1.)
            alpha=df.match_alpha(df.spectrum(g),target_df)
            head=torch.linalg.solve(g+alpha*torch.eye(2305,dtype=torch.float64),target)
            torch.save(dict(head=head,alpha=alpha,factors=fac),root/f'{arm}_heads_{stage}.private.pt')
            audits[arm].append(dict(stage=stage,classes=n,target_df=target_df,alpha=alpha))
            status('fit_heads',arm=arm,stage=stage)
    write(root/'FIT_SEAL.json',dict(frozen_at=time.time(),query_opened=False,heads=12,stages=audits,factors=scales,training_parity=parity))
    write(root/'QUERY_OPEN.private.json',dict(opened_at=time.time(),validation_encodes=0))
    old_query=torch.load(Path(cfg['joint_run'])/'query768.private.pt',weights_only=True)
    query={};vparity={}
    for label in cfg['order']:
        rows=[r for r in val if r['label']==label];assert len(rows)==15
        rawparts=[];errors=[]
        for c,p,idx in extract(rows):
            nv+=len(idx);rawparts.append(raw_dictionary(c,p).cpu())
            errors.append(max(float((c.cpu()-old_query[label][0]).abs().max()),float((p.cpu()-old_query[label][1]).abs().max())))
            status('validation_features',class_id=label)
        assert max(errors)<=cfg['feature_atol'];vparity[str(label)]=max(errors);query[label]=torch.cat(rawparts)
    torch.save(query,root/'query.private.pt');write(root/'VALIDATION_PARITY.json',vparity)
    write(root/'costs.json',dict(wall_seconds=time.time()-cfg['original_start_epoch'],science_training_encodes=nt,qualification_training_encodes=nq,total_training_encodes=nt+nq,validation_encodes=nv,optimizer_updates=0,attempts=1,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    write(root/'status.private.json',dict(state='FIT_COMPLETE_PENDING_VERIFICATION',updated=time.time(),science_training_images=nt,qualification_encodes=nq,validation_images=nv))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        root=Path(cfg['output']);root.mkdir(exist_ok=True)
        state=json.loads((root/'status.private.json').read_text()) if (root/'status.private.json').exists() else {}
        write(root/'FAILURE.private.json',dict(updated=time.time(),last_status=state,error=traceback.format_exc()))
        state.update(state='FAILED',updated=time.time());write(root/'status.private.json',state)
        raise
