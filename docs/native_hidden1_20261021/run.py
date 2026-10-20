"""One frozen native-MLP pair, shared encoding, base-only coordinate seal."""
import importlib.util
import json, os, time, traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core import NativeCapture, choose_hidden, feature_indices, self_check


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
    data=module(base/'case_hierarchical1_20261018/data.py','native_data')
    common=module(base/'matched_heads1_20261018/core.py','native_weights')
    df=module(base/'joint_controls1_20261019/core.py','native_df')
    nt=nv=nq=0
    def status(phase,**kw):
        write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),
              science_training_images=nt,validation_images=nv,qualification_encodes=nq,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60
        assert nt<=68058 and nv<=300 and nt+nq<=cfg['new_training_encode_cap']
        if torch.cuda.is_initialized():assert torch.cuda.max_memory_reserved()<=cfg['gpu_reserved_budget_bytes']
    status('math_qualification');self_check(df)
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
    qrows=[r for r in train if r['label']==cfg['order'][0]][:32]
    qstart=time.time();capture=NativeCapture(encoder);full={}
    def save_full(name):
        def hook(m,args,output):full[name]=output.detach()
        return hook
    mlp=encoder.blocks[11].mlp
    qhooks=[mlp.act.register_forward_hook(save_full('hidden')),mlp.fc2.register_forward_hook(save_full('post'))]
    for images,idx in DataLoader(data.Images(qrows,cfg),batch_size=64,shuffle=False,num_workers=0):
        images=images.cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=encoder.forward_features(images);nq+=len(idx);raw=capture.take(z)
            reconstructed=torch.nn.functional.linear(full['hidden'],mlp.fc2.weight,mlp.fc2.bias)
        status('hook_fc2_qualification')
        fc2_error=float((reconstructed-full['post']).abs().max())
        assert torch.equal(reconstructed,full['post'])
        assert torch.equal(raw[:,768:1536],full['post'][:,0].double())
        assert torch.equal(raw[:,1536:],full['hidden'][:,0].double())
        for h in qhooks:h.remove()
        capture.close()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            original=encoder.forward_features(images);nq+=len(idx)
        status('hook_identity_qualification')
        cls_error=float((original[:,0]-z[:,0]).abs().max())
        assert torch.equal(original,z)
    del images,z,original,raw,reconstructed,full
    torch.cuda.synchronize();qseconds=time.time()-qstart
    estimate=qseconds/32*68058*1.5+1200
    assert estimate<cfg['max_wall_seconds']-300,'Qualification cannot support frozen wall budget'
    write(root/'QUALIFICATION.json',dict(state='PASS',unique_training_images=32,training_encodes=nq,
          val_encodes=0,optimizer_updates=0,math_checks=True,fc2_reconstruction_error=fc2_error,
          hooked_unhooked_CLS_error=cls_error,all_final_tokens_identical=True,
          wall_seconds=qseconds,conditional_projected_seconds=estimate,forecast_is_not_upper_bound=True))
    capture=NativeCapture(encoder)
    def extract(rows):
        for images,idx in DataLoader(data.Images(rows,cfg),batch_size=64,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=encoder.forward_features(images.cuda())
            raw=capture.take(z)
            yield torch.cat([raw,torch.ones(len(raw),1,dtype=torch.float64,device='cuda')],1),idx
    grams=[];parity={};selection=None
    cls_ids=torch.tensor([*range(768),4608])
    for ci,c in enumerate(cfg['order']):
        assert ci<10 or (root/'BASE_SELECTION.json').exists()
        rows=[r for r in train if r['label']==c]
        old=torch.load(prior/f'class_{c:02d}.private.pt',weights_only=True)
        weights=common.weights([r['group'] for r in rows],'cuda')
        assert len({r['group'] for r in rows})==old['groups']
        g=torch.zeros(4609,4609,dtype=torch.float64,device='cuda')
        for x,idx in extract(rows):
            nt+=len(idx);g+=x.T@(x*weights[idx.cuda(),None])
            status('training_moments',class_id=c,class_images=int(idx[-1])+1)
        g=g.cpu();error=float((g[cls_ids][:,cls_ids]-old['ridge_xx']).abs().max())
        assert error<cfg['feature_atol'] and abs(float(g[-1,-1])-1)<1e-9
        parity[str(c)]=error
        torch.save(dict(gram=g,target=g[:,-1].clone(),images=len(rows),groups=old['groups']),root/f'class_{c:02d}.private.pt')
        grams.append(g);print(json.dumps(dict(event='class_complete',class_id=c,training_images=nt,CLS_gram_error=error)),flush=True)
        if ci==9:
            selection=choose_hidden(grams)
            write(root/'BASE_SELECTION.json',dict(base_classes=cfg['order'][:10],raw_hidden_indices=selection.tolist(),
                  frozen_at=time.time(),training_images=nt,future_class_images=0,validation_images=0))
    assert nt==68058
    ids_by_arm=feature_indices(selection);scales={};audits={}
    for arm,ids in ids_by_arm.items():
        sub=[g[ids][:,ids] for g in grams]
        ec=sum(float(g[:768,:768].trace()) for g in sub[:10])/10
        ep=sum(float(g[768:1536,768:1536].trace()) for g in sub[:10])/10
        assert ec>0 and ep>0;scale=(ec/ep)**.5;scales[arm]=scale;audits[arm]=[]
        for stage in range(6):
            n=sum(cfg['task_sizes'][:stage+1]);total=torch.stack(sub[:n]).sum(0)
            target=torch.stack([g[:,-1] for g in sub[:n]],1)
            g,t,factors=df.scaled(total,target,scale)
            cls_g=torch.stack([s[cls_ids][:,cls_ids] for s in grams[:n]]).sum(0)
            target_df=df.degrees(df.spectrum(cls_g),1.)
            alpha=df.match_alpha(df.spectrum(g),target_df)
            head=torch.linalg.solve(g+alpha*torch.eye(1537,dtype=torch.float64),t)
            torch.save(dict(head=head,alpha=alpha,factors=factors,indices=ids),root/f'{arm}_heads_{stage}.private.pt')
            audits[arm].append(dict(stage=stage,target_df=target_df,alpha=alpha))
            status('fit_heads',arm=arm,stage=stage)
    write(root/'FIT_SEAL.json',dict(query_opened=False,frozen_at=time.time(),scales=scales,stages=audits,training_CLS_parity=parity,heads=12))
    query={}
    for c in cfg['order']:
        rows=[r for r in val if r['label']==c];assert len(rows)==15
        xs=[]
        for x,idx in extract(rows):nv+=len(idx);xs.append(x.cpu())
        x=torch.cat(xs);ref=torch.load(prior/f'val_{c:02d}.private.pt',weights_only=True)[2].double()
        assert float((x[:,:768]-ref).abs().max())<=cfg['feature_atol']
        query[c]=x;status('validation_features',class_id=c)
    capture.close();torch.save(query,root/'query.private.pt')
    write(root/'costs.json',dict(wall_seconds=time.time()-cfg['original_start_epoch'],science_training_encodes=nt,
          qualification_unique_images=32,qualification_training_encodes=nq,total_training_encodes=nt+nq,
          validation_encodes=nv,optimizer_updates=0,GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
          GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),attempts=1))
    write(root/'status.private.json',dict(state='FIT_COMPLETE_PENDING_VERIFICATION',updated=time.time(),science_training_images=nt,qualification_encodes=nq,validation_images=nv))


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        r=Path(cfg['output']);r.mkdir(exist_ok=True)
        write(r/'failure.private.json',dict(traceback=traceback.format_exc(),time=time.time()))
        raise
