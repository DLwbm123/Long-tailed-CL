import importlib.util,json,math,os,signal,time,traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from core import Flow,ARMS,layout,aggregate,context,whiten,conditions,math_check

def write(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['output']);root.mkdir(exist_ok=False)
    cost=dict(qualification_encodes=0,reference_encodes=0,science_training_encodes=0,validation_encodes=0,qualification_updates=0,optimizer_updates=0,arm_updates={a:0 for a in ARMS},feature_image_exposures={a:0 for a in ARMS},encoder_seconds={})
    def status(phase,**kw):
        cost.update(wall_from_first=time.time()-cfg['original_start_epoch'],GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
        write(root/'COSTS.private.json',cost);write(root/'status.private.json',dict(state='RUNNING',phase=phase,updated=time.time(),cost=cost,**kw))
        assert time.time()<cfg['hard_deadline_epoch']-60
        assert cost['GPU_peak_reserved_bytes']<=cfg['gpu_reserved_budget_bytes']
    def stopped(sig,frame):
        status('TERM_SAVED');raise SystemExit(143)
    signal.signal(signal.SIGTERM,stopped)
    status('math_qualification');maths=math_check()
    data=module(Path(cfg['remote_base'])/'case_hierarchical1_20261018/data.py','data_fixed')
    train,val=data.read_rows(cfg,'train'),data.read_rows(cfg,'val')
    assert len(train)==68058 and len(val)==300 and not ({r['group'] for r in train}&{r['group'] for r in val})
    counts={str(c):sum(r['label']==c for r in train) for c in cfg['order']}
    write(root/'COVERAGE.json',dict(train_counts=counts,patient_overlap=0,train_patients=len({r['group'] for r in train}),val_patients=len({r['group'] for r in val})))
    import timm
    from safetensors.torch import load_file
    encoder=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    encoder.load_state_dict({k:v for k,v in load_file(cfg['weight']).items() if not k.startswith('head.')},strict=True)
    encoder.eval().requires_grad_(False).cuda()
    def extract(rows,field):
        tick=time.time()
        for images,idx in DataLoader(data.Images(rows,cfg),batch_size=32,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=encoder.forward_features(images.cuda())
            assert z.shape[1:]==(197,768) and torch.isfinite(z).all()
            torch.cuda.synchronize();cost.setdefault('encoder_seconds',{}).setdefault(field,0.);cost['encoder_seconds'][field]+=time.time()-tick
            cost[field]+=len(idx)
            yield z[:,1:].float(),z[:,0].float(),idx
            tick=time.time()
    m,prior,pos,perm=layout('cuda',torch.float64);gen=torch.Generator().manual_seed(cfg['seed'])
    projection=torch.linalg.qr(torch.randn(768,32,generator=gen,dtype=torch.float64)).Q.cuda()
    def weights(rows):
        groups={}
        for r in rows:groups[r['group']]=groups.get(r['group'],0)+1
        return torch.tensor([1/(len(groups)*groups[r['group']]) for r in rows],dtype=torch.float64,device='cuda')
    def moments(rows,field):
        w=weights(rows);g=torch.zeros(33,33,dtype=torch.float64,device='cuda');xz=torch.zeros(196,33,768,dtype=torch.float64,device='cuda');zz=torch.zeros(196,768,dtype=torch.float64,device='cuda')
        for patch,cls,idx in extract(rows,field):
            x=context(cls,projection);v=patch.double();mass=w[idx.cuda()]
            g+=x.T@(x*mass[:,None]);xz+=torch.einsum('bi,bjd,b->jid',x,v,mass);zz+=torch.einsum('bjd,b->jd',v.square(),mass)
            status(field,class_id=rows[0]['label'])
        return g,xz,zz
    def reference(g,xz,zz):
        reg=torch.eye(33,dtype=torch.float64,device='cuda')*.001;reg[-1,-1]=0
        beta=torch.linalg.solve(g+reg,xz.permute(1,0,2).reshape(33,-1)).reshape(33,196,768).permute(1,0,2).contiguous()
        var=zz-2*(beta*xz).sum(1)+torch.einsum('jid,ik,jkd->jd',beta,g,beta)
        assert torch.isfinite(var).all() and var.min()>-1e-6
        return dict(projection=projection,beta=beta,sigma=var.clamp_min(.0001).sqrt())
    qrows=[r for r in train if r['label']==cfg['order'][0]][:32]
    start=time.time();g,xz,zz=moments(qrows,'qualification_encodes');q=reference(g,xz,zz)
    # One engineering update per arm on the same 32 training images; no second encoder pass.
    # moments deliberately retains only aggregates, so a new 32-image forward is separately charged.
    qual_updates={}
    for patch,cls,idx in extract(qrows,'qualification_encodes'):
        u,c=whiten(patch,cls,q);cond=conditions(c,pos)
        opt_start=time.time()
        for ai,arm in enumerate(ARMS):
            torch.manual_seed(cfg['seed']);f=Flow().double().cuda();assert f.ratio(u,cond).abs().max()==0
            opt=torch.optim.AdamW(f.parameters(),lr=.0001,weight_decay=.0001)
            loss=-aggregate(f.ratio(u,cond),arm,m,prior,perm).mean()/196
            opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(f.parameters(),5.,error_if_nonfinite=True);assert norm>0
            opt.step();cost['qualification_updates']+=1
            from verify import manual_ratio,manual_score
            with torch.no_grad():
                actual=aggregate(f.ratio(u,cond),arm,m,prior,perm)
                independent=manual_score(manual_ratio(u.double(),cond.double(),{k:v.double() for k,v in f.state_dict().items()}),arm)
                error=float((actual.double()-independent).abs().max());assert error<=cfg['score_absolute_tolerance']
            qual_updates[arm]=dict(loss=float(loss),gradient_norm=float(norm),independent_score_error=error)
        torch.cuda.synchronize();qualification_training_seconds=time.time()-opt_start
    assert cost['qualification_encodes']==64
    projected=1.5*(cost['encoder_seconds']['qualification_encodes']/64*128187+qualification_training_seconds/32*340290)+7200
    assert projected<cfg['max_wall_seconds']-600,'Qualification conditional cost screen exceeds fixed budget'
    write(root/'QUALIFICATION.json',dict(state='PASS',math=maths,training_step_three_arms_seconds=qualification_training_seconds,real_training_encodes=64,validation_encodes=0,updates=qual_updates,wall_seconds=time.time()-start,conditional_projected_seconds=projected,not_formal_throughput_or_upper_bound=True))
    del f,opt,u,c,cond,q,g,xz,zz;torch.cuda.empty_cache()
    gs=torch.zeros(33,33,dtype=torch.float64,device='cuda');xs=torch.zeros(196,33,768,dtype=torch.float64,device='cuda');zs=torch.zeros(196,768,dtype=torch.float64,device='cuda')
    for label in cfg['order'][:10]:
        rows=[r for r in train if r['label']==label];g,xz,zz=moments(rows,'reference_encodes');gs+=g/10;xs+=xz/10;zs+=zz/10
    assert cost['reference_encodes']==60129
    q=reference(gs,xs,zs);torch.save({k:v.cpu() for k,v in q.items()},root/'reference.private.pt')
    # Refit provenance from sufficient statistics is retained without any training sample vectors.
    torch.save(dict(g=gs.cpu(),xz=xs.cpu(),zz=zs.cpu()),root/'reference_moments.private.pt')
    write(root/'REFERENCE_SEAL.json',dict(at=time.time(),base_images=60129,future_images=0,val_images=0,reference_parameter_bytes=sum(v.numel()*v.element_size() for v in q.values())))
    del gs,xs,zs,g,xz,zz
    class_audits=[];curve=[]
    for ci,label in enumerate(cfg['order']):
        rows=[r for r in train if r['label']==label];n=len(rows);status('class_encoding',class_id=label)
        # Transient current-class RAM only; released before the next class. Never written as a cache.
        tokens=torch.empty(n,196,768,dtype=torch.float16);global_cls=torch.empty(n,768,dtype=torch.float32)
        for patch,cls,idx in extract(rows,'science_training_encodes'):
            tokens[idx]=patch.cpu().half();global_cls[idx]=cls.cpu();status('class_encoding',class_id=label,encoded=int(idx[-1])+1)
        w=weights(rows).cpu().float()*n;models={};opts={}
        for arm in ARMS:
            torch.manual_seed(cfg['seed']+label);models[arm]=Flow().double().cuda();opts[arm]=torch.optim.AdamW(models[arm].parameters(),lr=.0001,weight_decay=.0001)
        torch.save(models['G'].state_dict(),root/'initial.private.pt')
        for epoch in range(1,6):
            order=torch.randperm(n,generator=torch.Generator().manual_seed(cfg['seed']+label*1009+epoch))
            sums={a:0. for a in ARMS};steps=0
            for ids in order.split(32):
                u,c=whiten(tokens[ids].cuda().float(),global_cls[ids].cuda(),q);cond=conditions(c,pos);mass=w[ids].cuda()
                for arm in ARMS:
                    opts[arm].zero_grad();score=aggregate(models[arm].ratio(u,cond),arm,m,prior,perm)
                    loss=-(score*mass).mean()/196;assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(models[arm].parameters(),5.,error_if_nonfinite=True);opts[arm].step()
                    cost['optimizer_updates']+=1;cost['arm_updates'][arm]+=1;cost['feature_image_exposures'][arm]+=len(ids);sums[arm]+=float(loss.detach())
                steps+=1
                if steps%8==0:status('class_training',class_id=label,epoch=epoch,epoch_steps=steps)
            record=dict(class_id=label,epoch=epoch,steps=steps,mean_training_NLL_relative_per_patch={a:sums[a]/steps for a in ARMS})
            curve.append(record);write(root/'TRAINING_CURVES.json',curve);status('class_epoch_complete',**record)
        for arm in ARMS:
            torch.save({k:v.cpu() for k,v in models[arm].state_dict().items()},root/f'{arm}_{label:02d}.private.pt')
        class_audits.append(dict(class_id=label,at=time.time(),images=n,patient_count=len({r['group'] for r in rows}),steps_per_arm=5*math.ceil(n/32),parameters_per_arm=sum(v.numel() for v in models['G'].parameters()),val_images=0,future_class_encodes=0))
        write(root/'CLASS_SEALS.json',class_audits)
        del tokens,global_cls,models,opts,u,c,cond;torch.cuda.empty_cache()
    expected=5*sum(math.ceil(int(n)/32) for n in counts.values())
    assert cost['science_training_encodes']==68058 and all(n==expected for n in cost['arm_updates'].values())
    assert all(n==5*68058 for n in cost['feature_image_exposures'].values())
    write(root/'FIT_SEAL.json',dict(at=time.time(),models=60,classes=class_audits,query_opened=False,updates_per_arm=expected))
    write(root/'QUERY_OPEN.private.json',dict(at=time.time(),validation_encodes=0))
    query=[];clss=[];labels=[]
    for label in cfg['order']:
        rows=[r for r in val if r['label']==label];assert len(rows)==15
        for patch,cls,idx in extract(rows,'validation_encodes'):query.append(patch.cpu().half());clss.append(cls.cpu());labels += [label]*len(idx)
    query=torch.cat(query);clss=torch.cat(clss);labels=torch.tensor(labels);torch.save(dict(patch=query,cls=clss,labels=labels),root/'query.private.pt')
    allscores={a:torch.empty(300,20,dtype=torch.float64) for a in ARMS};posterior={}
    for ci,label in enumerate(cfg['order']):
        for arm in ARMS:
            f=Flow().double().cuda();f.load_state_dict(torch.load(root/f'{arm}_{label:02d}.private.pt',weights_only=True));f.eval()
            area=[]
            with torch.no_grad():
                for ids in torch.arange(300).split(16):
                    u,c=whiten(query[ids].cuda().float(),clss[ids].cuda(),q);ratio=f.ratio(u,conditions(c,pos));v=aggregate(ratio,arm,m,prior,perm);allscores[arm][ids,ci]=v.cpu()
                    if arm=='E':
                        post=(ratio@m.T+prior).softmax(-1);area.append((post@(m.sum(1)/196)).cpu())
            if area:posterior[str(label)]=dict(mean_area_fraction=float(torch.cat(area).mean()),global_like_fraction=float((torch.cat(area)>.95).float().mean()))
            status('validation_density_scoring',class_id=label,arm=arm)
    torch.save(dict(scores=allscores,labels=labels,order=cfg['order']),root/'scores.private.pt');write(root/'AREA_DIAGNOSTICS.json',posterior)
    assert cost['validation_encodes']==300
    status('FIT_COMPLETE_PENDING_VERIFICATION');write(root/'COSTS.json',cost)

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    try:main(cfg)
    except BaseException:
        root=Path(cfg['output']);root.mkdir(exist_ok=True);write(root/'FAILURE.private.json',dict(at=time.time(),error=traceback.format_exc()));raise
