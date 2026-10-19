"""Fixed-checkpoint native null/range information diagnostic, no tuning."""
import json,os,time,sys,resource,traceback,math
from pathlib import Path
import torch
import numpy as np
from core_depth import PERM
from core_attribution import weights,norm_fit

def write(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)

def unitq(x):
    q,r=torch.linalg.qr(x,mode='reduced');assert float(r.diag().abs().min())>1e-8
    assert float((q.T@q-torch.eye(q.shape[1],dtype=q.dtype,device=q.device)).abs().max())<1e-8
    return q

def state_bytes():return 8*(1415**2+2*1415*6+1537+3073+1415+3072*96)+64

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CPU,(1200,1200));root=Path(cfg['output'])
    assert state_bytes()==18561200 and state_bytes()<19058800
    def check(phase,**kw):
        assert time.monotonic()-start<880 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30
        if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<4*2**30
        write(root/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    image_root=None;allowed=set();cache=Path(cfg['cache']).resolve();base=Path(cfg['base_bundle']).resolve()
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(cache.parent):assert p==cache and args[1] in ['r','rb']
            if p.is_relative_to(base.parent.parent):assert p==base and args[1] in ['r','rb']
            if image_root is not None and p.is_relative_to(image_root):assert p in allowed
            assert not any(s in p.name for s in ['test.rows','test.features','predictions.private','heads.private','transforms.private'])
    sys.addaudithook(guard)
    o=torch.load(cache,weights_only=False);g,y,groups,fold=o['g'],o['y'],o['groups'],np.asarray(o['fold']);del o
    o=torch.load(base,weights_only=False);idx=torch.where(~o['valid'])[0].tolist();rows=[o['rows'][i] for i in idx]
    assert torch.equal(y,o['y'][idx]) and groups==[o['groups'][i] for i in idx];del o
    assert g.shape==(2555,1536) and len(rows)==2555 and [int((y==c).sum()) for c in range(4)]==[307,20,170,2058]
    expected=np.full(len(y),-1);rg=np.random.default_rng(74125)
    for c in range(4):
        ks=sorted({groups[i] for i in torch.where(y==c)[0].tolist()});rg.shuffle(ks);assignment={k:j%4 for j,k in enumerate(ks)}
        for i in torch.where(y==c)[0].tolist():expected[i]=assignment[groups[i]]
    assert np.array_equal(fold,expected)
    bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text());image_root=Path(bc['datasets']['ISIC']['development_images']).resolve();allowed={Path(image_root,r['relative_path']).resolve() for r in rows}
    import timm
    from safetensors.torch import load_file
    from run_images import Images
    from torch.utils.data import DataLoader
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0);model.load_state_dict({k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda();b=model.blocks[5];assert isinstance(b.mlp.norm,torch.nn.Identity)
    W=b.mlp.fc2.weight.detach().double();assert W.shape==(768,3072)
    gen=torch.Generator().manual_seed(74130);R=torch.randn(3072,96,generator=gen,dtype=torch.float64).cuda();A=W@W.T;e=torch.linalg.eigvalsh(A);assert float(e[0])>1e-10*float(e[-1])
    vis=W.T@torch.linalg.solve(A,W@R);Pn=unitq(R-vis);Pv=unitq(vis);Pa=unitq(R)
    null_error=float((W@Pn).norm()/W.norm());cross=float((Pn.T@Pv).abs().max());assert null_error<1e-8 and cross<1e-8
    Ph=unitq(torch.randn(768,96,generator=gen,dtype=torch.float64).cuda())
    projections=dict(NULL=Pn,RANGE=Pv,NATIVE=Pa,CLS=Ph);torch.save({k:v.cpu() for k,v in projections.items()},root/'projections.private.pt')
    feats={a:[] for a in ['RANGE','NATIVE','NULL','UNIFORM_NULL','CLS']};extract=time.monotonic()
    for x,ii in DataLoader(Images(rows,dict(images=str(image_root))),batch_size=32,num_workers=0,shuffle=False):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            z=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x.cuda()))))
            for bb in model.blocks[:5]:z=bb(z)
            bn=b.norm1(z);qkv=b.attn.qkv(bn).reshape(len(z),len(z[0]),3,b.attn.num_heads,-1).permute(2,0,3,1,4);q,k,v=qkv.unbind(0);q=b.attn.q_norm(q);k=b.attn.k_norm(k)
            # Pooling weights are derived from the same frozen q/k, not fed back.
            with torch.autocast('cuda',enabled=False):
                att=((q[:,:,0:1].float()*b.attn.scale)@k.float().transpose(-2,-1)).softmax(-1)[:,:,0,1:].sum(1);att=att/att.sum(1,keepdim=True).clamp_min(1e-12)
            sa=z+b.drop_path1(b.ls1(b.attn(bn)));native=b.mlp.act(b.mlp.fc1(b.norm2(sa)));patch=native[:,1:].float()
            pooled=torch.einsum('bp,bpd->bd',att,patch);uniform=patch.mean(1)
            post=sa+b.drop_path2(b.ls2(b.mlp(b.norm2(sa))))
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            pd=pooled.double();ud=uniform.double();last=post[:,0].double()
            for name in ['RANGE','NATIVE','NULL']:feats[name].append((pd@projections[name]).cpu())
            feats['UNIFORM_NULL'].append((ud@Pn).cpu());feats['CLS'].append((last@Ph).cpu())
        check('features',processed=int(ii[-1])+1)
    extract_seconds=time.monotonic()-extract;del model;torch.cuda.empty_cache();feats={k:torch.cat(v) for k,v in feats.items()};assert all(x.shape==(2555,96) and bool(torch.isfinite(x).all()) for x in feats.values())
    torch.save(feats,root/'features.private.pt');arms=['G_PAD','RANGE','NATIVE','NULL','UNIFORM_NULL','CLS'];pred={a:np.full(len(y),-1,dtype=int) for a in arms};audits=[];heads=0
    for f in range(4):
        ti=np.where(fold!=f)[0];qi=np.where(fold==f)[0];assert not {groups[i] for i in ti}&{groups[i] for i in qi}
        yy=y[ti];ww=weights(yy,[groups[i] for i in ti]);core=g[:,PERM[:1318]];mu,scale=norm_fit(core[ti],ww);core=(core-mu)/scale
        for a in arms:
            if a=='G_PAD':extra=torch.zeros(len(y),96,dtype=torch.float64)
            else:
                m,s=norm_fit(feats[a][ti],ww);extra=(feats[a]-m)/s
            x=torch.cat([core/math.sqrt(2),extra/math.sqrt(2),torch.ones(len(y),1,dtype=torch.float64)],1);xt=x[ti];A=xt.T@(xt*ww[:,None])/4;A.diagonal()[:-1].add_(.01);B=xt.T@(torch.nn.functional.one_hot(yy,4).double()*ww[:,None])/4
            score=x[qi]@torch.cholesky_solve(B,torch.linalg.cholesky(A));assert bool(torch.isfinite(score).all());pred[a][qi]=score.argmax(1).numpy();heads+=1
        audits.append(dict(fold=f,train_rows=len(ti),query_rows=len(qi),class1_query_rows=int((y[qi]==1).sum()),overlap=0,normalization_and_head_train_only=True));check('fits',fold=f,heads=heads)
    assert heads==24 and all(bool((v>=0).all()) for v in pred.values());yn=y.numpy();c1=np.where(yn==1)[0];ks=sorted({groups[i] for i in c1});members=[np.array([i for i in c1 if groups[i]==k]) for k in ks];assert len(ks)==16
    risk={a:np.array([(pr[m]!=yn[m]).mean() for m in members]) for a,pr in pred.items()};gf=np.array([fold[m[0]] for m in members]);rng=np.random.default_rng(74131);draw=np.concatenate([rng.choice(np.where(gf==f)[0],(10000,4),replace=True) for f in range(4)],1)
    primary={}
    for a in ['G_PAD','RANGE','NATIVE']:
        dv=risk[a]-risk['NULL'];bs=100*dv[draw].mean(1);primary[a]=dict(null_benefit_pp=float(100*dv.mean()),CI95_pp=np.quantile(bs,[.025,.975]).tolist(),family3_CI_pp=np.quantile(bs,[.05/6,1-.05/6]).tolist())
    gate=primary['G_PAD']['null_benefit_pp']>=5 and all(v['family3_CI_pp'][0]>0 for v in primary.values())
    summaries=[dict(arm=a,class1_group_error_pp=float(100*risk[a].mean()),class1_image_error_pp=float(100*(pr[c1]!=yn[c1]).mean()),BA_pp=float(100*np.mean([(pr[yn==c]==c).mean() for c in range(4)]))) for a,pr in pred.items()]
    result=dict(status='COMPLETE',source_commit=cfg['source_commit'],training_rows=len(y),class1_query_rows=len(c1),class1_query_groups=len(ks),core=1318,extra=96,heads=heads,fold_audit=audits,summaries=summaries,primary=primary,necessary_null_gate=bool(gate),method_success=False,next_decision='ANALYZE_NOT_METHOD_SUCCESS' if gate else 'STOP_NO_NULL_SPECIFIC_EVIDENCE',numerics=dict(W2_gram_eigen_ratio=float(e[0]/e[-1]),null_relative_error=null_error,null_range_cross=cross),planned_max_state_bytes=state_bytes(),test_used=False,permanent_validation_used=False,patient_isolation='UNKNOWN',costs=dict(body_wall_seconds=time.monotonic()-start,feature_extraction_seconds=extract_seconds))
    write(root/'aggregate.json',result);write(root/'status.json',dict(state='COMPLETE',necessary_gate=bool(gate)))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);start=time.time()
    try:main(cfg);write(root/'EXIT.json',dict(exit_code=0))
    except Exception:write(root/'failure.private.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise
    finally:
        u=resource.getrusage(resource.RUSAGE_SELF);write(root/'COST_RECEIPT.json',dict(entry_wall_seconds=time.time()-start,cpu_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
