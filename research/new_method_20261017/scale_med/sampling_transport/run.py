"""Privileged offline audit of sampling influence, never a deployed learner."""
import json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from core_depth import mapped
from core_attribution import weights

def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def solve(h,b):
    return torch.cholesky_solve(b,torch.linalg.cholesky(h))

def selfcheck():
    g=torch.Generator().manual_seed(74118)
    x=torch.randn(25,7,generator=g,dtype=torch.float64)
    h=x.T@x+torch.eye(7,dtype=x.dtype)
    b=torch.randn(7,3,generator=g,dtype=x.dtype);w=solve(h,b)
    da=.0001*(x[:3].T@x[:3]);db=torch.randn(7,3,generator=g,dtype=x.dtype)*.0001
    exact=solve(h+da,b+db)-w;linear=solve(h,db-da@w)
    assert float((exact-linear).norm()/exact.norm())<.001
    assert torch.allclose((h+da)@exact,db-da@w,atol=1e-12)

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);selfcheck()
    resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
    out=Path(cfg['output']);old=Path(cfg['reference']);cache=Path(cfg['training_cache'])
    def checkpoint(phase,**kw):
        assert time.monotonic()-start<3600,'wall budget'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<16*2**30,'RSS budget'
        write(out/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=str(Path(os.fsdecode(args[0])).resolve())
            assert not any(s in p for s in ['predictions.private','test.features','test.rows','/public_results/','/results.json']), 'Test/result read denied'
    sys.addaudithook(guard)
    trans=torch.load(old/'transforms.private.pt',weights_only=False)
    ys=[];groups=[];valid_bundles=[];stages=[];metadata=set()
    for stage in range(3):
        obj=torch.load(old/'inputs'/f'{stage}.private.pt',weights_only=False)
        tr=~obj['valid'];ys.append(obj['y'][tr]);groups += [obj['groups'][i] for i in torch.where(tr)[0].tolist()]
        stages += [stage]*int(tr.sum())
        ids=torch.where(obj['valid'])[0].tolist()
        assert not {obj['groups'][i] for i in ids}&{obj['groups'][i] for i in torch.where(tr)[0].tolist()}
        rr=[obj['rows'][i] for i in ids]
        for row in rr:metadata.update(row.keys())
        valid_bundles.append(dict(rows=rr,g=obj['g'][obj['valid']],y=obj['y'][obj['valid']],groups=[obj['groups'][i] for i in ids]))
    y=torch.cat(ys);stageids=torch.tensor(stages)
    assert np.array_equal(y.numpy(),np.load(cache/'labels.private.npy')) and len(y)==4395
    assert sum(len(v['y']) for v in valid_bundles)==1131
    bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text())
    import timm
    from safetensors.torch import load_file
    from run_images import Images
    from torch.utils.data import DataLoader
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda();valx={a:[] for a in ['G','C']};vy=[];vg=[];gpu_start=time.monotonic()
    for stage,v in enumerate(valid_bundles):
        values=[]
        for x,idx in DataLoader(Images(v['rows'],dict(images=bc['datasets']['ISIC']['development_images'])),batch_size=64,num_workers=0,shuffle=False):
            assert time.monotonic()-gpu_start<1800,'GPU phase budget'
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x.cuda()))))
                for block in model.blocks[:6]:z=block(z)
                values.append(z[:,0].float().cpu().double())
            checkpoint('validation_features',stage=stage,rows=int(idx[-1])+1)
        h=torch.cat(values)
        for a in valx:valx[a].append(mapped(v['g'],h,trans[a]))
        vy.append(v['y']);vg+=v['groups']
    gpu_seconds=time.monotonic()-gpu_start;peak_gpu=torch.cuda.max_memory_allocated()
    del model;torch.cuda.empty_cache()
    valx={a:torch.cat(v) for a,v in valx.items()};vy=torch.cat(vy)
    assert not set(groups)&set(vg)
    train={a:torch.from_numpy(np.load(cache/f'{a}.features.private.npy')) for a in valx}
    ww=weights(y,groups);hot=torch.nn.functional.one_hot(y,6).double();rng=np.random.default_rng(74119)
    perturb={};group_counts={}
    for c in [1,3,4]:
        keys=sorted({groups[i] for i in torch.where(y==c)[0].tolist()});group_counts[str(c)]=len(keys)
        ids=torch.where(y==c)[0];lookup={k:i for i,k in enumerate(keys)};gid=np.array([lookup[groups[i]] for i in ids.tolist()])
        perturb[c]=[]
        for rep in range(8):
            counts=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys))
            dw=torch.zeros_like(ww);dw[ids]=ww[ids]*torch.from_numpy(counts[gid]-1)
            assert abs(float(dw.sum()))<1e-10 and bool((ww+dw>=0).all())
            perturb[c].append(dw)
    parity={};baselines={};results=[]
    def metrics(scores,seen):
        mask=vy<seen;p=scores[mask].argmax(1);yy=vy[mask]
        recalls=[float((p[yy==c]==c).double().mean()) for c in range(seen)]
        return dict(BA=sum(recalls)/seen,recall=recalls)
    for arm,x in train.items():
        vv=valx[arm];states={};base={};chols={}
        for stage in range(3):
            seen=stage+4;mask=stageids<=stage
            A=x[mask].T@(x[mask]*ww[mask,None]);B=x[mask].T@(hot[mask]*ww[mask,None])
            saved=torch.load(old/f'{stage}.statistics.private.pt',weights_only=False)[arm]
            error=max(float((A-saved[0]).abs().max()),float((B-saved[1]).abs().max()));assert error<1e-8
            parity[f'{arm}.{stage}']=error;states[stage]=(A,B)
            hh=torch.load(old/f'{stage}.heads.private.pt',weights_only=False)[arm]
            for alpha in [.0001,.01,1.]:
                H=A.clone();H.diagonal()[:-1].add_(seen*alpha);L=torch.linalg.cholesky(H)
                W=torch.cholesky_solve(B[:,:seen],L)
                err=float((vv@W-vv@hh[str(alpha)]).abs().max());assert err<1e-7
                base[stage,alpha]=W;chols[stage,alpha]=L
                baselines[f'{arm}.{stage}.{alpha}']=metrics(vv@W,seen)
        for c in [1,3,4]:
            birth=0 if c<4 else 1;other=1 if c==3 else 3;query=vv[vy==c]
            deltas=[]
            for dw in perturb[c]:
                ids=torch.where(y==c)[0];xx=x[ids]
                deltas.append((xx.T@(xx*dw[ids,None]),xx.T@(hot[ids]*dw[ids,None])))
            for alpha in [.0001,.01,1.]:
                birthW=base[birth,alpha];birthL=chols[birth,alpha]
                for stage in range(birth,3):
                    seen=stage+4;W=base[stage,alpha];L=chols[stage,alpha];A,B=states[stage]
                    exacts=[];dynamic=[];static=[];mm=[]
                    for rep,(da,db) in enumerate(deltas):
                        H=A+da;H.diagonal()[:-1].add_(seen*alpha)
                        Wp=solve(H,(B+db)[:,:seen]);delta=Wp-W
                        dyn=torch.cholesky_solve(db[:,:seen]-da@W,L)
                        sta=torch.cholesky_solve(db[:,:birth+4]-da@birthW,birthL)
                        def contrast(D):return (query@(D[:,c]-D[:,other])).numpy()
                        exacts.append(contrast(delta));dynamic.append(contrast(dyn));static.append(contrast(sta));mm.append(metrics(vv@Wp,seen))
                        checkpoint('refits',arm=arm,class_index=c,alpha=alpha,stage=stage,replicate=rep)
                    e=np.stack(exacts);d=np.stack(dynamic);s=np.stack(static)
                    rms=float(np.sqrt(np.mean(e**2)));de=float(np.sqrt(np.mean((d-e)**2)));se=float(np.sqrt(np.mean((s-e)**2)))
                    rec=np.array([m['recall'] for m in mm]);ba=np.array([m['BA'] for m in mm])
                    row=dict(arm=arm,source_class=c,stage=stage,alpha=alpha,query_count=len(query),contrast_class=other,exact_margin_RMS=rms,margin_repeat_variance=float(e.var(axis=0,ddof=1).mean()),dynamic_RMSE=de,static_RMSE=se,dynamic_relative_error=de/max(rms,1e-30),dynamic_to_static_RMSE=de/max(se,1e-30),BA_mean=float(ba.mean()),BA_sd=float(ba.std(ddof=1)),recall_mean=rec.mean(0).tolist(),recall_sd=rec.std(0,ddof=1).tolist())
                    results.append(row);write(out/'partial.json',dict(cells=results,baselines=baselines,parity=parity))
    def cell(c):return next(r for r in results if r['arm']=='C' and r['source_class']==c and r['stage']==2 and r['alpha']==.01)
    main=cell(4);aux=cell(1)
    passed=main['exact_margin_RMS']>1e-8 and main['dynamic_to_static_RMSE']<=.8 and main['dynamic_relative_error']<=.25 and aux['dynamic_RMSE']<aux['static_RMSE']
    costs=dict(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),GPU_phase_seconds=gpu_seconds,peak_GPU_bytes=peak_gpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    assert len(results)==48
    write(out/'aggregate.json',dict(source_commit=cfg['source_commit'],status='COMPLETE',test_access=False,offline_privileged_audit=True,fixed_representation_conditional=True,groups=group_counts,metadata_keys=sorted(metadata),patient_isolation='UNKNOWN',distribution_shift_identification='UNRESOLVED_METADATA_NOT_ESTABLISHED',training_rows=len(y),validation_rows=len(vy),cells=results,baselines=baselines,parity=parity,costs=costs,gate=bool(passed),next_decision='ANALYZE_NOT_METHOD_SUCCESS' if passed else 'STOP_LINEAR_TRANSPORT_ROUTE'))
    write(out/'status.json',dict(state='COMPLETE',cells=len(results),gate=bool(passed)))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());root=Path(cfg['output']);t=time.time()
    try:main(cfg);write(root/'EXIT.json',dict(exit_code=0,wall_seconds=time.time()-t))
    except Exception:
        write(root/'failure.private.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1,wall_seconds=time.time()-t));raise
