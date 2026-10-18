"""Separate class-count ridge scaling from new-class covariance; audit only."""
import json,os,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from core_depth import mapped
from core_attribution import weights
from audit import selfcheck,solve,write

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CPU,(600,600));selfcheck()
    out=Path(cfg['output']);old=Path(cfg['reference']);cache=Path(cfg['training_cache'])
    def check(phase,**kw):
        assert time.monotonic()-start<600
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*2**30
        write(out/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            s=str(Path(os.fsdecode(args[0])).resolve())
            assert not any(x in s for x in ['predictions.private','test.rows','test.features','/public_results/','/results.json'])
    sys.addaudithook(guard)
    ys=[];groups=[];stages=[];queries={}
    for st in range(3):
        obj=torch.load(old/'inputs'/f'{st}.private.pt',weights_only=False);tr=~obj['valid']
        ys.append(obj['y'][tr]);groups += [obj['groups'][i] for i in torch.where(tr)[0].tolist()];stages += [st]*int(tr.sum())
        for c in [1,4]:
            ids=torch.where(obj['valid']&(obj['y']==c))[0].tolist()
            if ids:queries[c]=dict(rows=[obj['rows'][i] for i in ids],g=obj['g'][ids],groups=[obj['groups'][i] for i in ids])
    y=torch.cat(ys);stages=torch.tensor(stages);assert len(y)==4395 and np.array_equal(y.numpy(),np.load(cache/'labels.private.npy'))
    assert len(queries[1]['rows'])==3 and len(queries[4]['rows'])==442
    assert not set(groups)&set(queries[1]['groups']+queries[4]['groups'])
    bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text())
    trans=torch.load(old/'transforms.private.pt',weights_only=False)
    import timm
    from safetensors.torch import load_file
    from run_images import Images
    from torch.utils.data import DataLoader
    model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')},strict=True)
    model.eval().requires_grad_(False).cuda();gpu_start=time.monotonic();q={}
    for c,v in queries.items():
        hh=[]
        for x,idx in DataLoader(Images(v['rows'],dict(images=bc['datasets']['ISIC']['development_images'])),batch_size=64,num_workers=0,shuffle=False):
            assert time.monotonic()-gpu_start<300
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x.cuda()))))
                for b in model.blocks[:6]:z=b(z)
                hh.append(z[:,0].float().cpu().double())
            check('features',class_index=c,processed=int(idx[-1])+1)
        h=torch.cat(hh)
        for a in ['G','C']:q[a,c]=mapped(v['g'],h,trans[a])
    gpu_seconds=time.monotonic()-gpu_start;peak_gpu=torch.cuda.max_memory_allocated();del model;torch.cuda.empty_cache()
    w=weights(y,groups);hot=torch.nn.functional.one_hot(y,6).double();rng=np.random.default_rng(74119);perturb={}
    for c in [1,3,4]:
        ids=torch.where(y==c)[0];keys=sorted({groups[i] for i in ids.tolist()});lookup={g:i for i,g in enumerate(keys)}
        gid=np.array([lookup[groups[i]] for i in ids.tolist()]);perturb[c]=[]
        for rep in range(8):
            count=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys));dw=torch.zeros_like(w);dw[ids]=w[ids]*torch.from_numpy(count[gid]-1)
            assert abs(float(dw.sum()))<1e-10 and bool((w+dw>=0).all());perturb[c].append(dw)
    cells=[];parity={};effects=[]
    for a in ['G','C']:
        x=torch.from_numpy(np.load(cache/f'{a}.features.private.npy'));states={}
        for st in [0,1,2]:
            ix=stages<=st;A=x[ix].T@(x[ix]*w[ix,None]);B=x[ix].T@(hot[ix]*w[ix,None])
            saved=torch.load(old/f'{st}.statistics.private.pt',weights_only=False)[a]
            err=max(float((A-saved[0]).abs().max()),float((B-saved[1]).abs().max()));assert err<1e-8;parity[f'{a}.{st}']=err;states[st]=(A,B)
        for c in [1,4]:
            birth=0 if c==1 else 1;k=birth+4;Ab,Bb=states[birth];Af,Bf=states[2]
            assert float((Bb[:,:k]-Bf[:,:k]).abs().max())<1e-12
            data={};vecs={};preds={};target=[];query=q[a,c]
            for name,A,kk in [('BIRTH',Ab,k),('RIDGE_ONLY',Ab,6),('GRAM_ONLY',Af,k),('FULL',Af,6)]:
                H=A.clone();H.diagonal()[:-1].add_(kk*.01);L=torch.linalg.cholesky(H);W=torch.cholesky_solve(Bb[:,:k],L)
                data[name]=(H,L,W);vecs[name]=[];preds[name]=[]
            ids=torch.where(y==c)[0];xx=x[ids]
            for rep,dw in enumerate(perturb[c]):
                da=xx.T@(xx*dw[ids,None]);db=(xx.T@(hot[ids]*dw[ids,None]))[:,:k]
                for name,(H,L,W) in data.items():
                    exact=solve(H+da,Bb[:,:k]+db)-W;linear=torch.cholesky_solve(db-da@W,L)
                    vecs[name].append((query@(exact[:,c]-exact[:,3])).numpy());preds[name].append((query@(linear[:,c]-linear[:,3])).numpy())
                check('refits',arm=a,class_index=c,replicate=rep)
            full=np.stack(vecs['FULL']);rms=float(np.sqrt(np.mean(full**2)))
            for name in data:
                e=np.stack(vecs[name]);p=np.stack(preds[name]);err=float(np.sqrt(np.mean((p-full)**2)))
                cells.append(dict(arm=a,source_class=c,operator=name,query_count=len(query),own_exact_RMS=float(np.sqrt(np.mean(e**2))),own_linear_relative_error=float(np.linalg.norm(p-e)/max(np.linalg.norm(e),1e-30)),RMSE_to_full_exact=err,relative_error_to_full=err/max(rms,1e-30)))
            def row(n):return next(r for r in cells if r['arm']==a and r['source_class']==c and r['operator']==n)
            b,r,g,f=[row(n) for n in ['BIRTH','RIDGE_ONLY','GRAM_ONLY','FULL']]
            gram_specific=rms>1e-8 and f['relative_error_to_full']<=.25 and f['RMSE_to_full_exact']<=.8*r['RMSE_to_full_exact']
            ridge_sufficient=r['RMSE_to_full_exact']<=1.1*f['RMSE_to_full_exact'] and r['RMSE_to_full_exact']<=.5*b['RMSE_to_full_exact']
            effects.append(dict(arm=a,source_class=c,FULL_to_RIDGE_ONLY_RMSE=f['RMSE_to_full_exact']/max(r['RMSE_to_full_exact'],1e-30),RIDGE_ONLY_to_BIRTH_RMSE=r['RMSE_to_full_exact']/max(b['RMSE_to_full_exact'],1e-30),gram_specific=bool(gram_specific),ridge_sufficient=bool(ridge_sufficient)))
            write(out/'partial.json',dict(cells=cells,effects=effects))
    assert len(cells)==16
    costs=dict(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),GPU_phase_seconds=gpu_seconds,peak_GPU_bytes=peak_gpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'aggregate.json',dict(status='COMPLETE',source_commit=cfg['source_commit'],alpha=.01,train_rows=4395,query_rows=445,test_access=False,offline_privileged_audit=True,patient_isolation='UNKNOWN',parity=parity,cells=cells,effects=effects,costs=costs,method_success=False))
    write(out/'status.json',dict(state='COMPLETE',cells=16))

if __name__=='__main__':
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);start=time.time()
    try:main(cfg);write(out/'EXIT.json',dict(exit_code=0,wall_seconds=time.time()-start))
    except Exception:
        write(out/'failure.private.json',dict(traceback=traceback.format_exc()));write(out/'EXIT.json',dict(exit_code=1,wall_seconds=time.time()-start));raise
