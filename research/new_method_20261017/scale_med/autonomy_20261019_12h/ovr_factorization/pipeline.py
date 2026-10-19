"""One privileged prefix CE diagnostic; all scientific limits are fixed."""
import inspect, json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from core_depth import mapped
from core_attribution import weights

ARMS=['G','P','C']; MODELS=[a+'_'+v for a in ARMS for v in ['S','B']]; LAMBDAS=[.0001,.01,1.]

def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else (_ for _ in ()).throw(TypeError(type(v).__name__))));q.replace(p)

def loss_grad(w,x,y,p,lam):
    s=x@w;prob=s.softmax(1);ce=(p*(s.logsumexp(1)-s[torch.arange(len(y),device=y.device),y])).sum()
    loss=ce+lam/2*w[:-1].square().sum()
    hot=torch.nn.functional.one_hot(y,w.shape[1]).to(x.dtype)
    grad=x.T@((prob-hot)*p[:,None]);grad[:-1]+=lam*w[:-1]
    return loss,grad

from solver import fit as binary_fit, selfcheck
from evaluation import evaluate

def metrics(s,y,groups,k):
    pred=s.argmax(1).numpy();yy=y.numpy();rec=[float((pred[yy==c]==c).mean()) for c in range(k)]
    grec=[]
    for c in range(k):
        gs=sorted({groups[i] for i in np.where(yy==c)[0]});grec.append(float(np.mean([np.mean(pred[[i for i in np.where(yy==c)[0] if groups[i]==g]]==c) for g in gs])))
    return dict(BA=sum(rec)/k,recall=rec,correct=[int(((pred==c)&(yy==c)).sum()) for c in range(k)],counts=[int((yy==c).sum()) for c in range(k)],group_equal_BA=sum(grec)/k,group_recall=grec)

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);out=Path(cfg['output']);old=Path(cfg['reference']).resolve();sealed=False;stage=-1;allowed_images=set();gpu_tick=None
    def check(phase=None,**kw):
        from space import check_space
        check_space(cfg['window_root'])
        if phase:write(out/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    write(out/'SELFCHECK.json',selfcheck())
    def solve(arm,st,x,y,q,k,lam,variant):
        check('fit',arm=arm,stage=st,variant=variant,lambda_value=lam)
        w,info=binary_fit(x.numpy(),y.numpy(),q.numpy(),k,lam,variant,check);return torch.from_numpy(w),info
    bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text());testroot=Path(bc['output']).resolve()
    image_roots=[Path(bc['datasets']['ISIC'][key]).resolve() for key in ['images','development_images']]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(out):return
        if p.is_relative_to(testroot) or p==old/'predictions.private.pt' or p==Path(cfg['ce_predictions']).resolve():assert sealed,'test before model seal'
        if p.is_relative_to(old/'inputs'):assert p==old/'inputs'/f'{stage}.private.pt','future bundle access'
        if any(p.is_relative_to(z) for z in image_roots):assert p in allowed_images,'image outside active stage'
    sys.addaudithook(guard)
    try:guard('open',(str(old/'predictions.private.pt'),'rb',0))
    except AssertionError:pass
    else:raise AssertionError('test guard probe failed')
    trans=torch.load(old/'transforms.private.pt',map_location='cpu',weights_only=False)
    import timm
    from safetensors.torch import load_file
    from run_images import Images
    from torch.utils.data import DataLoader
    torch.manual_seed(74002);model=timm.create_model('vit_base_patch16_224',pretrained=False,num_classes=0)
    model.load_state_dict({k:v for k,v in load_file(src['weight']).items() if not k.startswith('head.')},strict=True)
    gpu_tick=time.monotonic();model.eval().requires_grad_(False).cuda();parity={};fit_records=[];heads={};chosen={};training=[];all_groups=[];trainx={a:[] for a in ARMS};feature_cost=[];counts=[];all_valid=set()
    def extract(rows,g,image_root,tag):
        nonlocal allowed_images
        t=time.monotonic();allowed_images={Path(image_root,r['relative_path']).resolve() for r in rows};values=[]
        for images,idx in DataLoader(Images(rows,dict(images=image_root)),batch_size=64,num_workers=0,shuffle=False):
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                z=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(images.cuda()))))
                for block in model.blocks[:6]:z=block(z)
                v=z[:,0].float()
                if not values:
                    got=[];hook=model.blocks[5].register_forward_hook(lambda m,a,o:got.append(o[:,0].float()));model.forward_features(images.cuda());hook.remove()
                    error=float((v-got[0]).abs().max());assert error<1e-5;parity[tag+'.hook']=error
            values.append(v.cpu().double());check('features',tag=tag,rows=int(idx[-1])+1,total=len(rows))
        h=torch.cat(values);allowed_images=set();feature_cost.append(dict(tag=tag,images=len(rows),wall_seconds=time.monotonic()-t))
        result={a:mapped(g,h,trans[a]) for a in ARMS}
        from space import check_space
        check_space(cfg['window_root'],sum(v.numel()*v.element_size() for v in result.values())+1048576)
        torch.save(result,out/(tag+'.features.private.pt'));return result
    for stage in range(3):
        k=stage+4;obj=torch.load(old/'inputs'/f'{stage}.private.pt',map_location='cpu',weights_only=False);tr=~obj['valid'];va=obj['valid'];y=obj['y'];groups=obj['groups']
        tg=[groups[i] for i in torch.where(tr)[0].tolist()];vg={groups[i] for i in torch.where(va)[0].tolist()};assert not set(tg)&vg
        assert len(y)==[3236,2254,36][stage] and int(tr.sum())==[2555,1812,28][stage]
        assert not all_valid&set(tg) and not set(all_groups)&vg;all_valid|=vg
        data=extract(obj['rows'],obj['g'],bc['datasets']['ISIC']['development_images'],'stage'+str(stage))
        training.append(y[tr]);all_groups+=tg;yy=torch.cat(training);ww=weights(yy,all_groups);p=ww/k
        assert abs(float(p.sum())-1)<1e-12 and all(abs(float(p[yy==c].sum())-1/k)<1e-12 for c in range(k))
        counts.append(dict(stage=stage,training_rows=len(yy),new_training_rows=int(tr.sum()),new_validation_rows=int(va.sum()),class_counts=[int((yy==c).sum()) for c in range(k)]))
        original_stats=torch.load(old/f'{stage}.statistics.private.pt',map_location='cpu',weights_only=False);ridge=torch.load(old/f'{stage}.heads.private.pt',map_location='cpu',weights_only=False)
        for arm in ARMS:
            trainx[arm].append(data[arm][tr]);x=torch.cat(trainx[arm]);hot=torch.nn.functional.one_hot(yy,6).double();xx=x.cuda();wx=ww.cuda();A=(xx.T@(xx*wx[:,None])).cpu();B=(xx.T@(hot.cuda()*wx[:,None])).cpu();del xx,wx
            err=max(float((A-original_stats[arm][0]).abs().max()),float((B-original_stats[arm][1]).abs().max()));assert err<=1e-8,(arm,stage,err);parity[f'{arm}.{stage}.statistics']=err
            if stage==0:
                expected=json.loads((old/'BASE_SELECTION.json').read_text())['all_arm_validation_BA'][arm]['0.01']
                valba=float(np.mean([float(((data[arm][va]@ridge[arm]['0.01']).argmax(1)[y[va]==c]==c).double().mean()) for c in range(4)]));assert abs(valba-expected)<1e-10
                parity[arm+'.ridge_base_validation_BA']=valba
            for variant in ['S','B']:
                name=arm+'_'+variant
                if stage==0:
                    candidates=[]
                    for lam in LAMBDAS:
                        W,info=solve(arm,stage,x,yy,ww,k,lam,variant);info.update(stage=stage,arm=name)
                        if info['status']=='CONVERGED':info['base_validation_BA']=metrics(data[arm][va]@W,y[va],[groups[i] for i in torch.where(va)[0].tolist()],4)['BA']
                        fit_records.append(info);torch.save(W,out/f'head_{name}_{stage}_{lam}.private.pt');candidates.append((lam,W,info));write(out/'partial.json',dict(fits=fit_records,chosen=chosen,counts=counts,parity=parity))
                    if all(v[2]['status']=='CONVERGED' for v in candidates):
                        best=max(candidates,key=lambda v:(v[2]['base_validation_BA'],v[0]));chosen[name]=best[0];heads[name,stage]=best[1]
                    else:chosen[name]=None
                elif chosen[name] is not None:
                    lam=chosen[name];W,info=solve(arm,stage,x,yy,ww,k,lam,variant);info.update(stage=stage,arm=name);fit_records.append(info);torch.save(W,out/f'head_{name}_{stage}_{lam}.private.pt')
                    if info['status']=='CONVERGED':heads[name,stage]=W
                write(out/'partial.json',dict(fits=fit_records,chosen=chosen,counts=counts,parity=parity))
        if stage==0:write(out/'SELECTION_SEAL.json',dict(chosen=chosen,grid=LAMBDAS,metric='base_validation_image_BA',validation_rows=681,test_used=False,fit_records=fit_records))
        del obj,data,original_stats,ridge
    assert len(yy)==4395 and not set(all_groups)&all_valid
    modelseal=dict(source_commit=cfg['source_commit'],fit_units=len(fit_records),expected_units=30,chosen=chosen,converged_units=sum(v['status']=='CONVERGED' for v in fit_records),complete_arms=[a for a in MODELS if all((a,s) in heads for s in range(3))],test_used=False,all_models_sealed_at=time.time())
    write(out/'MODEL_SEAL.json',modelseal);sealed=True;stage=3
    rr=json.loads((testroot/'ISIC.test.rows.private.json').read_text());cache=torch.load(testroot/'ISIC.test.features.private.pt',map_location='cpu',weights_only=False);ty=torch.tensor([v['label'] for v in rr]);gg=[v['identity_component'] for v in rr]
    assert len(ty)==586 and len(set(gg))==500 and torch.equal(cache['y'],ty) and cache['components']==gg and not set(gg)&(set(all_groups)|all_valid)
    g=torch.nn.functional.normalize(cache['x'].double(),dim=1);testx=extract(rr,g,bc['datasets']['ISIC']['images'],'test');prior=torch.load(old/'predictions.private.pt',map_location='cpu',weights_only=False)
    assert torch.equal(prior['y'],ty) and prior['groups']==gg
    scores={};summary={};harm=[]
    for st in range(3):
        k=st+4;mask=ty<k;vgs=[gg[i] for i in torch.where(mask)[0].tolist()];rheads=torch.load(old/f'{st}.heads.private.pt',map_location='cpu',weights_only=False)
        for a in ARMS:
            key=a+'_ridge';s=testx[a]@rheads[a]['0.01'];oldscore=prior['scores'][str(st)][a]['0.01'];err=float((s-oldscore).abs().max());assert err<=1e-7 and torch.equal(s.argmax(1),oldscore.argmax(1)),(a,st,err);parity[f'{a}.{st}.test_scores']=err;scores[key,st]=oldscore;summary[key,st]=metrics(oldscore[mask],ty[mask],vgs,k)
            for variant in ['S','B']:
                name=a+'_'+variant
                if (name,st) in heads:
                    score=testx[a]@heads[name,st];assert bool(torch.isfinite(score).all());scores[name,st]=score
    ce=torch.load(Path(cfg['ce_predictions']),map_location='cpu',weights_only=False);assert torch.equal(ce['y'],ty) and ce['groups']==gg
    for st in range(3):
        for a in ARMS:scores[a+'_CE',st]=ce['scores'][a+'_CE',st]
    torch.save(dict(scores=scores,y=ty,groups=gg),out/'predictions.private.pt')
    del model;torch.cuda.empty_cache();gpu_residence=time.monotonic()-gpu_tick
    result=evaluate(scores,ty,gg,modelseal['complete_arms'])
    write(out/'EVALUATION.json',result)
    costs=dict(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_residence_upper_seconds=gpu_residence,feature_extraction=feature_cost,output_bytes=sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()),privileged_old_training_access=True,full_history_memory_cap_compliant=False)
    check('complete')
    result.update(status='COMPLETE',source_commit=cfg['source_commit'],fits=fit_records,chosen=chosen,counts=counts,model_seal=modelseal,parity=parity,costs=costs,historical_test_exposed=True,patient_isolation='UNKNOWN',new_parameter_search_after_base=False)
    decision=result['decision']
    write(out/'aggregate.json',result);write(out/'COST_RECEIPT.json',costs);write(out/'status.json',dict(state='COMPLETE',decision=decision))

