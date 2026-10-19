"""One privileged prefix CE diagnostic; all scientific limits are fixed."""
import json, os, resource, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from core_depth import mapped
from core_attribution import weights

ARMS=['G','P','C']; LAMBDAS=[.0001,.01,1.]

def write(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)

class HeadLimit(Exception):pass

def loss_grad(w,x,y,p,lam):
    s=x@w;prob=s.softmax(1);ce=(p*(s.logsumexp(1)-s[torch.arange(len(y),device=y.device),y])).sum()
    loss=ce+lam/2*w[:-1].square().sum()
    hot=torch.nn.functional.one_hot(y,w.shape[1]).to(x.dtype)
    grad=x.T@((prob-hot)*p[:,None]);grad[:-1]+=lam*w[:-1]
    return loss,grad

def fit(x,y,p,k,lam,device,check,limit=60):
    tick=time.monotonic();x=x.to(device);y=y.to(device);p=p.to(device)
    diagonal=.5*(x.square()*p[:,None]).sum(0);diagonal[:-1]+=lam
    scale=diagonal.sqrt().clamp_min(1e-12)[:,None]
    theta=torch.zeros((x.shape[1],k),dtype=torch.float64,device=device,requires_grad=True)
    opt=torch.optim.LBFGS([theta],lr=1,max_iter=300,max_eval=400,tolerance_grad=1e-9,tolerance_change=1e-14,history_size=20,line_search_fn='strong_wolfe')
    trace=[];reason='OPTIMIZER_STOP'
    def closure():
        if time.monotonic()-tick>limit or len(trace)>=500:raise HeadLimit()
        check();opt.zero_grad();w=theta.detach()/scale;loss,g=loss_grad(w,x,y,p,lam)
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(g).all()),'nonfinite objective/gradient'
        theta.grad=g/scale
        trace.append((float(loss),float(g.abs().max())))
        return loss
    try:opt.step(closure)
    except HeadLimit:reason='HEAD_BUDGET_LIMIT'
    with torch.no_grad():
        w=theta.detach()/scale;loss,g=loss_grad(w,x,y,p,lam);grad=float(g.abs().max());obj=float(loss)
        finite=bool(torch.isfinite(w).all()) and np.isfinite(obj) and np.isfinite(grad)
        converged=finite and grad<=1e-7 and bool(trace) and obj<=trace[0][0]+1e-10 and reason!='HEAD_BUDGET_LIMIT'
        assert finite,'nonfinite final state'
        info=dict(status='CONVERGED' if converged else 'UNRESOLVED',stop_reason=reason,lambda_value=lam,iterations=int(opt.state[theta].get('n_iter',0)),closure_evaluations=len(trace),initial_objective=trace[0][0] if trace else None,final_objective=obj,gradient_max_abs=grad,wall_seconds=time.monotonic()-tick)
        return w.cpu(),info

def selfcheck():
    gen=torch.Generator().manual_seed(74149);x=torch.randn(18,5,generator=gen,dtype=torch.float64);x[:,-1]=1
    y=torch.arange(18)%3;p=torch.full((18,),1/18,dtype=x.dtype);w=torch.randn(5,3,generator=gen,dtype=x.dtype,requires_grad=True)
    s=x@w;ref=(p*(-s.log_softmax(1)[torch.arange(18),y])).sum()+.01/2*w[:-1].square().sum()
    v,g=loss_grad(w.detach(),x,y,p,.01);assert torch.allclose(ref,v,atol=1e-12) and torch.allclose(torch.autograd.grad(ref,w)[0],g,atol=1e-12)
    xx=x.repeat_interleave(2,0);yy=y.repeat_interleave(2);pp=p.repeat_interleave(2)/2
    vv,gg=loss_grad(w.detach(),xx,yy,pp,.01);assert torch.allclose(v,vv,atol=1e-12) and torch.allclose(g,gg,atol=1e-12)
    f,info=fit(x,y,p,3,.01,'cpu',lambda:None,10);assert info['status']=='CONVERGED'
    assert float(loss_grad(f,x,y,p,.01)[1].abs().max())<=1e-7
    return dict(analytic_gradient=True,duplicate_weight_invariance=True,scaled_optimizer=True)

def metrics(s,y,groups,k):
    pred=s.argmax(1).numpy();yy=y.numpy();rec=[float((pred[yy==c]==c).mean()) for c in range(k)]
    grec=[]
    for c in range(k):
        gs=sorted({groups[i] for i in np.where(yy==c)[0]});grec.append(float(np.mean([np.mean(pred[[i for i in np.where(yy==c)[0] if groups[i]==g]]==c) for g in gs])))
    return dict(BA=sum(rec)/k,recall=rec,correct=[int(((pred==c)&(yy==c)).sum()) for c in range(k)],counts=[int((yy==c).sum()) for c in range(k)],group_equal_BA=sum(grec)/k,group_recall=grec)

def main(cfg):
    start=time.monotonic();torch.set_num_threads(4);resource.setrlimit(resource.RLIMIT_CPU,(3600,3600));out=Path(cfg['output']);old=Path(cfg['reference']).resolve();sealed=False;stage=-1;allowed_images=set();gpu_tick=None
    def check(phase=None,**kw):
        assert time.monotonic()-start<1790,'global wall cap'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30,'RSS cap'
        if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<4*2**30,'GPU cap'
        if phase:write(out/'status.json',dict(state='RUNNING',phase=phase,elapsed_seconds=time.monotonic()-start,**kw))
    write(out/'SELFCHECK.json',selfcheck())
    bc=json.loads(Path(cfg['benchmark_config']).read_text());src=json.loads(Path(bc['transfer_config']).read_text());testroot=Path(bc['output']).resolve()
    image_roots=[Path(bc['datasets']['ISIC'][key]).resolve() for key in ['images','development_images']]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(out):return
        if p.is_relative_to(testroot) or p==old/'predictions.private.pt':assert sealed,'test before model seal'
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
        return {a:mapped(g,h,trans[a]) for a in ARMS}
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
                valba=float(np.mean([float(((data[arm][va]@ridge[arm]['0.01']).argmax(1)[y[va]==c]==c).double().mean()) for c in range(4)]));assert abs(valba-expected)<1e-10;parity[arm+'.ridge_base_validation_BA']=valba
                candidates=[]
                for lam in LAMBDAS:
                    check('CE_fit',stage=stage,arm=arm,lambda_value=lam);W,info=fit(x,yy,p,k,lam,'cuda',check);info.update(stage=stage,arm=arm)
                    if info['status']=='CONVERGED':info['base_validation_BA']=float(np.mean([float(((data[arm][va]@W).argmax(1)[y[va]==c]==c).double().mean()) for c in range(4)]))
                    fit_records.append(info);torch.save(W,out/f'head_{arm}_{stage}_{lam}.private.pt');candidates.append((lam,W,info));write(out/'partial.json',dict(fits=fit_records,chosen=chosen,parity=parity,counts=counts))
                if all(v[2]['status']=='CONVERGED' for v in candidates):
                    best=max(candidates,key=lambda v:(v[2]['base_validation_BA'],v[0]));chosen[arm]=best[0];heads[arm,stage]=best[1]
                else:chosen[arm]=None
            elif chosen[arm] is not None:
                lam=chosen[arm];check('CE_fit',stage=stage,arm=arm,lambda_value=lam);W,info=fit(x,yy,p,k,lam,'cuda',check);info.update(stage=stage,arm=arm);fit_records.append(info);torch.save(W,out/f'head_{arm}_{stage}_{lam}.private.pt')
                if info['status']=='CONVERGED':heads[arm,stage]=W
            write(out/'partial.json',dict(fits=fit_records,chosen=chosen,parity=parity,counts=counts))
        if stage==0:write(out/'SELECTION_SEAL.json',dict(chosen=chosen,grid=LAMBDAS,metric='base_validation_image_BA',validation_rows=681,fit_records=fit_records,test_used=False,sealed_at=time.time()))
        del obj,data,original_stats,ridge
    assert len(yy)==4395 and not set(all_groups)&all_valid
    modelseal=dict(source_commit=cfg['source_commit'],fit_units=len(fit_records),expected_units=15,chosen=chosen,converged_units=sum(v['status']=='CONVERGED' for v in fit_records),complete_arms=[a for a in ARMS if all((a,s) in heads for s in range(3))],test_used=False,all_models_sealed_at=time.time())
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
            if (a,st) in heads:
                key=a+'_CE';s=testx[a]@heads[a,st];assert bool(torch.isfinite(s).all());scores[key,st]=s;summary[key,st]=metrics(s[mask],ty[mask],vgs,k)
                rp=oldscore[mask].argmax(1);cp=s[mask].argmax(1);truth=ty[mask]
                harm.append(dict(arm=a,stage=st,rescue=int(((rp!=truth)&(cp==truth)).sum()),harm=int(((rp==truth)&(cp!=truth)).sum())))
    torch.save(dict(scores=scores,y=ty,groups=gg),out/'predictions.private.pt')
    del model;torch.cuda.empty_cache();gpu_residence=time.monotonic()-gpu_tick
    names=[a+'_ridge' for a in ARMS]+[a+'_CE' for a in modelseal['complete_arms']];truth=ty.numpy();rng=np.random.default_rng(74150);rec=np.zeros((len(names),2,10000,6));check('paired_bootstrap')
    for c in range(6):
        ids=np.where(truth==c)[0];keys=sorted({gg[i] for i in ids});members=[[i for i in ids if gg[i]==g] for g in keys];sizes=np.array([len(v) for v in members]);draw=rng.integers(0,len(keys),(10000,len(keys)));den=sizes[draw].sum(1)
        for ni,name in enumerate(names):
            for si,st in enumerate([1,2]):
                if c>=st+4:continue
                ok=(scores[name,st].argmax(1).numpy()==truth);hits=np.array([ok[ix].sum() for ix in members]);rec[ni,si,:,c]=hits[draw].sum(1)/den
    jboot=(rec[:,0,:,:5].mean(-1)+rec[:,1].mean(-1))/2*100
    J={n:100*(summary[n,1]['BA']+summary[n,2]['BA'])/2 for n in names};comparisons={};gates={}
    for a in modelseal['complete_arms']:
        n=a+'_CE';diffs={}
        for r in ARMS:
            b=jboot[names.index(n)]-jboot[names.index(r+'_ridge')];diffs[r]=dict(J_gain_pp=J[n]-J[r+'_ridge'],CI95_pp=np.quantile(b,[.025,.975]).tolist(),CI9_adjusted_pp=np.quantile(b,[.05/18,1-.05/18]).tolist())
        own=a+'_ridge';stagegains=[100*(summary[n,s]['BA']-summary[own,s]['BA']) for s in [1,2]]
        birth=[100*(summary[n,1]['recall'][4]-summary[own,1]['recall'][4]),100*(summary[n,2]['recall'][5]-summary[own,2]['recall'][5]),100*(summary[n,2]['recall'][4]-summary[own,2]['recall'][4])]
        oldg=[100*np.mean(np.array(summary[n,s]['recall'][:4])-summary[own,s]['recall'][:4]) for s in [1,2]]+[100*np.mean(np.array(summary[n,2]['recall'][:5])-summary[own,2]['recall'][:5])]
        gate=dict(J_vs_G_at_least1pp=diffs['G']['J_gain_pp']>=1,J_vs_P_C_positive=all(diffs[r]['J_gain_pp']>0 for r in ['P','C']),all3_adjusted_lower_positive=all(z['CI9_adjusted_pp'][0]>0 for z in diffs.values()),both_stage_gain_nonnegative=min(stagegains)>=-1e-12,future_birth_and_retirement_nonnegative=min(birth)>=-1e-12,old_macro_cost_within1pp=min(oldg)>=-1-1e-12)
        comparisons[a]=dict(vs_ridge=diffs,stage_gain_own_pp=stagegains,class4_birth_class5_birth_class4_retired_gain_own_pp=birth,old_macro_gains_pp=oldg);gates[a]=dict(**gate,pass_all=all(gate.values()))
    passed=[a for a,v in gates.items() if v['pass_all']];decision='TARGET_HEADROOM_EXPLORATORY' if passed else ('STOP_EVIDENCE_INSUFFICIENT_NO_EXTENSION' if len(modelseal['complete_arms'])==3 else 'UNRESOLVED_OPTIMIZATION_NO_EXTENSION')
    costs=dict(wall_seconds=time.monotonic()-start,cpu_seconds=sum(resource.getrusage(resource.RUSAGE_SELF)[i] for i in [0,1]),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_residence_upper_seconds=gpu_residence,feature_extraction=feature_cost,output_bytes=sum(p.stat().st_size for p in out.parent.rglob('*') if p.is_file()),privileged_old_training_access=True,full_history_memory_cap_compliant=False)
    check('complete')
    result=dict(status='COMPLETE',source_commit=cfg['source_commit'],fits=fit_records,chosen=chosen,counts=counts,model_seal=modelseal,parity=parity,summary={n:{str(s):summary[n,s] for s in range(3) if (n,s) in summary} for n in [a+'_'+q for a in ARMS for q in ['ridge','CE']]},J_pp=J,comparisons=comparisons,gates=gates,passing_arms=passed,method_success=False,decision=decision,rescue_harm=harm,costs=costs,bootstrap=dict(draws=10000,seed=74150,primary_family=9),historical_test_exposed=True,patient_isolation='UNKNOWN',new_parameter_search_after_base=False)
    write(out/'aggregate.json',result);write(out/'COST_RECEIPT.json',costs);write(out/'status.json',dict(state='COMPLETE',decision=decision))

if __name__=='__main__':
    if os.environ.get('SELFTEST')=='1':print(json.dumps(selfcheck()));sys.exit(0)
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=Path(cfg['output']);t=time.monotonic()
    try:main(cfg);write(out/'EXIT.json',dict(exit_code=0,wall_seconds=time.monotonic()-t))
    except Exception as e:
        write(out/'FAILURE.private.json',dict(error=type(e).__name__,traceback=traceback.format_exc()));write(out/'EXIT.json',dict(exit_code=1,status='UNRESOLVED_ENGINEERING_OR_RESOURCE',wall_seconds=time.monotonic()-t));raise
