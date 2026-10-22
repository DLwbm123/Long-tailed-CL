"""A-D support-only nested cross-validation; one bounded experiment."""
import os, sys, json, time, math, resource, traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from core import Adapter, weights, loss, selfcheck, paired
from design import SEEDS, CLASSES, validate, score, flows, select_temperature
from backbones import get_backbone

cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text()); root=Path(cfg['output']); out=root
torch.set_num_threads(4)
counts=dict(image_encoder_rows=0,image_encoder_calls=0,text_encoder_rows=0,text_encoder_calls=0,
            backward=0,optimizer_steps=0,ridge_solves=0,fits=0,query_rows=0)

def write(name, value):
    p=out/name; tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(p)

def status(**v):
    write('STATUS.json',v); write('ACTUAL_COUNTS.json',counts)
    p=root/'CURRENT.json'; p.write_text(json.dumps(dict(backbone=out.name,**v,counts=counts)))

def fit(x, text, y, groups, ids, seed, name):
    xx=x[ids].cuda(); yy=y[ids].cuda(); gs=[groups[i] for i in ids]
    sampling=weights(yy.cpu(),gs)
    orders=torch.multinomial(sampling,600*64,replacement=True,generator=torch.Generator().manual_seed(seed)).reshape(600,64)
    model=Adapter().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    trace=[]
    for step,batch in enumerate(orders):
        opt.param_groups[0]['lr']=.001*.5*(1+math.cos(math.pi*step/600))
        opt.zero_grad(set_to_none=True)
        val,_,_=loss(model,xx[batch],yy[batch],text,'flat',list(range(13)))
        assert torch.isfinite(val); val.backward(); counts['backward']+=1
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)); opt.step(); counts['optimizer_steps']+=1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if step%100==0 or step==599:
            trace.append(dict(step=step+1,loss=float(val.detach()),gradient_norm=gn))
            status(phase='TRAIN',fit=name,step=step+1,total=600)
    model.eval(); proto=[]
    for c in range(13):
        keys=sorted(set(gs[j] for j in torch.where(yy.cpu()==c)[0].tolist()))
        proto.append(torch.stack([xx[torch.tensor([g==k for g in gs],device='cuda')].mean(0) for k in keys]).mean(0))
    proto=F.normalize(torch.stack(proto),dim=1)
    torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    prototype=proto.cpu(),train_indices=ids,seed=seed,steps=600),out/(name+'.private.pt'))
    write(name+'.trace.json',trace); counts['fits']+=1
    del opt,xx,yy
    return model,proto

def predict(model, proto, x, text, ids):
    with torch.no_grad():
        z=x[ids].cuda(); n=(10*z@proto.T).cpu(); a=model.logits(z,text)[:,:13].softmax(1).cpu()
    assert torch.isfinite(n).all() and torch.isfinite(a).all()
    return n,a

def bootstrap(predictions,y,groups):
    # Paired, stratified known-group bootstrap of fixed OOF predictions. No refits;
    # conditional uncertainty only, not an independent-fold generalization CI.
    gen=torch.Generator().manual_seed(86421); result={}
    for base in ['B','C']:
        differences=torch.stack([(p['D'].argmax(1)==y).double()-(p[base].argmax(1)==y).double() for p in predictions]).mean(0)
        byclass=[]
        for c in CLASSES:
            keys=sorted({g for g,v in zip(groups,y.tolist()) if v==c}); vals=[]
            for g in keys:
                mask=torch.tensor([q==g for q in groups])
                vals.append(float(differences[mask].mean()))
            values=torch.tensor(vals,dtype=torch.double)
            sample=torch.randint(len(values),(2000,len(values)),generator=gen)
            byclass.append(values[sample].mean(1))
        ds=torch.stack(byclass).mean(0)
        result['D-'+base]=dict(mean=float(ds.mean()),lower=float(torch.quantile(ds,.025)),upper=float(torch.quantile(ds,.975)),replicates=2000)
    return result

def main():
    global out
    plan=json.loads((root/'PLAN.private.json').read_text()); validate(plan)
    rows=[r for r in json.loads(Path(cfg['support_rows']).read_text()) if r['label']<13]
    assert rows==plan['rows']
    y=torch.tensor([r['label'] for r in rows]); groups=[r['identity_component'] for r in rows]
    image_root=Path(cfg['image_root']).resolve()
    allowed={(image_root/r['relative_path']).resolve() for r in rows}
    assert all(p.is_relative_to(image_root) for p in allowed)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(image_root): assert p in allowed,'NON_SUPPORT_IMAGE'
            assert not any(s in p.name for s in ['HK.E.','HK.Qgate.','responses.private']), 'QUERY_ARTIFACT_FORBIDDEN'
    sys.addaudithook(guard)
    write('SYNTHETIC.json',selfcheck())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    final={}
    for kind in ['CLIP','BiomedCLIP']:
        out=root/kind;out.mkdir(exist_ok=True)
        backbone,preprocess,text,versions=get_backbone(kind,cfg,out,write,status,counts)
        parts=[]
        with torch.no_grad():
            for start in range(0,len(rows),32):
                images=[]
                for r in rows[start:start+32]:
                    with Image.open(image_root/r['relative_path']) as im: images.append(preprocess(im.convert('RGB')))
                z=F.normalize(backbone.encode_image(torch.stack(images).cuda()).float(),dim=1).cpu()
                assert torch.isfinite(z).all();parts.append(z)
                counts['image_encoder_rows']+=len(z);counts['image_encoder_calls']+=1
                status(phase='ENCODING_SUPPORT',done=start+len(z),total=len(rows))
        x=torch.cat(parts); assert x.shape==(3140,512)
        assert all(not p.requires_grad and p._version==versions[n] for n,p in backbone.named_parameters())
        torch.save(dict(x=x,text=text.cpu(),row_indices=list(range(len(rows)))),out/'support_features.private.pt')
        del backbone,preprocess;torch.cuda.empty_cache()
        eligible=[i for i,r in enumerate(rows) if r['label']!=1]; ey=y[eligible];eg=[groups[i] for i in eligible]
        reports={}; saved_predictions=[]; selection=[]
        for seed in SEEDS:
            allp={arm:torch.full((len(rows),13),float('nan')) for arm in ['A','B','C','D']}
            fold_reports=[]
            for k,fold in enumerate(plan['folds']):
                inner_n=[];inner_a=[];inner_ids=[]
                for j,part in enumerate(fold['inner']):
                    model,proto=fit(x,text,y,groups,part['train'],seed,f'{seed}.o{k}.i{j}')
                    n,a=predict(model,proto,x,text,part['validation']);inner_n.append(n);inner_a.append(a);inner_ids+=part['validation']
                    del model,proto
                temperature,losses=select_temperature(torch.cat(inner_n),torch.cat(inner_a),y[inner_ids],[groups[i] for i in inner_ids])
                seal=dict(seed=seed,outer_fold=k,temperature=temperature,inner_balanced_NLL=losses,outer_seen=False)
                selection.append(seal);write('TEMPERATURE_SELECTION.json',selection)
                model,proto=fit(x,text,y,groups,fold['train'],seed,f'{seed}.o{k}.outer')
                ids=fold['test'];n,a=predict(model,proto,x,text,ids);pn=n.softmax(1)
                pp=dict(A=pn,B=a,C=.5*(pn+a),D=.5*((n/temperature).softmax(1)+a))
                for arm in pp:allp[arm][ids]=pp[arm]
                fy=y[ids];fg=[groups[i] for i in ids]
                fold_reports.append(dict(outer_fold=k,temperature=temperature,metrics={arm:score(p,fy,fg) for arm,p in pp.items()},
                                         paired={base:paired(pp[base],pp['D'],fy) for base in ['B','C']},
                                         flows={arm:flows(pp['A'],pp['B'],pp[arm],fy) for arm in ['C','D']}))
                write(f'{seed}.folds.json',fold_reports);del model,proto
            allp={arm:p[eligible] for arm,p in allp.items()};saved_predictions.append(allp)
            reports[str(seed)]=dict(metrics={arm:score(p,ey,eg) for arm,p in allp.items()},folds=fold_reports,
                                    paired={base:paired(allp[base],allp['D'],ey) for base in ['B','C']},
                                    flows={arm:flows(allp['A'],allp['B'],allp[arm],ey) for arm in ['C','D']})
            torch.save(dict(predictions=allp,indices=eligible),out/f'{seed}.oof.private.pt')
        ci=bootstrap(saved_predictions,ey,eg)
        delta={base:[reports[str(s)]['metrics']['D']['group_BA']-reports[str(s)]['metrics'][base]['group_BA'] for s in SEEDS] for base in ['B','C']}
        point=sum(delta['C'])/3>=.01 and all(d>0 for d in delta['C']) and sum(delta['B'])/3>0 and all(d>=0 for d in delta['B'])
        protection=[]
        for c in CLASSES:
            for base in ['B','C']:
                ds=[reports[str(s)]['metrics']['D']['recall'][c]-reports[str(s)]['metrics'][base]['recall'][c] for s in SEEDS]
                limit=0 if c in [2,12] else -.01
                protection.append(dict(class_id=c,baseline=base,mean_delta=sum(ds)/3,limit=limit,passed=sum(ds)/3>=limit-1e-12))
        safe=all(v['passed'] for v in protection);uncertain=not all(v['lower']>0 for v in ci.values())
        final[kind]=dict(seed_reports=reports,delta=delta,point_gate=point,protection_gate=safe,protection=protection,
                         conditional_group_bootstrap=ci,uncertainty_gate=not uncertain,
                         decision='SUPPORT_SCALE_CANDIDATE' if point and safe and not uncertain else 'NO_QUALIFIED_SCALE_SIGNAL',
                         method_success=False,singleton_class_1='NOT_ESTIMABLE',patient_independence='UNKNOWN')
        write('aggregate.json',final[kind]);del x,text,saved_predictions;torch.cuda.empty_cache()
    out=root
    assert counts['fits']==96 and counts['optimizer_steps']==57600 and counts['image_encoder_rows']==6280 and counts['query_rows']==0
    write('aggregate.json',dict(backbones=final,counts=counts,source_commit=cfg['source_commit'],scientific_complete=True,
                                followup_training_authorized=False,query_used=False,scope='Support-only conditional OOF evidence'))
    status(phase='COMPLETE_ANALYZE_THEN_STOP')

start=time.time();code=1
try:main();code=0
except BaseException as e:
    out=root;write('FAILURE.json',dict(type=type(e).__name__,scientific_complete=False));(root/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED')
finally:
    out=root;u=resource.getrusage(resource.RUSAGE_SELF)
    write('WORKER_RESOURCE.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,
                                    peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
    write('ACTUAL_COUNTS.json',counts);write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
