"""Fixed internal OPSA experiment; no query use before every model is sealed."""
import os, sys, json, time, resource, traceback, math
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from common import OPSAAdapter, LinearControl, weights, transform, to_device, lr_scale, metrics, pairs, risk_gate, synthetic_check

cfg = json.loads(Path(os.environ['RUN_CONFIG']).read_text())
out = Path(cfg['output'])
torch.set_num_threads(4)
counts = dict(encoder_images=0, encoder_calls=0, backward=0, optimizer_steps=0, ridge_solves=0, mapper_fits=0)
allowed_images = set()

def write(name, value):
    p = out/name
    t = p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    t.replace(p)

def status(**kw):
    write('STATUS.json', kw)
    write('ACTUAL_COUNTS.json', counts)

def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def same(a,b):
    if isinstance(a, torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def image_paths(rows, ds):
    root = Path(cfg['image_roots'][ds]).resolve()
    paths=[]
    for row in rows:
        rel=Path(row['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
        p=(root/rel).resolve();assert p.is_relative_to(root);paths.append(p)
    return paths

def read_batch(paths, ids):
    parts=[]
    for i in ids.tolist():
        p=paths[i];assert p in allowed_images
        with Image.open(p) as im: parts.append(preprocess(im.convert('RGB')))
    return torch.stack(parts).cuda()

active_adapter = None
active_kind = None
training_forward = False
hook_stats = {}

def hook(module, args):
    if active_adapter is None:return None
    x=args[0]
    if training_forward:
        with torch.no_grad():
            a=F.linear(x.detach()/x.detach().norm(dim=-1,keepdim=True)*math.sqrt(768),active_adapter.down)
            hook_stats['max_abs_A']=max(hook_stats.get('max_abs_A',0),float(a.abs().max()))
            hook_stats['boundary_values']=hook_stats.get('boundary_values',0)+int((a.abs()>=math.pi/2-.25).sum())
            hook_stats['activation_values']=hook_stats.get('activation_values',0)+a.numel()
    return (active_adapter(x,collect_mapc=training_forward and active_kind=='M'),)+args[1:]

def encode(xx):
    counts['encoder_images']+=len(xx);counts['encoder_calls']+=1
    o=raw(model,pointwise,xx)
    return F.normalize(F.normalize(torch.cat([o['pre_logits'],o['pre_logits_few']],1),dim=1).double(),dim=1)

def frozen_check():
    assert all(not p.requires_grad and p._version==versions[n] for n,p in model.named_parameters()),'BACKBONE_CHANGED'


def train_producer(tag, ds, rows, y, groups, cached, teacher, seed):
    global allowed_images, active_adapter, active_kind, training_forward, hook_stats
    paths=image_paths(rows,ds);allowed_images=set(paths)
    classes=int(y.max())+1
    q=weights(y,groups,classes)
    orders=torch.multinomial(q,800*16,replacement=True,generator=torch.Generator().manual_seed(seed)).reshape(800,16)
    torch.save(dict(orders=orders,seed=seed),out/(tag+'.sampling.private.pt'))
    write(tag+'.PLAN_SEAL.json',dict(rows=len(rows),classes=classes,groups=len(set(groups)),steps=800,batch=16,query_excluded=True,source_commit=cfg['source_commit']))
    mapper=to_device(teacher['mapper']);w0=teacher['W'].cuda()
    assert w0.shape==(1536,classes)
    active_adapter=None;training_forward=False
    probe=torch.arange(min(32,len(rows)))
    with torch.no_grad():
        pg=encode(read_batch(paths,probe))
        feature_error=float((pg.cpu()-cached[probe]).abs().max())
        ps=transform(pg,mapper)@w0[:-1]+w0[-1]
        old=transform(cached[probe],teacher['mapper'])@teacher['W'][:-1]+teacher['W'][-1]
        score_error=float((ps.cpu()-old).abs().max())
    write(tag+'.REPLAY.json',dict(feature_maxabs=feature_error,score_maxabs=score_error,rows=len(probe)))
    assert feature_error<=1e-6 and score_error<=1e-5,'F_SUPPORT_REPLAY_FAILED'
    torch.manual_seed(seed)
    init=OPSAAdapter(768).state_dict()
    manifests=[]
    for kind in ['H','L','N','M']:
        active_kind=kind
        active_adapter=None if kind=='H' else (LinearControl(768) if kind=='L' else OPSAAdapter(768)).cuda()
        if active_adapter is not None:active_adapter.load_state_dict(init);active_adapter.train()
        w=w0.detach().clone().requires_grad_(True)
        optgroups=[dict(params=[w],lr=2e-4)]
        if active_adapter is not None:optgroups.insert(0,dict(params=list(active_adapter.parameters()),lr=1.2e-4))
        opt=torch.optim.AdamW(optgroups,weight_decay=.01,betas=(.9,.999),eps=1e-8)
        params=[w]+([] if active_adapter is None else list(active_adapter.parameters()))
        trace=[];resets=0;hook_stats={};firstloss=None;lastloss=None
        # Pre-update support diagnostic, never an argmax/risk selection.
        with torch.no_grad():
            initialized=pg if kind=='H' else encode(read_batch(paths,probe))
            initial_shift=float((initialized-pg).abs().max())
        for step,ids in enumerate(orders):
            for g,base_lr in zip(opt.param_groups,([2e-4] if kind=='H' else [1.2e-4,2e-4])):g['lr']=base_lr*lr_scale(step)
            opt.zero_grad(set_to_none=True)
            training_forward=kind!='H'
            g=cached[ids].cuda() if kind=='H' else encode(read_batch(paths,ids))
            logits=transform(g,mapper)@w[:-1]+w[-1]
            loss=F.cross_entropy(logits,y[ids].cuda());assert torch.isfinite(loss)
            loss.backward();counts['backward']+=1
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            if step==0:assert all(p.grad.norm()>0 for p in params),'ZERO_INITIAL_GRADIENT'
            gradnorm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            if kind=='M':resets+=active_adapter.reset_mapc(opt)
            opt.step();counts['optimizer_steps']+=1
            assert all(torch.isfinite(p).all() for p in params)
            val=float(loss.detach());firstloss=val if firstloss is None else firstloss;lastloss=val
            if step%25==0 or step==799:
                trace.append(dict(step=step+1,CE=val,gradient_norm_before_clip=gradnorm,reset_entries_cumulative=resets))
                status(phase='TRAIN',producer=tag,arm=kind,step=step+1,total=800)
                write(tag+'.'+kind+'.TRACE.json',trace)
            del loss,logits,g
        training_forward=False;frozen_check()
        art=dict(adapter=None if active_adapter is None else {k:v.detach().cpu() for k,v in active_adapter.state_dict().items()},head=w.detach().cpu(),mapper=teacher['mapper'],kind=kind,tag=tag,seed=seed,steps=800,site='blocks.11.input',shared_dual_stream=True,source_commit=cfg['source_commit'])
        path=out/(tag+'.'+kind+'.model.private.pt');torch.save(art,path)
        assert path.stat().st_size<19058800
        back=load(path);assert same(art,back),'MODEL_READBACK'
        with torch.no_grad():
            sg=pg if kind=='H' else encode(read_batch(paths,probe))
            ss=transform(sg,mapper)@w[:-1]+w[-1]
            replay=transform(sg.cpu(),back['mapper'])@back['head'][:-1]+back['head'][-1]
            err=float((ss.cpu()-replay).abs().max())
        assert err<=1e-10,'HEAD_READBACK_SCORE'
        distance=0 if active_adapter is None else math.sqrt(sum(float((v.cpu()-init[n]).square().sum()) for n,v in active_adapter.state_dict().items()))
        rec=dict(arm=kind,steps=800,initial_feature_shift=initial_shift,first_CE=firstloss,last_CE=lastloss,adapter_parameter_L2=distance,head_parameter_L2=float((w-w0).norm()),reset_entries=resets,hook_activation_statistics=hook_stats,artifact_bytes=path.stat().st_size,readback_exact=True,support_readback_score_maxabs=err,backbone_unchanged=True)
        manifests.append(rec);write(tag+'.'+kind+'.MODEL_SEAL.json',rec)
        active_adapter=None;del w,opt,params;torch.cuda.empty_cache()
    write(tag+'.PRODUCER_SEAL.json',dict(arms=manifests,query_pending=True))
    return manifests


def query_scores(tag, ds, rows, teacher):
    global allowed_images, active_adapter, active_kind, training_forward
    assert (out/'MODEL_SEAL.json').exists()
    paths=image_paths(rows,ds);allowed_images=set(paths);training_forward=False
    result={};fresh_g=None
    for kind in ['F','H','L','N','M']:
        active_kind=kind
        if kind in ['F','H']:active_adapter=None
        else:
            active_adapter=(LinearControl(768) if kind=='L' else OPSAAdapter(768)).cuda().eval()
        art=None if kind=='F' else load(out/(tag+'.'+kind+'.model.private.pt'))
        if active_adapter is not None:active_adapter.load_state_dict(art['adapter'])
        mapper=to_device(teacher['mapper'] if kind=='F' else art['mapper'])
        w=(teacher['W'] if kind=='F' else art['head']).cuda();values=[];gf=[]
        with torch.no_grad():
            for ids in torch.arange(len(rows)).split(32):
                g=fresh_g[ids].cuda() if kind=='H' else encode(read_batch(paths,ids))
                values.append((transform(g,mapper)@w[:-1]+w[-1]).cpu())
                if kind=='F':gf.append(g.cpu())
                status(phase='QUERY_AFTER_SEAL',producer=tag,arm=kind,done=int(ids[-1])+1,total=len(rows))
        if kind=='F':fresh_g=torch.cat(gf)
        result[kind]=torch.cat(values);assert torch.isfinite(result[kind]).all()
        active_adapter=None;torch.cuda.empty_cache()
    frozen_check();return result


def main():
    global model,raw,pointwise,versions,preprocess
    write('SYNTHETIC.json',synthetic_check())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    sys.path.insert(0,cfg['producer_root'])
    from train_base import setup,raw as base_raw
    raw=base_raw
    model,_,_,_,_,pointwise=setup(cfg)
    assert len(model.net.backbone.blocks)==12
    versions={k:p._version for k,p in model.named_parameters()}
    from run_medical_v2 import transform as canonical_transform
    preprocess=canonical_transform(False)
    handle=model.net.backbone.blocks[11].register_forward_pre_hook(hook)
    prod=Path(cfg['producer_root']);prior=Path(cfg['prior_root'])
    plan=load(prod/'prep/PLAN.private.pt')
    rr=json.loads((prior/'ISIC.S_old.rows.private.json').read_text());rr=[r for r in rr if r['label']<4]
    ir=[rr[i] for i in plan['row_ids'].tolist()]
    assert len(ir)==2555 and [r['identity_component'] for r in ir]==plan['groups'] and torch.equal(torch.tensor([r['label'] for r in ir]),plan['y'])
    hr=json.loads((prior/'HK.S_old.rows.private.json').read_text());hm=torch.tensor([r['label']<13 for r in hr]);hr=[r for r in hr if r['label']<13]
    hy=torch.tensor([r['label'] for r in hr]);hg=[r['identity_component'] for r in hr];assert len(hr)==3140
    hf=load(prior/'HK.S_old.features.private.pt');assert torch.equal(hf['y'][hm],hy)
    hc=F.normalize(hf['x'][hm].double(),dim=1)
    ht=load(Path(cfg['window_root'])/'group_consistency_pilot/HK.FCR.artifact.private.pt')
    hteacher=dict(mapper=ht['mapper'],W=ht['W']);assert hteacher['W'].shape==(1536,13)
    its=[load(prod/'prep'/f'{k}.teacher.private.pt') for k in [0,1]]
    # File-access enforcement: only current support or post-seal query images.
    roots=[Path(p).resolve() for p in cfg['image_roots'].values()]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if any(p.is_relative_to(root) for root in roots):assert p in allowed_images,'IMAGE_OUTSIDE_CURRENT_SUPPORT_OR_SEALED_QUERY'
    sys.addaudithook(guard)
    producers=[]
    for k in [0,1]:
        mask=plan['fold']==k;ids=torch.where(mask)[0].tolist();other=torch.where(~mask)[0].tolist()
        assert not {plan['groups'][i] for i in ids}&{plan['groups'][i] for i in other}
        producers.append(train_producer('ISIC'+str(k),'ISIC',[ir[i] for i in ids],plan['y'][mask],[plan['groups'][i] for i in ids],plan['cached_g'][mask],its[k],74440+k))
    producers.append(train_producer('HK','HK',hr,hy,hg,hc,hteacher,74442))
    write('MODEL_SEAL.json',dict(producers=3,models=12,all_arms_sealed=True,query_risk_pending=True,source_commit=cfg['source_commit']))
    iy=plan['y'];igs=plan['groups'];iscores={n:torch.empty(len(iy),4,dtype=torch.double) for n in ['F','H','L','N','M']}
    foldtab={}
    old=load(prod/'evaluate/responses.private.pt');assert torch.equal(old['y'],iy) and old['groups']==igs
    parity=[]
    for k in [0,1]:
        mask=plan['fold']!=k;ids=torch.where(mask)[0].tolist();yy=iy[mask];gg=[igs[i] for i in ids]
        scores=query_scores('ISIC'+str(k),'ISIC',[ir[i] for i in ids],its[k])
        err=float((scores['F']-old['score']['F_out'][mask]).abs().max());assert err<=1e-5
        assert torch.equal(scores['F'].argmax(1),old['score']['F_out'][mask].argmax(1))
        parity.append(dict(dataset='ISIC'+str(k),Fscore_maxabs=err,prediction_exact=True))
        for n,v in scores.items():iscores[n][mask]=v
        foldtab[str(k)]={n:metrics(v,yy,gg,4) for n,v in scores.items()}
    itab={n:metrics(v,iy,igs,4) for n,v in iscores.items()};assert itab['F']['correct']==[237,13,64,1626]
    ipa={n:pairs(iscores[n],iscores['M'],iy,4) for n in ['F','H','L','N']}
    it=dict(summary=itab,folds=foldtab,paired=ipa,gate=risk_gate(itab,foldtab,ipa))
    hparts={n:[] for n in iscores};ys=[];ggall=[];cohorts={};seenp={r['relative_path'] for r in hr};seeng=set(hg)
    for role in ['E','Qgate']:
        rows=json.loads((prior/f'HK.{role}.rows.private.json').read_text());keep=torch.tensor([r['label']<13 for r in rows]);rows=[r for r in rows if r['label']<13]
        yy=torch.tensor([r['label'] for r in rows]);gg=[r['identity_component'] for r in rows];pp={r['relative_path'] for r in rows}
        assert len(rows)=={'E':986,'Qgate':388}[role] and not pp&seenp and not set(gg)&seeng;seenp|=pp;seeng|=set(gg)
        scores=query_scores('HK','HK',rows,hteacher)
        of=load(prior/f'HK.{role}.features.private.pt');assert torch.equal(of['y'][keep],yy)
        og=F.normalize(of['x'][keep].double(),dim=1);reference=transform(og,hteacher['mapper'])@hteacher['W'][:-1]+hteacher['W'][-1]
        err=float((scores['F']-reference).abs().max());assert err<=1e-5
        tab={n:metrics(v,yy,gg,13) for n,v in scores.items()};assert tab['F']['correct']==cfg['HK_expected'][role]['correct']
        parity.append(dict(dataset='HK_'+role,Fscore_maxabs=err,class_counts_exact=True))
        cohorts[role]=dict(summary=tab,paired={n:pairs(scores[n],scores['M'],yy,13) for n in ['F','H','L','N']})
        for n,v in scores.items():hparts[n].append(v)
        ys.append(yy);ggall+=gg
    yy=torch.cat(ys);hs={n:torch.cat(v) for n,v in hparts.items()};htab={n:metrics(v,yy,ggall,13) for n,v in hs.items()};hpa={n:pairs(hs[n],hs['M'],yy,13) for n in ['F','H','L','N']}
    hk=dict(summary=htab,cohorts=cohorts,paired=hpa,gate=risk_gate(htab,{n:v['summary'] for n,v in cohorts.items()},hpa,hk=True))
    torch.save(dict(ISIC=iscores,HK=hs),out/'responses.private.pt')
    passed=it['gate']['pass_all'] and hk['gate']['pass_all']
    write('aggregate.json',dict(decision='OPSA_INTERNAL_POINT_SIGNAL' if passed else 'STOP_OPSA_INTERNAL_PILOT',method_success=False,scientific_complete=True,ISIC=it,HK=hk,producers=producers,parity=parity,counts=counts,bootstrap=False,CI=None,local_MAPC_contract=True,historically_exposed=True,source_commit=cfg['source_commit']))
    handle.remove();status(phase='COMPLETE')

start=time.time();code=1
try:
    main();code=0
except BaseException as e:
    write('FAILURE.json',dict(type=type(e).__name__,scientific_complete=False,no_retry=True))
    (out/'FAILURE.private.txt').write_text(traceback.format_exc());status(phase='FAILED',error_type=type(e).__name__)
finally:
    write('ACTUAL_COUNTS.json',counts);u=resource.getrusage(resource.RUSAGE_SELF)
    write('WORKER_RESOURCE_RECEIPT.json',dict(wall_seconds=time.time()-start,CPU_seconds=u.ru_utime+u.ru_stime,peak_RSS_bytes=u.ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0))
    write('WORKER_EXIT.json',dict(exit_code=code))
raise SystemExit(code)
