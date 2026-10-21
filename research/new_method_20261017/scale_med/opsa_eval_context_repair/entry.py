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


qualified = {}
def batches(tag, ds, rows):
    role='S_old' if ds=='ISIC' else tag.split('_',1)[1]
    original=json.loads((Path(cfg['prior_root'])/(ds+'.'+role+'.rows.private.json')).read_text())
    lookup={r['relative_path']:i for i,r in enumerate(original)}
    groups={}
    for j,row in enumerate(rows):
        i=lookup[row['relative_path']];assert row==original[i]
        groups.setdefault(i//32,[]).append((j,i%32))
    result=[]
    for block,items in sorted(groups.items()):
        size=min(32,len(original)-block*32)
        padded=[items[0][0]]*size
        for j,slot in items:padded[slot]=j
        result.append((torch.tensor(padded),torch.tensor([j for j,_ in items]),torch.tensor([slot for _,slot in items])))
    assert sorted(j for _,ids,_ in result for j in ids.tolist())==list(range(len(rows)))
    return result

def qualify(tag,ds,rows,teacher,cached,reference):
    global allowed_images,active_adapter,training_forward
    active_adapter=None;training_forward=False
    paths=image_paths(rows,ds);allowed_images=set(paths)
    g=torch.empty_like(cached);receipts=[]
    with torch.no_grad():
        for padded,ids,slots in batches(tag,ds,rows):
            value=encode(read_batch(paths,padded))[slots].cpu();g[ids]=value
            receipts.append(dict(input_rows=len(padded),retained_rows=len(ids),feature_maxabs=float((value-cached[ids]).abs().max())))
            status(phase='ORIGINAL_F_QUALIFICATION',producer=tag,batches=len(receipts))
    score=transform(g,teacher['mapper'])@teacher['W'][:-1]+teacher['W'][-1]
    fe=float((g-cached).abs().max());se=float((score-reference).abs().max())
    pred=torch.equal(score.argmax(1),reference.argmax(1))
    write(tag+'.QUALIFICATION.json',dict(feature_maxabs=fe,score_maxabs=se,prediction_exact=pred,rows=len(rows),batches=receipts))
    assert torch.isfinite(g).all() and torch.isfinite(score).all()
    assert fe<=1e-6 and se<=1e-5 and pred,'ORIGINAL_CONTEXT_QUALIFICATION_FAILED'
    frozen_check();qualified[tag]=(g,score)

def query_scores(tag,ds,rows,teacher):
    global allowed_images,active_adapter,active_kind,training_forward
    if ds=='HK':tag=cfg['current_HK_tag']
    assert (out/'ALL_QUERY_F_QUALIFIED.json').exists()
    paths=image_paths(rows,ds);allowed_images=set(paths);training_forward=False
    fresh_g,fs=qualified[tag];result={'F':fs}
    modeltag=tag if ds=='ISIC' else 'HK'
    for kind in ['H','L','N','M']:
        active_kind=kind;active_adapter=None if kind=='H' else (LinearControl(768) if kind=='L' else OPSAAdapter(768)).cuda().eval()
        art=load(Path(cfg['model_root'])/(modeltag+'.'+kind+'.model.private.pt'))
        assert art['kind']==kind and art['tag']==modeltag and art['steps']==800
        if active_adapter is not None:active_adapter.load_state_dict(art['adapter'])
        mapper=to_device(art['mapper']);w=art['head'].cuda();values=torch.empty(len(rows),w.shape[1],dtype=torch.double)
        with torch.no_grad():
            for padded,ids,slots in batches(tag,ds,rows):
                g=fresh_g[ids].cuda() if kind=='H' else encode(read_batch(paths,padded))[slots]
                values[ids]=(transform(g,mapper)@w[:-1]+w[-1]).cpu()
                status(phase='SEALED_MODEL_EVALUATION',producer=tag,arm=kind,done=int(ids[-1])+1,total=len(rows))
        result[kind]=values;assert torch.isfinite(values).all()
        active_adapter=None;torch.cuda.empty_cache()
    frozen_check();return result


def main():
    global model,raw,pointwise,versions,preprocess
    write('SYNTHETIC.json',synthetic_check())
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    sys.path.insert(0,cfg['producer_root'])
    from train_base import setup,raw as base_raw
    raw=base_raw
    model,_,_,_,_,pointwise=setup(cfg)
    assert len(model.net.backbone.blocks)==12
    versions={k:p._version for k,p in model.named_parameters()}
    from run_medical_v2 import transform as canonical_transform
    preprocess=canonical_transform(False)
    from PIL import features
    signature=dict(Pillow=Image.__version__,jpg_abi=features.version('jpg'),libjpeg_turbo=features.version('libjpeg_turbo'))
    assert signature==dict(Pillow='10.4.0',jpg_abi='6.2',libjpeg_turbo='3.0.3')
    write('NUMERICAL_BINDING.json',dict(decoder=signature,matmul_tf32=False,cudnn_tf32=True,benchmark=False))
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
    seal=json.loads((Path(cfg['model_root'])/'MODEL_SEAL.json').read_text());assert seal['all_arms_sealed'] and seal['models']==12
    write('MODEL_SEAL.json',dict(seal,read_only_reuse=True,training_updates_this_run=0))
    producers=[json.loads((Path(cfg['model_root'])/(tag+'.PRODUCER_SEAL.json')).read_text())['arms'] for tag in ['ISIC0','ISIC1','HK']]
    old=load(prod/'evaluate/responses.private.pt');assert torch.equal(old['y'],plan['y']) and old['groups']==plan['groups']
    for k in [0,1]:
        mask=plan['fold']!=k;ids=torch.where(mask)[0].tolist()
        qualify('ISIC'+str(k),'ISIC',[ir[i] for i in ids],its[k],plan['cached_g'][mask],old['score']['F_out'][mask])
    for role in ['E','Qgate']:
        rows=json.loads((prior/f'HK.{role}.rows.private.json').read_text());keep=torch.tensor([r['label']<13 for r in rows]);rows=[r for r in rows if r['label']<13]
        of=load(prior/f'HK.{role}.features.private.pt');assert torch.equal(of['y'][keep],torch.tensor([r['label'] for r in rows]))
        og=F.normalize(of['x'][keep].double(),dim=1);ref=transform(og,hteacher['mapper'])@hteacher['W'][:-1]+hteacher['W'][-1]
        qualify('HK_'+role,'HK',rows,hteacher,og,ref)
    write('ALL_QUERY_F_QUALIFIED.json',dict(producers=list(qualified),feature_tolerance=1e-6,score_tolerance=1e-5,all_prediction_exact=True,counts=dict(counts),learned_models_not_yet_loaded=True))
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
        cfg['current_HK_tag']='HK_'+role
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
    write('aggregate.json',dict(decision='OPSA_INTERNAL_POINT_SIGNAL' if passed else 'STOP_OPSA_INTERNAL_PILOT',method_success=False,scientific_complete=True,ISIC=it,HK=hk,producers=producers,parity=parity,counts=counts,bootstrap=False,CI=None,local_MAPC_contract=True,historically_exposed=True,source_commit=cfg['source_commit'],evaluation_only=True,reused_training_source='bdc4f0f099405593f9d416ff1ff704b46f271927'))
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
