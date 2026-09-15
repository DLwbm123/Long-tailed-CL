"""Frozen CLS baselines with current-arrival-only sufficient statistics."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms as T
from run_medical_v2 import sha,write_json
from diagnose_medical_v3 import dataset,loader,scored,SEEDS
from evaluate_medical_v2 import write_csv
from utils.medical_v2 import FrozenOriginal,WEIGHT_SHA

DINO_SHA='d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841'
DINO_REV='f9e44c814b77203eaa57a6bdbbd535f21ede1415'

@torch.no_grad()
def extract(config,encoder,out):
    weight=Path(config['weight']) if encoder=='A' else Path(config['v3_root'])/'weights/model.safetensors'
    expected=WEIGHT_SHA if encoder=='A' else DINO_SHA
    assert sha(weight)==expected,'BLOCKED_WEIGHT_SHA'
    if encoder=='A':
        model=FrozenOriginal(weight).cuda().eval();mean=std=[.5]*3
    else:
        from transformers import Dinov2Model
        model=Dinov2Model.from_pretrained(str(weight.parent),local_files_only=True,trust_remote_code=False).cuda().eval()
        assert model.config.hidden_size==768 and model.config.patch_size==14
        mean=[.485,.456,.406];std=[.229,.224,.225]
    model.requires_grad_(False)
    prep={'geometry':'RGB Resize224x224 bicubic antialias','mean':mean,'std':std,'feature':'final layer norm CLS; sample L2','dimension':768}
    import hashlib
    prep_sha=hashlib.sha256(json.dumps(prep,sort_keys=True).encode()).hexdigest()
    lock={'encoder':encoder,'model_sha256':expected,'preprocess':prep,'preprocess_sha256':prep_sha,'test_predictions':0,'splits':{}}
    if encoder=='B':lock['metadata_sha256']={name:sha(weight.parent/name) for name in ('config.json','preprocessor_config.json')};lock['revision']=DINO_REV
    for split in ('train','val'):
        target=out/(split+'.npz');association=out/(split+'_associations.json')
        assert not target.exists(),'BLOCKED_DUPLICATE_FEATURE_EXTRACTION'
        d=dataset(config,list(range(8)),8,split)
        d.transform=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.ToTensor(),T.Normalize(mean,std)])
        parts=[];labels=[];start=time.monotonic()
        for _,x,y in loader(d):
            x=x.cuda();z=model(x)['pre_logits'] if encoder=='A' else model(pixel_values=x).last_hidden_state[:,0]
            assert z.shape==(len(x),768) and torch.isfinite(z).all() and (z.norm(dim=1)>0).all()
            parts.append(torch.nn.functional.normalize(z,dim=1).cpu().numpy());labels.append(y.numpy())
        z=np.concatenate(parts);y=np.concatenate(labels)
        np.savez(target,z=z,y=y);write_json(association,d.rows)
        lock['splits'][split]={'n':len(d),'seconds':time.monotonic()-start,'manifest_sha256':sha(Path(config['protocol'])/(split+'.csv')),
                               'cache_sha256':sha(target),'associations_sha256':sha(association)}
    write_json(out/'CACHE_LOCK.json',lock);del model;gc.collect();torch.cuda.empty_cache()
    return lock

class Arrivals:
    def __init__(self,z,y,order):self.z=z;self.y=y;self.order=order;self.stage=0;self.access=[]
    def next(self):
        bounds=(0,4,6,8);cs=self.order[bounds[self.stage]:bounds[self.stage+1]];self.stage+=1
        self.access.append(list(cs))
        # Nothing except current arrivals is returned to the fitting interface.
        for c in cs:yield c,self.z[self.y==c].astype(np.float64)

def solve_stats(stats,order):
    c=len(order);g=sum(stats[k]['Q']/stats[k]['n'] for k in order)/c
    b=np.stack([stats[k]['s']/stats[k]['n'] for k in order],axis=1)/c
    a=g+1e-3*np.eye(768);w=np.linalg.solve(a,b)
    proto=np.stack([stats[k]['s']/stats[k]['n'] for k in order]);proto/=np.linalg.norm(proto,axis=1,keepdims=True)
    residual=np.linalg.norm(a@w-b)/np.linalg.norm(b)
    assert np.isfinite(w).all() and residual<1e-10,'BLOCKED_RIDGE_SOLVE'
    return w,proto,{'condition_number':float(np.linalg.cond(a)),'relative_residual':float(residual),'finite':True}

def baselines(config,encoder,out):
    with np.load(out/'train.npz') as c:z=c['z'];y=c['y']
    with np.load(out/'val.npz') as c:v=c['z'].astype(np.float64);vy=c['y']
    rows=json.loads((out/'val_associations.json').read_text());allmetrics=[];allpc=[];audits=[];finals=[]
    for seed in SEEDS:
        order=config['class_orders'][str(seed)];arrival=Arrivals(z,y,order);stats={}
        for task,seen in enumerate((4,6,8)):
            for label,current in arrival.next():
                assert label not in stats and label in order[:seen]
                stats[label]={'n':len(current),'s':current.sum(0,dtype=np.float64),'Q':current.T@current}
            assert set(stats)==set(order[:seen])
            w,proto,diag=solve_stats(stats,order[:seen]);mask=np.isin(vy,order[:seen]);indices=np.where(mask)[0]
            targets=np.array([order.index(int(k)) for k in vy[mask]]);rs=[rows[i] for i in indices]
            for method,raw in [('NCM',v[mask]@proto.T),('CBRidge',v[mask]@w)]:
                m,pc=scored(raw,targets,rs,order,[0,4,6][task]);prefix={'encoder':encoder,'classifier':method,'order_seed':seed,'session':task,'split':'val'}
                allmetrics.append(dict(**prefix,**m));allpc += [dict(**prefix,**p) for p in pc]
            audits.append({'order_seed':seed,'session':task,'current_access':arrival.access[-1],'stats_classes':list(stats),'class_counts':{str(k):s['n'] for k,s in stats.items()},**diag})
        canonical=[order.index(k) for k in range(8)];finals.append({'W':w[:,canonical],'NCM':proto[canonical],'stats':stats})
    # Offline equivalence check, isolated from the class-incremental fit interface.
    zz=z.astype(np.float64);n=np.bincount(y,minlength=8);weights=1/(8*n[y])
    a=zz.T@(zz*weights[:,None])+1e-3*np.eye(768)
    b=zz.T@(np.eye(8)[y]*weights[:,None]);batch_w=np.linalg.solve(a,b)
    eq={'batch_stream_max_abs':float(np.max(np.abs(batch_w-finals[0]['W']))),'orders':[]}
    assert np.allclose(batch_w,finals[0]['W'],atol=1e-10,rtol=1e-9),'BLOCKED_RIDGE_EQUIVALENCE'
    for f in finals[1:]:
        e={k:float(np.max(np.abs(f[k]-finals[0][k]))) for k in ('W','NCM')}
        for k in ('W','NCM'):assert np.allclose(f[k],finals[0][k],atol=1e-10,rtol=1e-9)
        assert np.array_equal((v@f['W']).argmax(1),(v@finals[0]['W']).argmax(1))
        assert np.array_equal((v@f['NCM'].T).argmax(1),(v@finals[0]['NCM'].T).argmax(1))
        for label in range(8):
            for key in ('n','s','Q'):assert np.array_equal(f['stats'][label][key],finals[0]['stats'][label][key])
        eq['orders'].append(e)
    write_json(out/'engineering.json',{'stages':audits,'equivalence':eq,'access':'current arrivals only; batch equivalence offline','lambda':1e-3,'bias':False,'statistics_dtype':'float64'})
    write_csv(out/'frozen_val_metrics.csv',allmetrics);write_csv(out/'frozen_per_class_metrics.csv',allpc)
    return allmetrics,allpc,eq

def p1(config):
    out=Path(config['output'])/'p1';out.mkdir(exist_ok=True,parents=True);summary=[];ms=[];pcs=[]
    for encoder in ('A','B'):
        dst=out/encoder;dst.mkdir(exist_ok=True)
        if encoder=='B' and not (Path(config['v3_root'])/'weights/model.safetensors').exists():
            summary.append({'encoder':'B','status':'PARTIAL_BACKBONE_COMPARISON','reason':'Pinned Hugging Face download: remote Network is unreachable; local TCP connection timed out. No replacement.'});continue
        if not (dst/'CACHE_LOCK.json').exists():extract(config,encoder,dst)
        m,p,eq=baselines(config,encoder,dst);ms+=m;pcs+=p;summary.append({'encoder':encoder,'status':'COMPLETE','equivalence':eq})
    write_csv(out/'frozen_val_metrics.csv',ms);write_csv(out/'frozen_per_class_metrics.csv',pcs);write_json(out/'COMPLETE.json',{'encoders':summary,'test_predictions':0})
    lines=['# Frozen representation baselines','','Both classifiers use fixed final-normalized CLS, sample L2 normalization, no bias or classifier hyperparameter search. CBRidge lambda=0.001 with float64 class-balanced sufficient statistics and solve. Each encoder extracts train/val once; no test features.','','| Encoder | Classifier | Final BA | Old recall | Current recall |','|---|---|---:|---:|---:|']
    for r in ms:
        if r['session']==2 and r['order_seed']==1993:lines.append(f"| {r['encoder']} | {r['classifier']} | {r['balanced_accuracy']:.4f} | {r['old_macro_recall']:.4f} | {r['current_macro_recall']:.4f} |")
    lines+=['','Final eight-class statistics/predictions are order invariant and are reported once, not as three independent repetitions. Old/current partitions depend on order; full stage and per-class tables retain all three orders. Streaming/batch equivalence, condition numbers and residuals are in engineering.json.','','Availability: '+json.dumps(summary), '\nA/B compares encoder combinations, not an isolated causal effect of self-supervision. Validation development evidence does not establish test generalization.']
    (out/'P1_FROZEN_FEATURE_BASELINES.md').write_text('\n'.join(lines)+'\n')
