"""Immutable roles, strong teacher anchor, fixed S support throughout deployment."""
import csv,json,io
from pathlib import Path
import torch
from common import write

def metadata(c):
    src=json.loads(Path(c['source_binding']).read_text());rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()))
    assert rows==list(csv.DictReader(io.StringIO(src['train_manifest']))) and len(rows)==6821
    y=torch.tensor([c['order'].index(int(v['original_label'])) for v in rows]);assert torch.bincount(y,minlength=23).tolist()==c['train_counts']
    roles={k:src['roles'][v] for k,v in [('S','S'),('B','W'),('F','F')]}
    assert [len(roles[k]) for k in roles]==[3405,1713,1703]
    sets=[set(v) for v in roles.values()];assert len(set.union(*sets))==6821 and all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    components=[{rows[i]['identity_component'] for i in roles[k]} for k in roles]
    assert all(not a&b for i,a in enumerate(components) for b in components[i+1:])
    write(Path(c['output'])/'ASSET_CONTRACT.json',{'passed':True,'TRAIN':6821,'roles':{k:len(v) for k,v in roles.items()},'component_disjoint':True,'teacher_full_TRAIN_history':True,'F_historical_development':True,'new_encoder_optimizer_F_VAL_TEST':0,'support_deployment':'Fixed S, anchored to original teacher; same all arms','no_new_hashing':'Reuse prior teacher provenance; manifest ordered equality and feature filled/labels checked'})
    return rows,y,roles

def load(c,s,meta):
    rows,y,roles=meta;C=sum(map(len,c['stages'][:s+1]));pack=torch.load(Path(c['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False)
    expected=[i for i,v in enumerate(y.tolist()) if v<C]
    assert pack['stage']==s and pack['ids']==expected and torch.equal(pack['labels'],y) and pack['filled'][expected].all()
    si=sorted([i for i in roles['S'] if int(y[i])<C],key=lambda i:(int(y[i]),i));bi=sorted(i for i in roles['B'] if int(y[i])<C)
    S=pack['z'][si].cuda();B=pack['z'][bi].cuda();sy=y[si].cuda();by=y[bi].cuda()
    assert torch.isfinite(S).all() and torch.isfinite(B).all() and torch.allclose(S.norm(dim=1),torch.ones(len(S),device='cuda'),atol=2e-5)
    ix=[torch.where(sy==k)[0] for k in range(C)];assert all(len(i)>0 for i in ix)
    head=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);base=head['prototypes'].cuda()
    assert base.shape==(C,512) and head['train_counts'].tolist()==c['train_counts'][:C]
    births=[b for b,cl in enumerate(c['stages'][:s+1]) for _ in cl]
    return dict(stage=s,C=C,S=S,B=B,Sy=sy,By=by,ix=ix,Sids=si,Bids=bi,base=base,raw_mean=torch.stack([S[j].mean(0) for j in ix]),rawcounts=c['train_counts'][:C],births=births,nB=torch.bincount(by,minlength=C).float(),tail=torch.tensor(c['train_counts'][:C],device='cuda')<250,protected=torch.tensor([b<s or c['train_counts'][i]<250 for i,b in enumerate(births)],device='cuda'),teacher_scores=20*B@base.T)
