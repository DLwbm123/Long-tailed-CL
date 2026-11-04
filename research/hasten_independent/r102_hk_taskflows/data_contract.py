"""New original-label/task binding, never old-order teacher tensors."""
import csv,json,io
from pathlib import Path
import torch
from common import write

def metadata(c):
    src=json.loads(Path(c['source_binding']).read_text());rows=list(csv.DictReader((Path(c['manifest'])/'train.csv').open()))
    assert rows==list(csv.DictReader(io.StringIO(src['train_manifest']))) and len(rows)==6821
    y=torch.tensor([c['order'].index(int(v['original_label'])) for v in rows]);assert torch.bincount(y,minlength=23).tolist()==c['train_counts']
    roles=src['roles'];assert [len(roles[k]) for k in ['S','B','F']]==[3405,1713,1703]
    for k,ids in roles.items():assert [rows[i]['sample_id'] for i in ids]==src['stable_sample_ids'][k]
    for s in range(len(c['stages'])):
        C=sum(map(len,c['stages'][:s+1]));assert all((torch.bincount(y[[i for i in ids if y[i]<C]],minlength=C)>0).all() for ids in roles.values())
    write(Path(c['output'])/'ASSET_CONTRACT.json',{'passed':True,'TRAIN':6821,'role_counts':{k:len(v) for k,v in roles.items()},'stable_id_binding':True,'new_order':True,'tiny_new_indices':c['tiny_indices'],'history_access':'all arrived TRAIN teachers; fixed arrived S/B modules','bounded_memory':False,'old_supervised_weights_features_read':False,'VAL_TEST_read':False})
    return rows,y,roles

def load(c,s,meta):
    rows,y,roles=meta;C=sum(map(len,c['stages'][:s+1]));pack=torch.load(Path(c['cache_source'])/f'teacher_s{s}.private.pt',map_location='cpu',weights_only=False)
    expected=[i for i,v in enumerate(y.tolist()) if v<C];assert pack['new_stream']==c['stream_name'] and pack['stage']==s and pack['ids']==expected and torch.equal(pack['labels'],y) and pack['filled'][expected].all()
    assert not pack['filled'][[i for i in range(len(y)) if i not in set(expected)]].any()
    si=sorted([i for i in roles['S'] if int(y[i])<C],key=lambda i:(int(y[i]),i));bi=sorted(i for i in roles['B'] if int(y[i])<C)
    S=pack['z'][si].cuda();B=pack['z'][bi].cuda();sy=y[si].cuda();by=y[bi].cuda();ix=[torch.where(sy==k)[0] for k in range(C)]
    assert torch.isfinite(S).all() and torch.isfinite(B).all() and all(len(v)>0 for v in ix)
    head=torch.load(Path(c['strong_heads'])/f'head{s}.private.pt',map_location='cpu',weights_only=False);base=head['prototypes'].cuda();assert head['new_stream']==c['stream_name'] and head['seen']==C and head['train_counts'].tolist()==c['train_counts'][:C]
    births=[b for b,cl in enumerate(c['stages'][:s+1]) for _ in cl]
    return dict(stage=s,stage_denominator=len(c['stages'])-1,C=C,S=S,B=B,Sy=sy,By=by,ix=ix,Sids=si,Bids=bi,base=base,raw_mean=torch.stack([S[j].mean(0) for j in ix]),rawcounts=c['train_counts'][:C],births=births,nB=torch.bincount(by,minlength=C).float(),tail=torch.tensor(c['train_counts'][:C],device='cuda')<250,protected=torch.tensor([b<s or c['train_counts'][i]<250 for i,b in enumerate(births)],device='cuda'),teacher_scores=20*B@base.T)
