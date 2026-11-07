"""Small synthetic checks for byte accounting, permissions and paired updates."""
import importlib,json,os,tempfile,time
from pathlib import Path
import numpy as np
import torch
from . import evidence as E
from . import roots
from .supply import birth_roles


def run(output,device='cpu'):
    start=time.time();torch.set_num_threads(1 if device=='cpu' else 2);torch.manual_seed(112)
    rows=[{'original_label':str(i//24),'identity_component':f'unit_{i}'} for i in range(144)]
    z=torch.nn.functional.normalize(torch.randn(144,6,512,device=device),dim=-1)
    full=E.mem.from_views(z,torch.arange(144,device=device),torch.arange(144,device=device)//24)
    current=E.born(full.parents_only(torch.arange(72,device=device)),rows,0)
    compressed=E.compressed(current,74003);compressed.validate()
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'a.bin';size=compressed.serialize(path);restored=E.Evidence.restore(path,device)
        assert size==len(compressed.bank)*2064+184<=528568
        assert torch.equal(restored.bank.z,compressed.bank.z) and torch.equal(restored.flags,compressed.flags)
        assert np.array_equal(restored.stats,compressed.stats)
    q=compressed.role(1);x,y,ids=q.bags();assert x.shape==(3,1025) and bool((x[:,512:]==0).all())
    m=roots.learner.Learner(112).to(device);m.arrive(compressed.role(0),3);assert m(x).shape==(3,3)
    m.forbidden=set(q.parents.tolist());o=roots.learner.optimizer(m)
    try:roots.learner.learn(m,o,q,1,112,{},None,64,'forbidden');raise AssertionError('QUERY_FIREWALL_NOT_TRIGGERED')
    except AssertionError as exc:assert str(exc)=='SHADOW_QUERY_IN_GRADIENT_OR_REPLAY'
    # With no additional core beyond original coverage, constructors agree.
    S=compressed.role(0);old=E.qchoice(S,3,200)
    core=(S.pid*8+S.view)[old.cover];new=E.Choice(S,3,200,core)
    for seed in [74003,74004]:
        for mode in ['RANDOM','BALANCED']:
            a=old.construct(mode,generator=torch.Generator(device=device).manual_seed(seed))[0]
            b=new.construct(mode,generator=torch.Generator(device=device).manual_seed(seed))[0]
            assert torch.equal(a.pid*8+a.view,b.pid*8+b.view)
    second=E.born(full.parents_only(torch.arange(72,144,device=device)),rows,1)
    cfg={'stages':[[0,1,2],[3,4,5]],'train_counts':[24]*6};counter={}
    pool,previous,future,tail,meta,m,o,teacher=roots.root(compressed,second,cfg,1,0,counter,True)
    candidates,ledger=roots.generate(pool,previous,m,o,1,meta['seed'],counter,teacher)
    assert 1<len(candidates)<=12 and tail==set(range(3))
    altered=E.Evidence(E.Bank(pool.bank.z.clone(),pool.bank.pid,pool.bank.y,pool.bank.view),pool.flags,pool.stats)
    mask=(altered.flags&3)>0;altered.bank.z[mask]=-altered.bank.z[mask]
    ca,_=roots.generate(altered,previous,m,o,1,meta['seed'],{},teacher)
    assert [c.key for c in candidates]==[c.key for c in ca],'QUERY_FEATURES_AFFECT_CANDIDATE_GENERATION'
    before={k:v.clone() for k,v in m.state_dict().items()}
    run=roots.rollout(pool,future,m,o,teacher,candidates[0],31001,counter)
    again=roots.rollout(pool,future,m,o,teacher,candidates[0],31001,counter)
    result=roots.score([run,again],tail)
    for role in ['FIT','AUDIT']:
        assert result[role][1]['R']==0 and result[role][1]['full_oldtail_evaluable']
        assert all(x['denominator']==[1]*len(x['denominator']) for x in result[role][0]['endpoints'])
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    assert sum(counter[k] for k in counter if k=='paired_classifier_updates')==1400
    assert all(x['total_bytes']<=528568 and x['total_tokens']<=256 for x in run['bytes'])
    out={'passed':True,'device':device,'synthetic_only':True,'checks':['charged flags/stats serialization','global-only 1025-d interface','query training firewall','query-feature perturbation leaves candidates unchanged','original constructor equivalence when core overlaps','protected core and quota constraints','same-origin paired null','source parameters unchanged','three-horizon complete risk and byte bounds'],
         'counts':counter,'wall_seconds':time.time()-start,'real_TRAIN_encoder_rows':0,'real_classifier_updates':0}
    Path(output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out

if __name__=='__main__':run(os.environ['Q_OUTPUT'],os.environ.get('Q_DEVICE','cpu'))
