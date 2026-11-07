"""Prospective input coverage check; no real classifier initialization or prediction.

The frozen uniform ZERO constructor ignores learned scores. Its geometry and
feasibility transition are reused, with inert state embeddings. Only current
arrivals and each seed's retained 256 tokens are accessible.
"""
import csv,gc,importlib,json,os,random,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F

base=importlib.import_module('r110_ecv_opportunity.base.legacy_manager')
memory=importlib.import_module('r110_ecv_opportunity.base.memory')
Roles=importlib.import_module('r110_ecv_opportunity.episodes').Roles
common=importlib.import_module('r110_ecv_opportunity.common')
write=common.write


def query_free_choice(pool,C,capacity=256):
    x=base.Choice.__new__(base.Choice);x.pool=pool;x.C=C;x.B=min(capacity,len(pool))
    x.emb=torch.zeros(len(pool),32,device=pool.z.device);x.val=torch.zeros(len(pool),device=pool.z.device)
    g=pool.gidx;gy=pool.y[g]
    x.cent=F.normalize(torch.stack([pool.z[g[gy==c]].mean(0) for c in range(C)]),dim=-1)
    x.cover=torch.stack([g[gy==c][(pool.z[g[gy==c]]@x.cent[c]).argmax()] for c in range(C)])
    x.localcap=torch.bincount(pool.parent_index[pool.view>0],minlength=len(g));x.optional=g[~torch.isin(g,x.cover)]
    x.capacity_hist=torch.bincount(x.localcap[pool.parent_index[x.optional]],minlength=6)
    x.covercap=int(x.localcap[pool.parent_index[x.cover]].sum())
    ng=torch.arange(C,min(x.B,len(g))+1,device=g.device)
    x.ng=ng[base.top_capacity(x.capacity_hist.expand(len(ng),-1),ng-C)+x.covercap>=x.B-ng]
    assert len(x.ng)>0
    return x


def zero(pool,C,seed,capacity=256):
    return query_free_choice(pool,C,capacity).construct('RANDOM',generator=torch.Generator(device=pool.z.device).manual_seed(seed))[0]


def synthetic_check(device='cpu'):
    Learner=importlib.import_module('r110_ecv_opportunity.base.learner').Learner
    checks=0
    for n,B,partial in [(18,48,False),(24,64,True),(12,256,False)]:
        torch.manual_seed(111)
        p=memory.from_views(F.normalize(torch.randn(n,6,512,device=device),dim=-1),torch.arange(n,device=device),torch.arange(n,device=device)//(n//3))
        if partial:p=p.take(torch.where((p.view==0)|((p.pid+p.view)%3==0))[0])
        for modelseed in [111,221]:
            m=Learner(modelseed).to(device);m.arrive(p,3);v=base.ValueNet().to(device)
            actual=base.Choice(p,m,[n//3]*3,[0]*3,0,B,v);q=query_free_choice(p,3,B)
            for seed in [74003,74004]:
                a=actual.construct('RANDOM',generator=torch.Generator(device=device).manual_seed(seed))[0]
                b=q.construct('RANDOM',generator=torch.Generator(device=device).manual_seed(seed))[0]
                assert torch.equal(a.pid*8+a.view,b.pid*8+b.view),'ZERO_KERNEL_MISMATCH'
                a=actual.construct('BALANCED')[0];b=q.construct('BALANCED')[0]
                assert torch.equal(a.pid*8+a.view,b.pid*8+b.view),'BAL_KERNEL_MISMATCH'
                checks+=2
    return {'passed':True,'checks':checks,'device':device,'synthetic_only':True,'real_classifier_forward':0}


def support_root(split,cfg,stage,index,rows):
    """Called only after complete real old-tail input coverage is established."""
    supported=sorted(set(split['S'].y.tolist()));seed=75110+100*stage+index
    future=[c for c in cfg['stages'][stage] if c in supported];random.Random(seed).shuffle(future)
    if len(future)<2:return {'eligible':False,'reason':'less_than_two_future_classes'}
    prefix=sorted(set(supported)-set(future[:2]));order=prefix+future[:2];mapping=[-1]*(max(cfg['stages'][stage])+1)
    for j,c in enumerate(order):mapping[c]=j
    s=split['S'];s=memory.Bank(s.z,s.pid,torch.tensor(mapping,device=s.z.device)[s.y],s.view)
    kept=None;phases=[(t,[mapping[c] for c in prefix if c in cs]) for t,cs in enumerate(cfg['stages'][:stage+1])]
    phases=[(t,cs) for t,cs in phases if cs]
    for j,(t,cs) in enumerate(phases):
        data=s.take(torch.where(torch.isin(s.y,torch.tensor(cs,device=s.z.device)))[0]);pool=memory.join(kept,data)
        C=max(cs)+1
        if j<len(phases)-1:kept=zero(pool,C,seed+2000+j)
    x=query_free_choice(pool,len(prefix));a=x.construct('RANDOM',generator=torch.Generator(device=pool.z.device).manual_seed(seed))[0];b=x.construct('BALANCED')[0]
    distinct=not torch.equal(a.pid*8+a.view,b.pid*8+b.view)
    if not distinct and len(pool)>256:
        edit=importlib.import_module('r110_ecv_opportunity.base.anchor_candidates')._edit
        from types import SimpleNamespace
        state=SimpleNamespace(pool=pool,observed=[0]*len(prefix),groups={int(i):rows[int(i)]['identity_component'] for i in pool.parents})
        alt,n=edit(state,a,torch.zeros(len(pool),device=pool.z.device),2,seed+4,True)
        distinct=n>0
    return {'eligible':len(pool)>256 and distinct,'root_tokens':len(pool),'capacity':256,'concrete_A_or_L_alternative':distinct,'reason':'informative' if len(pool)>256 and distinct else 'no_pressure_or_action'}


def run(cfg):
    out=Path(cfg['output']);counter={};cpu=time.process_time();start=time.time()
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(out/'KERNEL_CHECK.json',synthetic_check('cuda'))
        rows=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));assert len(rows)==6821
        assert len({r['identity_component'] for r in rows})==len(rows)
        y=torch.tensor([cfg['order'].index(int(r['original_label'])) for r in rows]);assert torch.bincount(y,minlength=23).tolist()==cfg['train_counts']
        Encoder=importlib.import_module('r110_ecv_opportunity.encoding').Encoder
        encoder=Encoder(cfg);roles=Roles(947110);kept={s:None for s in cfg['donor_seeds']};records=[];details=[]
        for stage,classes in enumerate(cfg['stages']):
            common.guard(cfg);ids=[i for i in range(len(rows)) if int(y[i]) in classes];roles.arrive(ids,rows)
            write(out/'STATUS.json',{'phase':'CURRENT_ARRIVAL_ENCODING','stage':stage,'counts':counter})
            z=encoder.encode(cfg,rows,ids,counter,'readiness_TRAIN',ids);data=memory.from_views(z,torch.tensor(ids,device='cuda'),y[ids].cuda());del z
            for seed in cfg['donor_seeds']:
                previous=kept[seed];pool=memory.join(previous,data);C=max(classes)+1
                if stage in [0,2,3,4]:
                    split=roles.split(pool,rows);T={c for c in range(min(classes)) if cfg['train_counts'][c]<250};missing={}
                    for c in sorted(T):
                        counts={role:int((bank.y[bank.gidx]==c).sum()) for role,bank in split.items()}
                        absent=[role for role,n in counts.items() if not n]
                        details.append({'stream':cfg['setting'],'seed':seed,'stage':stage,'real_class':c,'original_label':cfg['order'][c],**counts,'missing_roles':absent,'patient_denominator':'NA'})
                        if absent:missing[str(c)]=absent
                    full=bool(T) and not missing
                    for index in range(4):
                        rec={'stream':cfg['setting'],'seed':seed,'stage':stage,'scenario':index,'required_real_oldtail':sorted(T),'full_input_coverage':full,'missing':missing,'previous_tokens':0 if previous is None else len(previous),'current_plus_previous_tokens':len(pool),'eligible':False,'patient_denominator':'NA'}
                        if full:rec.update(support_root(split,cfg,stage,index,rows))
                        else:rec['reason']='risk_not_applicable' if not T else 'missing_support_or_query'
                        records.append(rec)
                    write(out/'ROOT_INPUT_CHECKS.json',records);write(out/'CLASS_INPUT_COUNTS.json',details)
                    del split
                kept[seed]=zero(pool,C,seed+70000+stage)
                write(out/f'DONOR_{seed}_{stage}.json',{'stage':stage,'seed':seed,'tokens':len(kept[seed]),'parents':len(kept[seed].parents),'serialized_equivalent_bytes':len(kept[seed])*2064+184,'real_classifier_initializations':0,'optimizer':0})
                del previous,pool;gc.collect()
            del data;gc.collect();torch.cuda.empty_cache()
        byseed={str(s):{'informative_roots':sum(r['eligible'] for r in records if r['seed']==s),'informative_snapshots':len({r['stage'] for r in records if r['seed']==s and r['eligible']})} for s in cfg['donor_seeds']}
        passed=all(v['informative_roots']>=4 and v['informative_snapshots']>=1 for v in byseed.values())
        write(out/'READINESS_RESULT.json',{'passed':passed,'stream':cfg['setting'],'by_seed':byseed,'roots_checked':len(records),'new_seed_selection':False,'same_TRAIN_components':True,'patient_denominator':'NA','VAL_TEST':0,'real_classifier_forward':0,'real_classifier_initializations':0,'optimizer':0,'counts':counter,'scientific_P1b_started':False})
        write(out/'STATUS.json',{'phase':'COMPLETE_INPUT_READINESS','passed':passed})
    except BaseException:
        (out/'FAILURE.private.txt').write_text(traceback.format_exc());raise
    finally:write(out/'COSTS.json',{'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'counts':counter,'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'optimizer':0})

if __name__=='__main__':run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
