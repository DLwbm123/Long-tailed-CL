"""One bounded construction check, sharing only current arrivals across arms."""
import csv,gc,importlib,json,time
from pathlib import Path
import numpy as np
import torch
from . import evidence as E
from . import roots
from .supply import birth_roles,write,table

common=importlib.import_module('r110_ecv_opportunity.common')
legacy_root=importlib.import_module('r111_risk_qualification.readiness').support_root


def reference_birth(full,rows,stage):
    roles=birth_roles(full.parents.tolist(),rows);flags=torch.tensor([{'S':0,'FIT':1,'AUDIT':2}[roles[i]] for i in full.pid.tolist()],device=full.z.device)
    stats=np.zeros(23,dtype=E.STATS);stats['birth']=-1
    for c in set(full.y.tolist()):
        ids=full.parents[full.y[full.gidx]==c].tolist();stats[c]=(len(ids),stage,len(ids))
    return E.Evidence(full,flags,stats)


def reference_compressed(pool,seed):
    q=E.qchoice(pool.bank,pool.C,256);bank=q.construct('RANDOM',generator=torch.Generator(device=pool.bank.z.device).manual_seed(seed))[0]
    return E.subset(pool,bank)


def step(new_previous,reference_previous,current,raw_current,cfg,stage,seed,rows):
    records=[];reference_pool=E.join(reference_previous,raw_current);split={name:reference_pool.role(i) for i,name in enumerate(['S','FIT','AUDIT'])}
    T={c for c in range(min(cfg['stages'][stage])) if int(reference_pool.stats[c]['birth_total'])<250}
    original_full=bool(T) and all(all(int((b.y==c).sum())>0 for b in split.values()) for c in T)
    if stage in [0,2,3,4]:
        for index in range(4):
            common.guard(cfg)
            pool,previous,future,tail,meta,m,o,teacher=roots.root(new_previous,current,cfg,stage,index,train=False)
            candidates,ledger=roots.generate(pool,previous,None,None,stage,meta['seed'],{},construction=True)
            old=legacy_root(split,cfg,stage,index,rows)
            full=bool(tail) and meta['full_oldtail_roles'];different=len(candidates)>1;pressure=meta['S_root_tokens']>meta['S_capacity']
            eligible=full and pressure and different
            old_n=old.get('root_tokens');query_tokens=ledger['FIT_tokens']+ledger['AUDIT_tokens']
            rec={'stream':cfg['setting'],'seed':seed,'stage':stage,'scenario':index,'required_oldtail_classes':len(tail),'full_oldtail_roles':full,
                 'original_real_pool_tokens':len(reference_pool.bank),'original_role_S_pool_tokens':len(split['S']),
                 'original_S_root_tokens':old_n,'original_capacity':256,'original_full_oldtail_roles':original_full,
                 'new_real_pool_tokens':len(E.join(new_previous,current).bank),'new_S_root_tokens':meta['S_root_tokens'],'new_S_capacity':meta['S_capacity'],
                 'new_total_anchor_tokens':len(candidates[0].memory)+query_tokens,'new_total_anchor_bytes':(len(candidates[0].memory)+query_tokens)*2064+184,
                 'new_query_tokens':query_tokens,'new_pinned_support_globals':pool.C,'mandatory_support_global_union':ledger['mandatory_support_globals_union'],
                 'concrete_unique_anchor_random_candidates':len(candidates),'full_scientific_candidate_count':'not generated without classifier',
                 'original_pressure':old_n is not None and old_n>256,'new_pressure':pressure,
                 'new_pool_pressure_only_under_paid_S_capacity':pressure and meta['S_root_tokens']<=256,
                 'new_pressure_with_original_no_pressure':pressure and old_n is not None and old_n<=256,
                 'eligible':eligible,'reason':'INFORMATIVE' if eligible else 'risk_not_applicable' if not tail else 'PREFIX_LOSS' if not full else 'NO_LEGAL_ACTION',
                 'query_package':'one fixed global per class and role','patient_denominator':'NA'}
            records.append(rec)
            del pool,previous,future,candidates
    new_pool=E.join(new_previous,current);new=E.compressed(new_pool,seed+70000+stage)
    reference=reference_compressed(reference_pool,seed+70000+stage)
    return new,reference,records


def run(cfg):
    start=time.time();cpu=time.process_time();out=Path(cfg['output']);counter={}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));assert len(rows)==6821
    assert len({r['identity_component'] for r in rows})==len(rows),'COMPONENT_BINDING_CHANGED'
    y=torch.tensor([cfg['order'].index(int(r['original_label'])) for r in rows]);encoder=importlib.import_module('r110_ecv_opportunity.encoding').Encoder(cfg)
    new={s:None for s in [74003,74004]};old={s:None for s in new};records=[];retention=[]
    for stage,cs in enumerate(cfg['stages']):
        ids=[i for i in range(len(rows)) if int(y[i]) in cs];common.guard(cfg)
        write(out/'STATUS.json',{'phase':'CONSTRUCTION_CURRENT_ENCODING','stage':stage,'counts':counter})
        z=encoder.encode(cfg,rows,ids,counter,'construction_TRAIN',ids);full=E.mem.from_views(z,torch.tensor(ids,device='cuda'),y[ids].cuda());del z
        current=E.born(full,rows,stage);reference_current=reference_birth(full,rows,stage)
        for seed in new:
            new[seed],old[seed],rec=step(new[seed],old[seed],current,reference_current,cfg,stage,seed,rows);records.extend(rec)
            size=new[seed].serialize(out/f'NEW_DONOR_{seed}_CURRENT.private.bin')
            # Verify that the charged representation, rather than an external
            # role table, is sufficient for the next-stage retained evidence.
            new[seed]=E.Evidence.restore(out/f'NEW_DONOR_{seed}_CURRENT.private.bin','cuda')
            for c in range(new[seed].C):
                row={'stream':cfg['setting'],'seed':seed,'stage':stage,'real_class':c,'original_label':cfg['order'][c],'total_bytes':size,'total_tokens':len(new[seed].bank),'S_capacity':256-2*new[seed].C}
                for role,name in enumerate(['S','FIT','AUDIT']):
                    b=new[seed].role(role);mask=b.y==c;row[name+'_parents']=int((b.y[b.gidx]==c).sum());row[name+'_global_tokens']=int((mask&(b.view==0)).sum());row[name+'_local_tokens']=int((mask&(b.view>0)).sum())
                retention.append(row)
            write(out/'ROOT_CONSTRUCTION.json',records);write(out/'RETENTION.json',retention)
        del current,reference_current,full;gc.collect();torch.cuda.empty_cache()
    byseed={str(s):{'informative_roots':sum(r['eligible'] for r in records if r['seed']==s),'informative_snapshots':len({r['stage'] for r in records if r['seed']==s and r['eligible']})} for s in new}
    passed=all(x['informative_roots']>=4 and x['informative_snapshots']>=1 for x in byseed.values())
    result={'passed':passed,'stream':cfg['setting'],'by_seed':byseed,'planned_roots':32,'counts':counter,'real_classifier_forward':0,'real_classifier_updates':0,'VAL_TEST':0,'role_seed':947110,'role_reassignment':False,'patient_denominator':'NA',
            'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    write(out/'CONSTRUCTION_RESULT.json',result);write(out/'STATUS.json',{'phase':'CONSTRUCTION_COMPLETE','passed':passed})
