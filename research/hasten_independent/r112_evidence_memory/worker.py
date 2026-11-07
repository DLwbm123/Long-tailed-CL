"""Construction, then conditional complete four-rule classifier experiment."""
import csv,gc,importlib,json,os,time,traceback
from pathlib import Path
import torch
from . import evidence as E
from . import roots
from .supply import write

common=importlib.import_module('r110_ecv_opportunity.common')


def scientific(cfg,counter):
    from .native import run as native
    out=Path(cfg['output']);parent=Path(cfg['root_output']);rows=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()))
    assert len(rows)==6821 and len({r['identity_component'] for r in rows})==len(rows)
    y=torch.tensor([cfg['order'].index(int(r['original_label'])) for r in rows]);assert torch.bincount(y,minlength=23).tolist()==cfg['train_counts']
    encoder=importlib.import_module('r110_ecv_opportunity.encoding').Encoder(cfg);first=[i for i in range(len(rows)) if int(y[i])<5]
    write(out/'STATUS.json',{'phase':'COMPLETE_NATIVE_RESOURCE_ADMISSION'})
    native(cfg,encoder,rows,first,counter)
    while not (parent/'GLOBAL_NATIVE_ADMISSION.json').exists():
        if (parent/'GLOBAL_NATIVE_FAIL.json').exists():write(out/'STATUS.json',{'phase':'COMPLETE_NOT_RESOURCE_ADMITTED','science_optimizer':0});return
        common.guard(cfg);time.sleep(2)
    counter.phase='science';write(out/'SCIENCE_STARTED.json',{'epoch':time.time(),'original_start':cfg['start'],'frozen_protocol':'SCIENTIFIC_FREEZE.json'})
    models={s:roots.learner.Learner(s).cuda() for s in [74003,74004]};opts={s:roots.learner.optimizer(m) for s,m in models.items()};memory={s:None for s in models};records=[]
    for stage,cs in enumerate(cfg['stages']):
        ids=[i for i in range(len(rows)) if int(y[i]) in cs];common.guard(cfg)
        write(out/'STATUS.json',{'phase':'SCIENCE_CURRENT_TRAIN_ENCODING','stage':stage,'counts':dict(counter)})
        z=encoder.encode(cfg,rows,ids,counter,'science_TRAIN',ids);full=E.mem.from_views(z,torch.tensor(ids,device='cuda'),y[ids].cuda());del z
        current=E.born(full,rows,stage);del full
        for seed in models:
            previous=memory[seed]
            if stage in [0,2,3,4]:
                for index in range(4):
                    common.guard(cfg);write(out/'STATUS.json',{'phase':'SCIENCE_P1C_ROOT','seed':seed,'stage':stage,'scenario':index,'completed_roots':len(records),'counts':dict(counter)})
                    pool,previous_root,future,tail,meta,m,o,teacher=roots.root(previous,current,cfg,stage,index,counter,True)
                    candidates,ledger=roots.generate(pool,previous_root,m,o,stage,meta['seed'],counter,teacher)
                    tapes=[]
                    for tape in [31001,31002]:
                        runs=[roots.rollout(pool,future,m,o,teacher,c,tape+meta['seed'],counter) for c in candidates]
                        scored=roots.score(runs,tail);tapes.append(scored)
                        raw=[]
                        for candidate,run in zip(candidates,runs):
                            data={'candidate_key':candidate.key,'support_memory_keys':roots.contracts.token_keys(candidate.memory),'bytes':run['bytes']}
                            for role in ['FIT','AUDIT']:
                                data[role]=[{'ids':x.ids,'good':x.good.tolist(),'correct':x.correct.tolist(),'denominator':x.denominator.tolist()} for x in run[role]]
                            raw.append(data)
                        write(out/f'RAW_{seed}_{stage}_{index}_{tape}.private.json',{'metadata':meta,'runs':raw})
                        del runs
                    rec={'stream':cfg['setting'],'seed':seed,'stage':stage,'scenario':index,'metadata':meta,'byte_ledger':ledger,
                         'candidates':[{'family':c.family,'anchor':c.anchor,'removals':c.removals} for c in candidates],'tapes':tapes}
                    records.append(rec);write(out/f'ROOT_{seed}_{stage}_{index}.json',rec)
                    del pool,previous_root,future,m,o,teacher,candidates;gc.collect();torch.cuda.empty_cache()
            m=models[seed];o=opts[seed];pool=E.join(previous,current);teacher=roots.learner.frozen(m) if m.C else None
            m.forbidden={int(i) for r in [1,2] for i in pool.role(r).parents};m.arrive(current.role(0),pool.C)
            roots.learner.learn(m,o,pool.role(0),200,seed+stage*100+1,counter,teacher,64,'science_donor')
            memory[seed]=E.compressed(pool,seed+70000+stage)
            roots.learner.learn(m,o,memory[seed].role(0),100,seed+stage*100+2,counter,teacher,64,'science_donor')
            size=memory[seed].serialize(out/f'DONOR_{seed}_CURRENT.private.bin')
            # Exercise the persisted flags/stats as the sole history decoder.
            memory[seed]=E.Evidence.restore(out/f'DONOR_{seed}_CURRENT.private.bin','cuda')
            write(out/f'DONOR_{seed}_{stage}.json',{'seed':seed,'stage':stage,'tokens':len(memory[seed].bank),'bytes':size,'S_tokens':len(memory[seed].role(0)),'S_capacity':256-2*memory[seed].C,'FIT_tokens':memory[seed].C,'AUDIT_tokens':memory[seed].C,'classifier_updates':300})
            del previous,pool,teacher
        del current;gc.collect();torch.cuda.empty_cache()
    science=sum(counter.get(k,0) for k in ['science_donor_classifier_optimizer','science_prefix_classifier_optimizer','paired_classifier_updates','engineering_virtual_classifier_optimizer'])
    # Counts remaps the latter two into the scientific phase.
    assert science<=585432,'COMPLETE_SCIENCE_UPDATE_BOUND'
    write(out/'RESULTS.json',{'stream':cfg['setting'],'roots':records,'counts':dict(counter),'science_classifier_updates':science,'controller_RL_updates':0,'VAL_TEST':0})
    write(out/'STATUS.json',{'phase':'COMPLETE_P1C','roots':len(records),'science_classifier_updates':science})


def run(cfg):
    start=time.time();cpu=time.process_time();out=Path(cfg['output']);counter=common.Counts()
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((12*2**30-64*2**20)/total,0)
        if cfg['phase']=='construction':
            from .construction import run as construct
            construct(cfg)
        else:scientific(cfg,counter)
    except BaseException:
        (out/'FAILURE.private.txt').write_text(traceback.format_exc());write(out/'FAILURE_COUNTS.json',dict(counter));raise
    finally:write(out/'COSTS.json',{'phase':cfg['phase'],'CPU_seconds':time.process_time()-cpu,'wall_seconds':time.time()-start,'counts':dict(counter),
                                 'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'preparation_CPU_seconds':'nonzero not fully sampled'})

if __name__=='__main__':run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))
