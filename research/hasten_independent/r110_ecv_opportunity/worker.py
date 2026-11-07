"""Native admission, fresh ZERO donor stream, 16 prospective scenarios."""
import os,json,csv,time,gc,traceback,copy
from pathlib import Path
import torch
from .common import write,guard,Counts
from .encoding import Encoder
from .native import run as native
from .episodes import Roles,root,score_runs
from .candidates import generate
from .base.memory import from_views,join
from .base.learner import Learner,optimizer,learn,frozen
from .base.legacy_manager import Choice,ValueNet
from .base.state_features import describe
from .base.contracts import token_keys


def choose_zero(pool,model,observed,birth,stage,seed):
    with torch.random.fork_rng(devices=[]):torch.manual_seed(0);v=ValueNet().cuda()
    return Choice(pool,model,observed,birth,stage,256,v).construct('RANDOM',generator=torch.Generator(device='cuda').manual_seed(seed))[0]


def run(cfg):
    output=Path(cfg['output']);parent=Path(cfg['root_output']);counter=Counts();start=time.time();cpu=time.process_time()
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rows=list(csv.DictReader((Path(cfg['manifest'])/'train.csv').open()));assert len(rows)==6821
        assert len({r['identity_component'] for r in rows})==len(rows),'IDENTITY_BINDING_CHANGED'
        y=torch.tensor([cfg['order'].index(int(r['original_label'])) for r in rows]);assert torch.bincount(y,minlength=23).tolist()==cfg['train_counts']
        first=[i for i in range(len(rows)) if int(y[i])<5];encoder=Encoder(cfg)
        write(output/'STATUS.json',{'phase':'NATIVE_FULL_P1_ADMISSION','counts':dict(counter)})
        admission=native(cfg,encoder,rows,first,counter)
        while not (parent/'GLOBAL_NATIVE_ADMISSION.json').exists():
            if (parent/'GLOBAL_NATIVE_FAIL.json').exists():
                write(output/'STATUS.json',{'phase':'COMPLETE_NOT_ADMITTED','science_updates':0});return
            guard(cfg);time.sleep(2)
        counter.phase='science';write(output/'SCIENCE_STARTED.json',{'epoch':time.time(),'original_start':cfg['start'],'source_commit':cfg['source_commit']})
        roles=Roles(947110);donor=Learner(74002).cuda();donor_opt=optimizer(donor);memory=None;observed=[];birth=[]
        records=[]
        for stage,classes in enumerate(cfg['stages']):
            guard(cfg);ids=[i for i in range(len(rows)) if int(y[i]) in classes];roles.arrive(ids,rows)
            write(output/'STATUS.json',{'phase':'ENCODE_CURRENT_TRAIN','stage':stage,'counts':dict(counter)})
            z=encoder.encode(cfg,rows,ids,counter,'science_TRAIN',ids);data=from_views(z,torch.tensor(ids,device='cuda'),y[ids].cuda());del z
            previous=memory
            # Fixed stage/scenario slots; no favorable replacement after a skip.
            if stage in [0,2,3,4]:
                for index in range(4):
                    guard(cfg);write(output/'STATUS.json',{'phase':'P1_ROOT','stage':stage,'scenario':index,'completed_roots':len(records),'counts':dict(counter)})
                    built,skip=root(previous,data,roles,rows,cfg,stage,index,counter)
                    if skip is not None:
                        rec={'stage':stage,'scenario':index,'skipped':skip};records.append(rec);write(output/f'ROOT_{stage}_{index}.json',rec);continue
                    env,m,o,state,subsets,meta=built
                    desc=describe(state,m,o,counter,teacher=state.teacher,seed=meta['seed'])
                    candidates,matching=generate(state,m,desc,meta['seed'])
                    tapes=[]
                    for tape in [31001,31002]:
                        runs=[env.rollout(c.memory,tape+meta['seed']) for c in candidates]
                        scores=score_runs(runs,env,subsets);tapes.append(scores)
                        private=[]
                        for candidate,run in zip(candidates,runs):
                            private.append({'candidate_key':candidate.key,'memory_keys':token_keys(candidate.memory),
                                            'endpoints':[{'ids':c.ids,'good':c.good.tolist(),'brier':c.brier.tolist()} for c in run['endpoints']]})
                        write(output/f'RAW_{stage}_{index}_{tape}.private.json',{'mapping_context':meta,'query_labels':[q.bags()[1].cpu().tolist() for q in env.queries],
                                  'query_subsets':{str(h):{k:sorted(v) for k,v in sub.items()} for h,sub in subsets.items()},'runs':private})
                        del runs
                    rec={'stage':stage,'scenario':index,'metadata':meta,'candidates':[{'family':c.family,'anchor':c.anchor,'removals':c.removals} for c in candidates],
                         'matched_reconstructions':matching,'tapes':tapes}
                    records.append(rec);write(output/f'ROOT_{stage}_{index}.json',rec)
                    del env,m,o,state,subsets,desc,candidates,built;gc.collect();torch.cuda.empty_cache()
            pool=join(previous,data);teacher=frozen(donor) if donor.C else None
            observed.extend([int((y[ids]==cl).sum()) for cl in classes]);birth.extend([stage]*len(classes));donor.arrive(data,len(observed))
            learn(donor,donor_opt,pool,200,74002+stage*100+1,counter,teacher,64,'science_donor')
            memory=choose_zero(pool,donor,observed,birth,stage,74002+70000+stage)
            learn(donor,donor_opt,memory,100,74002+stage*100+2,counter,teacher,64,'science_donor')
            memory.serialize(output/f'DONOR_{stage}.private.bin',observed,birth)
            write(output/f'DONOR_{stage}.json',{'stage':stage,'tokens':len(memory),'global_tokens':len(memory.parents),'bytes':len(memory)*2064+184,'classifier_updates':300})
            del previous,pool,data,teacher;gc.collect();torch.cuda.empty_cache()
        science=sum(counter.get(k,0) for k in ['science_donor_classifier_optimizer','science_prefix_classifier_optimizer','science_paired_classifier_updates','science_virtual_classifier_optimizer'])
        assert science<=382316,'SCIENCE_UPDATE_BOUND'
        write(output/'RESULTS.json',{'stream':cfg['setting'],'roots':records,'counts':dict(counter),'science_classifier_updates':science,'actor_verifier_optimizer':0,'VAL_TEST':0})
        write(output/'STATUS.json',{'phase':'COMPLETE_P1','science_classifier_updates':science,'roots':len(records),'counts':dict(counter)})
    except BaseException:
        (output/'FAILURE.private.txt').write_text(traceback.format_exc());write(output/'FAILURE_COUNTS.json',dict(counter));raise
    finally:
        write(output/'COSTS.json',{'counts':dict(counter),'worker_core_CPU_seconds':time.process_time()-cpu,'worker_core_wall_seconds':time.time()-start,
                                 'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,'GPU_kernel_seconds':'not measured','preparation_import_CPU':'nonzero not fully sampled'})

if __name__=='__main__':run(json.loads(Path(os.environ['Q_CONFIG']).read_text()))

