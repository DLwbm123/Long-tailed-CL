"""Execution-only two-worker amendment; the frozen training sources stay unchanged."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PAIRS=[f'{seed}_{branch}' for seed in (1993,1994,1995) for branch in ('B','C')]

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def run_queue(config,entry):
    """Only this parent owns scheduling; each child exclusively owns one trajectory."""
    output=Path(config['output']);pending=list(PAIRS);active={};events=[]
    recoveries=json.loads((output/'recovery.json').read_text())['attempt'] if (output/'recovery.json').exists() else 0
    deadline=config['formal_started_at']+86400
    try:
        while pending or active:
            while pending and len(active)<2:
                assert time.time()<deadline,'BLOCKED_FORMAL_BUDGET'
                pair=pending.pop(0);log=(output/f'parallel_{pair}.log').open('a')
                process=subprocess.Popen([sys.executable,'-u',entry],env=dict(os.environ,Q8_PAIR=pair),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[process.pid]=(process,pair,log)
                events.append(dict(event='start',pair=pair,pid=process.pid,time=time.time()))
                save(output/'parallel_scheduler.json',dict(max_parallel=2,events=events,active={str(pid):row[1] for pid,row in active.items()}))
            pid,status=os.wait();process,pair,log=active.pop(pid)
            process.returncode=os.waitstatus_to_exitcode(status);log.close()
            events.append(dict(event='finish',pair=pair,pid=pid,time=time.time(),exit_code=process.returncode))
            save(output/'parallel_scheduler.json',dict(max_parallel=2,events=events,active={str(pid):row[1] for pid,row in active.items()}))
            if process.returncode==75 and recoveries<1 and time.time()<deadline:
                recoveries+=1;pending.insert(0,pair)
                save(output/'parallel_recovery.json',dict(attempt=recoveries,pair=pair,time=time.time(),same_config=True))
            elif process.returncode:
                raise RuntimeError(f'BLOCKED_WORKER {pair} exit={process.returncode}')
        return events
    finally:
        for process,pair,log in active.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM);process.wait()
            log.close()

def check_sources(config):
    from run_medical_v2 import sha
    root=Path(__file__).resolve().parents[1]
    for path,digest in config['code_sha256'].items():assert sha(root/path)==digest,('BLOCKED_FROZEN_SOURCE_DRIFT',path)
    amendment=json.loads((Path(config['output'])/'PARALLEL_EXECUTION_AMENDMENT.json').read_text())
    assert sha(__file__)==amendment['scheduler_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==amendment['execution_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    assert sha(Path(config['output'])/'PROTOCOL_LOCK_V2.json')==config['protocol_sha256']
    return amendment

def trajectory(config,pair):
    import math
    import torch
    from run_medical_v2 import MedicalLearner,args_for,seed_all,network_hash,sha
    assert pair in PAIRS;check_sources(config)
    seed,branch=pair.split('_');seed=int(seed);output=Path(config['output']);run=output/pair
    seed_all(seed);args,order=args_for(config,seed,branch,False)
    assert args['tuned_epoch']==10 and args['batch_size']==48
    learner=MedicalLearner(args,config,order,run)
    receipt=dict(pair=pair,initial_hash=network_hash(learner._network),sessions=[],checkpoints=[])
    save(run/'actual_config.json',dict(args,device=['cuda:0']))
    original_hook=learner._v2_gradient_hook
    def guard(epoch,batch,loss):
        assert time.time()<config['formal_started_at']+86400,'BLOCKED_FORMAL_BUDGET'
        return original_hook(epoch,batch,loss)
    learner._v2_gradient_hook=guard
    summary=json.loads((Path(config['protocol'])/'V2_DATASET_SUMMARY.json').read_text())
    counts={x['original_label']:x['n_train_images'] for x in summary['class_counts']}
    for task in range(3):
        checkpoint=run/f'session{task}.pt'
        if checkpoint.exists():
            state=learner.restore(checkpoint)
            assert state['phase']=='session_complete' and state['epoch']==10 and state['task']==task
            stat=dict(session=task,restored_completed=True)
            del state
        else:
            resume=run/'resume.pt';use_resume=None
            if resume.exists():
                state=torch.load(resume,map_location='cpu',weights_only=False)
                if state['task']==task:use_resume=resume
                del state
            stat=learner.session(task,False,use_resume)
        epoch_rows=[json.loads(line) for line in (run/'epochs.jsonl').read_text().splitlines()]
        assert [x['epoch'] for x in epoch_rows if x['session']==task]==list(range(1,11)),'BLOCKED_EPOCH_COVERAGE'
        current=order[[0,4,6][task]:[4,6,8][task]];n=sum(counts[c] for c in current)
        stat.update(seed=seed,branch=branch,expected_images=n,expected_optimizer_steps=10*math.ceil(n/48))
        if 'train_images' in stat:assert stat['train_images']==n
        stat['statistics_memory_bytes']=sum(v.numel()*v.element_size() for s in learner.concm_stage1_memory.values() for v in s.values() if torch.is_tensor(v))
        stat['parameter_bytes']={name:sum(p.numel()*p.element_size() for n,p in learner._network.named_parameters() if condition(n,p)) for name,condition in {
            'all':lambda n,p:True,'frozen':lambda n,p:not p.requires_grad,'trainable_pool':lambda n,p:p.requires_grad and '.pool' in n,
            'heads':lambda n,p:n.startswith(('backbone.head.','backbone.head_few.')),'assigner':lambda n,p:n.startswith('backbone.assigner.')}.items()}
        if task==0:receipt['s0_hash']=network_hash(learner._network)
        receipt['sessions'].append(stat)
        receipt['checkpoints'].append(dict(seed=seed,branch=branch,session=task,path=str(checkpoint),sha256=sha(checkpoint),epoch=10))
        save(run/'trajectory_progress.json',receipt)
    save(run/'TRAJECTORY_COMPLETE.json',receipt)

def collect(config):
    output=Path(config['output']);entries=[];usage=[];initial={};s0={}
    for pair in PAIRS:
        receipt=json.loads((output/pair/'TRAJECTORY_COMPLETE.json').read_text());assert receipt['pair']==pair
        assert len(receipt['checkpoints'])==3 and [x['session'] for x in receipt['checkpoints']]==[0,1,2]
        for x in receipt['checkpoints']:
            assert f"{x['seed']}_{x['branch']}"==pair and x['epoch']==10
            assert Path(x['path'])==output/pair/f"session{x['session']}.pt" and Path(x['path']).is_file()
        initial[pair]=receipt['initial_hash'];s0[pair]=receipt['s0_hash']
        entries.extend(receipt['checkpoints']);usage.extend(receipt['sessions'])
    for seed in (1993,1994,1995):
        assert initial[f'{seed}_B']==initial[f'{seed}_C'],'BLOCKED_PAIRED_INITIALIZATION'
        assert s0[f'{seed}_B']==s0[f'{seed}_C'],'BLOCKED_S0_EQUIVALENCE'
    assert len({(x['seed'],x['branch'],x['session']) for x in entries})==18
    save(output/'ALL_CHECKPOINTS_LOCK.json',dict(checkpoints=entries,initial_hashes=initial,s0_hashes=s0,protocol_sha256=config['protocol_sha256'],code_commit=config['code_commit'],execution_amendment='PARALLEL_EXECUTION_AMENDMENT.json',test_predictions_before_lock=0))
    save(output/'resource_usage.json',dict(status='TRAINING_COMPLETE',elapsed_seconds=time.time()-config['formal_started_at'],budget_seconds=86400,sessions=usage,completed_trajectories=6,completed_checkpoints=18,model_test_predictions=0,max_parallel=2))

def main(config,entry):
    output=Path(config['output'])
    try:
        if 'Q8_PAIR' in os.environ:
            import torch
            torch.set_num_threads(4);trajectory(config,os.environ['Q8_PAIR'])
        else:
            check_sources(config)
            assert not (output/'TEST_BATCH_STARTED.json').exists(),'BLOCKED_TEST_ALREADY_STARTED'
            save(output/'STATUS.json',dict(status='PARALLEL_RUNNING',max_parallel=2,test_predictions=0))
            run_queue(config,entry);collect(config)
            assert time.time()<config['formal_started_at']+86400,'BLOCKED_FORMAL_BUDGET'
            # The unchanged evaluator validates all 18 hashes and creates its one-time test marker.
            with (output/'test_batch.log').open('a') as log:
                subprocess.run([sys.executable,'-u',entry],env=dict(os.environ,Q8_EVALUATE='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
            save(output/'STATUS.json',dict(status='COMPLETE',max_parallel=2))
        return 0
    except Exception as error:
        import errno,traceback
        infrastructure=isinstance(error,OSError) and error.errno in (errno.EIO,errno.ESTALE,errno.ETIMEDOUT)
        name='FAILURE_'+os.environ['Q8_PAIR']+'.json' if 'Q8_PAIR' in os.environ else 'STOPPED.json'
        save(output/name,dict(status='INFRASTRUCTURE_INTERRUPTION' if infrastructure else 'BLOCKED',error=str(error),traceback=traceback.format_exc(),time=time.time()))
        traceback.print_exc();return 75 if infrastructure else 1
