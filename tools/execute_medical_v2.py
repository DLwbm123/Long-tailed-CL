"""Fixed six-trajectory worker; the neutral supervisor starts a separate test process."""
import gc
import json
import math
import subprocess
import time
from pathlib import Path
import torch
from run_medical_v2 import MedicalLearner,args_for,seed_all,sha,write_json,network_hash

def train(config):
    output=Path(config['output']);protocol=Path(config['protocol'])
    lock=json.loads((output/'PROTOCOL_LOCK_V2.json').read_text())
    assert lock['status']=='LOCKED_TECHNICAL_PASS' and lock['formal_epochs_per_session']==10
    assert sha(output/'PROTOCOL_LOCK_V2.json')==config['protocol_sha256']
    assert sha(protocol/'V2_DATASET_SUMMARY.json')==lock['dataset_summary_sha256'],'BLOCKED_DATA_SUMMARY_DRIFT'
    assert config['class_orders']==lock['class_orders'],'BLOCKED_ORDER_DRIFT'
    root=Path(__file__).resolve().parents[1]
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==config['code_commit'],'BLOCKED_RUNTIME_COMMIT'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(),'BLOCKED_RUNTIME_DIRTY'
    for path,expected in config['code_sha256'].items():
        assert sha(Path(__file__).resolve().parents[1]/path)==expected,('BLOCKED_CODE_DRIFT',path)
    started=config['formal_started_at'];deadline=started+86400
    entries=[];usage=[];initial={};s0={}
    summary=json.loads((protocol/'V2_DATASET_SUMMARY.json').read_text())
    counts={r['original_label']:r['n_train_images'] for r in summary['class_counts']}
    for seed in (1993,1994,1995):
        order_file=json.loads((output/f'order_{seed}.json').read_text())
        for branch in ('B','C'):
            assert time.time()<deadline,'BLOCKED_FORMAL_BUDGET'
            seed_all(seed);args,order=args_for(config,seed,branch,False)
            assert order==order_file['class_order']
            run=output/f'{seed}_{branch}';learner=MedicalLearner(args,config,order,run)
            initial[f'{seed}_{branch}']=network_hash(learner._network)
            if branch=='C':assert initial[f'{seed}_B']==initial[f'{seed}_C'],'BLOCKED_INITIALIZATION'
            public_args=dict(args,device=['cuda:0']);write_json(run/'actual_config.json',public_args)
            original_hook=learner._v2_gradient_hook
            def guarded_hook(epoch,batch,loss):
                assert time.time()<deadline,'BLOCKED_FORMAL_BUDGET'
                return original_hook(epoch,batch,loss)
            learner._v2_gradient_hook=guarded_hook
            for task in range(3):
                checkpoint=run/f'session{task}.pt'
                if checkpoint.exists():
                    state=learner.restore(checkpoint)
                    assert state['phase']=='session_complete' and state['epoch']==10 and state['task']==task
                    stat=dict(session=task,restored_completed=True)
                else:
                    resume=run/'resume.pt';use_resume=None
                    if resume.exists():
                        state=torch.load(resume,map_location='cpu',weights_only=False)
                        if state['task']==task:use_resume=resume
                        del state
                    stat=learner.session(task,False,use_resume)
                current=order[[0,4,6][task]:[4,6,8][task]]
                stat.update(seed=seed,branch=branch,expected_images=sum(counts[c] for c in current),expected_optimizer_steps=10*math.ceil(sum(counts[c] for c in current)/48))
                if 'train_images' in stat:assert stat['train_images']==stat['expected_images']
                stat['statistics_memory_bytes']=sum(v.numel()*v.element_size() for s in learner.concm_stage1_memory.values() for v in s.values() if torch.is_tensor(v))
                stat['parameter_bytes']={name:sum(p.numel()*p.element_size() for n,p in learner._network.named_parameters() if condition(n,p)) for name,condition in {
                    'all':lambda n,p:True,'frozen':lambda n,p:not p.requires_grad,
                    'trainable_pool':lambda n,p:p.requires_grad and '.pool' in n,
                    'heads':lambda n,p:n.startswith(('backbone.head.','backbone.head_few.')),
                    'assigner':lambda n,p:n.startswith('backbone.assigner.')}.items()}
                if task==0:
                    s0[f'{seed}_{branch}']=network_hash(learner._network)
                    if branch=='C':assert s0[f'{seed}_B']==s0[f'{seed}_C'],'BLOCKED_S0_EQUIVALENCE'
                entry=dict(seed=seed,branch=branch,session=task,path=str(checkpoint),sha256=sha(checkpoint),epoch=10)
                entries.append(entry);usage.append(stat)
                write_json(output/'resource_usage.json',dict(status='TRAINING',elapsed_seconds=time.time()-started,budget_seconds=86400,sessions=usage,completed_trajectories=len(entries)//3,completed_checkpoints=len(entries),model_test_predictions=0))
            del learner;gc.collect();torch.cuda.empty_cache()
    assert len(entries)==18 and time.time()<deadline
    write_json(output/'ALL_CHECKPOINTS_LOCK.json',dict(checkpoints=entries,initial_hashes=initial,s0_hashes=s0,protocol_sha256=config['protocol_sha256'],code_commit=config['code_commit'],test_predictions_before_lock=0))
    write_json(output/'resource_usage.json',dict(status='TRAINING_COMPLETE',elapsed_seconds=time.time()-started,budget_seconds=86400,sessions=usage,completed_trajectories=6,completed_checkpoints=18,model_test_predictions=0))
    return 'TRAINING_COMPLETE'
