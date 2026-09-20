"""Audited S0 forks; only the three real-CE candidate sets change."""
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_medical_v2 import MedicalLearner,args_for,seed_all,write_json,sha,network_hash,rng_equal
from diagnose_medical_v3 import real_terms,gradient_probe,evaluate,head_stats

class PilotComplete(Exception):pass

class ForkLearner(MedicalLearner):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.epoch_component=None;self.pause_after_epoch=False;self.pair_hash=hashlib.sha256()
    def _v3_batch_hook(self,epoch,batch,o,targets,sum_ce,few_ce,match,replay,replay_weight):
        if batch==0:
            self.epoch_component={'batches':0,'optimizer_steps':0,'real_exposures':[0]*8,'synthetic_exposures':[0]*8,
                                  'correct_all':[0]*8,'correct_restricted':[0]*8,'losses':{},'norm_main_sum':0.,'norm_few_sum':0.}
        r=self.epoch_component;k=self._known_classes;t=self._total_classes
        r['batches']+=1;r['optimizer_steps']+=1
        y=targets.detach();raw=(o['logits']+o['logits_few'])[:,:t].detach();pred=raw.argmax(1);restricted=raw[:,k:].argmax(1)+k
        for c in range(k,t):
            mask=y==c;r['real_exposures'][c]+=int(mask.sum());r['correct_all'][c]+=int(((pred==y)&mask).sum());r['correct_restricted'][c]+=int(((restricted==y)&mask).sum())
        if replay is not None:
            # Original fixed 4/class, at most 6 old classes: cap48 never truncates.
            assert 4*k<=48
            for c in range(k):r['synthetic_exposures'][c]+=4
        terms=real_terms(o,targets,k,t,self.args['real_ce_scope'])
        pull=-self.args['pull_constraint_coeff']*(o['reduce_sim']+o['reduce_sim_few']) if self.args['pull_constraint'] else 0.
        vals={**terms,'pool_match':match,'pull':pull,'replay_raw':replay if replay is not None else 0.,'replay_weighted':replay*replay_weight if replay is not None else 0.}
        for name,v in vals.items():r['losses'][name]=r['losses'].get(name,0.)+float(v.detach() if torch.is_tensor(v) else v)
        r['norm_main_sum']+=float(o['pre_logits'].detach().norm(dim=1).sum());r['norm_few_sum']+=float(o['pre_logits_few'].detach().norm(dim=1).sum())
    def _v2_input_hook(self,epoch,batch,inputs,targets):
        # Auditable stream fingerprint; no RNG draw or extra training data access.
        if batch==0:self.pair_hash=hashlib.sha256()
        self.pair_hash.update(inputs.detach().cpu().numpy().tobytes());self.pair_hash.update(targets.cpu().numpy().tobytes())
    def _v2_epoch_hook(self,epoch,optimizer,scheduler,stats):
        torch.cuda.synchronize();training_seconds=time.monotonic()-self.epoch_start
        r=self.epoch_component;k=self._known_classes;t=self._total_classes;n=sum(r['real_exposures'])
        assert n==len(self.train_dataset) and r['optimizer_steps']==len(self.train_loader)
        for name,col in [('train_current_all_seen','correct_all'),('train_current_restricted','correct_restricted')]:
            r[name+'_acc']=100*sum(r[col])/n
            r[name+'_macro_recall']=100*float(np.mean([r[col][c]/r['real_exposures'][c] for c in range(k,t)]))
        r['losses']={name:value/r['batches'] for name,value in r['losses'].items()}
        r['feature_main_norm_mean']=r.pop('norm_main_sum')/n;r['feature_few_norm_mean']=r.pop('norm_few_sum')/n
        r.update(train_images=n,real_stream_sha256=self.pair_hash.hexdigest(),training_seconds=training_seconds,
                 fixed_train_probe=gradient_probe(self,epoch=epoch),head=head_stats(self))
        # Probe is deterministic and restores model modes and every RNG stream before checkpointing.
        stats=dict(stats,components=r)
        super()._v2_epoch_hook(epoch,optimizer,scheduler,stats)
        if self.pause_after_epoch:
            self.pause_after_epoch=False;raise PilotComplete()

def fork_from_s0(config,seed,branch):
    assert branch in ('D','E');parent_branch={'D':'B','E':'C'}[branch]
    v2=json.loads(Path(config['v2_runtime']).read_text());out=Path(config['output'])/f'{seed}_{branch}'
    lock=json.loads((Path(config['v2_output'])/'ALL_CHECKPOINTS_LOCK.json').read_text())
    entry=next(r for r in lock['checkpoints'] if (r['seed'],r['branch'],r['session'])==(seed,parent_branch,0))
    assert sha(entry['path'])==entry['sha256'],'BLOCKED_PARENT_HASH'
    a,order=args_for(v2,seed,parent_branch,False);seed_all(seed)
    learner=ForkLearner(a,v2,order,out)
    state=learner.restore(entry['path'])  # Unmodified ordinary restore validates the complete V2 contract.
    assert (state['task'],state['known'],state['total'],state['epoch'],state['phase'])==(0,0,4,10,'session_complete')
    before=network_hash(learner._network);rng=learner._capture_rng_state();loader_rng=learner.loader_generator.get_state();synth=learner.synth_rng.clone()
    expected=json.loads((Path(config['output'])/'p0/s0_network_hashes.json').read_text())
    assert before==expected[f'{seed}_B']==expected[f'{seed}_C'],'BLOCKED_PARENT_NETWORK_PAIR'
    if branch=='E':
        assert set(learner.concm_stage1_memory)==set(range(4))
        for c,m in learner.concm_stage1_memory.items():assert m['task']==0 and m['count']==a['lt_list'][c]
    else:assert not learner.concm_stage1_memory
    learner.config=copy.deepcopy(config);learner.protocol_hash=config['protocol_sha256'];learner.args=dict(a,real_ce_scope='all_seen')
    learner._cur_task=1;learner._known_classes=4;learner._total_classes=6
    # session(1) builds the original incremental optimizer, not the S0 optimizer/scheduler.
    assert network_hash(learner._network)==before and rng_equal(rng,learner._capture_rng_state())
    assert torch.equal(loader_rng,learner.loader_generator.get_state()) and torch.equal(synth,learner.synth_rng)
    record={'seed':seed,'branch':branch,'parent':entry,'parent_network_sha256':before,
            'parent_code_commit':v2['code_commit'],'parent_code_sha256':state['code_sha256'],
            'new_code_commit':config['code_commit'],'new_code_sha256':config['code_sha256'],
            'only_training_intervention':'real_ce_scope current -> all_seen for main CE, sum CE and pool-weighted few CE',
            'ordinary_restore':'PASS','network_rng_loader_synthesis_inheritance':'PASS','memory_classes':sorted(learner.concm_stage1_memory),
            'next_boundary':[1,4,6],'optimizer':'rebuilt original incremental AdamW fixed LR',
            'training_args':dict(learner.args,device=[str(d) for d in learner.args['device']])}
    marker=out/'FORK_FROM_S0.json'
    if marker.exists():assert json.loads(marker.read_text())==record,'BLOCKED_FORK_DRIFT'
    else:write_json(marker,record)
    write_json(out/'actual_config.json',record['training_args'])
    return learner

def trajectory(config,seed,branch,pilot=False):
    out=Path(config['output'])/f'{seed}_{branch}';done=out/'TRAJECTORY_COMPLETE.json'
    if done.exists():return json.loads(done.read_text())
    learner=fork_from_s0(config,seed,branch);sessions=[];resume=out/'resume.pt'
    for task in (1,2):
        final=out/f'session{task}.pt'
        if final.exists():
            state=learner.restore(final);assert state['phase']=='session_complete' and state['task']==task and state['epoch']==10
        else:
            restore=None
            if resume.exists():
                state=torch.load(resume,map_location='cpu',weights_only=False)
                if state['task']==task:restore=resume
                del state
            learner.pause_after_epoch=pilot
            try:r=learner.session(task,resume=restore)
            except PilotComplete:
                row=learner.records[-1]
                write_json(out/'PILOT.json',{'epoch':row,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'test_predictions':0})
                return {'status':'PILOT_COMPLETE'}
            write_json(out/f'session{task}_training.json',r)
        result_path=out/f'session{task}_val.json'
        if not result_path.exists():
            val=evaluate(learner,'val');write_json(result_path,val)
        sessions.append({'session':task,'path':str(final),'sha256':sha(final),'epoch':10})
    result={'seed':seed,'branch':branch,'status':'COMPLETE','sessions':sessions,'new_epochs':20,'test_predictions':0}
    write_json(done,result);return result
