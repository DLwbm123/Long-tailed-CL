"""One targeted intervention: keep the C-S0 feature extractor fixed after S0."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from run_medical_v2 import MedicalLearner,args_for,seed_all,write_json,sha,network_hash,rng_equal
from train_medical_v3 import ForkLearner,PilotComplete
from diagnose_medical_v3 import evaluate,SEEDS
from evaluate_medical_v2 import write_csv
from report_medical_v3 import flatten

HEADS={'backbone.head.weight','backbone.head.bias','backbone.head_few.weight','backbone.head_few.bias'}

class StationaryLearner(ForkLearner):
    def assert_old_rows(self,optimizer=None):
        if self.args.get('old_classifier_rows')!='session_fixed':return
        ref=self.old_row_reference;k=ref['known']
        assert (ref['task'],k)==(self._cur_task,self._known_classes),'BLOCKED_OLD_ROW_STAGE'
        for name,p in self._network.named_parameters():
            if name in HEADS:
                assert torch.equal(p[:k].detach().cpu(),ref['rows'][name].cpu()),'BLOCKED_OLD_ROW_DRIFT '+name
                if optimizer is not None:
                    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                        v=optimizer.state.get(p,{}).get(key)
                        if v is not None:assert torch.count_nonzero(v[:k])==0,'BLOCKED_OLD_ROW_MOMENT'

    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        if self.args.get('old_classifier_rows')!='session_fixed':
            return super()._init_train(train_loader,test_loader,optimizer,scheduler)
        k=self._known_classes
        if not hasattr(self,'old_row_reference') or self.old_row_reference['task']!=self._cur_task:
            self.old_row_reference={'task':self._cur_task,'known':k,'steps':0,
                'rows':{n:p[:k].detach().cpu().clone() for n,p in self._network.named_parameters() if n in HEADS}}
        self.assert_old_rows(optimizer)
        heads=[(p,self.old_row_reference['rows'][n].to(p.device)) for n,p in self._network.named_parameters() if n in HEADS]
        def before_step(opt,args,kwargs):
            for p,_ in heads:
                if p.grad is not None:p.grad[:k].zero_()
        @torch.no_grad()
        def after_step(opt,args,kwargs):
            # Gradient masking alone does not prevent AdamW decay or residual moments.
            for p,value in heads:
                p[:k].copy_(value)
                for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                    v=opt.state.get(p,{}).get(key)
                    if v is not None:v[:k].zero_()
            self.old_row_reference['steps']+=1
        pre=optimizer.register_step_pre_hook(before_step);post=optimizer.register_step_post_hook(after_step)
        try:return super()._init_train(train_loader,test_loader,optimizer,scheduler)
        finally:pre.remove();post.remove()

    def _v2_epoch_hook(self,epoch,optimizer,scheduler,stats):
        if self.args.get('old_classifier_rows')=='session_fixed':
            self.assert_old_rows(optimizer)
            stats=dict(stats,old_classifier_rows_exact='PASS',old_classifier_moments_zero='PASS',
                       constrained_optimizer_steps=self.old_row_reference['steps'])
        return super()._v2_epoch_hook(epoch,optimizer,scheduler,stats)

    def _concm_stage1_effective_weight(self,epoch):
        weight=super()._concm_stage1_effective_weight(epoch)
        if self.args.get('replay_weight_rule','constant')=='old_current_count':
            assert 0<=self._known_classes<self._total_classes<=8
            weight*=self._known_classes/(self._total_classes-self._known_classes)
        return weight

    def freeze_features(self,parent,state):
        self.delta_parent=parent
        self.base_network={k:v.detach().cpu().clone() for k,v in state['network'].items()}
        for name,p in self._network.named_parameters():p.requires_grad_(name in HEADS)
        assert {n for n,p in self._network.named_parameters() if p.requires_grad}==HEADS

    def assert_stationary(self):
        for name,value in self._network.state_dict().items():
            if name not in HEADS:
                assert torch.equal(value.detach().cpu(),self.base_network[name]),'BLOCKED_FEATURE_DRIFT '+name

    def checkpoint(self,path,optimizer,scheduler,epoch,phase):
        if phase=='session_complete':self.assert_stationary()
        state=self._checkpoint_state(optimizer,scheduler,epoch,phase)
        state['network']={k:v for k,v in state['network'].items() if k in HEADS}
        state['checkpoint_format']='S0_HEAD_DELTA_V1';state['immutable_parent']=self.delta_parent
        if self.args.get('old_classifier_rows')=='session_fixed':
            self.assert_old_rows(optimizer);state['old_row_reference']=self.old_row_reference
        tmp=Path(str(path)+'.part');torch.save(state,tmp);tmp.replace(path)

    def expand_delta(self,state):
        assert state['checkpoint_format']=='S0_HEAD_DELTA_V1'
        assert state['immutable_parent']==self.delta_parent,'BLOCKED_DELTA_PARENT'
        assert set(state['network'])==HEADS,'BLOCKED_DELTA_KEYS'
        if self.args.get('old_classifier_rows')=='session_fixed':
            ref=state['old_row_reference'];k=state['known']
            assert (ref['task'],ref['known'])==(state['task'],k) and set(ref['rows'])==HEADS,'BLOCKED_OLD_ROW_REFERENCE'
            assert all(torch.equal(state['network'][n][:k],ref['rows'][n]) for n in HEADS),'BLOCKED_OLD_ROW_REFERENCE'
        state=dict(state,network={**self.base_network,**state['network']})
        return state

    def restore(self,path):
        if not hasattr(self,'delta_parent'):return super().restore(path)
        state=self.expand_delta(torch.load(path,map_location='cpu',weights_only=False))
        # Use precisely the normal protocol/config/seed/branch/state guards.
        result=self._restore_checkpoint_state(state)
        if self.args.get('old_classifier_rows')=='session_fixed':
            self.old_row_reference=state['old_row_reference'];self.assert_old_rows(self.optimizer)
        return result

def fork(config,seed,out=None):
    branch=config.get('candidate_branch','F');weight=config.get('replay_weight',.05);rule=config.get('replay_weight_rule','constant')
    assert (branch,weight,rule) in [('F',.05,'constant'),('G',1.0,'constant'),('H',1.0,'old_current_count'),('I',1.0,'old_current_count')],'Unsupported fixed contrast'
    out=Path(out or Path(config['output'])/f'{seed}_{branch}')
    v2=json.loads(Path(config['v2_runtime']).read_text())
    entries=json.loads((Path(config['v2_output'])/'ALL_CHECKPOINTS_LOCK.json').read_text())['checkpoints']
    parent=next(e for e in entries if (e['seed'],e['branch'],e['session'])==(seed,'C',0))
    assert sha(parent['path'])==parent['sha256'],'BLOCKED_PARENT_SHA'
    a,order=args_for(v2,seed,'C',False);seed_all(seed)
    learner=StationaryLearner(a,v2,order,out);state=learner.restore(parent['path'])
    assert (state['task'],state['known'],state['total'],state['epoch'],state['phase'])==(0,0,4,10,'session_complete')
    assert set(learner.concm_stage1_memory)==set(range(4))
    for c,m in learner.concm_stage1_memory.items():assert m['task']==0 and m['count']==a['lt_list'][c]
    before=network_hash(learner._network);rng=learner._capture_rng_state()
    expected=json.loads((Path(config['v3_output'])/'p0/s0_network_hashes.json').read_text())
    assert before==expected[f'{seed}_C']==expected[f'{seed}_B']
    learner.freeze_features(parent,state)
    learner.config=copy.deepcopy(config);learner.protocol_hash=config['protocol_sha256']
    learner.args=dict(a,real_ce_scope='all_seen',feature_update_scope='s0_frozen',concm_stage1_loss_weight=weight,replay_weight_rule=rule)
    if branch=='I':
        assert config['old_classifier_rows']=='session_fixed'
        learner.args['old_classifier_rows']='session_fixed'
    learner.concm_stage1_loss_weight=weight
    learner._cur_task=1;learner._known_classes=4;learner._total_classes=6
    assert network_hash(learner._network)==before and rng_equal(rng,learner._capture_rng_state())
    record={'seed':seed,'branch':branch,'parent':parent,'parent_network_sha256':before,
            'ordinary_parent_restore':'PASS','network_and_rng_inheritance':'PASS',
            'intervention':{'F':'freeze all non-head parameters at the corresponding C S0 state',
                            'G':'relative to F, replay coefficient 0.05 -> 1.0; all other settings unchanged',
                            'H':'relative to G, multiply replay coefficient by known/current class count: S1=2, S2=3',
                            'I':'relative to H, preserve existing classifier rows and zero their Adam moments at each session; losses unchanged'}[branch],
            'trainable_names':sorted(HEADS),'trainable_parameters':sum(p.numel() for p in learner._network.parameters() if p.requires_grad),
            'memory_classes':sorted(learner.concm_stage1_memory),'code_commit':config['code_commit'],
            'checkpoint_format':'S0_HEAD_DELTA_V1: exact parent plus changed heads, optimizer, memory and complete RNG',
            'training_args':dict(learner.args,device=[str(d) for d in learner.args['device']])}
    marker=out/'FORK_FROM_S0.json'
    if marker.exists():assert json.loads(marker.read_text())==record,'BLOCKED_FORK_DRIFT'
    else:write_json(marker,record)
    write_json(out/'actual_config.json',record['training_args'])
    return learner

def trajectory(config,seed,pilot=False):
    branch=config.get('candidate_branch','F')
    out=Path(config['output'])/f'{seed}_{branch}';done=out/'TRAJECTORY_COMPLETE.json'
    if done.exists():return json.loads(done.read_text())
    l=fork(config,seed);sessions=[]
    for task in (1,2):
        final=out/f'session{task}.pt';resume=out/'resume.pt'
        if final.exists():
            state=l.restore(final);assert state['phase']=='session_complete' and state['task']==task and state['epoch']==10
        else:
            restore=None
            if resume.exists():
                state=torch.load(resume,map_location='cpu',weights_only=False)
                if state['task']==task:restore=resume
                del state
            l.pause_after_epoch=pilot
            try:r=l.session(task,resume=restore)
            except PilotComplete:
                l.assert_stationary()
                result={'status':'PILOT_COMPLETE','epoch':l.records[-1],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'checkpoint_bytes':resume.stat().st_size,'test_predictions':0}
                write_json(out/'PILOT.json',result);return result
            write_json(out/f'session{task}_training.json',r)
        l.assert_stationary()
        p=out/f'session{task}_val.json'
        if not p.exists():write_json(p,evaluate(l,'val'))
        sessions.append({'session':task,'path':str(final),'sha256':sha(final),'epoch':10,'bytes':final.stat().st_size,'frozen_features_exact':'PASS'})
    result={'seed':seed,'branch':branch,'status':'COMPLETE','sessions':sessions,'new_epochs':20,'test_predictions':0,'immutable_parent':l.delta_parent}
    write_json(done,result);return result

def report(config):
    import csv
    root=Path(config['output']);out=root/'results';out.mkdir(exist_ok=True)
    branch=config.get('candidate_branch','F');metrics=[];classes=[];epochs=[];locks=[]
    for seed in SEEDS:
        d=root/f'{seed}_{branch}';e=[json.loads(x) for x in (d/'epochs.jsonl').read_text().splitlines()]
        if branch=='I':assert all(r['old_classifier_rows_exact']=='PASS' and r['old_classifier_moments_zero']=='PASS' for r in e),'BLOCKED_OLD_ROW_AUDIT'
        assert len(e)==20 and {(r['session'],r['epoch']) for r in e}=={(t,k) for t in (1,2) for k in range(1,11)}
        ref=[json.loads(x) for x in (Path(config['v3_output'])/f'{seed}_E/epochs.jsonl').read_text().splitlines()]
        assert all(a['components']['real_stream_sha256']==b['components']['real_stream_sha256'] for a,b in zip(e,ref)),'BLOCKED_REAL_STREAM_PAIR'
        for t in (0,1,2):
            value=json.loads((Path(config['v3_output'])/'p0'/f'{seed}_C_s0.json').read_text())['diagnostics']['val'] if t==0 else json.loads((d/f'session{t}_val.json').read_text())
            prefix=dict(seed=seed,branch=branch,session=t,inherited_s0=t==0)
            metrics += [dict(**prefix,**m) for m in value['metrics']]
            classes += [dict(**prefix,**p) for p in value['per_class']]
        epochs += [dict(seed=seed,branch=branch,session=r['session'],epoch=r['epoch'],**flatten(r['components'])) for r in e]
        locks += json.loads((d/'TRAJECTORY_COMPLETE.json').read_text())['sessions']
    baseline=list(csv.DictReader((Path(config['v3_output'])/'results/val_session_metrics.csv').open()))
    oldpc=list(csv.DictReader((Path(config['v3_output'])/'results/val_per_class_metrics.csv').open()))
    comparisons=['C','E']
    if branch in ('G','H','I'):
        reference=Path(config['v4_output'])/'results'
        baseline += list(csv.DictReader((reference/'val_session_metrics.csv').open()))
        oldpc += list(csv.DictReader((reference/'val_per_class_metrics.csv').open()))
        comparisons.append('F')
    if branch in ('H','I'):
        reference=Path(config['v5_output'])/'results'
        baseline += list(csv.DictReader((reference/'val_session_metrics.csv').open()))
        oldpc += list(csv.DictReader((reference/'val_per_class_metrics.csv').open()))
        comparisons.append('G')
    if branch=='I':
        reference=Path(config['v6_output'])/'results'
        baseline += list(csv.DictReader((reference/'val_session_metrics.csv').open()))
        oldpc += list(csv.DictReader((reference/'val_per_class_metrics.csv').open()))
        comparisons.append('H')
    pairs=[];decisions={};means=[]
    fields=['balanced_accuracy','old_macro_recall','current_macro_recall','old_current_hm_macro_recall','tail_rank2','current_to_old_rate','old_to_current_rate','restricted_current_ba']
    for b in comparisons:
        newzero=[];improved=0
        for seed in SEEDS:
            for task in (1,2):
                x=next(r for r in metrics if (r['seed'],r['session'],r['head'])==(seed,task,'sum'))
                y=next(r for r in baseline if (r['seed'],r['session'],r['head'],r['branch'])==(str(seed),str(task),'sum',b))
                if task==2:improved+=int(x['balanced_accuracy']>float(y['balanced_accuracy']) and x['current_macro_recall']>float(y['current_macro_recall']))
                for field in fields:pairs.append(dict(contrast=branch+'-'+b,seed=seed,session=task,metric=field,difference_pp=x[field]-float(y[field])))
                for p in classes:
                    if (p['seed'],p['session'],p['head'])==(seed,task,'sum') and p['head_index']>=(4,6)[task-1] and p['recall']==0:
                        q=next(r for r in oldpc if (r['seed'],r['session'],r['head'],r['branch'],r['original_label'])==(str(seed),str(task),'sum',b,str(p['original_label'])))
                        if float(q['recall'])>0:newzero.append([seed,task,p['original_label']])
        for task in (1,2):
            for f in fields:
                v=[r['difference_pp'] for r in pairs if (r['contrast'],r['session'],r['metric'])==(branch+'-'+b,task,f)]
                assert len(v)==3 and all(math.isfinite(x) for x in v)
                means.append(dict(contrast=branch+'-'+b,session=task,metric=f,mean_difference_pp=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1))))
        ds={r['metric']:r['mean_difference_pp'] for r in means if (r['contrast'],r['session'])==(branch+'-'+b,2)}
        passed=ds['balanced_accuracy']>0 and ds['current_macro_recall']>=10 and ds['old_macro_recall']>=-5 and improved>=2 and not newzero
        decisions[branch+'-'+b]=dict(final_mean_deltas=ds,seeds_improved_ba_and_current=improved,new_zero_current=newzero,success_criterion_pass=passed)
    for name,rs in [('val_session_metrics',metrics),('val_per_class_metrics',classes),('training_components',epochs),('paired_val_differences',pairs),('paired_val_summary',means)]:write_csv(out/(name+'.csv'),[flatten(r) for r in rs])
    result={'status':'COMPLETE','candidate':branch,'new_epochs':60,'new_session_checkpoints':6,'test_predictions':0,'paired_real_streams_vs_E':'PASS','contrasts':decisions,'success':decisions[branch+'-C']['success_criterion_pass']}
    write_json(root/'CHECKPOINT_LOCK.json',{'sessions':locks,'format':'S0_HEAD_DELTA_V1','parent_reference_required':True,'test_predictions':0})
    write_json(root/'FINAL_STATUS.json',result);write_json(root/'STATUS.json',result)
    (out/'DECISION.md').write_text('# Stationary-feature experiment\n\n'+json.dumps(result,indent=2)+'\n\nDevelopment validation only. Full three-seed matrix; no epoch selection. Repeated validation use introduces adaptive selection bias. Frozen features do not eliminate Gaussian or augmentation mismatch. V2/V3 originals are unchanged.\n')
    return result
