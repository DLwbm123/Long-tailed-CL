"""FD-only scheduling with permanently disjoint fitting and reward statistics."""
import copy
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import core1_competition as competition
import fdrl_control as ctl
import next1_support as support
import prototype_coherent as method
import affine_moments
from pcrl_control import Snapshot, allocation
from run_multilabel import ApartFeatures
from run_pcrl import IndexedImages, common_shift, cpu_bank
from run_prototype_single import Images, extract, manifests, save


def device_bank(bank):
    return dict(mu=bank['mu'].cuda(),Q=bank['Q'].cuda(),n=bank['n'],
        components=[{k:v.cuda() if torch.is_tensor(v) else v for k,v in c.items()} for c in bank['components']])


def run(config):
    arm=config['method']
    if arm not in ('R','FD5','FD10','FD20','RANDOM','GRADIENT','GREEDY','FDRL','SHUFFLED','COMP10','COMP_FDRL'):
        raise ValueError('Unknown FD arm')
    learned=arm in ('FDRL','SHUFFLED','COMP_FDRL')
    compete=arm!='R' and (config.get('base_competition',True) or arm in ('COMP10','COMP_FDRL'))
    full_task=config.get('full_task_returns',False)
    varied=config.get('varied_warmup',False);warm_episodes=int(config.get('warmup_episodes',4))
    drift=config.get('drift','translation')
    if drift not in ('translation','affine'):raise ValueError('Unknown drift model')
    if warm_episodes not in (4,16):raise ValueError('Unregistered warmup length')
    expected_warm=warm_episodes*32+(warm_episodes//2*8 if varied else 0)
    update_cap=476+600+expected_warm
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    save(output/'INPUT.private.json',config)
    start=time.monotonic();steps=0;initial_steps=0;rollouts=0;warm_updates=0;diagnostic_seconds=0.
    policy=ctl.Policy(config['seed']);decisions=[];diagnostics=[];history=[]
    choice_rng=torch.Generator().manual_seed(config['seed']+62531)
    shuffle_rng=torch.Generator().manual_seed(config['seed']+77231)
    def budget():
        if time.monotonic()-start>=config['max_wall_seconds']:raise RuntimeError('INCOMPLETE_WALL_BUDGET')
    def status(**fields):
        save(output/'STATUS.json',dict(steps=steps,actual_updates=steps-initial_steps+rollouts,
            retained_updates=steps-initial_steps,rollout_updates=rollouts,warmup_updates=warm_updates,
            policy_updates=policy.updates,diagnostic_gpu_seconds=diagnostic_seconds,
            elapsed_seconds=time.monotonic()-start,**fields))
    try:
        seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
        train,_=manifests(config)
        encoder=ApartFeatures(config['legacy_repo'],config['weight'],8,'cuda:0',seed)
        from run_medical_v2 import transform
        def loader(rows,training,task,sampler=None):
            ds=(IndexedImages if training else Images)(rows,config['images'],transform(training),seed+task*100003)
            workers=int(config.get('workers',4))
            opts=dict(multiprocessing_context='spawn',prefetch_factor=2,persistent_workers=not training) if workers else {}
            return DataLoader(ds,batch_size=64,shuffle=training and sampler is None,sampler=sampler,num_workers=workers,
                generator=torch.Generator().manual_seed(seed+task*2003),**opts)
        @torch.no_grad()
        def features(model,data):
            x,y=extract(model,data,budget)
            return torch.as_tensor(x,device='cuda',dtype=torch.float64),torch.as_tensor(y,device='cuda')
        def fitted(b,old):
            native,_=method.head(b,0.)
            if not old or not compete:return native,dict(relative_residual=0.,iterations=0)
            a=allocation(competition.competition(b,native),old,1)
            return competition.bank_head(b,a,.5)
        bank=method.empty(encoder.dim,'cuda');meta_bank=method.empty(encoder.dim,'cuda');seen=[];W=None
        tasks=[config['order'][i:i+2] for i in range(0,8,2)];start_task=1
        tail_labels=set(sorted(config['order'],key=lambda c:(-sum(r['label']==c for r in train),c))[4:])
        if config.get('prefix'):
            state=torch.load(config['prefix'],map_location='cpu',weights_only=False)
            if state['steps']!=276 or state['seen']!=tasks[0]:raise ValueError('Wrong T1 prefix')
            encoder.load_state_dict(state['model'],strict=True);seen=state['seen'].copy();steps=initial_steps=276
            if not state.get('fit_only'):
                status(status='RUNNING',phase='prepare_disjoint_prefix')
                rows=[r for r in train if r['label'] in seen];fi,mi=support.fixed_split(rows,config)
                assert not ({rows[i]['identity_component'] for i in fi}&{rows[i]['identity_component'] for i in mi})
                data=loader(rows,False,1);x,raw=features(encoder,data)
                y=torch.tensor([seen.index(int(c)) for c in raw],device='cuda')
                ix=torch.tensor(fi,device='cuda');im=torch.tensor(mi,device='cuda')
                seeds=method.seed_components(x[ix],y[ix]);g,d=method.memberships(x[ix],y[ix],seeds)
                weights=method.sample_weights(y[ix],g,d,x.new_zeros(len(seeds)))
                bank=method.append(bank,x[ix],y[ix],g,weights);meta_bank=ctl.meta_append(meta_bank,x[im],y[im]);W,_=fitted(bank,0)
                state=dict(model=state['model'],adapter=state['adapter'],head=W.float().cpu(),bank=cpu_bank(bank),
                    meta_bank=cpu_bank(meta_bank),seen=seen,steps=276,rng=state['rng'],fit_only=True)
                torch.save(state,output/'prepared_prefix.pt')
                del data,x,raw,y,seeds,g,d,weights
            else:
                bank=device_bank(state['bank']);meta_bank=device_bank(state['meta_bank']);W=state['head'].double().cuda()
            torch.save(state,output/'stage_1.pt');support.restore_rng(state['rng']);start_task=2
            save(output/'PREFIX.json',dict(reused_adapter_updates=276,fit_only=True,meta_moments_separate=True,
                fit_counts=bank['n'],meta_counts=meta_bank['n'],independent_repeat=False))
        for task,classes in enumerate(tasks,1):
            if task<start_task:continue
            seen+=classes;old_count=len(bank['n']);beta=.5 if old_count and compete else 0.
            rows=[r for r in train if r['label'] in classes];fi,mi=support.fixed_split(rows,config)
            if {rows[i]['identity_component'] for i in fi}&{rows[i]['identity_component'] for i in mi}:raise ValueError('Fit-meta identity overlap')
            fi=torch.tensor(fi,device='cuda');mi=torch.tensor(mi,device='cuda')
            canonical=loader(rows,False,task);training=loader([rows[i] for i in fi.tolist()],True,task)
            teacher=copy.deepcopy(encoder).requires_grad_(False).eval()
            status(status='RUNNING',phase='task_start_features',task=task)
            before,raw_y=features(teacher,canonical);lookup=torch.tensor([seen.index(c) if c in seen else -1 for c in range(8)],device='cuda');y=lookup[raw_y];current=before
            seeds=method.seed_components(before[fi],y[fi]);group,difficulty=method.memberships(before,y,seeds)
            weights=method.sample_weights(y[fi],group[fi],difficulty[fi],before.new_zeros(len(seeds)))
            tail=[i for i,c in enumerate(seen) if c in tail_labels]
            def subset(ids,offset):
                chosen=[]
                for c in sorted(y[ids].unique().tolist()):
                    indices=ids[y[ids]==c].tolist();random.Random(seed+task*100003+c*2003+offset).shuffle(indices);chosen+=indices[:64]
                return torch.tensor(chosen,device='cuda')
            pf,pm=subset(fi,11),subset(mi,29);pi=torch.cat([pf,pm]);probe_loader=loader([rows[i] for i in pi.tolist()],False,task)
            @torch.no_grad()
            def transported(base,observed,labels):
                shift=common_shift(base,observed,labels)
                if drift=='affine' and old_count:
                    mapping=affine_moments.fit(base,observed,labels)
                    return affine_moments.transport(bank,mapping),affine_moments.transport(meta_bank,mapping),shift,mapping['diagnostics']
                return method.translate(bank,shift),method.translate(meta_bank,shift),shift,dict(kind='translation')
            @torch.no_grad()
            def readout(observed,fit_ids,meta_ids,labels,base):
                old,old_meta,shift,drift_audit=transported(base[fit_ids],observed[fit_ids],labels[fit_ids])
                g,d=method.memberships(observed[fit_ids],labels[fit_ids],seeds)
                sw=method.sample_weights(labels[fit_ids],g,d,observed.new_zeros(len(seeds)))
                b=method.append(old,observed[fit_ids],labels[fit_ids],g,sw);head,solve=fitted(b,old_count)
                risk=ctl.risks(old_meta,head,observed[meta_ids],labels[meta_ids])
                return risk,head,b,shift,dict(solve,drift=drift_audit),old_meta
            @torch.no_grad()
            def probe():
                observed,labels=features(encoder,probe_loader)
                if not torch.equal(lookup[labels],y[pi]):raise ValueError('Probe order changed')
                ix=torch.arange(len(pf),device='cuda');im=torch.arange(len(pf),len(pi),device='cuda')
                return readout(observed,ix,im,y[pi],before[pi])
            initial_full=readout(before,fi,mi,y,before)[0]
            params=[p for p in encoder.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(params,lr=config['lr'],weight_decay=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,2,eta_min=1e-5)
            previous_risk=None;trajectory=[]
            for epoch in (1,2):
                budget();encoder.eval();training.dataset.epoch=epoch
                old=transported(before[fi],current[fi],y[fi])[0]
                candidate=method.append(old,current[fi],y[fi],group[fi],weights)
                W,R=method.head(candidate,0.);pairs=competition.competition(candidate,W)
                if beta:W,_=competition.bank_head(candidate,pairs,beta);pairs=allocation(pairs,old_count,1)
                inverse,cross=method.proximal_base(old,R,len(seen))
                def update(batch,head,action,kind=None,diagnose=False):
                    nonlocal rollouts,warm_updates,diagnostic_seconds
                    budget();x,labels,indices=[v.cuda() for v in batch];optimizer.zero_grad(set_to_none=True)
                    z=encoder(x)
                    with torch.no_grad():
                        ref=teacher(x);target=F.one_hot(lookup[labels],len(seen)).double();alpha=len(fi)*weights[indices]/(len(seen)*len(x));previous=head
                        head=method.proximal_head(z.detach().double(),target,alpha,inverse,cross,previous)
                        solve=dict(relative_residual=0.,iterations=0)
                        if beta:
                            q,mu,mass=competition.moments(old,z.detach().double(),lookup[labels],alpha,len(seen))
                            head,solve=competition.solve(q,mu,mass,pairs,beta,False,previous,.01,inverse,z.detach().double(),alpha,head)
                    fit=.5*(alpha*(z.double()@head-target).square().sum(1)).sum()
                    pair=beta*competition.pair_loss(old,z.double(),lookup[labels],alpha,head,pairs) if beta else fit.new_zeros(())
                    fd=(z-ref).square().sum(1).mean();loss=fit+pair+ctl.ACTIONS[action]*fd
                    grad=None
                    if diagnose:
                        raw,cost=support.gradient_diagnostic(fit+pair,fd,params);diagnostic_seconds+=cost
                        grad=dict(task_norm=raw['fit_gradient_norm'],weighted_fd_norm=raw['weighted_fd_gradient_norm'],cosine=raw['cosine'] or 0.)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite adapter loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True)
                    if kind:
                        if steps-initial_steps+rollouts>=update_cap:raise RuntimeError('ADAPTER_UPDATE_CAP')
                        optimizer.step()
                        if kind!='retained':rollouts+=1
                        if kind=='warmup':warm_updates+=1
                    return head,dict(fit=float(fit.detach()),fd=float(fd.detach()),norm=float(norm),solve=solve,gradient=grad)
                def context(batch,head,risk,prior,shift,progress):
                    snap=Snapshot(encoder,optimizer);_,detail=update(batch,head,1,diagnose=True);snap.restore(optimizer)
                    return ctl.features(risk,prior,old_count,tail,detail['gradient'],float(shift.norm()),meta_bank,progress),detail['gradient']
                if config.get('preflight'):
                    if task!=2 or epoch!=1:raise ValueError('Preflight requires T2 prefix')
                    batch=next(iter(training));snap=Snapshot(encoder,optimizer)
                    # Probe a nonzero displacement without an optimizer update, then restore exactly.
                    with torch.no_grad():
                        for p in params:p.add_(1e-4)
                    perturbed=probe()
                    post=Snapshot(encoder,optimizer);norms=[];heads=[]
                    for action in (0,2):
                        post.restore(optimizer);head,detail=update(batch,W,action,diagnose=True)
                        norms.append(detail['norm']);heads.append(head)
                    post.restore(optimizer)
                    if abs(norms[0]-norms[1])<=1e-12:raise ValueError('FD actions do not affect adapter gradient')
                    snap.restore(optimizer);snap.verify(optimizer)
                    risk,_,_,shift,solve,_=probe();state,grad=context(batch,W,risk,risk,shift,[.5,.5,0.])
                    sample,p=policy.sample(state)
                    save(output/'PREFLIGHT.json',dict(status='PASS',optimizer_updates=0,restore_verified=True,
                        action_gradient_norms=norms,meta_fit_identity_disjoint=True,fit_counts=bank['n'],meta_counts=meta_bank['n'],
                        initial_risk=risk.tolist(),state=state.tolist(),probabilities=p.tolist(),sampled_action=sample,
                        drift=drift,perturbed_drift=perturbed[4]['drift'],perturbed_risk=perturbed[0].tolist(),
                        max_residual=solve['relative_residual'],peak_gpu_bytes=torch.cuda.max_memory_allocated()))
                    status(status='COMPLETE',preflight=True,optimizer_updates=0);return
                if task==2 and epoch==1:
                    fit_rows=[rows[i] for i in fi.tolist()]
                    if not varied:
                        warm_loader=loader(fit_rows,True,task+20);warm_loader.dataset.epoch=1
                        warm_batches=list(itertools.islice(warm_loader,32));del warm_loader
                    snap=Snapshot(encoder,optimizer);entry=W.clone();warm_audits=[]
                    for episode in range(warm_episodes):
                        snap.restore(optimizer);head=entry.clone();warm_trajectory=[]
                        depth=8*(episode%2) if varied else 0
                        if varied:
                            mass=(.25,.5,.75,.5)[episode%4]
                            counts={c:sum(r['label']==c for r in fit_rows) for c in classes}
                            sw=[(mass if r['label']==classes[0] else 1-mass)/counts[r['label']] for r in fit_rows]
                            sampler=WeightedRandomSampler(sw,(32+depth)*64,replacement=True,
                                generator=torch.Generator().manual_seed(seed+9227+episode*2003))
                            warm_loader=loader(fit_rows,True,task+20+episode,sampler);warm_loader.dataset.epoch=episode+1
                            warm_batches=list(warm_loader);del warm_loader
                            for batch in warm_batches[:depth]:head,_=update(batch,head,1,'warmup')
                            warm_batches=warm_batches[depth:]
                        risk,_,_,shift,_,_=probe();prior=risk;warm_initial=risk.clone()
                        for block in range(4):
                            cached=[warm_batches[(block*8+j)%len(warm_batches)] for j in range(8)]
                            state,_=context(cached[0],head,risk,prior,shift,[.5,0.,block/4]);action,p=policy.sample(state)
                            for batch in cached:head,_=update(batch,head,action,'warmup')
                            after,_,_,next_shift,_,_=probe();r=ctl.reward(risk,after,old_count,tail)
                            if block==3:
                                r+=ctl.reward(warm_initial,after,old_count,tail) if varied else ctl.reward(initial_full,readout(features(encoder,canonical)[0],fi,mi,y,before)[0],old_count,tail)
                            warm_trajectory.append(dict(policy_state=state,action=action,behavior=p,reward=r));prior,risk,shift=risk,after,next_shift
                            status(status='RUNNING',phase='policy_warmup',task=task,episode=episode+1,block=block+1)
                        audit=policy.update(warm_trajectory)
                        warm_audits.append(dict(episode=episode+1,start_depth=depth,first_class_sampling_mass=mass if varied else None,
                            actions=[r['action'] for r in warm_trajectory],rewards=[r['reward'] for r in warm_trajectory],**audit))
                    snap.restore(optimizer);snap.verify(optimizer);W=entry;del warm_batches
                    save(output/'WARMUP.json',dict(episodes=warm_audits,adapter_updates=warm_updates,discarded_model_updates=True,
                        scope='arrived Task 2 varied class sampling and 0/8-step starts' if varied else 'arrived Task 2 short continuations, reset to the same start',validation_accessed=False))
                if not full_task:trajectory=[]
                iterator=iter(training);n=len(training);block_audits=[]
                risk,_,_,shift,_,_=probe();previous_risk=risk if previous_risk is None else previous_risk
                for first in range(0,n,8):
                    cached=list(itertools.islice(iterator,min(8,n-first)))
                    if not cached:raise ValueError('Empty training block')
                    terminal=epoch==2 and first+len(cached)==n
                    state,gradient=context(cached[0],W,risk,previous_risk,shift,[task/4,epoch/2,first/max(n,1)])
                    history.append(state.clone());policy_state=state
                    if arm=='SHUFFLED':policy_state=history[int(torch.randint(len(history),(1,),generator=shuffle_rng))]
                    branch_rewards=[];branch_risks=[];max_residual=0.
                    if task>1:
                        snap=Snapshot(encoder,optimizer)
                        for action in range(3):
                            snap.restore(optimizer);head=W.clone()
                            for batch in cached:
                                head,detail=update(batch,head,action,'branch');max_residual=max(max_residual,detail['solve']['relative_residual'])
                            after,_,_,_,solve,_=probe();r=ctl.reward(risk,after,old_count,tail)
                            if terminal:r+=ctl.reward(initial_full,readout(features(encoder,canonical)[0],fi,mi,y,before)[0],old_count,tail)
                            branch_rewards.append(r);branch_risks.append(after.tolist());max_residual=max(max_residual,solve['relative_residual'])
                        snap.restore(optimizer);snap.verify(optimizer)
                    p,_=policy.forward(policy_state);p=p.detach()
                    if task==1 or arm in ('R','FD10','COMP10'):action=1
                    elif arm=='FD5':action=0
                    elif arm=='FD20':action=2
                    elif arm=='RANDOM':action=int(torch.randint(3,(1,),generator=choice_rng))
                    elif arm=='GRADIENT':
                        ratio=gradient['weighted_fd_norm']/max(gradient['task_norm'],1e-12)
                        target=10./max(ratio,1e-12) if gradient['cosine']<0 else 10.
                        action=min(range(3),key=lambda a:abs(np.log(ctl.ACTIONS[a]/target)))
                    elif arm=='GREEDY':action=max(range(3),key=lambda a:(branch_rewards[a],a==1,-a))
                    else:action,p=policy.sample(policy_state)
                    if not learned or task==1:
                        p=torch.full((3,),1/3,dtype=torch.float64) if arm=='RANDOM' and task>1 else F.one_hot(torch.tensor(action),3).double()
                    for batch in cached:
                        W,detail=update(batch,W,action,'retained');steps+=1;max_residual=max(max_residual,detail['solve']['relative_residual'])
                    after,_,_,next_shift,solve,_=probe();r=ctl.reward(risk,after,old_count,tail);terminal_reward=0.
                    if terminal:
                        current,after_y=features(encoder,canonical)
                        if not torch.equal(after_y,raw_y):raise ValueError('Canonical order changed')
                        full=readout(current,fi,mi,y,before);terminal_reward=ctl.reward(initial_full,full[0],old_count,tail);r+=terminal_reward
                    if task>1:
                        if abs(r-branch_rewards[action])>1e-8:raise ValueError('Retained path differs from matched branch')
                        trajectory.append(dict(policy_state=policy_state,action=action,behavior=p,reward=r))
                        audit=dict(task=task,epoch=epoch,batch=first,horizon=len(cached),state=state.tolist(),policy_state=policy_state.tolist(),
                            probabilities=p.tolist(),selected_action=action,fd_weight=ctl.ACTIONS[action],reward=r,terminal_reward=terminal_reward,
                            branch_rewards=branch_rewards,branch_risks=branch_risks,before_risk=risk.tolist(),after_risk=after.tolist(),
                            gradient=gradient,max_residual=max(max_residual,solve['relative_residual']),drift=solve['drift'],behavior='sampled' if learned else arm)
                        decisions.append(audit);block_audits.append(audit)
                        save(output/'CONTROLLER.json',dict(decisions=decisions,policy_updates=policy.updates,rollout_updates=rollouts))
                    previous_risk,risk,shift=risk,after,next_shift
                    status(status='RUNNING',phase='train_blocks',task=task,epoch=epoch,batch=first)
                update_audit=policy.update(trajectory) if task>1 and learned and (not full_task or epoch==2) else None
                scheduler.step()
                if epoch!=2:current,after_y=features(encoder,canonical)
                diagnostics.append(dict(task=task,epoch=epoch,steps=steps,fit_n=len(fi),meta_n=len(mi),
                    block_count=len(block_audits),return_scope='task' if full_task else 'epoch',
                    trajectory_length=len(trajectory),policy_update=update_audit))
                save(output/'diagnostics.json',diagnostics)
            full=readout(current,fi,mi,y,before);_,W,new_bank,shift,solve,old_meta=full
            meta_bank=ctl.meta_append(old_meta,current[mi],y[mi]);bank=new_bank
            model={k:v.detach().cpu() for k,v in encoder.state_dict().items()}
            torch.save(dict(model=model,adapter={k:p.detach().cpu() for k,p in encoder.named_parameters() if p.requires_grad},
                head=W.float().cpu(),bank=cpu_bank(bank),meta_bank=cpu_bank(meta_bank),seen=seen.copy(),steps=steps,
                fit_only=True,rng=support.rng_state(),configuration=config,policy=policy.saved()),output/f'stage_{task}.pt')
            save(output/f'boundary_{task}.json',dict(task=task,fit_counts=bank['n'],meta_counts=meta_bank['n'],
                class_meta_risks=full[0].tolist(),solve=solve,steps=steps,fit_only=True,meta_prototypes=0))
            del teacher,canonical,training,probe_loader,before,current,seeds,group,difficulty
        expected=200 if initial_steps else 476
        if steps!=476 or steps-initial_steps!=expected or rollouts!=600+expected_warm or warm_updates!=expected_warm:
            raise ValueError('Frozen update accounting mismatch')
        status(status='TRAINED',stages=4,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status(status='INCOMPLETE',error=str(exc));raise


if __name__=='__main__':run(json.load(sys.stdin))
