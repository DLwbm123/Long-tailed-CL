"""Paired fixed actions: immediate feedback, one epoch, and boundary refits."""
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import core1_competition as competition
import edge_competition as edge
import prototype_coherent as native
from diagnose_edge_learning import summarize, paired
from diagnose_holdout_feedback import rebuild, to_device
from run_multilabel import ApartFeatures
from run_prototype_coherent import IndexedImages, common_shift, fit_split
from run_prototype_single import Images, extract, manifests, save


CUTS=('immediate','encoder_only','trained','fit_refit','all_refit')


def actions():
    vectors=[torch.zeros(8,dtype=torch.float64)]
    for dimension in range(8):
        for direction in (1.,-1.):
            x=torch.zeros(8,dtype=torch.float64);x[dimension]=direction;vectors.append(x)
    return vectors


def head_for(bank,coefficients,old_count):
    w,metric=native.head(bank,1.)
    base=competition.competition(bank,w)
    static,static_audit=competition.bank_head(bank,base,regularizer=metric)
    pairs=edge.allocation(base,edge.descriptors(bank,static,old_count),coefficients.to(base))
    head,audit=competition.bank_head(bank,pairs,regularizer=metric)
    return head,static,metric,base,pairs,max(audit['relative_residual'],static_audit['relative_residual'])


def feedback(old,head,reference,x,y,tail,pairs,base):
    values=edge.risks(old,head,x,y);baseline=edge.risks(old,reference,x,y)
    return dict(class_risks=values.tolist(),reference_class_risks=baseline.tolist(),
        reward=float(edge.reward(values,baseline,len(old['n']),tail)-.01*edge.action_kl(pairs,base)),
        allocation_kl=float(edge.action_kl(pairs,base)),
        relative_mass_l1=float((pairs-base).abs().sum()/base.sum()),
        current_meta_macro_risk=float(values[len(old['n']):].mean()),
        old_moment_macro_risk=float(values[:len(old['n'])].mean()))


def cut_scores(initial,after,trace):
    heads={k:trace[k].float().numpy() for k in ('initial_head','trained_head','fit_refit','all_refit')}
    return dict(immediate=initial@heads['initial_head'],encoder_only=after@heads['initial_head'],
        trained=after@heads['trained_head'],fit_refit=after@heads['fit_refit'],all_refit=after@heads['all_refit'])


def barrier_ready(jobs):
    ready=True
    for job in jobs:
        root=Path(job['output']);marker=root/'TRAIN_COMPLETE.json';status=root/'STATUS.json'
        if job.get('suite'):
            suite=Path(job['suite'])/'STATUS.json'
            if suite.exists() and json.loads(suite.read_text())['status']=='INCOMPLETE':
                raise RuntimeError('A paired suite failed; development remains sealed')
        if status.exists() and json.loads(status.read_text())['status']=='INCOMPLETE':
            raise RuntimeError('A paired training worker failed; development remains sealed')
        if not marker.exists():
            ready=False;continue
        record=json.loads(marker.read_text())
        if record['completed_candidates']!=job['candidate_ids'] or record['dataset']!=job['dataset']:
            raise ValueError('Training barrier coverage mismatch')
    return ready


def run(config):
    source=Path(config['run']);original=json.loads((source/'INPUT.private.json').read_text())
    assert config['dataset'] in ('ISIC','HK') and original['method']=='static_pc'
    assert original['evaluation_split']=='development_validation' and config['task']==2
    assert original['seed']==original['split_seed']==74002
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    save(output/'INPUT.private.json',config)
    started,cpu=time.monotonic(),time.process_time();steps=0;trained=[];evaluated=[];maximum_residual=0.
    def budget():
        if time.time()>=config['original_deadline'] or time.monotonic()-started>=config['max_wall_seconds']:
            raise TimeoutError('Original diagnostic deadline exceeded')
    def status(state,**extra):
        save(output/'STATUS.json',dict(status=state,completed_candidates=trained,evaluated_candidates=evaluated,
            adapter_updates=steps,policy_updates=0,test_accessed=False,elapsed_seconds=time.monotonic()-started,
            cpu_process_seconds=time.process_time()-cpu,max_solve_residual=maximum_residual,**extra))
    try:
        status('RUNNING',phase='initialize');budget()
        seed=original['seed'];task=config['task'];torch.set_num_threads(4)
        def reset_random():
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        reset_random()
        encoder=ApartFeatures(original['legacy_repo'],original['weight'],len(original['order']),'cuda:0',seed)
        previous=torch.load(source/f'stage_{task-1}.pt',map_location='cpu',weights_only=False)
        encoder.load_state_dict(previous['adapter'],strict=False);encoder.eval()
        initial_adapter={k:p.detach().cpu().clone() for k,p in encoder.named_parameters() if p.requires_grad}
        teacher=copy.deepcopy(encoder).requires_grad_(False).eval()
        old_bank=to_device(previous['bank'],'cuda')
        old_count=len(previous['seen']);seen=original['order'][:sum(original['task_sizes'][:task])]
        assert previous['seen']==seen[:old_count]
        new_classes=seen[old_count:];train,val=manifests(original)
        rows=[r for r in train if r['label'] in new_classes];fi,mi=fit_split(rows,original['split_seed'])
        assert len(fi)==config['expected_fit'] and len(mi)==config['expected_meta']
        held={rows[i]['identity_component'] for i in mi}
        assert not held&{rows[i]['identity_component'] for i in fi}
        assert not held&{r['identity_component'] for r in train if r['label'] in seen[:old_count]}
        expected_steps=math.ceil(len(fi)/original['batch_size'])
        from run_medical_v2 import transform
        def loader(selected,training=False):
            dataset=(IndexedImages if training else Images)(selected,original['images'],transform(training),seed+task*100003)
            dataset.epoch=1 if training else 0
            return DataLoader(dataset,batch_size=original['batch_size'],shuffle=training,num_workers=4,
                multiprocessing_context='spawn',generator=torch.Generator().manual_seed(seed+task*2003))
        def features(model,selected):
            z,labels=extract(model,loader(selected),budget)
            assert np.array_equal(labels,np.asarray([r['label'] for r in selected]))
            return torch.as_tensor(z,device='cuda',dtype=torch.float64),labels
        status('RUNNING',phase='initial_features')
        before,raw_y=features(teacher,rows)
        lookup=torch.tensor([seen.index(c) if c in seen else -1 for c in range(len(original['order']))],device='cuda')
        y=lookup[torch.as_tensor(raw_y,device='cuda')];fit=torch.tensor(fi,device='cuda');meta=torch.tensor(mi,device='cuda')
        fit_bank,group=rebuild(old_bank,before,before,y,fit,False)
        seeds=native.seed_components(before[fit],y[fit]);_,difficulty=native.memberships(before[fit],y[fit],seeds)
        weights=native.sample_weights(y[fit],group,difficulty,before.new_zeros(len(seeds)))
        ranked=sorted(original['order'],key=lambda c:(-sum(r['label']==c for r in train),c))
        tail=[i for i,c in enumerate(seen) if c in ranked[len(ranked)//2:]]
        zero,static,metric,base,_,residual=head_for(fit_bank,actions()[0],old_count)
        assert torch.equal(zero,static);maximum_residual=max(maximum_residual,residual)
        save(output/'DESIGN.json',dict(dataset=config['dataset'],task=task,seen=seen,current_classes=new_classes,
            fit_n=len(fi),meta_n=len(mi),expected_steps_per_candidate=expected_steps,candidate_ids=config['candidate_ids'],
            independent_confirmation=False,current_meta_excluded_before_all_refit=True,training_development_images_accessed=False,
            cuts=list(CUTS),scope='One-epoch paired forks from static_pc stage 1, not original two-epoch task continuation'))
        torch.save(dict(adapter=initial_adapter,features=before.cpu(),labels=raw_y,fit_indices=fi,meta_indices=mi,
                        previous_head=previous['head'],reference_head=static.cpu()),output/'INITIAL.private.pt')
        del previous,zero
        records={};batch_reference=None
        for index in config['candidate_ids']:
            budget();status('RUNNING',phase='candidate_setup',active_candidate=index)
            encoder.load_state_dict(initial_adapter,strict=False);encoder.eval();reset_random()
            coefficient=actions()[index]
            head,reference,metric,base,pairs,residual=head_for(fit_bank,coefficient,old_count)
            maximum_residual=max(maximum_residual,residual);initial_head=head.clone()
            record=dict(candidate=index,initial_feedback=feedback(old_bank,head,reference,before[meta],y[meta],tail,pairs,base))
            initial_pairs=pairs.cpu();initial_base=base.cpu()
            optimizer=torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad],lr=original['lr'],weight_decay=.01)
            # Original action duration is one epoch; no second-epoch LR step is executed here.
            inverse,cross=native.proximal_base(old_bank,metric,len(seen))
            sequence=[];start_steps=steps;losses=[];status('RUNNING',phase='train',active_candidate=index)
            for x,labels,indices in loader([rows[i] for i in fi],True):
                budget();sequence.extend(indices.tolist());x=x.cuda();labels=labels.cuda();indices=indices.cuda()
                optimizer.zero_grad(set_to_none=True);z=encoder(x)
                with torch.no_grad():
                    reference_z=teacher(x);target=F.one_hot(lookup[labels],len(seen)).double()
                    alpha=len(fi)*weights[indices]/(len(seen)*len(x));last=head
                    native_head=native.proximal_head(z.detach().double(),target,alpha,inverse,cross,last)
                    q,mu,mass=competition.moments(old_bank,z.detach().double(),lookup[labels],alpha,len(seen))
                    head,solve=competition.solve(q,mu,mass,pairs,previous=last,proximal=.01,inverse=inverse,
                        x=z.detach().double(),alpha=alpha,native=native_head,regularizer=metric)
                    maximum_residual=max(maximum_residual,solve['relative_residual'])
                fit_loss=.5*(alpha*(z.double()@head-target).square().sum(1)).sum()
                pair_loss=.5*competition.pair_loss(old_bank,z.double(),lookup[labels],alpha,head,pairs)
                old_loss=.5*native.old_square_losses(old_bank,head).sum()/len(seen)
                ridge=.0005*(head*(metric@head)).sum();proximal=.005*(head-last).square().sum()
                fd=(z-reference_z).square().sum(1).mean();loss=fit_loss+pair_loss+old_loss+ridge+proximal+10.*fd
                if not torch.isfinite(loss):raise ValueError('Nonfinite matched training objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in encoder.parameters() if p.requires_grad],5.,error_if_nonfinite=True)
                if config.get('preflight'):
                    assert torch.isfinite(norm) and norm>0 and steps==0
                    save(output/'PREFLIGHT.json',dict(status='PASS',gradient_norm=float(norm),loss=float(loss.detach()),
                        batch_solve_residual=solve['relative_residual'],initial_feedback=record['initial_feedback'],adapter_updates=0,policy_updates=0))
                    status('COMPLETE',preflight=True,peak_gpu_bytes=torch.cuda.max_memory_allocated());return
                optimizer.step();steps+=1;losses.append(float(loss.detach()))
            assert steps-start_steps==expected_steps
            if batch_reference is None:batch_reference=sequence
            assert sequence==batch_reference and sorted(sequence)==list(range(len(fi)))
            status('RUNNING',phase='after_features',active_candidate=index)
            after,_=features(encoder,rows)
            trained_head=head.clone();post_heads={};post_feedback={};post_pairs={}
            for partition,include in [('fit',False),('all',True)]:
                bank,_=rebuild(old_bank,before,after,y,fit,include)
                post,ref,_,base_post,pairs_post,residual=head_for(bank,coefficient,old_count)
                maximum_residual=max(maximum_residual,residual);post_heads[partition]=post
                use=torch.arange(len(y),device='cuda') if include else fit
                shifted=native.translate(old_bank,common_shift(before[use],after[use],y[use]))
                post_feedback[partition+'_refit']=feedback(shifted,post,ref,after[meta],y[meta],tail,pairs_post,base_post)
                post_pairs[partition]=pairs_post.cpu()
                if not include:
                    for cut,w in [('encoder_only',initial_head),('trained',trained_head)]:
                        values=edge.risks(shifted,w,after[meta],y[meta])
                        post_feedback[cut]=dict(class_risks=values.tolist(),current_meta_macro_risk=float(values[old_count:].mean()),old_moment_macro_risk=float(values[:old_count].mean()))
                del bank,shifted,post,ref
            branch=output/f'candidate_{index:02d}';branch.mkdir(exist_ok=False)
            torch.save(dict(adapter={k:p.detach().cpu() for k,p in encoder.named_parameters() if p.requires_grad},
                initial_head=initial_head.cpu(),trained_head=trained_head.cpu(),fit_refit=post_heads['fit'].cpu(),
                all_refit=post_heads['all'].cpu(),post_features=after.cpu(),labels=raw_y,
                initial_pairs=initial_pairs,initial_base=initial_base,post_pairs=post_pairs,
                optimizer=optimizer.state_dict()),branch/'TRACE.private.pt')
            record.update(adapter_updates=expected_steps,policy_updates=0,training_batch_order_verified=True,
                mean_training_loss=float(np.mean(losses)),post_feedback=post_feedback)
            records[str(index)]=record;trained.append(index);save(output/'TRAINING.json',records)
            del optimizer,inverse,cross,after,head,initial_head,trained_head,post_heads
            torch.cuda.empty_cache()
        assert steps==expected_steps*len(config['candidate_ids'])
        torch.save(batch_reference,output/'BATCH_ORDER.private.pt')
        save(output/'TRAIN_COMPLETE.json',dict(dataset=config['dataset'],completed_candidates=trained,adapter_updates=steps,ended=time.time()))
        status('RUNNING',phase='barrier')
        del teacher,fit_bank,old_bank,before,weights,metric,static,base
        torch.cuda.empty_cache()
        while not barrier_ready(config['barrier_jobs']):
            budget();time.sleep(5)
        status('RUNNING',phase='development_evaluation')
        encoder.eval().requires_grad_(False);encoder.load_state_dict(initial_adapter,strict=False)
        validation=[r for r in val if r['label'] in seen]
        start_features,validation_y=features(encoder,validation);start_features=start_features.float().cpu().numpy()
        torch.save(dict(features=start_features,labels=validation_y),output/'INITIAL_DEVELOPMENT.private.pt')
        expected=json.loads((source/'metrics.json').read_text())['stages'][task-2]
        initial_trace=torch.load(output/'INITIAL.private.pt',map_location='cpu',weights_only=False)
        original_head=initial_trace['previous_head'].numpy();initial_training=initial_trace['features'].float().numpy()
        old_mask=np.isin(validation_y,seen[:old_count]);old_scores=start_features[old_mask]@original_head
        old_summary=summarize(old_scores,validation_y[old_mask],seen[:old_count],
            [validation[j]['identity_component'] for j in np.flatnonzero(old_mask)])
        for c,v in old_summary['per_class'].items():
            assert v['n']==expected['per_class_n'][c] and abs(v['correct']/v['n']-expected['per_class_recall'][c])<1e-12
        results=dict(dataset=config['dataset'],task=task,seen=seen,current_classes=new_classes,
            independent_confirmation=False,development_after_all_training=True,initial_checkpoint_development_reproduced=True,
            policy_updates=0,test_accessed=False,candidates={})
        for index in config['candidate_ids']:
            budget();status('RUNNING',phase='development_evaluation',active_candidate=index)
            branch=output/f'candidate_{index:02d}';trace=torch.load(branch/'TRACE.private.pt',map_location='cpu',weights_only=False)
            encoder.load_state_dict(trace['adapter'],strict=False);z,_=features(encoder,validation);z=z.float().cpu().numpy()
            scores=cut_scores(start_features,z,trace)
            torch.save(dict(features=z,labels=validation_y,scores=scores),branch/'DEVELOPMENT.private.pt')
            views={}
            for view,keep in [('all',np.ones(len(validation),dtype=bool)),('current',np.isin(validation_y,new_classes)),('old',old_mask)]:
                ids=[validation[j]['identity_component'] for j in np.flatnonzero(keep)]
                views[view]=dict(heads={cut:summarize(v[keep],validation_y[keep],seen,ids) for cut,v in scores.items()},
                    transitions={a+'_to_'+b:paired(scores[a][keep],scores[b][keep],validation_y[keep],seen)
                                 for a,b in zip(CUTS[:-1],CUTS[1:])})
            training_scores=cut_scores(initial_training,trace['post_features'].float().numpy(),trace)
            training_views={}
            for view,ids_index in [('fit',fi),('meta',mi)]:
                ids=[rows[j]['identity_component'] for j in ids_index]
                training_views[view]={cut:summarize(v[ids_index],raw_y[ids_index],seen,ids) for cut,v in training_scores.items()}
            results['candidates'][str(index)]=dict(records[str(index)],development=views,training_views=training_views)
            evaluated.append(index);save(output/'RESULTS.json',results)
            del trace,z,scores
        status('COMPLETE',ended=time.time(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),development_after_all_training=True)
    except BaseException as exc:
        status('INCOMPLETE',error=str(exc));raise


def self_check():
    import tempfile
    assert len(actions())==17 and torch.equal(actions()[0],torch.zeros(8))
    assert all(torch.equal(actions()[2*i+1],-actions()[2*i+2]) for i in range(8))
    torch.manual_seed(8);torch.set_num_threads(2)
    x=torch.randn(18,5,dtype=torch.float64);y=torch.arange(3).repeat_interleave(6)
    old=native.append(native.empty(5),x[:6],y[:6],y[:6],torch.full((6,),1/6,dtype=x.dtype))
    fit=torch.tensor([0,1,2,3,6,7,8,9]);current=x[6:];labels=y[6:]
    bank,_=rebuild(old,current,current,labels,fit,False)
    w,static,metric,base,pairs,residual=head_for(bank,actions()[0],1)
    assert torch.equal(w,static) and torch.equal(pairs,base)
    f=feedback(old,w,static,current[[4,5,10,11]],labels[[4,5,10,11]],[1],pairs,base)
    assert f['reward']==f['allocation_kl']==f['relative_mass_l1']==0 and residual<1e-8
    for a in actions()[1:]:
        w,static,_,base,pairs,residual=head_for(bank,a,1)
        assert torch.all(pairs>=.5*base) and torch.allclose(pairs.sum(),base.sum()) and residual<1e-8
    trace={name:torch.tensor([[value]],dtype=torch.float64) for name,value in [('initial_head',1.),('trained_head',2.),('fit_refit',3.),('all_refit',4.)]}
    cut=cut_scores(np.array([[2.]]),np.array([[5.]]),trace)
    assert [float(cut[k][0,0]) for k in CUTS]==[2.,5.,10.,15.,20.]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);job=dict(output=tmp,dataset='HK',candidate_ids=[0,1])
        assert not barrier_ready([job])
        save(root/'TRAIN_COMPLETE.json',dict(dataset='HK',completed_candidates=[0,1]))
        assert barrier_ready([job])
        save(root/'STATUS.json',dict(status='INCOMPLETE'))
        try:barrier_ready([job])
        except RuntimeError:pass
        else:raise AssertionError('Failed worker opened evaluation barrier')
    print('PASS: 17 fixed actions, neutral head/feedback identity, allocation floor/budget, solver, five-cut score wiring and fail-closed evaluation barrier')


if __name__=='__main__':
    config=json.load(sys.stdin)
    if config.get('self_check'):self_check()
    else:run(config)
