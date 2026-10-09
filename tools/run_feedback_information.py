"""Frozen-head strength, empirical-feedback and conditional-transport diagnostics."""
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import prototype_coherent as native
from diagnose_holdout_feedback import rebuild, to_device
from diagnose_edge_learning import paired
from feedback_information import split_identities, gaussian_scores, choose
from run_action_trace import head_for, actions
from run_decision_probability import KEYS
from run_feedback_audit import metrics
from run_multilabel import ApartFeatures
from run_prototype_coherent import fit_split, common_shift
from run_prototype_single import Images, extract, manifests, save


HEADS = KEYS+['linear_ridge','prototype_ridge']
FAMILIES = dict(edge=list(range(17)), norm=[0]+list(range(17,33)))


def score(z, heads):
    return np.stack([np.asarray(z) @ w.float().cpu().numpy() for w in heads],axis=1)


@torch.no_grad()
def run(config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();cpu=time.process_time()
    prepared=[];evaluated=[];images=0;maximum_residual=0.;torch.set_num_threads(4)
    save(out/'INPUT.private.json',config)
    def budget():
        if time.time() >= config['original_deadline'] or time.monotonic()-started >= config['max_wall_seconds']:
            raise TimeoutError('Original deadline exhausted')
    def status(state,**extra):
        save(out/'STATUS.json',dict(status=state,prepared_states=prepared,evaluated_states=evaluated,
            elapsed_seconds=time.monotonic()-started,cpu_process_seconds=time.process_time()-cpu,
            image_rows=images,adapter_updates=0,policy_updates=0,test_accessed=False,**extra))
    def load(path):
        budget();return torch.load(path,map_location='cpu',weights_only=False)
    try:
        status('RUNNING',phase='prepare');records={}
        for spec in config['states']:
            key=spec['name'];task=spec['task'];target=out/key;target.mkdir();oldrun=Path(spec['previous_run'])
            previous_results=json.loads((oldrun/'RESULTS.json').read_text())[key]
            cached=load(oldrun/key/'HEADS.private.pt');original=json.loads((Path(spec['source'])/'INPUT.private.json').read_text())
            assert original['method']=='static_pc' and original['evaluation_split']=='development_validation'
            seen=previous_results['seen'];current=previous_results['current'];tail=previous_results['tail'];old_count=len(seen)-len(current)
            train,val=manifests(original);rows=[r for r in train if r['label'] in current];val=[r for r in val if r['label'] in seen]
            fi,mi=fit_split(rows,original['split_seed']);assert fi==cached['fit_indices'] and mi==cached['meta_indices']
            raw_y=np.asarray(cached['labels']);assert np.array_equal(raw_y,np.asarray([r['label'] for r in rows]))
            after=cached['current_features'].cuda().double();heads=cached['fit_heads'].cuda().double();assert len(heads)==33
            if spec.get('cached'):
                before=load(Path(spec['cached'])/'INITIAL.private.pt')['features'].cuda().double()
            else:
                random.seed(original['seed']);np.random.seed(original['seed']);torch.manual_seed(original['seed']);torch.cuda.manual_seed_all(original['seed'])
                model=ApartFeatures(original['legacy_repo'],original['weight'],len(original['order']),'cuda:0',original['seed'])
                from run_medical_v2 import transform
                previous=load(Path(spec['source'])/f'stage_{task-1}.pt');model.load_state_dict(previous['adapter'],strict=False)
                model.eval().requires_grad_(False);status('RUNNING',phase='current_training_before_features',active_state=key)
                loader=DataLoader(Images(rows,original['images'],transform(False),original['seed']+task*100003),batch_size=64,
                    num_workers=4,multiprocessing_context='spawn',generator=torch.Generator().manual_seed(original['seed']+task*2003))
                z,labels=extract(model,loader,budget);images+=len(rows);assert np.array_equal(labels,raw_y)
                before=torch.as_tensor(z,device='cuda',dtype=torch.float64);del model,previous,loader
            torch.save(dict(features=before.cpu(),labels=raw_y),target/'BEFORE.private.pt')
            previous=load(Path(spec['source'])/f'stage_{task-1}.pt');old=to_device(previous['bank'],'cuda')
            y=torch.tensor([seen.index(int(c)) for c in raw_y],device='cuda');fit=torch.tensor(fi,device='cuda');meta=torch.tensor(mi,device='cuda')
            bank,_=rebuild(old,before,after,y,fit,False);zero,_,_,_,_,residual=head_for(bank,actions()[0],old_count)
            maximum_residual=max(maximum_residual,residual);error=float((zero.float()-heads[0].float()).abs().max())
            assert torch.allclose(zero.float(),heads[0].float(),atol=1e-6,rtol=1e-5),f'Fit zero reconstruction error: {error}'
            extras=[]
            for strength in (0.,1.):
                w,R=native.head(bank,strength);A=bank['Q'].mean(0)+.001*R;rhs=bank['mu'].T/len(seen)
                residual=float((A@w-rhs).norm()/rhs.norm());assert residual<=1e-8;maximum_residual=max(maximum_residual,residual);extras.append(w.float().double())
            heads=torch.cat([heads,torch.stack(extras)])
            dev=load(Path(spec['cached'])/'candidate_00/DEVELOPMENT.private.pt' if spec.get('cached') else oldrun/key/'DEVELOPMENT.private.pt')
            z=np.asarray(dev['features']);labels=np.asarray(dev['labels']);assert np.array_equal(labels,np.asarray([r['label'] for r in val]))
            ai,bi,covered,support=split_identities(val,config['split_seed']);covered=[c for c in seen if c in covered]
            A=score(z[ai],heads)
            # Predictions retain all seen-class outputs even for a coverage-restricted diagnostic.
            emp=np.stack([(A[labels[ai]==c,:33].argmax(2)==seen.index(c)).mean(0) for c in covered],axis=1)
            gaussian=[];covariance=[];status('RUNNING',phase='feedback_integration',active_state=key)
            for c in covered:
                values,audit=gaussian_scores(torch.as_tensor(A[labels[ai]==c,:33],device='cuda'),seen.index(c),
                    config['integration_seed']+spec['seed_offset']+1009*seen.index(c),budget)
                gaussian.append(values);covariance.append(dict(label=c,**audit))
            g=torch.stack(gaussian,2);e=torch.tensor(emp,device='cuda',dtype=torch.float64)[None].expand(8,-1,-1)
            choices={};audits={}
            for source,values in [('empirical_A',e),('gaussian_A',g)]:
                for family,indices in FAMILIES.items():
                    chosen,audit=choose(values,covered,current,tail,indices)
                    choices[source+'_'+family]=KEYS[chosen];audits[source+'_'+family]={KEYS[i]:v for i,v in audit.items()}
            conditional={}
            for c in current:
                j=seen.index(c);own=fit[y[fit]==j];other=fit[y[fit]!=j];assert len(other)>0 and len(own)>=2
                shifted=before[own]+common_shift(before[other],after[other],y[other])
                direct_scores=score(after[own].cpu().numpy(),heads[:33]);transport_scores=score(shifted.cpu().numpy(),heads[:33])
                seed=config['integration_seed']+spec['seed_offset']+1009*j
                direct,da=gaussian_scores(torch.as_tensor(direct_scores,device='cuda'),j,seed,budget)
                transported,ta=gaussian_scores(torch.as_tensor(transport_scores,device='cuda'),j,seed,budget)
                meta_scores=score(after[meta[y[meta]==j]].cpu().numpy(),heads[:33])
                conditional[str(c)]=dict(fit_images=len(own),meta_images=len(meta_scores),
                    direct_batches=direct.cpu().tolist(),transported_batches=transported.cpu().tolist(),
                    empirical_fit=(direct_scores.argmax(2)==j).mean(0).tolist(),empirical_meta=(meta_scores.argmax(2)==j).mean(0).tolist(),
                    direct_covariance=da,transport_covariance=ta,target_excluded_from_shift=True,target_was_used_in_encoder_training=True)
            record=dict(dataset=spec['dataset'],task=task,cohort=previous_results['cohort'],seen=seen,current=current,tail=tail,
                covered_classes=covered,excluded_classes=[c for c in seen if c not in covered],coverage=support,
                original_selected=previous_results['selected'],selected=choices,choice_audits=audits,
                empirical_A=emp.tolist(),gaussian_A_batches=g.cpu().tolist(),gaussian_A_covariance=covariance,
                conditional_transport=conditional,zero_head_reconstruction_error=error,
                A_images=len(ai),B_images=len(bi),independent_confirmation=False,
                privileged_development_feedback=True,full_class_protection_testable=len(covered)==len(seen))
            save(target/'SELECTION.json',record)
            torch.save(dict(heads=heads.cpu(),A_indices=ai,B_indices=bi),target/'HEADS.private.pt')
            records[key]=record;prepared.append(key);save(out/'SELECTIONS.json',records)
            del before,after,bank,old,previous,cached,heads,zero,extras,dev,z,A,g,e
        save(out/'SELECT_COMPLETE.json',dict(states=prepared,selected_at=time.time(),B_metrics_evaluated=False))
        status('RUNNING',phase='B_evaluation_barrier')
        while not all((Path(p)/'SELECT_COMPLETE.json').exists() for p in config['barrier_jobs']):
            budget()
            for p in config['barrier_jobs']:
                f=Path(p)/'STATUS.json'
                if f.exists() and json.loads(f.read_text())['status']=='INCOMPLETE':raise RuntimeError('Another preparation failed; B metrics closed')
            time.sleep(1)
        for spec in config['states']:
            key=spec['name'];r=records[key];oldrun=Path(spec['previous_run']);saved=load(out/key/'HEADS.private.pt')
            dev=load(Path(spec['cached'])/'candidate_00/DEVELOPMENT.private.pt' if spec.get('cached') else oldrun/key/'DEVELOPMENT.private.pt')
            scores=score(dev['features'],saved['heads']);labels=np.asarray(dev['labels']);r['evaluation']={}
            prior=json.loads((oldrun/'RESULTS.json').read_text())[key]['development']['fit']
            for split,indices in [('full',np.arange(len(labels))),('A',saved['A_indices']),('B',saved['B_indices'])]:
                table={}
                for i,name in enumerate(HEADS):
                    m=metrics(scores[indices,i],labels[indices],r['seen'],r['current'])
                    table[name]=dict(metrics=m,paired_to_strong=paired(scores[indices,0],scores[indices,i],labels[indices],r['seen']))
                    if split=='full' and i<33:
                        assert {c:(v['n'],v['correct']) for c,v in m['per_class'].items()} == {
                            c:(v['n'],v['correct']) for c,v in prior[name]['metrics']['per_class'].items()}, 'Cached classification mismatch'
                r['evaluation'][split]=table
            r['B_evaluated_after_all_selections']=True;evaluated.append(key);save(out/'RESULTS.json',records)
            status('RUNNING',phase='B_evaluation',active_state=key)
        status('COMPLETE',maximum_solve_residual=maximum_residual,full_head_readouts=len(evaluated)*35*3)
    except BaseException as exc:
        status('INCOMPLETE',error=str(exc));raise


if __name__=='__main__':
    run(json.load(sys.stdin))
