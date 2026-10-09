"""Seven locked readout mechanisms and training-only point/stable selectors."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import core1_competition as competition
import edge_competition as edge
import prototype_coherent as native
from decision_probability import probabilities
from diagnose_holdout_feedback import rebuild, to_device
from diagnose_edge_learning import paired
from scarcity_factor import KEYS, allocations, select
from run_feedback_audit import metrics
from run_prototype_coherent import fit_split
from run_prototype_single import manifests, save


def score(z, heads):
    return np.stack([np.asarray(z) @ w.float().cpu().numpy() for w in heads],axis=1)


@torch.no_grad()
def run(config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();cpu=time.process_time()
    prepared=[];evaluated=[];maximum_residual=0.;torch.set_num_threads(4)
    save(out/'INPUT.private.json',config)
    def budget():
        if time.time() >= config['original_deadline'] or time.monotonic()-started >= config['max_wall_seconds']:
            raise TimeoutError('Frozen budget exhausted')
    def status(state,**extra):
        save(out/'STATUS.json',dict(status=state,prepared_states=prepared,evaluated_states=evaluated,
            elapsed_seconds=time.monotonic()-started,cpu_process_seconds=time.process_time()-cpu,
            image_rows=0,adapter_updates=0,policy_updates=0,test_accessed=False,**extra))
    def load(path):
        budget();return torch.load(path,map_location='cpu',weights_only=False)
    try:
        status('RUNNING',phase='prepare');records={}
        for spec in config['states']:
            key=spec['name'];target=out/key;target.mkdir();oldrun=Path(spec['previous_run']);info=Path(spec['information_run'])
            original=json.loads((Path(spec['source'])/'INPUT.private.json').read_text())
            assert original['method']=='static_pc' and original['evaluation_split']=='development_validation'
            previous_results=json.loads((oldrun/'RESULTS.json').read_text())[key]
            seen=previous_results['seen'];current=previous_results['current'];tail=previous_results['tail'];old_count=len(seen)-len(current)
            cached=load(oldrun/key/'HEADS.private.pt');before_record=load(info/key/'BEFORE.private.pt')
            train,_=manifests(original);rows=[r for r in train if r['label'] in current]
            fi,mi=fit_split(rows,original['split_seed']);assert fi==cached['fit_indices'] and mi==cached['meta_indices']
            raw_y=np.asarray(cached['labels']);assert np.array_equal(raw_y,before_record['labels'])
            assert np.array_equal(raw_y,np.asarray([r['label'] for r in rows]))
            assert not {rows[i]['identity_component'] for i in fi}&{rows[i]['identity_component'] for i in mi}
            before=before_record['features'].cuda().double();after=cached['current_features'].cuda().double()
            previous=load(Path(spec['source'])/f"stage_{spec['task']-1}.pt");old=to_device(previous['bank'],'cuda')
            y=torch.tensor([seen.index(int(c)) for c in raw_y],device='cuda');fit=torch.tensor(fi,device='cuda');meta=torch.tensor(mi,device='cuda')
            bank,_=rebuild(old,before,after,y,fit,False)
            initial,metric=native.head(bank,1.);base=competition.competition(bank,initial)
            reference,audit=competition.bank_head(bank,base,regularizer=metric);maximum_residual=max(maximum_residual,audit['relative_residual'])
            features=edge.descriptors(bank,reference,old_count);pairs=allocations(base,features[:,:,6]);heads=[];alloc_audits={}
            for name in KEYS:
                budget();w,audit=competition.bank_head(bank,pairs[name],regularizer=metric)
                maximum_residual=max(maximum_residual,audit['relative_residual']);heads.append(w.float().double())
                alloc_audits[name]=dict(relative_total_l1=float((pairs[name]-base).abs().sum()/base.sum()),
                    row_mass=pairs[name].sum(1).cpu().tolist(),base_row_mass=base.sum(1).cpu().tolist(),
                    minimum_base_ratio=float((pairs[name][base>0]/base[base>0]).min()),solve=audit)
            heads=torch.stack(heads);errors={}
            for ours,index in [('zero',0),('full_plus',13),('full_minus',14)]:
                w=heads[KEYS.index(ours)].float().cpu();saved=cached['fit_heads'][index].float()
                errors[ours]=float((w-saved).abs().max());assert torch.allclose(w,saved,atol=1e-6,rtol=1e-5),errors
            meta_scores=score(after[meta].cpu().numpy(),heads)
            correctness=torch.as_tensor(meta_scores.argmax(2)==y[meta].cpu().numpy()[:,None])
            status('RUNNING',phase='train_side_feedback',active_state=key)
            probability,covariance=probabilities(bank,heads,config['integration_seed']+spec['seed_offset'],budget)
            selected,choice_audits,stability,meta_acc,identity_records,support=select(probability,correctness,
                y[meta].cpu().tolist(),[rows[i]['identity_component'] for i in mi],old_count,[seen.index(c) for c in tail])
            record=dict(dataset=spec['dataset'],task=spec['task'],cohort=previous_results['cohort'],seen=seen,current=current,tail=tail,
                selected=selected,primary_method='within_stable',choice_audits=choice_audits,stability=stability,
                gaussian_batches=probability.cpu().tolist(),meta_accuracy=meta_acc.cpu().tolist(),
                meta_support={str(seen[c]):dict(identities=n,images=sum(raw_y[i]==seen[c] for i in mi)) for c,n in support.items()},
                allocations=alloc_audits,existing_head_reconstruction_errors=errors,gaussian_covariance=covariance,
                independent_confirmation=False,selection_used_development_labels=False,old_uncertainty_not_certified=True,
                identity_deletion_scope='current meta only; no population confidence guarantee')
            save(target/'META_IDENTITIES.private.json',identity_records)
            torch.save(dict(heads=heads.cpu()),target/'HEADS.private.pt');save(target/'SELECTION.json',record)
            records[key]=record;prepared.append(key);save(out/'SELECTIONS.json',records)
            del cached,before_record,before,after,previous,old,bank,heads,probability,meta_scores,correctness,pairs
        save(out/'SELECT_COMPLETE.json',dict(states=prepared,selected_at=time.time(),development_metrics_evaluated=False))
        status('RUNNING',phase='development_barrier')
        while not all((Path(p)/'SELECT_COMPLETE.json').exists() for p in config['barrier_jobs']):
            budget()
            for p in config['barrier_jobs']:
                f=Path(p)/'STATUS.json'
                if f.exists() and json.loads(f.read_text())['status']=='INCOMPLETE':raise RuntimeError('Preparation failed; evaluation closed')
            time.sleep(1)
        for spec in config['states']:
            key=spec['name'];r=records[key];oldrun=Path(spec['previous_run']);info=Path(spec['information_run'])
            heads=load(out/key/'HEADS.private.pt')['heads'];split=load(info/key/'HEADS.private.pt')
            prior=json.loads((info/'RESULTS.json').read_text())[key]
            dev=load(Path(spec['cached'])/'candidate_00/DEVELOPMENT.private.pt' if spec.get('cached') else oldrun/key/'DEVELOPMENT.private.pt')
            labels=np.asarray(dev['labels']);scores=score(dev['features'],heads)
            original=json.loads((Path(spec['source'])/'INPUT.private.json').read_text());_,val=manifests(original)
            assert np.array_equal(labels,np.asarray([v['label'] for v in val if v['label'] in r['seen']]))
            r.update(covered_classes=prior['covered_classes'],excluded_classes=prior['excluded_classes'],coverage=prior['coverage'],evaluation={})
            for partition,indices in [('full',np.arange(len(labels))),('A',split['A_indices']),('B',split['B_indices'])]:
                table={}
                for i,name in enumerate(KEYS):
                    m=metrics(scores[indices,i],labels[indices],r['seen'],r['current'])
                    table[name]=dict(metrics=m,paired_to_strong=paired(scores[indices,0],scores[indices,i],labels[indices],r['seen']))
                    if name in ('zero','full_plus','full_minus'):
                        oldkey={'zero':'edge_00','full_plus':'edge_13','full_minus':'edge_14'}[name]
                        assert {c:(v['n'],v['correct']) for c,v in m['per_class'].items()} == {
                            c:(v['n'],v['correct']) for c,v in prior['evaluation'][partition][oldkey]['metrics']['per_class'].items()}
                r['evaluation'][partition]=table
            r['development_evaluated_after_all_selections']=True;evaluated.append(key);save(out/'RESULTS.json',records)
            status('RUNNING',phase='development_evaluation',active_state=key)
        status('COMPLETE',maximum_solve_residual=maximum_residual,head_readouts=len(evaluated)*7*3)
    except BaseException as exc:
        status('INCOMPLETE',error=str(exc));raise


if __name__=='__main__':run(json.load(sys.stdin))
