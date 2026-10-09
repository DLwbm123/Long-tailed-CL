"""Current-task held-out head feedback on frozen static-policy encoders."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import core1_competition as competition
import prototype_coherent as native
from diagnose_edge_learning import summarize, paired
from run_multilabel import ApartFeatures
from run_prototype_coherent import common_shift, fit_split
from run_prototype_single import Images, extract, manifests, save


def to_device(bank, device):
    return dict(mu=bank['mu'].to(device), Q=bank['Q'].to(device), n=list(bank['n']),
        components=[{k:v.to(device) if torch.is_tensor(v) else v for k,v in c.items()}
                    for c in bank['components']])


def rebuild(old, before, after, labels, fit, include_meta):
    # Current meta must not influence the old-bank translation either.
    use=torch.arange(len(labels),device=labels.device) if include_meta else fit
    seeds=native.seed_components(before[fit],labels[fit])
    group,difficulty=native.memberships(before[use],labels[use],seeds)
    weights=native.sample_weights(labels[use],group,difficulty,before.new_zeros(len(seeds)))
    shifted=native.translate(old,common_shift(before[use],after[use],labels[use]))
    return native.append(shifted,after[use],labels[use],group,weights), group


def reliability(features, identities):
    groups=sorted(set(identities)); mean=features.mean(0)
    residual=features-mean
    g=len(groups)
    # Cluster-robust trace variance of the image-weighted mean; not a confidence guarantee.
    variance=None if g<2 else float(g/(g-1)*sum(
        np.square(residual[np.asarray(identities)==key].sum(0)).sum() for key in groups)/len(features)**2)
    return dict(images=len(features), identity_groups=g, centroid_variance_trace=variance,
        relative_centroid_standard_error=None if variance is None else float(np.sqrt(variance)/max(np.linalg.norm(mean),1e-12)))


def run(config):
    if config['dataset'] not in ('ISIC','HK'):
        raise ValueError('Only medical development diagnostics are authorized')
    source=Path(config['run']); original=json.loads((source/'INPUT.private.json').read_text())
    assert original['method']=='static_pc' and original['evaluation_split']=='development_validation'
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=False)
    started,cpu=time.monotonic(),time.process_time(); completed=[]
    def budget():
        if time.time()>=config['original_deadline'] or time.monotonic()-started>=config['max_wall_seconds']:
            raise TimeoutError('Frozen wall deadline exceeded')
    def status(state,**values):
        save(output/'STATUS.json',dict(status=state,completed_states=completed,
            elapsed_seconds=time.monotonic()-started,cpu_process_seconds=time.process_time()-cpu,
            optimizer_updates=0,policy_updates=0,test_accessed=False,**values))
    try:
        status('RUNNING',phase='initialize'); torch.set_num_threads(4)
        seed=original['seed']; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        encoder=ApartFeatures(original['legacy_repo'],original['weight'],len(original['order']),'cuda:0',seed)
        initial={k:p.detach().cpu().clone() for k,p in encoder.named_parameters() if p.requires_grad}
        encoder.eval().requires_grad_(False)
        train,validation=manifests(original)
        from run_medical_v2 import transform
        def get(rows,task):
            loader=DataLoader(Images(rows,original['images'],transform(False),seed+task*100003),
                batch_size=64,num_workers=4,multiprocessing_context='spawn',
                generator=torch.Generator().manual_seed(seed+task*2003))
            z,y=extract(encoder,loader,budget)
            assert np.array_equal(y,np.asarray([r['label'] for r in rows]))
            return z,y
        expected=json.loads((source/'metrics.json').read_text())
        result=dict(dataset=config['dataset'],states={},optimizer_updates=0,policy_updates=0,test_accessed=False,
            scope='Current-task meta excluded from rebuilt fit-only bank, shift and head; frozen static-policy encoder',
            independent_confirmation=False,old_meta_remains_in_historical_bank=True,
            original_meta_previously_reported=True,development_previously_used=True,
            feedback_rule='Choose native iff current meta macro square risk improves over static_pc by >1e-8; otherwise static_pc')
        for task in config['stages']:
            budget(); status('RUNNING',phase='teacher_features',task=task)
            state=torch.load(source/f'stage_{task}.pt',map_location='cpu',weights_only=False)
            seen=state['seen']; old_count=sum(original['task_sizes'][:task-1]); new=seen[old_count:]
            rows=[r for r in train if r['label'] in new]
            fi,mi=fit_split(rows,original['split_seed'])
            assert not ({rows[i]['identity_component'] for i in fi}&{rows[i]['identity_component'] for i in mi})
            assert not ({rows[i]['identity_component'] for i in mi}&
                        {r['identity_component'] for r in train if r['label'] in seen[:old_count]})
            if task==1:
                encoder.load_state_dict(initial,strict=False); old=native.empty(encoder.dim,'cuda')
            else:
                previous=torch.load(source/f'stage_{task-1}.pt',map_location='cpu',weights_only=False)
                encoder.load_state_dict(previous['adapter'],strict=False); old=to_device(previous['bank'],'cuda')
                del previous
            before,_=get(rows,task)
            encoder.load_state_dict(state['adapter'],strict=False)
            assert not any(p.requires_grad for p in encoder.parameters())
            status('RUNNING',phase='current_features',task=task)
            after,raw_y=get(rows,task)
            y=torch.tensor([seen.index(int(c)) for c in raw_y],device='cuda')
            fit=torch.tensor(fi,device='cuda'); before=torch.as_tensor(before,device='cuda',dtype=torch.float64)
            current=torch.as_tensor(after,device='cuda',dtype=torch.float64)
            heads={}; audits={}; bank_errors={}; support={}
            for partition,include in [('all',True),('fit_only',False)]:
                bank,groups=rebuild(old,before,current,y,fit,include)
                w,metric=native.head(bank,1.)
                pc,audit=competition.bank_head(bank,competition.competition(bank,w),regularizer=metric)
                heads[partition+'_native']=w.float().cpu().numpy()
                heads[partition+'_static_pc']=pc.float().cpu().numpy(); audits[partition]=audit
                if include:
                    saved=to_device(state['bank'],'cuda')
                    bank_errors={k:float((bank[k]-saved[k]).abs().max()) for k in ('mu','Q')}
                    assert all(torch.allclose(bank[k],saved[k],atol=1e-7,rtol=1e-5) for k in ('mu','Q'))
                    assert bank['n']==saved['n'] and len(bank['components'])==len(saved['components'])
                    assert np.allclose(heads['all_static_pc'],state['head'].numpy(),atol=1e-6,rtol=1e-5)
                    del saved
                else:
                    for i,c in enumerate(new,old_count):
                        indices=[j for j in fi if rows[j]['label']==c]
                        local=np.flatnonzero(y[fit].cpu().numpy()==i)
                        support[str(c)]=reliability(after[indices],[rows[j]['identity_component'] for j in indices])
                        support[str(c)]['components']=[dict(images=int((groups[local]==g).sum()),
                            identity_groups=len({rows[fi[int(j)]]['identity_component'] for j in local
                                                 if int(groups[j])==g})) for g in groups[local].unique().tolist()]
                del bank,w,metric,pc
            views={}
            for name,indices in [('fit_current',fi),('meta_current',mi)]:
                scores={k:after[indices]@w for k,w in heads.items()}
                ids=[rows[j]['identity_component'] for j in indices]
                views[name]=dict(heads={k:summarize(s,raw_y[indices],seen,ids) for k,s in scores.items()},
                    native_vs_static={p:paired(scores[p+'_static_pc'],scores[p+'_native'],raw_y[indices],seen)
                                      for p in ('all','fit_only')})
            status('RUNNING',phase='development_features',task=task)
            val=[r for r in validation if r['label'] in seen]; z,v_y=get(val,task)
            scores={k:z@w for k,w in heads.items()}
            assert np.array_equal(scores['all_static_pc'].argmax(1),(z@state['head'].numpy()).argmax(1))
            for name,keep in [('development_all',np.ones(len(val),dtype=bool)),
                              ('development_current',np.isin(v_y,new))]:
                ids=[val[j]['identity_component'] for j in np.flatnonzero(keep)]
                views[name]=dict(heads={k:summarize(s[keep],v_y[keep],seen,ids) for k,s in scores.items()},
                    native_vs_static={p:paired(scores[p+'_static_pc'][keep],scores[p+'_native'][keep],v_y[keep],seen)
                                      for p in ('all','fit_only')})
            observed=views['development_all']['heads']['all_static_pc']['per_class']
            for c,v in observed.items():
                assert v['n']==expected['stages'][task-1]['per_class_n'][c]
                assert abs(v['correct']/v['n']-expected['stages'][task-1]['per_class_recall'][c])<1e-12
            meta=views['meta_current']['heads']
            selected='native' if meta['fit_only_static_pc']['macro_square_risk']-meta['fit_only_native']['macro_square_risk']>1e-8 else 'static_pc'
            result['states'][str(task)]=dict(task=task,seen=seen,current_classes=new,
                bank_reconstruction_max_error=bank_errors,original_development_reproduced=True,
                fit_meta_identity_disjoint=True,solves=audits,splits=views,
                current_fit_support=support,meta_selected_head=selected)
            completed.append(task);save(output/'RESULTS.json',result)
            del state,old,heads,before,current,z,scores
            torch.cuda.empty_cache()
        status('COMPLETE',ended=time.time(),peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status('INCOMPLETE',error=str(exc));raise


def self_check():
    torch.manual_seed(7); before=torch.randn(12,4,dtype=torch.float64)
    after=before+.1*torch.randn_like(before); labels=torch.tensor([1]*6+[2]*6);fit=torch.tensor([0,1,2,3,6,7,8,9])
    old=native.append(native.empty(4),before[:2],torch.zeros(2,dtype=torch.long),torch.zeros(2,dtype=torch.long),torch.full((2,),.5,dtype=torch.float64))
    a,_=rebuild(old,before,after,labels,fit,False)
    altered_before=before.clone();altered_after=after.clone()
    altered_before[[4,5,10,11]]+=100;altered_after[[4,5,10,11]]-=100
    b,_=rebuild(old,altered_before,altered_after,labels,fit,False)
    assert torch.equal(a['mu'],b['mu']) and torch.equal(a['Q'],b['Q'])
    assert torch.equal(native.head(a)[0],native.head(b)[0])
    all_bank,_=rebuild(old,before,after,labels,fit,True)
    assert a['n']==[2,4,4] and all_bank['n']==[2,6,6]
    for c,d in zip(a['components'],b['components']):assert torch.equal(c['center'],d['center'])
    r=reliability(np.array([[1.,0.],[3.,0.]]),['a','b'])
    assert r['centroid_variance_trace']==1 and r['relative_centroid_standard_error']==.5
    assert reliability(np.ones((2,3)),['a','a'])['centroid_variance_trace'] is None
    print('PASS: excluded meta cannot change new/old moments, prototypes or head; sample counts and cluster uncertainty')


if __name__=='__main__':
    config=json.load(sys.stdin)
    if config.get('self_check'):self_check()
    else:run(config)
