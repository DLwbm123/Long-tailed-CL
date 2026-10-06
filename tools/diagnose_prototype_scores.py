"""Forward-only paired score diagnostics on already-used validation; aggregates only."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from run_prototype_single import Images, manifests, extract, save
from run_multilabel import ApartFeatures


def summarize(scores, labels, seen):
    index = np.array([seen.index(int(c)) for c in labels]); prediction=scores.argmax(1)
    other=scores.copy();other[np.arange(len(labels)),index]=-np.inf
    margin=scores[np.arange(len(labels)),index]-other.max(1)
    confusion=np.zeros((len(seen),len(seen)),dtype=int);np.add.at(confusion,(index,prediction),1)
    per_class={}
    for i,c in enumerate(seen):
        keep=index==i;group=np.arange(len(seen)-2,len(seen)) if i>=len(seen)-2 else np.arange(len(seen)-2)
        opposite=np.arange(len(seen)-2) if i>=len(seen)-2 else np.arange(len(seen)-2,len(seen))
        per_class[str(c)]=dict(n=int(keep.sum()),correct=int((prediction[keep]==i).sum()),
            restricted_group_correct=int((group[scores[keep][:,group].argmax(1)]==i).sum()),
            mean_true_score=float(scores[keep,i].mean()),mean_margin=float(margin[keep].mean()),
            median_margin=float(np.median(margin[keep])),
            mean_opposite_group_margin=float((scores[keep,i]-scores[keep][:,opposite].max(1)).mean()))
    return dict(class_order=seen,confusion=confusion.tolist(),per_class=per_class,
        balanced_accuracy=float(np.mean([v['correct']/v['n'] for v in per_class.values()])))


def run(config):
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def budget():
        if time.monotonic()-start>config['max_wall_seconds']:raise TimeoutError('Diagnostic budget exceeded')
    save(root/'STATUS.json',dict(status='RUNNING',started=time.time(),optimizer_updates=0))
    try:
        torch.set_num_threads(4);seed=config['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        encoder=ApartFeatures(config['legacy_repo'],config['weight'],len(config['order']),'cuda:0',seed)
        from run_medical_v2 import transform
        _,rows=manifests(config)
        loader=DataLoader(Images(rows,config['images'],transform(False),seed+4*100003),batch_size=config['batch_size'],num_workers=4,
            multiprocessing_context='spawn',persistent_workers=True,generator=torch.Generator().manual_seed(seed+4*2003))
        result={'models':{},'optimizer_updates':0,'test_accessed':False,'independent_confirmation':False};all_scores={}
        for name,path in config['models'].items():
            state=torch.load(Path(path)/'stage_4.pt',map_location='cpu',weights_only=False)
            encoder.load_state_dict(state['adapter'],strict=False);encoder.eval();z,y=extract(encoder,loader,budget)
            scores=z@state['head'].numpy();summary=summarize(scores,y,state['seen'])
            expected=json.loads((Path(path)/'metrics.json').read_text())
            assert abs(summary['balanced_accuracy']-expected['final_balanced_accuracy'])<1e-12
            for c,v in summary['per_class'].items():assert abs(v['correct']/v['n']-expected['stages'][-1]['per_class_recall'][c])<1e-12
            result['models'][name]=summary;all_scores[name]=scores
        a,b=all_scores['A'],all_scores['B'];seen=state['seen'];truth=np.array([seen.index(int(c)) for c in y]);pa=a.argmax(1);pb=b.argmax(1)
        changes={}
        for i,c in enumerate(seen):
            selected=truth==i;hurt=selected&(pa==i)&(pb!=i);helped=selected&(pa!=i)&(pb==i)
            changes[str(c)]=dict(n=int(selected.sum()),hurt=int(hurt.sum()),helped=int(helped.sum()),
                hurt_destinations={str(seen[j]):int((hurt&(pb==j)).sum()) for j in range(len(seen)) if (hurt&(pb==j)).any()},
                helped_previous_predictions={str(seen[j]):int((helped&(pa==j)).sum()) for j in range(len(seen)) if (helped&(pa==j)).any()})
        result['paired_changes']=changes;save(root/'RESULTS.json',result)
        save(root/'STATUS.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-start,ended=time.time(),optimizer_updates=0,validation_n=len(y)))
        print(json.dumps(result))
    except BaseException as exc:
        save(root/'STATUS.json',dict(status='INCOMPLETE',elapsed_seconds=time.monotonic()-start,error=str(exc),optimizer_updates=0));raise

def run_moments(config):
    """Last-task algebraic intervention; hybrid moments are diagnostics, not trained methods."""
    from prototype_analytic import append, transport as global_transport, ridge
    from prototype_graph import transport as graph_transport
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def budget():
        if time.monotonic()-start>config['max_wall_seconds']:raise TimeoutError('Diagnostic budget exceeded')
    save(root/'STATUS.json',dict(status='RUNNING',started=time.time(),optimizer_updates=0))
    try:
        torch.set_num_threads(4);seed=config['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        encoder=ApartFeatures(config['legacy_repo'],config['weight'],len(config['order']),'cuda:0',seed)
        from run_medical_v2 import transform
        train,val=manifests(config);classes=config['order'][-2:]
        def loader(rows):
            return DataLoader(Images(rows,config['images'],transform(False),seed+4*100003),batch_size=config['batch_size'],num_workers=4,
                multiprocessing_context='spawn',persistent_workers=True,generator=torch.Generator().manual_seed(seed+4*2003))
        current=loader([r for r in train if r['label'] in classes])
        path=Path(config['models']['B']);previous=torch.load(path/'stage_3.pt',map_location='cpu',weights_only=False)
        final=torch.load(path/'stage_4.pt',map_location='cpu',weights_only=False)
        encoder.load_state_dict(previous['adapter'],strict=False);encoder.eval();before,y=extract(encoder,current,budget)
        encoder.load_state_dict(final['adapter'],strict=False);after,after_y=extract(encoder,current,budget);assert np.array_equal(y,after_y)
        banks={}
        for name,fn in [('global',global_transport),('graph',graph_transport)]:
            shifted,_=fn(previous['bank'],before,after,y,classes)
            banks[name]=append(shifted,after,y,classes)
        z,vy=extract(encoder,loader(val),budget)
        result=dict(optimizer_updates=0,test_accessed=False,independent_confirmation=False,
            scope='B final encoder and task-3 bank fixed; last-task head only; hybrids are algebraic diagnostics',conditions={})
        for s_name in banks:
            for m_name in banks:
                bank=dict(S=banks[s_name]['S'],mu=banks[m_name]['mu'],n=banks['graph']['n'])
                W,residual=ridge(bank);W=W.astype(np.float32);scores=z@W
                name='S_'+s_name+'__mu_'+m_name
                result['conditions'][name]=dict(summary=summarize(scores,vy,final['seen']),ridge_residual=residual)
                if s_name==m_name=='graph':
                    assert np.allclose(W,final['head'].numpy(),rtol=1e-5,atol=1e-6)
                    expected=json.loads((path/'metrics.json').read_text())
                    assert abs(result['conditions'][name]['summary']['balanced_accuracy']-expected['final_balanced_accuracy'])<1e-12
        result['graph_reconstruction_matches_saved_head']=True
        save(root/'RESULTS.json',result);save(root/'STATUS.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-start,ended=time.time(),optimizer_updates=0,validation_n=len(vy)))
        print(json.dumps({k:v['summary']['balanced_accuracy'] for k,v in result['conditions'].items()}))
    except BaseException as exc:
        save(root/'STATUS.json',dict(status='INCOMPLETE',elapsed_seconds=time.monotonic()-start,error=str(exc),optimizer_updates=0));raise


if __name__=='__main__':
    config=json.load(sys.stdin)
    if config.get('self_check'):
        scores=np.array([[1.,0.,2.,0.],[0.,1.,0.,2.],[2.,0.,1.,0.],[0.,2.,0.,1.]])
        r=summarize(scores,np.arange(4),list(range(4)))
        assert r['balanced_accuracy']==0 and all(v['restricted_group_correct']==1 for v in r['per_class'].values())
        assert all(v['mean_opposite_group_margin']==-1 for v in r['per_class'].values());print('PASS: cross-group versus within-group diagnostic')
    elif config.get('moment_intervention'):run_moments(config)
    else:run(config)
