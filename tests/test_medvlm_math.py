import numpy as np
from shared.dual_moment_bank import DualMomentBank
from route_a.spectral_prior_ridge import spectral_prior_ridge, ridge
from tools.medvlm_math import fit,score

def test_fixed_readouts_match_direct_solvers():
    rng=np.random.default_rng(62)
    bank=DualMomentBank(dim_a=4,dim_u=3)
    for c in range(3):
        a=rng.normal(size=(8,4));a/=np.linalg.norm(a,axis=1,keepdims=True)
        u=rng.normal(size=(8,3));u/=np.linalg.norm(u,axis=1,keepdims=True)
        bank.add_class(c,np.concatenate((a,u),axis=1)/np.sqrt(2),task=c+1,component_ids=map(str,range(8)))
    t=np.eye(3);w=fit(bank,t,t,t);S,M,_=bank.class_balanced()
    np.testing.assert_allclose(w['J'],ridge(S/3,M/3,5e-4),rtol=1e-9,atol=1e-10)
    prior=np.vstack((np.zeros((4,3)),np.sqrt(2)*w['scale']*(t-t.mean(1,keepdims=True))))
    ref,_=spectral_prior_ridge(S,M,prior,w['gamma'],lam=5e-4)
    np.testing.assert_allclose(w['R'],ref,rtol=1e-9,atol=1e-10)
    ex=fit(bank,t,t,t,extensions=True)
    ref,_=spectral_prior_ridge(S,M,np.zeros_like(M),w['gamma'],lam=5e-4)
    np.testing.assert_allclose(ex['R-zero-prior'],ref,rtol=1e-9,atol=1e-10)
    for method in ['V','J','J1','Z','L','R','F','F2','L-permute','template-0']:
        assert score(w,method,a,u).shape==(8,3)

def test_complete_report_rejects_zero_gain():
    import tempfile,json
    from pathlib import Path
    from tools.finalize_medvlm import finalize
    from tools.run_nb2_vlm_r1 import _write,_sha,SIZES
    from route_a.run_ct13_real import _stage_metrics,_write_csv
    from tools.medvlm_math import CORE,EXTRA
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);old=root/'old';(old/'private').mkdir(parents=True);(root/'private').mkdir()
        stages=[];classes=[];digests={}
        for ds,sizes in SIZES.items():
            labels=np.repeat(np.arange(sum(sizes)),2);preds=[];index=[];oldpred=[];oldindex=[]
            for seed in (1993,1994,1995):
                position=0
                for task,size in enumerate(sizes,1):
                    current=list(range(position,position+size));position+=size;seen=list(range(position))
                    scores=(labels[:,None]==np.asarray(seen)).astype(float)
                    metric,cr,pred=_stage_metrics(scores,labels,seen,current,set(range(sum(sizes)-2,sum(sizes))))
                    for m in ['P','F','F2']+[tag+'.'+m for tag in ('G','B') for m in CORE+EXTRA]:
                        key={'dataset':ds,'seed':seed,'task':task,'method':m};stages.append({**key,**metric});classes.extend([{**key,**r} for r in cr]);index.append(key);preds.append(pred)
                    for m in ['P','A0','A1','A2']:
                        oldindex.append({'seed':seed,'task':task,'method':m});oldpred.append(pred)
            path=root/'private'/f'{ds}_PREDICTIONS.npz'
            np.savez_compressed(path,labels=labels,components=np.arange(len(labels)).astype(str),predictions=preds,index=np.asarray([json.dumps(x) for x in index]))
            digests[ds]=_sha(path)
            np.savez_compressed(old/'private'/f'{ds}_VAL_PREDICTIONS.npz',labels=labels,predictions=oldpred,index=np.asarray([json.dumps(x) for x in oldindex]))
        _write_csv(root/'stage_metrics.csv',stages);_write_csv(root/'class_metrics.csv',classes);_write(root/'PREDICTIONS_LOCK.json',{'files':digests})
        report=finalize(root,{'models':['G','B'],'generic_run':str(old)})
        assert report['stage_rows']==1215 and report['class_rows']==12393
        assert report['primary_utility_gate']=='FAIL'
        assert all(g['final_ba_gain_pp_by_seed']==[0,0,0] for g in report['gates'])

if __name__=='__main__':
    test_fixed_readouts_match_direct_solvers()
    test_complete_report_rejects_zero_gain()
    print('MEDVLM_MATH_AND_REPORT_TESTS_PASS')
