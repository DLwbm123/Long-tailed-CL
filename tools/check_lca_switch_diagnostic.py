"""CPU synthetic admission, dense-solve, signed counting and complete-report checks."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from lca_switch_diagnostic import heads,summarize_scores,validate_scope,read_val,JOBS,VERSION
import lca_supported as lca
import core1_competition as competition


def check(lock,temporary_root):
    begun=time.monotonic();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.manual_seed(165)
    d,k=6,4;z=torch.randn(k,9,d,dtype=torch.float64)
    bank=dict(n=torch.full((k,),9),mu=z.mean(1),Q=z.transpose(1,2)@z/9)
    pairs=(torch.ones(k,k,dtype=torch.float64)-torch.eye(k,dtype=torch.float64))/(k-1)
    R=torch.eye(d,dtype=torch.float64);v=torch.randn(d,2,dtype=torch.float64);S=.1*v@v.T
    M=lca.old_margin_metric(k,3,R);off,on,mechanical=heads(bank,pairs,R,S,M)
    q=bank['Q']/k;mu=bank['mu']/k;lap,targets=competition.laplacians(pairs)
    A=q.sum(0)+.001*R;rhs=mu.T+.5*mu.T@targets
    def operator(w):return A@w+.5*torch.einsum('cdk,ckl->dl',q@w,lap)+S@w@M
    eye=torch.eye(d*k,dtype=torch.float64)
    dense=torch.stack([operator(col.reshape(d,k)).flatten() for col in eye],1)
    direct=torch.linalg.solve(dense,rhs.flatten()).reshape(d,k)
    assert torch.allclose(on,direct,atol=1e-8,rtol=1e-8)
    w0,w1,a=heads(bank,pairs,R,torch.zeros_like(S),M)
    assert torch.equal(w0,w1) and a['delta_head_norm']==0
    assert mechanical['new_direct_force_norm']==0 and mechanical['delta_new_head_norm']>0
    heads(bank,pairs,R,S,M,on,on.float())
    try:heads(bank,pairs,R,S,M,on+1,on.float())
    except ValueError:pass
    else:raise AssertionError('Bad saved head accepted')
    data=dict(order=[10,30,20,40],task_sizes=[2,2],train_counts={str(c):9 for c in (10,30,20,40)},
        val_counts={'10':2,'30':2,'20':2,'40':2})
    y=np.array([10,10,30,30,20,20,40,40]);po=np.array([0,1,0,1,0,2,1,3]);pn=np.array([2,0,3,1,2,0,3,3])
    a=summarize_scores(np.eye(4,dtype=np.float32)[po],np.eye(4,dtype=np.float32)[pn],y,data)
    assert a['paired']['all']['rescued']==3 and a['paired']['all']['harmed']==2
    assert a['paired']['all']['wrong_to_different_wrong']==1 and a['paired']['all']['changed']==6
    assert a['paired']['all']['net_correct']==1
    assert a['confusion']['OFF']==dict(old_correct=2,old_to_new=0,old_to_old_wrong=2,new_correct=2,new_to_old=2,new_to_new_wrong=0)
    with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
        root=Path(folder);ds='ISIC';data=lock['data'][ds]
        old=dict(name=ds,job_name=JOBS[0],order=data['order'],task_sizes=data['task_sizes'],seed=74002,
            split_seed=74002,method='static_pc',source_commit=lock['checkpoint_source_commit'],
            lca_version='supported_old_margin_v1',lca_map_ridge=.001,lca_curvature_cap=.1,
            expected_steps=data['training_updates'],module='base',evaluation_split='development_validation',
            output=str(root/'parent/runs'/JOBS[0]))
        c=dict(job_name=JOBS[0],version=VERSION,modes=['OFF','ON'],train_allowed=False,test_accessed=False,
            batch_size=64,workers=0,task=len(data['task_sizes']),parent_root=str(root/'parent'),
            root=str(root/'new'),output=str(root/'new/runs'/JOBS[0]))
        validate_scope(c,lock,old)
        for key,value in [('modes',['ON']),('task',1),('train_allowed',True),('test_accessed',True),('batch_size',1)]:
            bad=copy.deepcopy(c);bad[key]=value
            try:validate_scope(bad,lock,old)
            except ValueError:pass
            else:raise AssertionError('Unauthorized diagnostic accepted')
        bad=copy.deepcopy(old);bad['source_commit']='wrong'
        try:validate_scope(c,lock,bad)
        except ValueError:pass
        else:raise AssertionError('Wrong source accepted')
        # Only val.csv exists: scope validation must not require train or test files.
        manifest=root/'manifest';manifest.mkdir();old['manifest']=str(manifest)
        import csv
        with (manifest/'val.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['relative_path','original_label','split','identity_component']);writer.writeheader()
            for label,count in data['val_counts'].items():
                for i in range(count):writer.writerow(dict(relative_path=f'{label}_{i}.png',original_label=label,split='val',identity_component=f'{label}_{i}'))
        assert len(read_val(old,data))==sum(data['val_counts'].values())
        from collect_lca_switch_diagnostic import report
        fixture_lock=copy.deepcopy(lock);runs={}
        for name in JOBS:
            ds=name.split('_')[0];data=lock['data'][ds];order=data['order']
            labels=np.concatenate([np.full(data['val_counts'][str(c)],c) for c in order])
            target=np.array([order.index(int(c)) for c in labels]);off_s=np.eye(len(order),dtype=np.float32)[target]
            on_s=off_s.copy();on_s[0]=np.roll(on_s[0],1)
            result=summarize_scores(off_s,on_s,labels,data)
            fixture_lock['on_references'][name]=result['metrics']['ON']
            result.update(status={'status':'COMPLETE'},dataset=ds,on_published_parity=True,encoder_state_unchanged=True,
                checkpoint_metadata_verified=True,optimizer_updates=0,policy_updates=0,train_accessed=False,test_accessed=False,
                checkpoint_source_commit=lock['checkpoint_source_commit'],diagnostic_source_commit='CPU_fixture',
                mechanical=mechanical,validation_image_exposures=len(labels))
            runs[name]=result
        snapshot=dict(program=dict(status='COMPLETE',active={},failures=[],costs=[dict(job=n,gpu=i%2,returncode=0,residence_seconds=1.) for i,n in enumerate(JOBS)],whole_round_wall_seconds=2.),
            runs=runs,lock=fixture_lock,source_commit='CPU_fixture',round_name='CPU_fixture',closed_round_cost_ledger={},cpu_checks={})
        a=report(snapshot,root/'report');assert a['status']=='PASS' and a['validation_image_exposures']==3986
        bad=copy.deepcopy(snapshot);bad['runs'][JOBS[0]]['metrics']['OFF']['per_class_n']['4']=0
        try:report(bad,root/'bad')
        except AssertionError:pass
        else:raise AssertionError('Bad denominator accepted')
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-begun,gpu_processes=0,medical_images_opened=0,
        optimizer_updates=0,test_accessed=False,dense_solve_and_zero_reduction=True,checkpoint_rebuild_rejection=True,
        new_direct_force_zero_does_not_imply_new_head_unchanged=True,paired_positive_negative_counts=True,
        old_new_confusion_conservation=True,scope_and_source_rejection=True,val_only_manifest=True,
        complete_four_state_report_and_bad_denominator_rejection=True)


if __name__=='__main__':
    c=json.loads(Path(os.environ['Q165_CPU_CONFIG']).read_text())
    result=check(json.loads(Path(c['lock']).read_text()),c['temporary_root'])
    Path(c['receipt']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
