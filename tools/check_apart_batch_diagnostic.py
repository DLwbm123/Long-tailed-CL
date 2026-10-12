"""CPU-only scope, paired-index, w summary and exact-author routing logic checks."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import types

import numpy as np
import torch

from apart_batch_diagnostic import (validate, ordered_rows, paired, describe_w,
    MODES, DIAGNOSTIC_VERSION)
from run_external_medical import PINNED, stage_report


def check(lock, temporary_root):
    begun=time.monotonic();assert os.environ['CUDA_VISIBLE_DEVICES']==''
    torch.set_num_threads(1)
    source=ast.parse((PINNED/'apart_vit.py').read_text())
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='AdapterPool')
    ns={'torch':torch}
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('forward','l2_normalize')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<author_route_fixture>','exec'),ns)
    pool=types.SimpleNamespace(embedding_key='cls',prompt_key=torch.eye(5),top_k=1,pool_size=5,batchwise_prompt=True)
    for name in ('forward','l2_normalize'):setattr(pool,name,types.MethodType(ns[name],pool))
    q=torch.eye(5)[0];mix=torch.stack([q,torch.eye(5)[1],torch.eye(5)[1]])
    assert pool.forward(torch.zeros(1,1,5),cls_features=q[None,:])['prompt_idx'][0].item()==0
    assert pool.forward(torch.zeros(3,1,5),cls_features=mix)['prompt_idx'][0].item()==1
    pool.batchwise_prompt=False
    assert pool.forward(torch.zeros(3,1,5),cls_features=mix)['prompt_idx'].flatten().tolist()==[0,1,1]
    rows=[dict(target=t,canonical_index=i) for i,t in enumerate([2,0,1,0])]
    assert [r['canonical_index'] for r in ordered_rows(rows,'class_concat')]==[1,3,2,0]
    assert ordered_rows(rows,'mixed_batch')==ordered_rows(rows,'individual_routing')==rows
    y=np.array([0,0,1,1]);a=np.array([0,1,0,1]);b=np.array([1,0,1,1]);routes=np.zeros((4,2),dtype=int)
    p=paired(a,b,y,routes,routes+1,[7,4],[7,4])['all']
    assert p==dict(n=4,predictions_changed=3,rescued=2,harmed=1,both_correct=1,both_wrong=0,either_pool_route_changed=4)
    assert describe_w([0.,0.,.01])['near_zero_screen']
    assert not describe_w([.1,.5,.9])['near_zero_screen']
    for bad in ([],[float('nan')],[-.1],[1.1]):
        try:describe_w(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid w accepted')
    stage=stage_report(1,[7,4],y,b,None,[7,4],{'7':3,'4':100})
    assert stage['per_class_n']=={'7':2,'4':2} and stage['balanced_accuracy']==.75
    with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
        root=Path(folder);parent=root/'parent';parent.mkdir();r=root/'new'
        data=lock['data']['ISIC']
        old=dict(dataset='ISIC',method='APART_full',source_commit=lock['checkpoint_source_commit'],
            version='air_apart_medical_native_v1',seed=74002,order=data['order'],task_sizes=data['task_sizes'],
            expected_steps=data['APART_total_updates'],test_accessed=False,evaluation_split='development_validation',
            output=str(parent/'runs/ISIC_APART_full'))
        (parent/'CONFIG.private.json').write_text(json.dumps(dict(jobs=[old])))
        (parent/'PROGRAM_STATE.json').write_text(json.dumps(dict(status='COMPLETE',active={})))
        path=root/'lock.json';path.write_text(json.dumps(lock))
        config=dict(diagnostic_lock=str(path),original_job=old,parent_root=str(parent),root=str(r),
            output=str(r/'runs/ISIC'),version=DIAGNOSTIC_VERSION,modes=list(MODES),batch_size=48,
            train_allowed=False,test_accessed=False)
        validate(config)
        for key,value in [('train_allowed',True),('test_accessed',True),('batch_size',1),('modes',['mixed_batch'])]:
            bad=copy.deepcopy(config);bad[key]=value
            try:validate(bad)
            except ValueError:pass
            else:raise AssertionError('Unauthorized diagnostic accepted')
        bad=copy.deepcopy(config);bad['original_job']['seed']=3
        try:validate(bad)
        except ValueError:pass
        else:raise AssertionError('Changed original checkpoint config accepted')
        (parent/'PROGRAM_STATE.json').write_text(json.dumps(dict(status='RUNNING',active={})))
        try:validate(config)
        except ValueError:pass
        else:raise AssertionError('Incomplete parent accepted')
        from collect_apart_batch_diagnostic import report
        original=json.loads(Path(os.environ['Q157_PUBLIC_ORIGINAL_RESULTS']).read_text())
        runs={}
        for ds in ('ISIC','HK'):
            metrics=original['runs'][ds+'_APART_full']['metrics'];data=lock['data'][ds]
            w_stages=[];pairs=[]
            for st in metrics['stages']:
                w_stages.append(dict(task=st['task'],groups={'all':describe_w([.02]*st['validation_n'])},
                    per_class={str(c):describe_w([.02]*st['per_class_n'][str(c)]) for c in st['seen']}))
                pairs.append(dict(task=st['task'],comparisons={a+'__'+b:{'all':dict(n=st['validation_n'],predictions_changed=0,rescued=0,harmed=0,either_pool_route_changed=0)}
                    for a,b in [('mixed_batch','class_concat'),('mixed_batch','individual_routing'),('class_concat','individual_routing')]}))
            runs[ds]=dict(optimizer_updates=0,train_accessed=False,test_accessed=False,parameters_unchanged=True,
                checkpoint_metadata_all_pass=True,model_source_commit=lock['checkpoint_source_commit'],
                metrics={mode:copy.deepcopy(metrics) for mode in MODES},w_stages=w_stages,paired_stages=pairs,
                validation_image_exposures=3*sum(st['validation_n'] for st in metrics['stages']))
        snapshot=dict(lock=lock,program=dict(status='COMPLETE',active={},failures=[],elapsed_seconds=2,
            costs=[dict(gpu=i,returncode=0,residence_seconds=1) for i in (0,1)]),runs=runs,
            original_results=original,source_commit='CPU_fixture',round_name='CPU_fixture',closed_round_cost_ledger={},cpu_checks={})
        audit=report(snapshot,root/'report_fixture')
        assert audit['status']=='PASS' and audit['validation_image_exposures']==23949
        assert len(json.loads((root/'report_fixture/RESULTS.json').read_text())['rows'])==6
        bad=copy.deepcopy(snapshot);bad['runs']['HK']['metrics']['individual_routing']['stages'][-1]['per_class_n']['5']=2
        try:report(bad,root/'invalid_report')
        except AssertionError:pass
        else:raise AssertionError('Changed class denominator accepted')
    return dict(status='PASS',cpu_pipeline_seconds=time.monotonic()-begun,gpu_processes=0,
        optimizer_updates=0,medical_images_opened=0,test_accessed=False,
        exact_author_batch_vote_counterexample=True,individual_route_batch_independence=True,
        stable_class_concatenation_and_canonical_pairing=True,rescue_harm_counts=True,
        fixed_w_screen_and_invalid_values=True,checkpoint_scope_and_sealed_permissions=True,
        complete_three_mode_report_and_bad_denominator_rejection=True)


if __name__=='__main__':
    c=json.loads(Path(os.environ['Q157_CPU_CONFIG']).read_text())
    result=check(json.loads(Path(c['lock']).read_text()),c['temporary_root'])
    Path(c['receipt']).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
