"""Deterministic graph edge cases and reference optimizer regression (CPU)."""
import ast
import copy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'third_party/APART')]

def check_cleaning():
    from prepare_isic_transfer_v2 import clean
    def row(i,label,split,sha): return dict(sample_id=i,original_label=label,split=split,file_hash=sha)
    rows=[row('a',0,'train','x'),row('b',1,'test','x'),
          row('c',0,'train','y'),row('d',0,'test','y'),
          row('e',0,'train','z'),row('f',0,'train','z'),
          row('g',0,'train','w')]
    meta={i:dict(lesion_id=l) for i,l in [('a','A'),('b','NA'),('c','C'),('d',''),('e','E'),('f','F'),('g','F')]}
    r={r['sample_id']:r for r in clean(rows,meta)}
    assert r['a']['disposition']==r['b']['disposition']=='quarantine_label_conflict'
    assert r['c']['disposition']=='excluded_lower_split_priority'
    assert r['d']['disposition']=='quarantine_unknown_lesion'
    assert r['e']['disposition']=='retained' and r['f']['disposition']=='excluded_duplicate_content'
    assert r['e']['identity_component']==r['g']['identity_component']
    assert clean(rows,meta)==clean(rows,meta)

def check_components():
    import torch
    from torch import nn,optim
    from types import SimpleNamespace
    from utils.medical_v2 import extend_embedding,effective_optimizer,effective_scheduler
    from backbone.vision_transformer_adapter_pool_a import PoolAssigner
    p=PoolAssigner();x=torch.randn(501,768);indices=torch.arange(501)
    expected=p(x,indices).detach();before=torch.get_rng_state();extend_embedding(p,1993)
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(p(x,indices),expected)
    counts=torch.tensor([500,501,12725,10529,3263,2835,1306,458,256,44,27])
    p(torch.zeros(len(counts),768),counts).sum().backward()
    assert p.cls_emb.weight.grad[501:].abs().sum()>0
    try:p.cls_emb(torch.tensor([12726]))
    except IndexError:pass
    else:raise AssertionError('invalid index accepted')
    tree=ast.parse((ROOT/'third_party/APART/models/apart.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Learner')
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('get_optimizer','get_scheduler')]
    ns={'optim':optim};exec(compile(ast.Module(body=methods,type_ignores=[]),'reference','exec'),ns)
    for session in range(3):
        a=nn.Module();a.pool=nn.Linear(2,2);a.head=nn.Linear(2,2);b=copy.deepcopy(a)
        state=SimpleNamespace(_network=SimpleNamespace(backbone=a),init_lr=.003,weight_decay=0,min_lr=1e-5,args=dict(optimizer='adamw',scheduler='constant',init_cls=4,increment=2,tuned_epoch=10))
        original=ns['get_optimizer'](state);sch=ns['get_scheduler'](state,original)
        if session:original=ns['get_optimizer'](state)
        explicit=effective_optimizer(b);s2=effective_scheduler(explicit,session)
        for epoch in range(10):
            for model,opt in [(a,original),(b,explicit)]:
                opt.zero_grad();sum(p.square().sum() for p in model.parameters()).backward();opt.step()
            sch.step()
            if s2:s2.step()
            assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
            assert [g['lr'] for g in original.param_groups]==[g['lr'] for g in explicit.param_groups]

def check_metrics():
    import numpy as np
    from evaluate_medical_v2 import metrics
    targets=np.array([0,0,0,1,2,3]);pred=np.array([0,0,1,1,2,3])
    raw=np.eye(4)[pred]*4
    rows=[dict(lesion_id=x,identity_component=x) for x in ('a','a','b','c','d','e')]
    m,pc=metrics(raw,targets,rows,[0,1,2,3],2,dict(head_rank2=[0,1],mid_rank4=[],tail_rank2=[2,3]))
    assert np.isclose(m['balanced_accuracy'],100*11/12)
    assert np.isclose(m['accuracy'],100*5/6)
    assert np.isclose(m['lesion_equal_balanced_accuracy'],87.5)
    assert m['current_to_old_count']==m['current_to_wrong_current_count']==0
    assert m['correct_current_rate']==100 and m['old_to_current_count']==0
    assert m['old_tail_rank2'] is None and m['current_tail_rank2']==100
    assert len(pc)==4 and m['macro_ovr_auroc'] is not None

def check_aggregation():
    import csv
    import tempfile
    import numpy as np
    from evaluate_medical_v2 import metrics,aggregate
    sessions=[];classes=[];groups=dict(head_rank2=[0,1],mid_rank4=[2,3,4,5],tail_rank2=[6,7])
    for seed in (1993,1994,1995):
        for variant in ('M0','M1','M2','M3'):
            for task,seen in enumerate((4,6,8)):
                y=np.repeat(np.arange(seen),20);pred=y.copy()
                errors=8-2*(seed-1992) if variant=='M3' else 8;pred[:errors]=1
                rows=[dict(lesion_id=str(i),identity_component=str(i)) for i in range(len(y))]
                m,pc=metrics(np.eye(seen)[pred]*4,y,rows,list(range(8)),(0,4,6)[task],groups)
                common=dict(train_seed=seed,eval_variant=variant,session=task,old_classes=(0,4,6)[task])
                sessions.append(dict(**common,**m));classes += [dict(**common,**c) for c in pc]
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);result=aggregate(sessions,classes,out,groups)
        paired=list(csv.DictReader((out/'paired_results.csv').open()))
        r=next(r for r in paired if r['metric']=='Final_BA' and r['contrast']=='M3-M0' and r['train_seed']=='paired_summary')
        assert np.isclose(float(r['mean']),2.5) and np.isclose(float(r['std_ddof1']),1.25)
        assert result['status']=='COMPLETE_MIXED_SIGNAL' and len(classes)==216 and len(sessions)==36
        r=next(r for r in paired if r['metric']=='old_tail_rank2' and r['contrast']=='M3-M0' and r['train_seed']=='paired_summary')
        assert r['mean']==r['std_ddof1']==''

def check_checkpoint_rejections():
    import torch
    from unittest.mock import patch
    from run_medical_v2 import MedicalLearner,WEIGHT_SHA
    learner=MedicalLearner.__new__(MedicalLearner)
    learner.order=list(range(8));learner.config={'code_sha256':{'fixture':'fixed'}}
    learner.manifest_hashes={'train':'fixed'};learner.protocol_hash='fixed';learner.concm_stage1_enabled=False
    learner.args={'seed':1993,'device':[torch.device('cpu')]}
    baseline=dict(order=learner.order,code_sha256=learner.config['code_sha256'],weight_sha256=WEIGHT_SHA,
                  manifest_sha256=learner.manifest_hashes,protocol_sha256='fixed',train_seed=1993,train_branch='B',training_args={'seed':1993,'device':['cpu']})
    for field,value in [('protocol_sha256','changed'),('manifest_sha256',{}),('train_seed',1994),('train_branch','C'),('training_args',{})]:
        state=dict(baseline);state[field]=value
        try:
            with patch('torch.load',return_value=state):learner.restore('unused')
        except AssertionError:pass
        else:raise AssertionError('accepted mismatched '+field)

if __name__=='__main__':
    check_cleaning();check_components();check_metrics();check_aggregation();check_checkpoint_rejections()
    print('CLEANING_CAPACITY_OPTIMIZER_METRICS_CHECKPOINT_CPU_PASS')
